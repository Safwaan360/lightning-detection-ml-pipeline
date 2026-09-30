"""
DECO3801 Lightning Detection — ML Pipeline
==========================================
Models:
  A — XGBoost on hand-crafted features (best: F1=0.954, Gap=0.785, ECE=0.039)
  B — Standalone 1D-CNN
  C — Hybrid: CNN encoder → XGBoost

Entry points:
  train()               — train all models on dataset_v5.csv, save weights
  run_inference()       — score raw waveforms, return DataFrame
  run_inference_json()  — score single waveform, return JSON string (for Flask/TDoA)

Usage:
  python ml_pipeline.py              # train all models
  python ml_pipeline.py --infer     # demo inference on one sample from DB
"""

import argparse
import ast
import csv
import json
import os
import warnings
import joblib

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss, classification_report, f1_score,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH       = os.getenv("LIGHTNING_DATA_PATH", "data/dataset_v5.csv")
OUTPUT_DIR      = "outputs"
MODEL_DIR       = "models"          # saved weights go here

SAMPLE_RATE_MHZ = 2.7              # ADC sample rate
ADC_MAX         = 4090             # 12-bit saturation threshold
FLAT_STD_THRESH = 50               # waveforms with std below this = flat
RANDOM_STATE    = 42

CNN_EPOCHS      = 30
CNN_LR          = 3e-4
CNN_BATCH       = 64
EARLY_STOP_PAT  = 7
XGB_N_ROUNDS    = 500
XGB_EARLY_STOP  = 30

SAMPLE_DT = 1.0 / (SAMPLE_RATE_MHZ * 1e6)   # seconds per sample

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def is_saturated(waveform, threshold=ADC_MAX):
    """Flag waveforms where any sample hits the ADC ceiling."""
    return max(waveform) >= threshold


def is_flat(waveform, std_threshold=FLAT_STD_THRESH):
    """Flag near-zero / flat waveforms (dead channels or mislabels)."""
    return np.std(waveform) < std_threshold


def normalise(waveform):
    """Mean-centre and scale to [-1, 1]."""
    arr = np.array(waveform, dtype=np.float32)
    arr = arr - arr.mean()
    mx  = np.abs(arr).max()
    if mx > 0:
        arr = arr / mx
    return arr


def is_glitch(waveform_norm, kurt_threshold=1000):
    """Catch near-flat waveforms with isolated spikes that survive the raw std filter."""
    return float(pd.Series(waveform_norm).kurt()) > kurt_threshold


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(waveform_norm):
    """
    Extract Tier 1 (statistical) + Tier 2 (morphological) + Tier 3 (frequency) features.
    Input:  normalised waveform (float32 array, mean-centred, scaled to [-1, 1])
    Returns: dict of 22 scalar features
    """
    arr = np.array(waveform_norm, dtype=np.float64)
    n   = len(arr)
    f   = {}

    # ── Tier 1: Statistical ──────────────────────────────────────────────────
    f["rms"]             = np.sqrt(np.mean(arr ** 2))
    f["std"]             = np.std(arr)
    f["skewness"]        = float(pd.Series(arr).skew())
    f["kurtosis"]        = float(pd.Series(arr).kurt())
    f["peak_amp"]        = np.abs(arr).max()
    f["crest_factor"]    = f["peak_amp"] / (f["rms"] + 1e-9)
    f["energy"]          = np.sum(arr ** 2)
    zcr                  = np.sum(np.diff(np.sign(arr)) != 0)
    f["zero_cross_rate"] = zcr / n

    # ── Tier 2: Morphological ────────────────────────────────────────────────
    peak_idx           = int(np.argmax(np.abs(arr)))
    f["peak_idx_norm"] = peak_idx / n

    pre_peak  = arr[:peak_idx + 1]
    peak_val  = arr[peak_idx]
    lo_thresh = 0.1 * abs(peak_val)
    hi_thresh = 0.9 * abs(peak_val)
    lo_idxs   = np.where(np.abs(pre_peak) >= lo_thresh)[0]
    hi_idxs   = np.where(np.abs(pre_peak) >= hi_thresh)[0]
    rise_time_samples = (
        (hi_idxs[0] - lo_idxs[0])
        if (len(lo_idxs) > 0 and len(hi_idxs) > 0) else 0
    )
    f["rise_time_us"] = rise_time_samples * SAMPLE_DT * 1e6

    post_peak  = arr[peak_idx:]
    half_idxs  = np.where(np.abs(post_peak) <= 0.5 * abs(peak_val))[0]
    f["fall_time_us"] = (
        half_idxs[0] * SAMPLE_DT * 1e6 if len(half_idxs) > 0
        else n * SAMPLE_DT * 1e6
    )

    pre_energy  = np.sum(arr[:peak_idx] ** 2)
    post_energy = np.sum(arr[peak_idx:] ** 2)
    f["pre_peak_energy_ratio"] = pre_energy / (f["energy"] + 1e-9)
    f["asymmetry"]             = (pre_energy - post_energy) / (f["energy"] + 1e-9)

    peaks, _ = find_peaks(np.abs(arr), height=0.2)
    f["n_peaks"] = len(peaks)

    half_max = 0.5 * abs(peak_val)
    above    = np.where(np.abs(arr) >= half_max)[0]
    f["half_width_samples"] = (above[-1] - above[0]) if len(above) > 1 else 0

    # ── Tier 3: Frequency-domain ─────────────────────────────────────────────
    fft_mag  = np.abs(rfft(arr))
    freqs    = rfftfreq(n, d=SAMPLE_DT)
    total_pw = np.sum(fft_mag ** 2) + 1e-9

    f["spectral_centroid_khz"] = np.sum(freqs * fft_mag ** 2) / total_pw / 1e3

    bands = [(0, 10e3), (10e3, 100e3), (100e3, 500e3), (500e3, np.inf)]
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        f[f"band_energy_{i}"] = np.sum(fft_mag[mask] ** 2) / total_pw

    f["dominant_freq_khz"] = freqs[np.argmax(fft_mag)] / 1e3

    cumsum      = np.cumsum(fft_mag ** 2)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total_pw)
    f["spectral_rolloff_khz"] = freqs[min(rolloff_idx, len(freqs) - 1)] / 1e3

    return f


# ─────────────────────────────────────────────────────────────────────────────
# CNN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class WaveformDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(
            np.stack(df["data_norm"].values), dtype=torch.float32
        ).unsqueeze(1)
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LightningCNN(nn.Module):
    """
    1D-CNN encoder + MLP classifier.
    Call .encode(x) to get the 128-d embedding (used by the hybrid model).
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 91, 128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def encode(self, x):
        return self.fc1(self.encoder(x))

    def forward(self, x):
        return self.fc2(self.encode(x)).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: weighted average calibration gap across equal-frequency bins."""
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)


def tune_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on the given set."""
    thresholds = np.arange(0.05, 0.95, 0.05)
    f1s = [
        f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = thresholds[int(np.argmax(f1s))]
    return best_t, max(f1s)


def evaluate_model(name, y_true, y_prob, val_y_true, val_y_prob):
    """Print full evaluation metrics and return summary dict."""
    gap = y_prob[y_true == 1].mean() - y_prob[y_true == 0].mean()
    ece = expected_calibration_error(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)

    best_thresh, _ = tune_threshold(val_y_true, val_y_prob)
    y_pred = (y_prob > best_thresh).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["Noise", "Lightning"],
        output_dict=True, zero_division=0,
    )
    prec = report["Lightning"]["precision"]
    rec  = report["Lightning"]["recall"]
    f1   = report["Lightning"]["f1-score"]

    print(f"\n  ── {name} ──")
    print(f"  Gap: {gap:.3f} | ECE: {ece:.3f} | Brier: {bs:.3f}")
    print(f"  Best threshold (val F1): {best_thresh:.2f}")
    print(classification_report(
        y_true, y_pred, target_names=["Noise", "Lightning"], zero_division=0
    ))

    return {
        "name": name, "precision": prec, "recall": rec, "f1": f1,
        "gap": gap, "ece": ece, "brier": bs, "threshold": best_thresh,
        "y_prob": y_prob, "y_true": y_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _train_cnn(df_tr, df_vl, model, device, tag="CNN"):
    pos_weight = torch.tensor(
        [(df_tr["label"] == 0).sum() / (df_tr["label"] == 1).sum()],
        dtype=torch.float32,
    ).to(device)
    print(f"\n  [{tag}] pos_weight: {pos_weight.item():.2f}")

    def loss_fn(p, y):
        w = torch.where(y == 1, pos_weight, torch.ones_like(y))
        return nn.functional.binary_cross_entropy(p, y, weight=w)

    opt   = torch.optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    tr_dl = DataLoader(WaveformDataset(df_tr), batch_size=CNN_BATCH, shuffle=True)
    vl_dl = DataLoader(WaveformDataset(df_vl), batch_size=CNN_BATCH)

    best_val, pat_ctr = float("inf"), 0
    save_path = os.path.join(MODEL_DIR, f"{tag}_best.pt")

    for epoch in range(CNN_EPOCHS):
        model.train()
        tr_loss = 0
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        vl_loss = 0
        with torch.no_grad():
            for Xb, yb in vl_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                vl_loss += loss_fn(model(Xb), yb).item()

        tr_loss /= len(tr_dl)
        vl_loss /= len(vl_dl)
        sched.step(vl_loss)
        print(f"  [{tag}] Epoch {epoch+1:02d} | Train: {tr_loss:.4f} | Val: {vl_loss:.4f}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), save_path)
            pat_ctr = 0
        else:
            pat_ctr += 1
            if pat_ctr >= EARLY_STOP_PAT:
                print(f"  [{tag}] Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(save_path))
    return model


def _get_cnn_probs(model, df, device):
    dl = DataLoader(WaveformDataset(df), batch_size=CNN_BATCH)
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for Xb, yb in dl:
            probs.extend(model(Xb.to(device)).cpu().numpy())
            targets.extend(yb.numpy().astype(int))
    return np.array(probs), np.array(targets)


def _get_cnn_embeddings(model, df, device):
    dl = DataLoader(WaveformDataset(df), batch_size=CNN_BATCH)
    model.eval()
    embs, targets = [], []
    with torch.no_grad():
        for Xb, yb in dl:
            embs.extend(model.encode(Xb.to(device)).cpu().numpy())
            targets.extend(yb.numpy().astype(int))
    return np.array(embs), np.array(targets)


def _augment_waveform(waveform_norm, n=2):
    augmented = []
    arr = np.array(waveform_norm, dtype=np.float32)
    for _ in range(n):
        aug  = arr.copy()
        aug += np.random.normal(0, 0.02, len(aug))
        aug *= np.random.uniform(0.9, 1.1)
        aug  = np.roll(aug, np.random.randint(-20, 20))
        augmented.append(aug)
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL STATE  (populated by train(), used by run_inference*)
# ─────────────────────────────────────────────────────────────────────────────
_state = {}


def train(data_path=DATA_PATH):
    """
    Full training pipeline. Trains all three models, saves weights to models/.
    Populates the module-level _state dict so run_inference* can be called
    without retraining.
    """
    global _state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    df_raw = pd.read_csv(
        data_path, quoting=csv.QUOTE_ALL, on_bad_lines="skip", engine="python"
    )
    df_raw["data"]  = df_raw["data"].apply(ast.literal_eval)
    df_raw["label"] = df_raw["label"].map({"lightning": 1, "noise": 0})
    df_raw = df_raw[["data", "label"]].dropna().copy()
    print(f"  Raw — pos: {(df_raw['label']==1).sum()}, neg: {(df_raw['label']==0).sum()}")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. DATA CLEANING")
    print("=" * 60)

    before = len(df_raw)
    df_raw["saturated"] = df_raw["data"].apply(is_saturated)
    df_raw["flat"]      = df_raw["data"].apply(is_flat)

    for label, name in [(1, "Lightning"), (0, "Noise")]:
        sub = df_raw[df_raw["label"] == label]
        print(f"  {name}:")
        print(f"    Saturated : {sub['saturated'].sum()} / {len(sub)}")
        print(f"    Flat      : {sub['flat'].sum()} / {len(sub)}")

    df_clean = df_raw[~df_raw["saturated"] & ~df_raw["flat"]].copy()
    print(f"\n  After raw filters — removed: {before - len(df_clean)}")

    df_pos = df_clean[df_clean["label"] == 1].copy()
    df_neg = df_clean[df_clean["label"] == 0].copy()

    df_pos["data_norm"] = df_pos["data"].apply(normalise)
    df_neg["data_norm"] = df_neg["data"].apply(normalise)
    df_pos["glitch"]    = df_pos["data_norm"].apply(is_glitch)
    df_neg["glitch"]    = df_neg["data_norm"].apply(is_glitch)

    df_pos = df_pos[~df_pos["glitch"]].copy()
    df_neg = df_neg[~df_neg["glitch"]].copy()
    print(f"  After glitch filter — pos: {len(df_pos)}, neg: {len(df_neg)}")

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. FEATURE EXTRACTION")
    print("=" * 60)

    all_df    = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    feat_list = [extract_features(row["data_norm"]) for _, row in all_df.iterrows()]
    df_feat   = pd.DataFrame(feat_list)
    df_feat["label"] = all_df["label"].values

    feature_cols = [c for c in df_feat.columns if c != "label"]
    print(f"  Features extracted: {len(feature_cols)}")

    # ── 4. Split ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. SPLITTING DATA")
    print("=" * 60)

    idx = np.arange(len(all_df))
    lbl = all_df["label"].values

    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.2, stratify=lbl, random_state=RANDOM_STATE
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.2,
        stratify=lbl[idx_trainval], random_state=RANDOM_STATE,
    )

    def subset(df, idxs):
        return df.iloc[idxs].reset_index(drop=True)

    wv_train = subset(all_df, idx_train)
    wv_val   = subset(all_df, idx_val)
    wv_test  = subset(all_df, idx_test)
    ft_train = subset(df_feat, idx_train)
    ft_val   = subset(df_feat, idx_val)
    ft_test  = subset(df_feat, idx_test)

    print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

    # ── 5. Augmentation (train positives only) ───────────────────────────────
    pos_train_df = wv_train[wv_train["label"] == 1]
    aug_rows = []
    for _, row in pos_train_df.iterrows():
        for aug_data in _augment_waveform(row["data_norm"], n=2):
            aug_rows.append({"data_norm": aug_data, "label": 1})
    df_aug = pd.DataFrame(aug_rows)

    def make_wv_split(wv_df, aug=None):
        parts = [wv_df[["data_norm", "label"]]]
        if aug is not None:
            parts.append(aug[["data_norm", "label"]])
        return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    df_train_wv = make_wv_split(wv_train, df_aug)
    df_val_wv   = make_wv_split(wv_val)
    df_test_wv  = make_wv_split(wv_test)

    print(f"\n  After augmentation — train: {len(df_train_wv)} (pos: {df_train_wv['label'].sum()})")

    # ── 6. Model A — XGBoost ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL A: XGBoost on hand-crafted features")
    print("=" * 60)

    X_train_ft = ft_train[feature_cols].values
    y_train_ft = ft_train["label"].values
    X_val_ft   = ft_val[feature_cols].values
    y_val_ft   = ft_val["label"].values
    X_test_ft  = ft_test[feature_cols].values
    y_test_ft  = ft_test["label"].values

    scaler_ft = StandardScaler()
    X_train_ft_sc = scaler_ft.fit_transform(X_train_ft)
    X_val_ft_sc   = scaler_ft.transform(X_val_ft)
    X_test_ft_sc  = scaler_ft.transform(X_test_ft)

    xgb_A = xgb.XGBClassifier(
        n_estimators          = XGB_N_ROUNDS,
        max_depth             = 5,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = (y_train_ft == 0).sum() / (y_train_ft == 1).sum(),
        eval_metric           = "aucpr",
        early_stopping_rounds = XGB_EARLY_STOP,
        random_state          = RANDOM_STATE,
        use_label_encoder     = False,
        verbosity             = 0,
    )
    xgb_A.fit(
        X_train_ft_sc, y_train_ft,
        eval_set=[(X_val_ft_sc, y_val_ft)],
        verbose=False,
    )

    cal_A = CalibratedClassifierCV(xgb_A, method="sigmoid", cv=5)
    cal_A.fit(X_train_ft_sc, y_train_ft)

    prob_A_test = cal_A.predict_proba(X_test_ft_sc)[:, 1]
    prob_A_val  = cal_A.predict_proba(X_val_ft_sc)[:, 1]
    results_A   = evaluate_model("XGBoost (features)", y_test_ft, prob_A_test, y_val_ft, prob_A_val)

    # ── 7. Model B — CNN ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL B: Standalone CNN")
    print("=" * 60)

    cnn_B = LightningCNN().to(device)
    cnn_B = _train_cnn(df_train_wv, df_val_wv, cnn_B, device, tag="CNN_B")

    prob_B_test, y_B_test = _get_cnn_probs(cnn_B, df_test_wv, device)
    prob_B_val,  y_B_val  = _get_cnn_probs(cnn_B, df_val_wv, device)
    results_B = evaluate_model("CNN (standalone)", y_B_test, prob_B_test, y_B_val, prob_B_val)

    # ── 8. Model C — Hybrid ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL C: Hybrid CNN → XGBoost")
    print("=" * 60)

    emb_train, y_emb_train = _get_cnn_embeddings(cnn_B, df_train_wv, device)
    emb_val,   y_emb_val   = _get_cnn_embeddings(cnn_B, df_val_wv, device)
    emb_test,  y_emb_test  = _get_cnn_embeddings(cnn_B, df_test_wv, device)

    xgb_C = xgb.XGBClassifier(
        n_estimators          = XGB_N_ROUNDS,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        scale_pos_weight      = (y_emb_train == 0).sum() / (y_emb_train == 1).sum(),
        eval_metric           = "aucpr",
        early_stopping_rounds = XGB_EARLY_STOP,
        random_state          = RANDOM_STATE,
        use_label_encoder     = False,
        verbosity             = 0,
    )
    xgb_C.fit(
        emb_train, y_emb_train,
        eval_set=[(emb_val, y_emb_val)],
        verbose=False,
    )

    cal_C = CalibratedClassifierCV(xgb_C, method="sigmoid", cv=5)
    cal_C.fit(emb_train, y_emb_train)

    prob_C_test = cal_C.predict_proba(emb_test)[:, 1]
    prob_C_val  = cal_C.predict_proba(emb_val)[:, 1]
    results_C   = evaluate_model("Hybrid CNN→XGBoost", y_emb_test, prob_C_test, y_emb_val, prob_C_val)

    # ── 9. Summary ───────────────────────────────────────────────────────────
    results_all = [results_A, results_B, results_C]
    summary = pd.DataFrame([{
        "Model"    : r["name"],
        "Precision": f"{r['precision']:.3f}",
        "Recall"   : f"{r['recall']:.3f}",
        "F1"       : f"{r['f1']:.3f}",
        "Gap"      : f"{r['gap']:.3f}",
        "ECE"      : f"{r['ece']:.3f}",
        "Brier"    : f"{r['brier']:.3f}",
        "Threshold": f"{r['threshold']:.2f}",
    } for r in results_all])

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    # ── Store everything needed for inference ─────────────────────────────────
    _state.update({
        "scaler_ft"   : scaler_ft,
        "feature_cols": feature_cols,
        "cal_A"       : cal_A,
        "cnn_B"       : cnn_B,
        "cal_C"       : cal_C,
        "device"      : device,
        "results_all" : results_all,
        "all_df"      : all_df,       # kept for demo purposes only
    })

    # Persist the trained artefacts used by the API.
    joblib.dump(scaler_ft, os.path.join(MODEL_DIR, "scaler_ft.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
    joblib.dump(cal_A, os.path.join(MODEL_DIR, "cal_A.joblib"))
    joblib.dump(cal_C, os.path.join(MODEL_DIR, "cal_C.joblib"))
    torch.save(cnn_B.state_dict(), os.path.join(MODEL_DIR, "cnn_B_final.pt"))

    print(f"\nTraining complete. Model artefacts saved to {MODEL_DIR}/")
    return _state


def load_models(model_dir=MODEL_DIR):
    """Load previously trained artefacts into module-level state for inference."""
    global _state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_B = LightningCNN().to(device)
    cnn_B.load_state_dict(
        torch.load(os.path.join(model_dir, "cnn_B_final.pt"), map_location=device)
    )
    cnn_B.eval()
    _state.update({
        "scaler_ft": joblib.load(os.path.join(model_dir, "scaler_ft.joblib")),
        "feature_cols": joblib.load(os.path.join(model_dir, "feature_cols.joblib")),
        "cal_A": joblib.load(os.path.join(model_dir, "cal_A.joblib")),
        "cnn_B": cnn_B,
        "cal_C": joblib.load(os.path.join(model_dir, "cal_C.joblib")),
        "device": device,
    })
    return _state


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(raw_waveforms, model_choice="xgb"):
    """
    Score a list of raw waveforms and return calibrated confidence scores.

    Args:
        raw_waveforms : list of lists — raw ADC integer values, 728 samples each
        model_choice  : 'xgb'    → Model A (hand-crafted features, best overall)
                        'cnn'    → Model B (standalone CNN)
                        'hybrid' → Model C (CNN encoder + XGBoost)

    Returns:
        pd.DataFrame with columns:
            prob_lightning  — calibrated confidence score [0, 1]
            prediction      — 1 if lightning, 0 if noise (threshold = 0.5)
            is_saturated    — True if waveform hit ADC ceiling
            is_flat         — True if waveform was near-flat
    """
    assert _state, "Call train() first (or load_models()) before running inference."

    scaler_ft    = _state["scaler_ft"]
    feature_cols = _state["feature_cols"]
    cal_A        = _state["cal_A"]
    cnn_B        = _state["cnn_B"]
    cal_C        = _state["cal_C"]
    device       = _state["device"]

    results_inf = []
    for wv in raw_waveforms:
        sat  = is_saturated(wv)
        flat = is_flat(wv)
        norm = normalise(wv)

        if model_choice == "xgb":
            feats = extract_features(norm)
            X     = scaler_ft.transform(pd.DataFrame([feats])[feature_cols].values)
            prob  = cal_A.predict_proba(X)[0, 1]

        elif model_choice == "cnn":
            t_in = torch.tensor(norm).unsqueeze(0).unsqueeze(0).to(device)
            cnn_B.eval()
            with torch.no_grad():
                prob = cnn_B(t_in).item()

        elif model_choice == "hybrid":
            t_in = torch.tensor(norm).unsqueeze(0).unsqueeze(0).to(device)
            cnn_B.eval()
            with torch.no_grad():
                emb = cnn_B.encode(t_in).cpu().numpy()
            prob = cal_C.predict_proba(emb)[0, 1]

        else:
            raise ValueError(f"Unknown model_choice: {model_choice!r}")

        results_inf.append({
            "prob_lightning": prob,
            "prediction"    : int(prob >= 0.5),
            "is_saturated"  : sat,
            "is_flat"       : flat,
        })

    return pd.DataFrame(results_inf)


def run_inference_json(detector_id, starttime, data, model_choice="xgb"):
    """
    Score a single waveform and return a JSON string for the TDoA web system.

    Args:
        detector_id  : int   — detector number (e.g. 13)
        starttime    : str   — ISO timestamp string from the DB
        data         : list  — raw ADC integer values (728 samples)
        model_choice : str   — 'xgb' | 'cnn' | 'hybrid'

    Returns:
        JSON string with keys: id, clktrim, starttime, data, confidence
    """
    prob = run_inference([data], model_choice=model_choice)["prob_lightning"].iloc[0]

    result = {
        "id"        : detector_id,
        "clktrim"   : None,   # TODO: look up detector_info table
        "starttime" : str(starttime),
        "data"      : data if isinstance(data, list) else data.tolist(),
        "confidence": round(float(prob), 4),
    }
    return json.dumps(result)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _demo_inference():
    """Pull one real waveform from the local DB and run inference on it."""
    import psycopg2

    print("\nConnecting to local PostgreSQL...")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME", "lightning"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT id, detector, data, rtsecs FROM sample "
        "WHERE label = 'lightning' LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()

    sample_id, detector_id, raw_data, starttime = row
    waveform = list(map(int, raw_data.strip("{}").split(",")))

    print(f"\nSample id={sample_id}, detector={detector_id}")
    print(run_inference_json(detector_id, starttime, waveform, model_choice="xgb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECO3801 Lightning ML Pipeline")
    parser.add_argument(
        "--infer", action="store_true",
        help="After training, demo inference on one waveform from local DB"
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help="Path to dataset CSV (default: dataset_v5.csv)"
    )
    args = parser.parse_args()

    train(data_path=args.data)

    if args.infer:
        _demo_inference()