import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

SAMPLE_RATE_MHZ = 2.7
ADC_MAX = 4090
FLAT_STD_THRESH = 50
SAMPLE_DT = 1.0 / (SAMPLE_RATE_MHZ * 1e6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_saturated(waveform, threshold=ADC_MAX):
    return max(waveform) >= threshold

def is_flat(waveform, std_threshold=FLAT_STD_THRESH):
    return np.std(waveform) < std_threshold

def normalise(waveform):
    arr = np.array(waveform, dtype=np.float32)
    arr = arr - arr.mean()
    mx = np.abs(arr).max()
    if mx > 0:
        arr = arr / mx
    return arr

def extract_features(waveform_norm):
    arr = np.array(waveform_norm, dtype=np.float64)
    n = len(arr)
    f = {}
    f['rms'] = np.sqrt(np.mean(arr ** 2))
    f['std'] = np.std(arr)
    f['skewness'] = float(pd.Series(arr).skew())
    f['kurtosis'] = float(pd.Series(arr).kurt())
    f['peak_amp'] = np.abs(arr).max()
    f['crest_factor'] = f['peak_amp'] / (f['rms'] + 1e-9)
    f['energy'] = np.sum(arr ** 2)
    zcr = np.sum(np.diff(np.sign(arr)) != 0)
    f['zero_cross_rate'] = zcr / n
    peak_idx = int(np.argmax(np.abs(arr)))
    f['peak_idx_norm'] = peak_idx / n
    pre_peak = arr[:peak_idx + 1]
    peak_val = arr[peak_idx]
    lo_thresh = 0.1 * abs(peak_val)
    hi_thresh = 0.9 * abs(peak_val)
    lo_idxs = np.where(np.abs(pre_peak) >= lo_thresh)[0]
    hi_idxs = np.where(np.abs(pre_peak) >= hi_thresh)[0]
    rise_time_samples = (hi_idxs[0] - lo_idxs[0]) if (len(lo_idxs) > 0 and len(hi_idxs) > 0) else 0
    f['rise_time_us'] = rise_time_samples * SAMPLE_DT * 1e6
    post_peak = arr[peak_idx:]
    half_idxs = np.where(np.abs(post_peak) <= 0.5 * abs(peak_val))[0]
    f['fall_time_us'] = (half_idxs[0] * SAMPLE_DT * 1e6) if len(half_idxs) > 0 else n * SAMPLE_DT * 1e6
    pre_energy = np.sum(arr[:peak_idx] ** 2)
    post_energy = np.sum(arr[peak_idx:] ** 2)
    f['pre_peak_energy_ratio'] = pre_energy / (f['energy'] + 1e-9)
    f['asymmetry'] = (pre_energy - post_energy) / (f['energy'] + 1e-9)
    peaks, _ = find_peaks(np.abs(arr), height=0.2)
    f['n_peaks'] = len(peaks)
    half_max = 0.5 * abs(peak_val)
    above = np.where(np.abs(arr) >= half_max)[0]
    f['half_width_samples'] = (above[-1] - above[0]) if len(above) > 1 else 0
    fft_mag = np.abs(rfft(arr))
    freqs = rfftfreq(n, d=SAMPLE_DT)
    total_pw = np.sum(fft_mag ** 2) + 1e-9
    f['spectral_centroid_khz'] = np.sum(freqs * fft_mag ** 2) / total_pw / 1e3
    bands = [(0, 10e3), (10e3, 100e3), (100e3, 500e3), (500e3, np.inf)]
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        f[f'band_energy_{i}'] = np.sum(fft_mag[mask] ** 2) / total_pw
    f['dominant_freq_khz'] = freqs[np.argmax(fft_mag)] / 1e3
    cumsum = np.cumsum(fft_mag ** 2)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total_pw)
    f['spectral_rolloff_khz'] = freqs[min(rolloff_idx, len(freqs) - 1)] / 1e3
    return f

class LightningCNN(nn.Module):
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
            nn.Linear(64 * 91, 128), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def encode(self, x):
        return self.fc1(self.encoder(x))

    def forward(self, x):
        return self.fc2(self.encode(x)).squeeze(1)