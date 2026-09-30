# Lightning Waveform Classification Pipeline

A portfolio version of the machine-learning component I worked on for **DECO3801 at the University of Queensland**. The wider team project used data from a real 18-detector lightning network in Brisbane and South East Queensland and aimed to distinguish lightning ADC waveforms from noise before downstream localisation and visualisation.

> **Portfolio scope:** this repository focuses on the ML/backend work. The original university project was completed by a seven-person cross-functional team spanning machine learning, frontend, databases, processing pipelines and localisation. This repository should not be read as a claim that I built the entire team system independently.

## What I worked on

My work focused on the machine-learning side of the project: training and evaluating XGBoost, 1D-CNN and hybrid approaches; comparing model behaviour; calibration and reliability analysis; threshold selection; and integrating classification output into a Python/REST workflow used by the broader system.

This cleaned portfolio version demonstrates:

- waveform cleaning and normalisation;
- statistical, morphological and frequency-domain feature extraction;
- XGBoost classification on engineered features;
- a PyTorch 1D-CNN for raw waveform learning;
- a hybrid model using CNN embeddings with an XGBoost classifier;
- stratified train/validation/test splitting and train-only augmentation;
- precision, recall, F1, Brier score and expected calibration error evaluation;
- probability calibration and validation-set threshold tuning;
- model persistence and reusable inference functions;
- a Flask REST API with `/predict` and `/health` endpoints.

## Architecture

```text
Raw ADC waveform
      |
      v
Cleaning / quality checks
      |
      +--------------------+
      |                    |
      v                    v
Engineered features     1D-CNN
      |                    |
      v                    +------> standalone CNN score
   XGBoost                 |
      |                    v
      |              128-d embedding
      |                    |
      |                    v
      |                 XGBoost
      |                    |
      +----------+---------+
                 v
       calibrated probability
                 |
                 v
          Flask /predict API
```

## Model comparison

The supplied project README for this code version recorded the following held-out results:

| Model | F1 | Confidence gap | ECE |
|---|---:|---:|---:|
| XGBoost (engineered features) | 0.948 | 0.838 | 0.025 |
| 1D-CNN | 0.983 | 0.908 | 0.024 |
| Hybrid CNN → XGBoost | **0.988** | **0.946** | **0.009** |

These figures are included as results from the project version supplied with this portfolio. The original training dataset is not included in this public repository.

## Repository structure

```text
.
├── app.py                 # Flask inference API
├── ml_pipeline.py         # training, evaluation, persistence and inference
├── utils.py               # reusable waveform/model utilities
├── tests/
│   ├── test_api.py        # API smoke test
│   └── test_real.py       # optional local PostgreSQL integration test
├── models/                # generated model artefacts (gitignored)
├── outputs/               # generated evaluation outputs (gitignored)
├── .env.example           # example local configuration
├── .gitignore
└── requirements.txt
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Place an authorised training CSV locally (not in Git) and provide its path:

```bash
# Windows PowerShell
$env:LIGHTNING_DATA_PATH="data/dataset_v5.csv"

# macOS/Linux
# export LIGHTNING_DATA_PATH="data/dataset_v5.csv"
```

## Train

```bash
python ml_pipeline.py --data data/dataset_v5.csv
```

Training creates model artefacts in `models/` and evaluation output in `outputs/`. Both directories are excluded from Git because model weights and project data should not be committed by default.

## Run the API

After training:

```bash
python app.py
```

Health check:

```text
GET http://localhost:5000/health
```

Prediction endpoint:

```text
POST http://localhost:5000/predict
Content-Type: application/json
```

Example request:

```json
{
  "id": 13,
  "starttime": "2025-10-31 18:46:50",
  "data": [1024, 1031, 1018],
  "model": "hybrid"
}
```

In the real project, waveforms contain 728 ADC samples. The shortened array above is only illustrative documentation.

Example response shape:

```json
{
  "id": 13,
  "starttime": "2025-10-31 18:46:50",
  "confidence": 0.9856,
  "label": "lightning",
  "model": "hybrid",
  "quality": {
    "is_saturated": false,
    "is_flat": false
  }
}
```

## Optional local database test

`tests/test_real.py` can pull a sample from a **local** PostgreSQL database and submit it to the API. Database credentials are read from environment variables; no passwords are stored in the repository.

```bash
# PowerShell example
$env:DB_HOST="localhost"
$env:DB_NAME="lightning"
$env:DB_USER="postgres"
$env:DB_PASSWORD="your-local-password"
python tests/test_real.py
```

## Security and data notes

This portfolio version intentionally excludes the original dataset, trained weights, private infrastructure details and credentials. Local file paths and database passwords from development have been replaced with configuration/environment variables. Only publish project data or model artefacts if you have permission to do so.

## Tech stack

**Python · PyTorch · XGBoost · scikit-learn · pandas · NumPy · SciPy · Flask · PostgreSQL · REST APIs**
