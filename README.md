<div align="center">

# PhishGuard AI

### AI-Powered Phishing Email Detection Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.41-FFD21E?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00d4ff?style=for-the-badge)

> A production-quality, multi-layer phishing detection system combining rule-based heuristics, machine learning, and transformer deep learning — with a cyberpunk-themed single-page UI.

</div>

---

## Overview

PhishGuard AI analyses every email through **three independent classifiers** and synthesises a weighted ensemble verdict with full explainability. Built as a cybersecurity portfolio project, it demonstrates end-to-end NLP, model training, REST API design, and frontend engineering.

| Classifier | Method | Accuracy | Notes |
|---|---|:---:|---|
| Rule-Based Heuristics | 14 hand-crafted rules, sigmoid scoring | — | Zero setup, always available |
| TF-IDF + Logistic Regression | 50k features, bigrams, class-weight balanced | **99.14%** | Fast, LIME-explainable |
| DistilBERT (fine-tuned) | HuggingFace Transformers, 3 epochs, fp16 | **99.72%** | Highest accuracy |

Trained on **158,577 real-world emails** from 7 public datasets.

---

## Features

- **Triple-layer ensemble** — Rule-Based + TF-IDF/LR + DistilBERT with weighted voting
- **Real-time explainability** — LIME token analysis with automatic TF-IDF coefficient fallback
- **Token highlighting** — colour-coded phishing indicators overlaid on the email body
- **Gmail integration** — OAuth2 inbox scanning and batch classification
- **Sample email library** — 10 pre-loaded phishing and safe emails for instant testing
- **Cyberpunk SPA UI** — Three.js 3D shield, tsParticles, GSAP animations, glassmorphism
- **4-sided live ticker** — continuous border ticker showing threat intelligence around the page
- **REST API** — FastAPI backend with full JSON responses and latency tracking

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PhishGuard AI                             │
│                                                                 │
│  Browser  ──►  FastAPI (src/app.py)  ──►  /api/analyze         │
│                         │                                       │
│             ┌───────────┼───────────┐                           │
│             ▼           ▼           ▼                           │
│        Rule-Based    ML Model   Transformer                     │
│        (rule_based) (ml_model)  (transformer_model)             │
│             │           │           │                           │
│             └───────────┴───────────┘                           │
│                         │                                       │
│                    compare.py                                   │
│             (ensemble + explainability)                         │
│                         │                                       │
│             ┌───────────┴──────────┐                            │
│        Agreement             LIME / Feature                     │
│        Scoring               Weights                            │
│                                                                 │
│  Gmail Inbox  ──►  gmail_fetch.py  ──►  batch classify          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
PhishGuard-AI/
│
├── src/                          # Backend source code
│   ├── app.py                    # FastAPI application (serves UI + REST API)
│   ├── compare.py                # Multi-classifier ensemble engine
│   ├── rule_based.py             # Heuristic rule engine (14 rule categories)
│   ├── ml_model.py               # TF-IDF + Logistic Regression classifier
│   ├── transformer_model.py      # DistilBERT fine-tuning + inference
│   ├── explainability.py         # LIME + TF-IDF feature explanations + token highlighting
│   ├── prepare_dataset.py        # Merges 7 source CSVs → emails.csv
│   ├── gmail_fetch.py            # Gmail API OAuth2 integration
│   └── __init__.py
│
├── templates/
│   └── index.html                # Cyberpunk SPA (Three.js, tsParticles, GSAP)
│
├── sample_emails/                # 5 phishing + 5 safe demo emails (.txt)
│
├── notebooks/
│   └── colab_train_distilbert.ipynb   # GPU training notebook (Colab T4, ~23 min)
│
├── dataset/                      # Raw source CSVs (not committed — see Datasets section)
│   ├── CEAS_08.csv
│   ├── Enron.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   ├── phishing_email.csv
│   └── SpamAssasin.csv
│
├── data/
│   └── processed/                # emails.csv generated by prepare_dataset.py
│
├── models/                       # Trained model artefacts (not committed — large files)
│   ├── tfidf_vectorizer.joblib
│   ├── lr_classifier.joblib
│   └── distilbert-phishing/      # Fine-tuned DistilBERT weights
│
├── tests/
│   └── test_classifiers.py       # pytest suite (no trained models required)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/zaidxahmed-cyber/PhishGuard-AI.git
cd PhishGuard-AI
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Default values work for local development
```

### 3. Run the server

```bash
uvicorn src.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

> The **rule-based classifier** works immediately with no training required. Load sample emails from the UI dropdown to test it. The ML and DistilBERT classifiers show `"unavailable"` until models are trained and placed in `models/`.

---

## Training the Models

### Datasets

Place the following CSVs in the `dataset/` folder before running `prepare_dataset.py`:

| File | Source | Description |
|---|---|---|
| `CEAS_08.csv` | CEAS 2008 | Spam challenge dataset |
| `Enron.csv` | Kaggle | Enron corporate email corpus |
| `Ling.csv` | Kaggle | Ling spam dataset |
| `Nazario.csv` | GitHub | Phishing emails by Jose Nazario |
| `Nigerian_Fraud.csv` | Kaggle | Nigerian advance-fee fraud emails |
| `phishing_email.csv` | Kaggle | Combined phishing email text |
| `SpamAssasin.csv` | Apache | SpamAssassin public corpus |

After merging and deduplication: **158,577 rows** — 50.7% phishing / 49.3% safe.

### Step 1 — Merge datasets

```bash
python -m src.prepare_dataset
```

Outputs `data/processed/emails.csv`.

### Step 2 — Train TF-IDF + Logistic Regression (~2 min on CPU)

```bash
python -m src.ml_model --train
```

Saves `models/tfidf_vectorizer.joblib` and `models/lr_classifier.joblib`.

### Step 3 — Fine-tune DistilBERT

**Option A — Google Colab (recommended, ~23 min on T4 GPU):**

1. Open `notebooks/colab_train_distilbert.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Run all cells
4. Download the output `distilbert-phishing.zip`
5. Extract into `models/distilbert-phishing/`

**Option B — Local CPU (slow, ~85 min with subsampling):**

```bash
python -m src.transformer_model --train --max-samples 10000 --epochs 3 --max-length 64
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the PhishGuard AI frontend |
| `/api/analyze` | POST | Analyse a single email |
| `/api/sample-emails` | GET | List available sample emails |
| `/api/sample-emails/{filename}` | GET | Get content of a sample email |
| `/api/gmail/fetch` | POST | Fetch and classify Gmail inbox |
| `/api/health` | GET | Service health and model status |

### POST /api/analyze

**Request:**
```json
{
  "subject": "URGENT: Verify your account",
  "body": "Click here to verify: http://bit.ly/fake",
  "sender": "support@paypal-secure.gmail.com",
  "include_explanation": true
}
```

**Response:**
```json
{
  "rule_based":  { "label": "phishing", "confidence": 0.87, "triggered_rules": [...] },
  "ml_model":    { "label": "phishing", "confidence": 0.99 },
  "transformer": { "label": "phishing", "confidence": 1.00 },
  "ensemble": {
    "ensemble_label": "phishing",
    "ensemble_confidence": 0.95,
    "threat_score": 95,
    "agreement_level": "full"
  },
  "explanation": {
    "rule_based": { "rules_fired": [...], "top_rule": {...} },
    "ml_model":   { "features": [...], "top_phishing_tokens": [...] },
    "token_highlights": [{ "text": "Click here", "type": "danger" }, ...]
  },
  "latency_ms": { "rule_based": 4, "ml_model": 18, "transformer": 142, "total": 145 }
}
```

---

## Gmail API Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the **Gmail API**
3. Create OAuth2 credentials → **Desktop app**
4. Download `credentials.json` and place it in the project root
5. On the first `/api/gmail/fetch` call, a browser window opens for OAuth consent

After consent, `token.json` is saved automatically for future runs.

> **Security:** Never commit `credentials.json` or `token.json` — both are in `.gitignore`.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover rule logic, text cleaning, ensemble agreement, and explainability. No trained models required.

---

## Benchmark Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Rule-Based Heuristics | — | — | — | — | — |
| TF-IDF + Logistic Regression | 99.14% | 98.95% | 99.35% | 99.15% | 99.94% |
| DistilBERT (fine-tuned, 3 epochs) | **99.72%** | — | — | **99.72%** | **99.99%** |

> DistilBERT trained on full 158,577 email dataset using Google Colab T4 GPU (fp16, ~23 minutes).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| ML | scikit-learn — TF-IDF, Logistic Regression |
| Deep Learning | HuggingFace Transformers, DistilBERT, PyTorch |
| Explainability | LIME, TF-IDF coefficient analysis |
| Gmail | google-auth-oauthlib, google-api-python-client |
| Frontend | Three.js, tsParticles, GSAP, CSS Glassmorphism |
| Fonts | Orbitron, Space Grotesk (Google Fonts) |
| Testing | pytest |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Uvicorn host |
| `PORT` | `8000` | Uvicorn port |
| `LOG_LEVEL` | `info` | Logging level |
| `USE_LIME` | `false` | Enable LIME (slower but more accurate explanations) |
| `RELOAD` | `true` | Auto-reload on file changes |

Copy `.env.example` to `.env` to configure.

---

## Author

**Zaid Ahmed**
- GitHub: [@zaidxahmed-cyber](https://github.com/zaidxahmed-cyber)
- Email: zaidahmed78654@gmail.com

---



<div align="center">
  <sub>Built as a cybersecurity portfolio project demonstrating NLP, model explainability, and production-quality API design.</sub>
</div>
