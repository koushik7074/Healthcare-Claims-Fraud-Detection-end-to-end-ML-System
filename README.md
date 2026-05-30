# Healthcare Claims Fraud Risk-Ranking System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi) ![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker) ![AWS](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikitlearn) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end, production-ready machine learning system that ranks healthcare providers by fraud risk using behavioral feature engineering, interpretable ML models, and a deployed REST API inference pipeline.

<img width="776" height="499" alt="image" src="https://github.com/user-attachments/assets/da203a67-be50-4851-ada7-749aecd0d79e" />

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Usage](#api-usage)
- [Model Explainability](#model-explainability)
- [Monitoring & Evaluation Strategy](#monitoring--evaluation-strategy)
- [Future Improvements](#future-improvements)

---

## Problem Statement

Healthcare fraud costs the U.S. system hundreds of billions of dollars annually. This project addresses the challenge of identifying fraudulent providers from high-volume claims data — where fraud labels are rare, noisy, and highly imbalanced.

The system produces a **risk level along with confidence**, enabling investigators to prioritize audits efficiently.

---

## System Architecture

```
Raw Claims Data
      │
      ▼
Feature Engineering (Provider-level aggregation)
      │
      ▼
Model Training & Benchmarking (Logistic Regression, Random Forest)
      │
      ▼
SHAP-based Explainability (Global + Local)
      │
      ▼
FastAPI Inference Server
      │
      ▼
Docker Container → AWS EC2 Deployment
```

---

## Key Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.93** |
| Fraud Recall | **85%** |
| Minority-class F1 | Optimized via precision-recall tuning |
| Deployment | REST API on AWS EC2 via Docker |
| Explainability | SHAP global + local feature attribution |

> The model was optimized for **high recall** over precision to minimize missed fraud cases, which carry higher real-world cost than false positives in an investigative triage context.

---

## Project Structure


```
├── 📁 dashboard
├── 📁 data
│   ├── 📁 processed
│   │   └── 📄 provider_features_train.csv
│   └── 📁 raw
│       ├── 📄 test.csv
│       ├── 📄 test_beneficiary.csv
│       ├── 📄 test_inpatient.csv
│       ├── 📄 test_outpatient.csv
│       ├── 📄 train.csv
│       ├── 📄 train_Inpatient.csv
│       ├── 📄 train_beneficiary.csv
│       └── 📄 train_outpatient.csv
├── 📁 models
│   ├── 📄 fraud_model.joblib
│   └── ⚙️ model_config.json
├── 📁 notebooks
│   ├── 📄 01_data_understanding.ipynb
│   ├── 📄 02_feature_engineering.ipynb
│   ├── 📄 03_model_training.ipynb
│   └── 📄 04_explainability.ipynb
├── 📁 src
│   ├── 📁 api
│   │   ├── 🐍 cache.py
│   │   ├── 🐍 main.py
│   │   └── 🐍 schemas.py
│   └── 📁 utils
│       ├── 🐍 __init__.py
│       └── 🐍 feature_schema.py
├── ⚙️ .dockerignore
├── ⚙️ .gitignore
├── 🐳 Dockerfile
├── 📝 README.md
├── ⚙️ docker-compose.yml
└── 📄 requirements.txt
```

---

## Features

**Feature Engineering (Provider-level)**
- total_claims
- total_reimbursed
- avg_reimbursed
- avg_duration_gap
- pct_claimed_gt_admitted
- avg_cost_per_day
- age_avg
- pct_chronic
- PotentialFraud

**Modeling**
- Benchmarked Logistic Regression and Random Forest 
- Precision-recall value at multiple thresholds for finding optimal threshold on imbalanced data
- Final model selected based on fraud recall and ROC-AUC
- High ROC (0.93) means -> the model will perform well across thresholds
- Threshold is configurable at inference based on risk taking ability of the business

**Explainability**
- Global SHAP summary plots for feature importance
- Local SHAP waterfall plots for individual provider risk explanation
- Designed to support investigator-facing audit reports

**Deployment**
- FastAPI REST API serving risk category, probability along with threshold
- Dockerized for reproducibility
- Deployed on AWS EC2

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| ML Modeling | Scikit-learn |
| Explainability | SHAP |
| API | FastAPI, Pydantic |
| Containerization | Docker |
| Cloud Deployment | AWS EC2 |
| Visualization | Matplotlib, Seaborn |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- AWS CLI (for EC2 deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/koushik7074/healthcare-fraud-risk-ranking.git
cd healthcare-fraud-risk-ranking

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
if model.pkl does not exists in model dir -> train and recreate the model.pkl file by running 03_model_training.ipynb notebook

# Start the API server
uvicorn api.main:app --reload --port 8000
```

### Docker Setup

```bash
# Build the Docker image
docker build -t fraud-risk-api .

# Run the container
docker run -p 8000:8000 fraud-risk-api
```

---

## API Usage

### Endpoint: `POST /predict`

**Request Body**

```json
{
  "total_claims": , 
 "total_reimbursed": ,
 "avg_reimbursed": ,
 "avg_duration_gap": ,
 "pct_claimed_gt_admitted": ,
 "avg_cost_per_day": ,
 "age_avg": ,
 "pct_chronic": ,
 "PotentialFraud":
}
```

**Response**

```json
 {
    "fraud_probability": ,
    "fraud_prediction": ,
    "decision_threshold": ,
    "risk_level": 
    }
```

### Endpoint: `GET /health`

```json
{ "status": "ok", "model_version": "1.0.0" }
```

---

## Model Explainability

SHAP (SHapley Additive exPlanations) is used at two levels:

**Global** — Which features drive fraud risk across all providers?
Visualized via SHAP summary bar plots and beeswarm plots.

**Local** — Why was this specific provider flagged?
Visualized via SHAP waterfall plots

This makes the system auditable and suitable for regulated healthcare environments where black-box decisions are not acceptable.

---

## Monitoring & Evaluation Strategy

| Concern | Strategy |
|---|---|
| Data drift | Monitor input feature distributions over time |
| Model degradation | Track ROC-AUC and recall on labeled batches monthly |
| Threshold tuning | Re-evaluate precision-recall tradeoff as fraud patterns evolve |
| Alerting | Flag providers crossing risk score threshold for investigator queue |

---

## Future Improvements

- Add MLflow for experiment tracking and model versioning
- Integrate a feature store for real-time feature serving
- Add CI/CD pipeline with GitHub Actions
- Expand to graph-based fraud detection (provider-patient network analysis)
- Fine-tune threshold dynamically based on investigator feedback loop

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Koushik Biswas](https://linkedin.com/in/koushik-biswas-juiitm)*
