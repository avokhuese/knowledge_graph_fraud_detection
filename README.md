# Graph-Based Provider Fraud Detection System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-prototype-orange)

A machine learning system that detects healthcare provider fraud by analyzing provider relationships as a graph, combining graph neural network embeddings with traditional machine learning for proactive fraud risk scoring.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Model Components](#model-components)
- [Evaluation](#evaluation)
- [Results Interpretation](#results-interpretation)
- [Production Deployment](#production-deployment)
- [Security & Privacy](#security--privacy)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Traditional fraud detection systems analyze providers in isolation, looking at billing patterns and statistical anomalies. However, sophisticated fraud schemes involve networks of colluding providers who share patients, bank accounts, facilities, and ownership structures.

This system represents healthcare providers as nodes in a heterogeneous graph, capturing relationships like:
- **Shared bank accounts** (straw provider schemes)
- **Co-billing patterns** (collusion rings)
- **Shared facilities** (shell clinics)
- **Referral networks** (kickback schemes)
- **Common ownership** (hidden control)

By applying graph-based machine learning, the system can identify suspicious network patterns before individual providers accumulate enough bad claims to trigger traditional alerts.

## Key Features

### 🔍 **Multi-Model Approach**
- **Graph-Structural Features**: Centrality, community detection, PageRank
- **Graph Embeddings**: Node2Vec, Spectral, Structural, Metapath2Vec
- **Claims-Behavioral Features**: Billing patterns, upcoding detection, temporal analysis
- **Hybrid Model**: Combines graph features with XGBoost for explainable predictions

### 📊 **Comprehensive Baselines**
- PageRank guilt-by-association from known exclusions
- Random Walk with Restart
- Label Propagation
- Traditional ML (claims-only features)
- Isolation Forest for unsupervised anomaly detection

### 🛡️ **Privacy-First Design**
- All PII/PHI hashed using HMAC-SHA256
- Configurable salt management
- Token-based pseudonymization
- No raw sensitive data stored in graph

### 📈 **Production-Ready Features**
- SHAP explanations for every prediction
- Temporal validation (no data leakage)
- Model comparison framework
- Investigator capacity-aware evaluation (Precision@K)
- Lift over baseline measurement

## Architecture
fraud_detection/
├── config/
│ └── settings.py # Configuration management
├── data/
│ ├── ingestion.py # Data loading and validation
│ └── preprocessing.py # Feature engineering
├── graph/
│ ├── construction.py # Heterogeneous graph builder
│ ├── features.py # Graph-structural features
│ └── embeddings.py # Node embeddings (Node2Vec, Spectral, etc.)
├── models/
│ ├── baseline.py # Baseline models (PageRank, RWR, Traditional ML)
│ ├── gnn.py # Graph Neural Networks
│ └── hybrid.py # Hybrid XGBoost + Graph model
├── evaluation/
│ └── metrics.py # Fraud-specific evaluation metrics
├── pipeline/
│ └── scoring.py # Main pipeline orchestrator
├── utils/
│ ├── hashing.py # Privacy-preserving hashing
│ └── validation.py # Data validation
├── run_fraud_detection.py # Complete demo pipeline
└── main.py # Production entry point


## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for GNN training

### Setup

1. **Clone the repository**
git clone https://github.com/your-org/fraud-detection.git
cd fraud-detection
2. **Create virtual environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install dependencies**
pip install -r requirements.txt
4. **Install optional dependencies for advanced features**
# For community detection
pip install python-louvain

# For GPU-accelerated GNN training
pip install torch-geometric

# For enhanced visualizations
pip install matplotlib seaborn
6.  Quick Demo
python run_fraud_detection.py

==============================================================
FRAUD DETECTION PIPELINE
==============================================================

[1/8] Generating sample data...
  Generated: 200 providers, 5000 claims, 15 exclusions
  Fraud ring: 15 providers sharing bank accounts and patients

[2/8] Validating data...
  Providers valid: True
  Claims valid: True
  Exclusions valid: True

[3/8] Applying privacy hashing...
  ✓ Hashed beneficiary IDs, bank accounts, and provider names

[4/8] Building provider relationship graph...
  ✓ Graph constructed successfully
  - Nodes: 234
  - Edges: 1,247

[5/8] Engineering features...
  ✓ Total graph-structural features: 45
  ✓ Claims-behavioral features: 23

[6/8] Training hybrid fraud detection models...
  ✓ Trained 3 model variants

[7/8] Generating risk scores and comparing models...

  Top 10 highest-risk providers (Full Model):
    P0001: Full=0.9234, Trad=0.4521, Status: ⚠️ FRAUD
    P0003: Full=0.8912, Trad=0.3890, Status: ⚠️ FRAUD
    ...

[8/8] Evaluating and comparing all models...

================================================================================
MODEL COMPARISON
================================================================================
Model                     AUC-PR     Precision@20    Recall@20    
--------------------------------------------------------------------------------
Full Hybrid Model         0.9234     0.8500          0.7333       
Graph-Only Model          0.8756     0.7500          0.6000       
Baseline-Enhanced         0.8123     0.7000          0.5333       
Traditional ML            0.6543     0.4500          0.3333       
Pure PageRank             0.7123     0.5500          0.4000       

✓ Best performing model: Full Hybrid Model (AUC-PR: 0.9234)

**Production Pipeline**
python main.py \
    --provider-file /path/to/providers.csv \
    --claims-file /path/to/claims.csv \
    --exclusions-file /path/to/exclusions.csv \
    --ownership-file /path/to/ownership.csv \
    --bank-file /path/to/bank_accounts.csv \
    --output-dir ./output \
    --investigator-capacity 50 \
    --seed 42


  **Using as a Python Library**
  from fraud_detection.pipeline.scoring import FraudDetectionPipeline
from fraud_detection.config.settings import Settings

# Initialize
settings = Settings()
settings.pipeline.output_dir = './output'
settings.pipeline.investigator_capacity = 50

# Run pipeline
pipeline = FraudDetectionPipeline(settings)
results = pipeline.run(
    provider_file='data/providers.csv',
    claims_file='data/claims.csv',
    exclusions_file='data/exclusions.csv'
)

# Access results
risk_scores = results['risk_scores']
evaluation = results['evaluation']
explanations = results['explanations']

**Development Setup**

# Clone repository
git clone https://github.com/your-org/fraud-detection.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black fraud_detection/

**Acknowledgments**
OIG LEIE: Exclusion data source

CMS NPPES: Provider enumeration data

NetworkX: Graph algorithms

SHAP: Model interpretability

XGBoost: Gradient boosting framework


Contact
For questions or support:

Project Lead: Dr. Alexander Okhuese Victor

Team: Payment Integrity Analytics

Email: PIA@schemasholdings.com
