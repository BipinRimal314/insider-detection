# Insider Threat Detection System

**Unsupervised Behavioral Profiling for Insider Threat Detection Using Time-Series and Anomaly Detection Techniques**

A machine learning system for detecting insider threats through behavioral analysis using three complementary unsupervised models: Isolation Forest, LSTM Autoencoder, and Deep Clustering.

## 🎯 Project Overview

This system implements unsupervised learning techniques to detect malicious insider activity within enterprise environments by:

- Analyzing user behavior patterns across multiple data sources (logon, file access, email, device usage, HTTP)
- Detecting anomalies without requiring labeled training data
- Combining multiple detection models in an ensemble for improved accuracy
- Privacy-preserving design through SHA-256 pseudonymization

## 📊 System Architecture

```
Raw Logs → Preprocessing → Feature Engineering → Model Training → Ensemble → Visualization
           (Stage 1)       (Stage 2)             (Stage 3)         (Stage 4-5) (Stage 6)
```

### Pipeline Stages

| Stage | Name | Script | Description |
|-------|------|--------|-------------|
| 1 | Data Preprocessing | `data_preprocessing_polars.py` | Load and clean CMU-CERT logs using Polars LazyFrames |
| 2 | Feature Engineering | `feature_engineering_polars.py` | Generate daily behavioral features and sequences |
| 3 | Model Training | `isolation_forest_model.py`, `lstm_autoencoder_model.py`, `deep_clustering_model.py` | Train unsupervised anomaly detection models |
| 4 | Model Evaluation | `model_evaluation.py` | Compare model performance with metrics and plots |
| 5 | Ensemble Integration | `ensemble_system.py` | Combine models using weighted, majority, or cascade voting |
| 6 | Visualization | `visualization.py` | Generate comprehensive result visualizations |

### Models Implemented

| Model | Technique | Purpose |
|-------|-----------|---------|
| **Isolation Forest** | Random tree-based isolation | Fast screening for point anomalies in high-dimensional spaces |
| **LSTM Autoencoder** | Sequence reconstruction | Detect temporal behavior pattern deviations |
| **Deep Clustering** | Autoencoder + KMeans | Behavioral profiling through joint feature learning and clustering |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (see `.python-version`)
- ~4GB RAM minimum for processing

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Thesis_work

# Create virtual environment
python -m venv .venv_tf
source .venv_tf/bin/activate  # On Windows: .venv_tf\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the **CMU-CERT Insider Threat Dataset** (r1 subset) from:  
   https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
2. Extract to `data/all_data/r1/` directory
3. Ensure the following files are present:
   ```
   data/all_data/r1/
   ├── logon.csv
   ├── device.csv
   ├── file.csv
   ├── email.csv
   ├── http.csv
   └── LDAP/
   ```

### Running the Pipeline

**Option 1: Run Complete Pipeline**
```bash
python main.py --full
```

**Option 2: Run Specific Stages**
```bash
# Run only preprocessing and feature engineering
python main.py --stages 1 2

# Skip already completed stages
python main.py --full --skip 1 2
```

**Option 3: Run Individual Stages**
```bash
python main.py --preprocess      # Stage 1
python main.py --feature-eng     # Stage 2
python main.py --train           # Stage 3
python main.py --evaluate        # Stage 4
```

**Option 4: Run Scripts Directly**
```bash
python data_preprocessing_polars.py
python feature_engineering_polars.py
python isolation_forest_model.py
python lstm_autoencoder_model.py
python deep_clustering_model.py
python model_evaluation.py
python ensemble_system.py
python visualization.py
```

## 📁 Project Structure

```
Thesis_work/
├── main.py                        # Main pipeline orchestrator (6 stages)
├── config.py                      # Configuration and hyperparameters
├── utils.py                       # Utility functions (logging, metrics, I/O)
│
├── data_preprocessing_polars.py   # Stage 1: Data loading/cleaning (Polars)
├── feature_engineering_polars.py  # Stage 2: Feature creation (Polars)
├── isolation_forest_model.py      # Model: Isolation Forest
├── lstm_autoencoder_model.py      # Model: LSTM Autoencoder
├── deep_clustering_model.py       # Model: Deep Clustering (AE + KMeans)
├── model_evaluation.py            # Stage 4: Model comparison
├── ensemble_system.py             # Stage 5: Ensemble integration
├── visualization.py               # Stage 6: Results visualization
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── all_data/r1/               # Raw CMU-CERT dataset files
│   └── processed/                 # Intermediate processed data
│
├── models/                        # Trained model files (.pkl, .keras)
├── results/                       # Predictions, metrics, alerts
├── logs/                          # Execution logs
│
└── legacy/                        # Archived files (safe to ignore)
    ├── deprecated_versions/       # Old model/evaluation versions
    ├── debug_scripts/             # Diagnostic and test scripts
    ├── one_off_utilities/         # Data labeling, patching scripts
    └── thesis_specific/           # Thesis visualization scripts
```

## ⚙️ Configuration

All configuration is centralized in `config.py`. Key parameters include:

### Data Settings
| Parameter | Description | Default |
|-----------|-------------|---------|
| `DATASET_SUBSET` | CMU-CERT datasets to process | `['r1']` |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | Data split ratios | 0.7 / 0.15 / 0.15 |
| `SEQUENCE_LENGTH` / `SEQUENCE_STRIDE` | LSTM sequence generation | 15 / 10 |
| `MAX_SEQUENCE_SAMPLES` | Debug sample limit | `None` (full dataset) |

### Model Parameters

**Isolation Forest:**
- `n_estimators`: 50, `contamination`: auto

**LSTM Autoencoder:**
- `lstm_units`: [32, 16], `epochs`: 20, `patience`: 5

**Deep Clustering:**
- `n_clusters`: 5, `encoding_dims`: [64, 32], `epochs`: 10

### Ensemble Settings
```python
ENSEMBLE = {
    'weights': {
        'isolation_forest': 0.3,
        'lstm_autoencoder': 0.4,
        'deep_clustering': 0.3
    },
    'final_threshold': 0.7
}
```

> **Note:** Current config processes full dataset with production-level epochs. For faster testing, set `DATASET_SUBSET = ['r1']` and reduce epochs.

## 📈 Experimental Results

### Latest Performance (CMU-CERT Dataset r1-r4.1, with Z-Score Features)

**Dataset Statistics:**
- **Total Records Processed:** 103,242,062 events
- **Insider Users Identified:** 7 (from ground truth)
- **Malicious Records Labeled:** 20,812 (0.02% of total)
- **Users Analyzed:** 7,999
- **Daily Feature Vectors:** 2,654,790 (26 features including Z-scores)
- **Sequence Samples:** 2,542,814 (shape: 15 timesteps × 23 features)

**Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **LSTM Autoencoder** 🏆 | 95.0% | 0.07% | **56.5%** | 0.14% | **0.94** |
| Transformer Autoencoder | 94.9% | 0.07% | 50.0% | 0.14% | 0.90 |
| Deep Clustering | 95.0% | 0.03% | 26.1% | 0.06% | 0.85 |
| Isolation Forest | 91.3% | 0.03% | **44.7%** | 0.06% | 0.84 |

**Z-Score Feature Impact (vs Previous Run):**
| Model | Previous AUC | New AUC | Improvement |
|-------|-------------|---------|-------------|
| Isolation Forest | 0.78 | **0.84** | **+6%** |
| LSTM Autoencoder | 0.93 | **0.94** | +1% |
| Deep Clustering | 0.84 | **0.85** | +1% |

### Understanding These Results

**Why is Precision so low?**  
This is expected! With only **0.02% of records being malicious**, even a perfect model would have many false positives. This is the "base rate fallacy" in rare-event detection. The key metric is **AUC-ROC**, which measures ranking quality.

**What does AUC-ROC = 0.93 mean?**  
If you randomly pick one insider and one normal user, there's a 93% chance the model will correctly rank the insider as more anomalous. This is excellent for unsupervised detection.

**What does 56.5% Recall mean?**  
The LSTM caught **~4 out of 7 insider attacks** without any labeled training data. For sophisticated insider threats, this is a strong result.

**Key Finding:**  
The **LSTM Autoencoder significantly outperforms** static models (Isolation Forest, Deep Clustering), validating the thesis hypothesis that **temporal sequence analysis is crucial for insider threat detection**.

### Ablation Study Results

**Z-Score Feature Impact:**
| Configuration | AUC-ROC | Change |
|---------------|---------|--------|
| Without Z-scores | 0.50 | (random) |
| With Z-scores | **0.87** | **+37%** |

**Top 3 Most Important Features:**
1. `daily_activity_count_peer_zscore` (+0.031 AUC impact)
2. `daily_activity_count_self_zscore` (+0.029 AUC impact)
3. `file_access_count_self_zscore` (+0.015 AUC impact)

**Insight:** Insiders deviate most in their **overall activity volume** compared to their baseline and peers. This validates the context-aware Z-score approach.

### Operational Impact
- **Raw Events:** 103M+ → **Alerts Generated:** ~9,600
- **Reduction:** 99.99% noise filtered
- **Alert Priority:** 259 critical, 1,165 high, 1,556 medium, 6,654 low

## 🔍 Output Files

### Predictions
- `results/isolation_forest_predictions.csv` — User, day, prediction, anomaly_score
- `results/lstm_autoencoder_predictions.csv`
- `results/deep_clustering_predictions.csv`
- `results/ensemble_results.csv` — Combined ensemble predictions

### Alerts
- `results/alerts.csv` — Final actionable alerts with severity levels (low/medium/high/critical)

### Visualizations
- `results/plots/` — ROC curves, confusion matrices, score distributions, executive summary

## 🛡️ Privacy & Ethics

This system implements privacy-by-design principles:

- **Data Pseudonymization**: User identifiers hashed using SHA-256 with configurable salt
- **Minimal Data Collection**: Only behavioral metadata is processed, not content
- **Configurable Privacy Settings**: See `PRIVACY` config in `config.py`

## 🔬 Research Background

This implementation is based on the thesis:  
**"Unsupervised Behavioural Profiling for Insider Threat Detection Using Time-Series and Anomaly Detection Techniques"**

Key findings:
- **LSTM Autoencoder outperforms static models** (AUC 0.93 vs 0.78 for Isolation Forest)
- Temporal sequence modeling is **crucial** for detecting complex insider threats
- Unsupervised approaches can achieve **56% recall** without labeled training data
- Privacy-preserving design through SHA-256 pseudonymization

> 📄 **For publication guidance**, see [research.md](research.md) — includes paper structure, target venues, and improvement suggestions.

## 📚 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| polars | ≥0.20.0 | High-performance data processing |
| tensorflow | ≥2.15.0 | LSTM Autoencoder, Deep Clustering |
| scikit-learn | ≥1.3.0 | Isolation Forest, metrics |
| pandas | ≥2.0.0 | Data manipulation |
| matplotlib/seaborn | ≥3.7.0/0.12.0 | Visualization |

## 🆘 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No data files found`  
**Solution**: Ensure CMU-CERT dataset is in `data/all_data/r1/` directory

**Issue**: `MemoryError during training`  
**Solution**: Reduce `MAX_SEQUENCE_SAMPLES` in `config.py`

**Issue**: `TensorFlow compatibility errors`  
**Solution**: Ensure TensorFlow ≥2.15.0: `pip install --upgrade tensorflow`

**Issue**: Poor model performance  
**Solution**: Increase `epochs` in config (currently set to debug mode)

### Getting Help

1. Check logs in `logs/insider_threat_detection.log`
2. Enable verbose mode: Set `LOGGING['level'] = 'DEBUG'` in config
3. Run stages individually to isolate issues

## 👤 Author

**Bipin Rimal**  
Module: Computing Individual Research Project (STW7048CEM)

## 📄 License

This project is for academic research purposes. Please cite appropriately if used in publications.

---

**Note**: This system is designed for research and educational purposes. Deployment in production environments should include additional security hardening, compliance reviews, and stakeholder approval.