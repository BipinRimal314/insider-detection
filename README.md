# Insider Threat Detection Using Unsupervised Temporal Behavioral Profiling

A rigorous comparative study of unsupervised anomaly detection methods for detecting insider threats in organizational networks. This research demonstrates that **temporal sequence modeling (LSTM Autoencoder) achieves 3.4x higher detection rates than static methods at operationally relevant thresholds**.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Findings](#key-findings)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Running Experiments](#running-experiments)
7. [Understanding the Code](#understanding-the-code)
8. [Results](#results)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

---

## Overview

### What is Insider Threat Detection?

**Insider threats** occur when individuals with legitimate access to an organization's systems (employees, contractors, partners) misuse that access to harm the organization. Unlike external hackers who must break in, insiders already have the keys—making their malicious activities extremely difficult to detect.

### The Problem with Current Approaches

Traditional security tools (firewalls, intrusion detection systems) focus on external threats. Machine learning approaches typically require **labeled examples** of insider attacks for training, but such labels are extremely rare in practice—most organizations have never experienced a confirmed insider incident.

### Our Approach

We investigate **unsupervised anomaly detection**—methods that learn "normal" behavior patterns and flag deviations without needing labeled attack examples. We specifically test the hypothesis that **temporal sequence modeling** (analyzing behavior over time) captures attack patterns that static methods miss.

### Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Can unsupervised anomaly detection effectively identify insider threats without labeled training data? |
| **RQ2** | Does temporal sequence modeling improve detection compared to static methods? |
| **RQ3** | Which behavioral features are most predictive of insider threat activity? |
| **RQ4** | How robust are these methods across different insider attack scenarios? |

---

## Key Findings

### Main Result

| Model | AUC-ROC | Recall@5%FPR | Recall@10%FPR |
|-------|---------|--------------|---------------|
| Isolation Forest | **0.799 ± 0.017** | 0.044 ± 0.009 | 0.220 ± 0.025 |
| PCA Reconstruction | 0.612 ± 0.000 | 0.049 ± 0.000 | 0.129 ± 0.000 |
| Dense Autoencoder | 0.659 ± 0.016 | 0.048 ± 0.006 | 0.118 ± 0.009 |
| **LSTM Autoencoder** | 0.770 ± 0.006 | **0.149 ± 0.021** | **0.254 ± 0.019** |

**Key Insight**: While Isolation Forest achieves the highest AUC-ROC (overall ranking), LSTM Autoencoder detects **3.4x more attacks** at the 5% false positive rate—the threshold most relevant for security operations.

### Why This Matters

- **AUC-ROC** measures how well a model ranks anomalies overall
- **Recall@5%FPR** measures how many true attacks are caught when limiting false alarms to 5% of normal users
- Security teams have limited capacity to investigate alerts—**low FPR thresholds are operationally critical**

### Additional Findings

1. **USB device activity** is the strongest indicator of insider attacks (correlation: 0.075)
2. **7-day temporal windows** are optimal; longer windows add noise without benefit
3. **"Boiling frog" attacks** (gradual, subtle behavior) evade detection—85% of attacks missed

---

## Project Structure

```
Thesis_work/
├── src/                           # Source code (modular, documented)
│   ├── data/                      # Data loading and preprocessing
│   │   ├── loader.py              # Load raw CMU-CERT CSV files
│   │   ├── preprocessing.py       # Clean and filter data
│   │   ├── features.py            # Extract 24 daily features
│   │   ├── sequences.py           # Create temporal sequences
│   │   └── splits.py              # Train/test splitting
│   │
│   ├── models/                    # Anomaly detection models
│   │   ├── base.py                # Abstract base class
│   │   ├── isolation_forest.py    # Isolation Forest detector
│   │   ├── pca_anomaly.py         # PCA reconstruction detector
│   │   ├── autoencoder.py         # Dense autoencoder
│   │   ├── lstm_autoencoder.py    # LSTM autoencoder (temporal)
│   │   └── transformer_autoencoder.py  # Transformer (experimental)
│   │
│   ├── experiments/               # Experiment infrastructure
│   │   ├── runner.py              # Multi-seed experiment runner
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── visualizations.py      # Plot generation
│   │
│   ├── utils/                     # Utilities
│   │   └── reproducibility.py     # Seed management
│   │
│   └── config.py                  # Hyperparameter configuration
│
├── paper/                         # LaTeX paper
│   ├── main.tex                   # Paper manuscript
│   ├── references.bib             # Bibliography (15+ citations)
│   └── figures/                   # Generated visualizations
│
├── data/                          # Dataset (not tracked in git)
│   ├── r4.2/                      # CMU-CERT r4.2 dataset
│   │   ├── logon.csv              # Authentication events
│   │   ├── device.csv             # USB device connections
│   │   ├── file.csv               # File operations
│   │   ├── email.csv              # Email metadata
│   │   └── http.csv               # Web browsing
│   └── answers/                   # Ground truth labels
│       └── insiders.csv           # Insider threat scenarios
│
├── results/                       # Experiment outputs
│   └── clean_experiments/         # Final results (JSON)
│
├── docs/                          # Additional documentation
│   └── THESIS_DEFENSE_GUIDE.md    # Comprehensive explanation
│
├── run_experiments.py             # Main experiment script
├── requirements-research.txt      # Python dependencies
├── REPRODUCIBILITY.md             # Step-by-step reproduction guide
└── README.md                      # This file
```

---

## Installation

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.11.x | `python3 --version` |
| pip | 23.0+ | `pip --version` |
| Git | 2.x | `git --version` |
| 16GB RAM | - | Recommended for LSTM training |

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Thesis_work
```

### Step 2: Create Virtual Environment

```bash
# Create isolated Python environment
python3.11 -m venv .venv_tf

# Activate it (run this every time you work on the project)
source .venv_tf/bin/activate  # macOS/Linux
# OR
.venv_tf\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-research.txt
```

**Note for Apple Silicon (M1/M2/M3) users**: The requirements include TensorFlow optimized for Apple Silicon. If you encounter issues, see [Troubleshooting](#troubleshooting).

**Note for Linux/Windows users**: Replace `tensorflow-macos` with `tensorflow` in requirements:
```bash
pip install tensorflow==2.15.0
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "from src.models import LSTMAutoencoder; print('Models imported successfully')"
```

---

## Dataset Setup

### About CMU-CERT Dataset

The **CMU-CERT Insider Threat Dataset** is the standard benchmark for insider threat research, created by the CERT Division of Carnegie Mellon University.

| Property | Value |
|----------|-------|
| Version | r4.2 |
| Users | ~1,000 simulated employees |
| Duration | 18 months of activity |
| Insiders | 70 malicious users (3 scenario types) |
| Events | Millions of log entries |

### Download the Dataset

1. Go to: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
2. Download the dataset archive (~2GB)
3. Extract to the `data/` directory

### Expected Directory Structure

After extraction, your `data/` folder should look like:

```
data/
├── r4.2/
│   ├── logon.csv      # 32M - User login/logout events
│   ├── device.csv     # 1.2M - USB device connections
│   ├── file.csv       # 445M - File copy operations
│   ├── email.csv      # 2.6G - Email send/receive
│   └── http.csv       # 1.6G - Web browsing activity
└── answers/
    └── insiders.csv   # Ground truth: who are the insiders
```

### Verify Dataset

```bash
# Check files exist
ls -la data/r4.2/
ls -la data/answers/

# Quick data check
head -5 data/r4.2/logon.csv
head -5 data/answers/insiders.csv
```

---

## Running Experiments

### Quick Test (Single Seed)

To verify everything works:

```bash
# Run Isolation Forest with 1 seed (fastest, ~2 minutes)
python run_experiments.py --model isolation_forest --quick
```

### Full Experiment Suite

To reproduce all paper results:

```bash
# Run all 4 models with 5 random seeds each
# WARNING: Takes ~4-5 hours (LSTM is slow on CPU)
python run_experiments.py --all --seeds 5
```

### Individual Models

```bash
# Static methods (fast, ~5 min each)
python run_experiments.py --model isolation_forest --seeds 5
python run_experiments.py --model pca --seeds 5
python run_experiments.py --model dense_autoencoder --seeds 5

# Temporal method (slow, ~60 min per seed on CPU)
python run_experiments.py --model lstm_autoencoder --seeds 5
```

### Custom Configuration

```bash
# Change sequence window size
python run_experiments.py --model lstm_autoencoder --window 14

# Use different dataset version
python run_experiments.py --all --dataset r5.2

# Custom output directory
python run_experiments.py --all --output results/my_experiment
```

### Generate Visualizations

After running experiments:

```bash
python -c "from src.experiments.visualizations import generate_all_plots; generate_all_plots()"
```

This creates:
- `paper/figures/roc_curves.png` - ROC curves for all models
- `paper/figures/model_comparison.png` - Bar charts comparing metrics
- `paper/figures/seed_variance.png` - Variance across random seeds
- `paper/figures/feature_importance.png` - Feature importance analysis

---

## Understanding the Code

### Data Pipeline

```
Raw CSV Files → Preprocessing → Feature Engineering → Sequences → Train/Test Split
```

1. **Loading** (`src/data/loader.py`): Reads raw CSV files
2. **Preprocessing** (`src/data/preprocessing.py`): Cleans data, filters inactive users
3. **Features** (`src/data/features.py`): Extracts 24 daily behavioral features
4. **Sequences** (`src/data/sequences.py`): Creates 7-day sliding windows for LSTM
5. **Splits** (`src/data/splits.py`): Temporal train/test split (70%/30%)

### Feature Categories (24 Features)

| Category | Features | Description |
|----------|----------|-------------|
| **Logon (6)** | logon_count, logoff_count, after_hours_logons, unique_pcs, first_logon_hour, last_logoff_hour | Authentication patterns |
| **Device (4)** | device_connects, device_disconnects, after_hours_connects, device_activity | USB/removable media usage |
| **HTTP (5)** | http_requests, unique_domains, upload_actions, download_actions, after_hours_browsing | Web activity |
| **Email (5)** | emails_sent, total_recipients, attachment_count, attachment_size, after_hours_emails | Email patterns |
| **File (4)** | file_operations, file_copies, exe_access, after_hours_files | File access behavior |

### Model Architecture

#### Isolation Forest
- Ensemble of 100 random trees
- Anomaly score = average path length to isolate a point
- Fast training and inference

#### PCA Reconstruction
- Projects data to principal components (95% variance retained)
- Anomaly score = reconstruction error
- No hyperparameters, deterministic

#### Dense Autoencoder
- Architecture: 24 → 64 → 32 → 16 → 32 → 64 → 24
- Anomaly score = MSE reconstruction error
- 50 epochs, early stopping

#### LSTM Autoencoder
- Encoder: LSTM(64) → LSTM(32) → Dense(16)
- Decoder: Dense(16) → LSTM(32) → LSTM(64) → Output
- Processes 7-day sequences
- Captures temporal dependencies

### Evaluation Metrics

```python
# AUC-ROC: Overall ranking ability (0-1, higher better)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, scores)

# Recall@K%FPR: Detection rate at fixed false positive rate
def recall_at_fpr(y_true, scores, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr)
    return tpr[idx]
```

---

## Results

### Output Files

After running experiments, find results in `results/clean_experiments/`:

```
results/clean_experiments/
├── IsolationForest_results.json
├── PCA_Reconstruction_results.json
├── DenseAutoencoder_results.json
├── LSTMAutoencoder_results.json
└── sequence_ablation.json
```

### Result Format

Each JSON file contains:

```json
{
  "model_name": "LSTMAutoencoder",
  "seeds": [42, 43, 44, 45, 46],
  "auc_roc_mean": 0.770,
  "auc_roc_std": 0.006,
  "recall_5fpr_mean": 0.149,
  "recall_5fpr_std": 0.021,
  "per_seed_results": [...]
}
```

### Compiling the Paper

```bash
cd paper

# Full compilation (recommended)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Output: main.pdf (6 pages, IEEE conference format)
```

---

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'src'"

```bash
# Option 1: Run from project root
cd /path/to/Thesis_work
python run_experiments.py --all

# Option 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### TensorFlow Issues on Apple Silicon

```bash
# If TensorFlow Metal causes issues, disable it
export CUDA_VISIBLE_DEVICES=""
python run_experiments.py --all
```

#### Memory Errors

LSTM training on 230K sequences requires significant memory:
```bash
# Reduce batch size in src/config.py
# Default: batch_size = 32
# Try: batch_size = 16
```

#### Dataset Not Found

```bash
# Verify paths
ls data/r4.2/logon.csv
ls data/answers/insiders.csv

# If answers folder is elsewhere, create symlink
ln -s /path/to/answers data/answers
```

#### Slow Training

LSTM training takes ~60 minutes per seed on CPU. Options:
1. Use GPU if available
2. Run with fewer seeds: `--seeds 1`
3. Use smaller dataset sample for debugging

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{rimal2025insider,
  title={Unsupervised Temporal Behavioral Profiling for Insider Threat Detection: A Comparative Study},
  author={Rimal, Bipin},
  institution={Coventry University},
  year={2025}
}
```

---

## Author

**Bipin Rimal**
Department of Computing
Coventry University
rimalb@uni.coventry.ac.uk

---

## License

This code is released for **research purposes only**. The CMU-CERT dataset is subject to its own license terms from Carnegie Mellon University.

---

## Acknowledgments

- CERT Division of the Software Engineering Institute at Carnegie Mellon University for the CMU-CERT Insider Threat Dataset
- Coventry University for research support
