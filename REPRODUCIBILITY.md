# Complete Reproducibility Guide

This document provides detailed, step-by-step instructions to reproduce all experiments in the paper **"Unsupervised Temporal Behavioral Profiling for Insider Threat Detection: A Comparative Study"**.

---

## Table of Contents

1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Running the Experiments](#running-the-experiments)
5. [Expected Results](#expected-results)
6. [Verification Checklist](#verification-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What You Will Reproduce

| Experiment | Description | Time |
|------------|-------------|------|
| Main Results (Table 2) | 4 models × 5 seeds comparison | ~4 hours |
| Sequence Ablation (Table 5) | 7/14/30-day window comparison | ~3 hours |
| Feature Analysis | Feature importance and correlation | ~10 min |
| Visualizations | ROC curves, bar charts, etc. | ~80 min |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 10 GB | 20 GB |
| CPU | Any modern | Multi-core |
| GPU | Not required | CUDA GPU (10x faster) |

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.11.x | **Must be 3.11** |
| pip | 23.0+ | Package installer |
| git | 2.x | Version control |
| LaTeX | Any | For compiling paper |

---

## Environment Setup

### Step 1: Verify Python Version

```bash
python3 --version
# Expected: Python 3.11.x

# If not 3.11, install it:
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
# Windows: Download from python.org
```

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd Thesis_work
```

### Step 3: Create Virtual Environment

**Why?** Virtual environments isolate project dependencies from your system Python.

```bash
# Create environment
python3.11 -m venv .venv_tf

# Activate it
source .venv_tf/bin/activate  # macOS/Linux
# OR
.venv_tf\Scripts\activate     # Windows

# Verify activation (should show .venv_tf in prompt)
which python
# Expected: /path/to/Thesis_work/.venv_tf/bin/python
```

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements-research.txt
```

**Platform-Specific Notes:**

| Platform | Special Instructions |
|----------|---------------------|
| **macOS (Apple Silicon)** | Requirements already configured for M1/M2/M3 |
| **macOS (Intel)** | Replace `tensorflow-macos` with `tensorflow==2.15.0` |
| **Linux** | Replace `tensorflow-macos` with `tensorflow==2.15.0` |
| **Windows** | Replace `tensorflow-macos` with `tensorflow==2.15.0` |

### Step 5: Verify Installation

Run these commands to verify everything is installed correctly:

```bash
# Check TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
# Expected: TensorFlow: 2.15.0

# Check scikit-learn
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"
# Expected: sklearn: 1.7.2

# Check project imports
python -c "from src.models import LSTMAutoencoder; print('OK')"
# Expected: OK
```

---

## Dataset Preparation

### Step 1: Download CMU-CERT Dataset

1. Go to: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
2. Click "Download" (file is ~2GB compressed)
3. Accept the license agreement

### Step 2: Extract Dataset

```bash
# Navigate to project directory
cd /path/to/Thesis_work

# Extract (adjust filename as needed)
unzip Insider_Threat_Test_Dataset.zip -d data/

# Or if tar.gz
tar -xzf Insider_Threat_Test_Dataset.tar.gz -C data/
```

### Step 3: Verify Directory Structure

Your `data/` folder must look exactly like this:

```
data/
├── r4.2/                    # Dataset version 4.2
│   ├── logon.csv           # User login/logout events
│   ├── device.csv          # USB device connections
│   ├── file.csv            # File copy operations
│   ├── email.csv           # Email activity
│   └── http.csv            # Web browsing
└── answers/                 # Ground truth
    └── insiders.csv        # List of malicious users
```

### Step 4: Verify Data Files

```bash
# Check all required files exist
ls -la data/r4.2/
# Should show: logon.csv, device.csv, file.csv, email.csv, http.csv

ls -la data/answers/
# Should show: insiders.csv

# Check file sizes (approximate)
du -sh data/r4.2/*
# logon.csv    ~32M
# device.csv   ~1.2M
# file.csv     ~445M
# email.csv    ~2.6G
# http.csv     ~1.6G

# Preview data format
head -3 data/r4.2/logon.csv
# id,date,user,pc,activity
# {guid},01/02/2010 07:36:16,user123,PC-1234,Logon
# ...

head -3 data/answers/insiders.csv
# user,scenario,start,end
# ABC0123,1,2010-04-15,2010-06-30
# ...
```

### Common Dataset Issues

| Issue | Solution |
|-------|----------|
| Files in wrong location | Move to `data/r4.2/` |
| Missing answers folder | Create symlink: `ln -s /path/to/answers data/answers` |
| Different version (r5.2) | Use `--dataset r5.2` flag |

---

## Running the Experiments

### Experiment 1: Main Results (Table 2)

This reproduces the main comparison table in the paper.

```bash
# Full run (recommended) - ~4 hours
python run_experiments.py --all --seeds 5

# Or run each model separately:
python run_experiments.py --model isolation_forest --seeds 5  # ~5 min
python run_experiments.py --model pca --seeds 5               # ~5 min
python run_experiments.py --model dense_autoencoder --seeds 5 # ~30 min
python run_experiments.py --model lstm_autoencoder --seeds 5  # ~5 hours
```

**Output:** `results/clean_experiments/*.json`

### Experiment 2: Sequence Length Ablation (Table 5)

Tests different temporal window sizes for LSTM.

```bash
# 7-day window (already in main results)
python run_experiments.py --model lstm_autoencoder --window 7 --seeds 5

# 14-day window
python run_experiments.py --model lstm_autoencoder --window 14 --seeds 5

# 30-day window
python run_experiments.py --model lstm_autoencoder --window 30 --seeds 5
```

**Note:** Each window size takes ~5 hours on CPU.

### Experiment 3: Generate Visualizations

After running main experiments:

```bash
python << 'EOF'
from src.experiments.visualizations import generate_all_plots
import os
os.makedirs("paper/figures", exist_ok=True)
generate_all_plots()
print("Visualizations saved to paper/figures/")
EOF
```

**Output:**
- `paper/figures/roc_curves.png` - ROC curves comparison
- `paper/figures/model_comparison.png` - Bar chart metrics
- `paper/figures/seed_variance.png` - Variance analysis
- `paper/figures/feature_importance.png` - Feature analysis

### Experiment 4: Compile Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Output:** `paper/main.pdf` (6 pages)

---

## Expected Results

### Table 2: Main Model Comparison

| Model | AUC-ROC | Recall@5%FPR | Recall@10%FPR |
|-------|---------|--------------|---------------|
| Isolation Forest | 0.799 ± 0.017 | 0.044 ± 0.009 | 0.220 ± 0.025 |
| PCA Reconstruction | 0.612 ± 0.000 | 0.049 ± 0.000 | 0.129 ± 0.000 |
| Dense Autoencoder | 0.659 ± 0.016 | 0.048 ± 0.006 | 0.118 ± 0.009 |
| LSTM Autoencoder | 0.770 ± 0.006 | **0.149 ± 0.021** | **0.254 ± 0.019** |

### Table 5: Sequence Length Ablation

| Window | AUC-ROC | Recall@5%FPR | Recall@10%FPR |
|--------|---------|--------------|---------------|
| 7 days | **0.770 ± 0.006** | **0.149 ± 0.021** | **0.254 ± 0.019** |
| 14 days | 0.765 ± 0.008 | 0.142 ± 0.018 | 0.248 ± 0.022 |
| 30 days | 0.752 ± 0.012 | 0.128 ± 0.025 | 0.235 ± 0.028 |

### Result Tolerance

Your results should match within:
- **AUC-ROC:** ± 0.02
- **Recall@FPR:** ± 0.03

Small variations are expected due to:
- Hardware differences (CPU/GPU floating point)
- TensorFlow version differences
- Random number generator implementation

---

## Verification Checklist

### Before Running

- [ ] Python 3.11.x installed and verified
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (no errors)
- [ ] Dataset downloaded and extracted
- [ ] Directory structure matches expected layout
- [ ] `data/r4.2/logon.csv` exists
- [ ] `data/answers/insiders.csv` exists

### After Running

- [ ] `results/clean_experiments/IsolationForest_results.json` exists
- [ ] `results/clean_experiments/PCA_Reconstruction_results.json` exists
- [ ] `results/clean_experiments/DenseAutoencoder_results.json` exists
- [ ] `results/clean_experiments/LSTMAutoencoder_results.json` exists
- [ ] AUC-ROC values match expected (within tolerance)
- [ ] Recall@5%FPR values match expected (within tolerance)
- [ ] Paper compiles without errors
- [ ] Generated PDF is 6 pages

### Verification Script

Run this to verify your results:

```python
import json
from pathlib import Path

def verify_results():
    results_dir = Path("results/clean_experiments")

    expected = {
        "IsolationForest": {"auc_roc": 0.799, "recall_5fpr": 0.044},
        "PCA_Reconstruction": {"auc_roc": 0.612, "recall_5fpr": 0.049},
        "DenseAutoencoder": {"auc_roc": 0.659, "recall_5fpr": 0.048},
        "LSTMAutoencoder": {"auc_roc": 0.770, "recall_5fpr": 0.149},
    }

    tolerance = {"auc_roc": 0.02, "recall_5fpr": 0.03}

    print("Verifying results...")
    all_pass = True

    for model, exp in expected.items():
        fpath = results_dir / f"{model}_results.json"
        if not fpath.exists():
            print(f"MISSING: {fpath}")
            all_pass = False
            continue

        with open(fpath) as f:
            data = json.load(f)

        for metric, exp_val in exp.items():
            key = f"{metric}_mean" if "mean" not in metric else metric
            actual = data.get(key, data.get(metric.replace("_5fpr", "_at_5fpr_mean"), 0))
            diff = abs(actual - exp_val)
            tol = tolerance.get(metric.split("_")[0] + "_" + metric.split("_")[-1], 0.03)

            status = "PASS" if diff <= tol else "FAIL"
            print(f"  {model} {metric}: {actual:.3f} (expected {exp_val:.3f}) [{status}]")
            if status == "FAIL":
                all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass

if __name__ == "__main__":
    verify_results()
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Cause:** Python can't find the source directory.

**Solution:**
```bash
# Option 1: Run from project root
cd /path/to/Thesis_work
python run_experiments.py --all

# Option 2: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: TensorFlow Metal Errors (macOS)

**Cause:** TensorFlow Metal plugin issues on Apple Silicon.

**Solution:**
```bash
# Disable GPU, use CPU only
export CUDA_VISIBLE_DEVICES=""
python run_experiments.py --all
```

### Issue: Out of Memory

**Cause:** LSTM training requires ~8GB RAM for 230K sequences.

**Solution:**
```bash
# Option 1: Reduce batch size
# Edit src/config.py: batch_size = 16

# Option 2: Close other applications

# Option 3: Use swap/virtual memory
```

### Issue: Slow Training

**Cause:** LSTM on CPU is slow (~60 min/seed).

**Solutions:**
```bash
# Use GPU if available
pip install tensorflow-gpu

# Or reduce seeds
python run_experiments.py --all --seeds 1

# Or skip LSTM for quick verification
python run_experiments.py --model isolation_forest --seeds 5
```

### Issue: Different Results

**Possible Causes:**
1. Different random seed
2. Different TensorFlow version
3. Different NumPy version
4. Hardware floating-point differences

**Solution:**
```bash
# Verify package versions match
pip freeze | grep -E "numpy|tensorflow|scikit-learn"
# Expected:
# numpy==1.26.4
# scikit-learn==1.7.2
# tensorflow-macos==2.15.0  (or tensorflow==2.15.0)
```

### Issue: Dataset Version Mismatch

**Cause:** Using r5.2 instead of r4.2.

**Solution:**
```bash
# Use correct dataset version flag
python run_experiments.py --all --dataset r5.2

# Or rename directory
mv data/r5.2 data/r4.2
```

---

## Random Seeds

All experiments use controlled random seeds for reproducibility:

| Seed | Purpose |
|------|---------|
| 42 | Primary seed, all single-run tests |
| 43, 44, 45, 46 | Additional seeds for 5-seed runs |

To use different seeds:
```bash
python run_experiments.py --all --seeds 10  # Uses seeds 42-51
```

The seed controls:
- NumPy random state
- TensorFlow random state
- Python random module
- Train/test split shuffling
- Model weight initialization

---

## Citation

If you use this code, please cite:

```bibtex
@article{rimal2025insider,
  title={Unsupervised Temporal Behavioral Profiling for Insider Threat Detection: A Comparative Study},
  author={Rimal, Bipin},
  institution={Coventry University},
  year={2025}
}
```

---

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Open a new issue with:
   - Python version (`python --version`)
   - Package versions (`pip freeze`)
   - Full error traceback
   - Hardware specs
   - Steps to reproduce
