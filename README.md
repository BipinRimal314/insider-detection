# The Metric That Misleads

**Unsupervised insider threat detection — and why the standard evaluation metric gives the wrong answer.**

Bipin Rimal | Coventry University | 2026

---

## The Problem

If someone breaks into your office, the alarm goes off. But if someone with a key walks in and starts stealing, the alarm stays silent. That's the insider threat problem — the attacker is already authorized.

Traditional security tools (firewalls, IDS, access controls) draw a line between authorized and unauthorized. Insiders are on the authorized side. These tools have nothing to detect.

## The Approach

We learn what "normal" behavior looks like — how each of ~1,000 employees uses their computer over 18 months — and flag deviations. No labeled attack data required (unsupervised). We test whether watching behavior **over time** (temporal sequences) catches attacks that **daily snapshots** miss.

Four models, from simple to complex:

| Model | What it asks | Temporal? |
|-------|-------------|-----------|
| **Isolation Forest** | How easy is this point to isolate from the crowd? | No |
| **PCA Reconstruction** | Does this point fit the dominant patterns? | No |
| **Dense Autoencoder** | Can this day be compressed and reconstructed? | No |
| **LSTM Autoencoder** | Can this *week* be compressed and reconstructed? | **Yes** |

## The Finding

The standard metric says Isolation Forest wins. The operational metric says the LSTM wins — by 3.4x.

| Model | AUC-ROC | Recall@5%FPR |
|-------|---------|--------------|
| Isolation Forest | **0.799** | 0.044 |
| LSTM Autoencoder | 0.770 | **0.149** |

**AUC-ROC** asks: "How well does the model rank anomalies overall?" Isolation Forest wins.

**Recall@5%FPR** asks: "If my security team can only investigate 50 alerts per day out of 1,000 users, how many real attacks will they find?" The LSTM catches 3.4x more.

The divergence happens because AUC-ROC averages across all thresholds — including ones no practitioner would use. The LSTM concentrates its discriminative power exactly where low-FPR thresholds operate, because temporal anomalies (unusual *sequences* of behavior) produce distinctly high reconstruction errors.

**The lesson: how you measure matters as much as what you build.**

## What Gives Insiders Away

USB device activity is the strongest signal (correlation: 0.075). After-hours activity is *not* significantly more predictive than regular activity (ratio: 1.02x) — what you do matters more than when you do it.

## What Doesn't Get Caught

85% of attack days evade detection. Detected attacks are loud (+9.92 SD in after-hours browsing). Missed attacks are quiet — the "boiling frog" pattern where insiders pace their behavior to stay within normal bounds. This is a fundamental limitation of anomaly-based detection, not a tuning problem.

---

## Running the Experiments

### Prerequisites

- Python 3.11
- ~5GB disk space for the CMU-CERT dataset
- GPU recommended for LSTM training (CPU works but slower)

### Setup

```bash
# Create environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify
python -c "import tensorflow as tf; print(f'TF {tf.__version__}, GPU: {tf.config.list_physical_devices(\"GPU\")}')"
```

### Dataset

Download CMU-CERT r4.2 from [KiltHub](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) and extract to `data/`:

```
data/
├── r4.2/
│   ├── logon.csv
│   ├── device.csv
│   ├── file.csv
│   ├── email.csv
│   └── http.csv
└── answers/
    └── insiders.csv
```

### Run

```bash
# Quick test (1 model, 1 seed)
python run_experiments.py --model isolation_forest --quick

# Full suite (4 models x 5 seeds — reproduces all paper results)
python run_experiments.py --all --seeds 5

# Generate figures
python -c "from src.experiments.visualizations import generate_all_plots; generate_all_plots()"
```

### Compile the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Project Structure

```
insider-detection/
├── paper/                  # IEEE conference paper
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── src/
│   ├── data/               # Loading, preprocessing, feature engineering
│   ├── models/             # Isolation Forest, PCA, Dense AE, LSTM AE
│   ├── experiments/        # Multi-seed runner, metrics, visualizations
│   ├── utils/              # Reproducibility, logging
│   └── config.py           # All hyperparameters (documented)
├── data/                   # CMU-CERT dataset (gitignored)
├── results/                # Experiment outputs (gitignored)
├── run_experiments.py      # Entry point
└── requirements.txt        # Python dependencies
```

## Related Work

This project has a companion piece: [threat-to-governance-pipeline](../threat-to-governance-pipeline/) — which takes these same insider threat detection models and applies them to AI agent behavioral monitoring, showing that the two problems are structurally identical.

---

*MSc Research, Department of Computing, Coventry University.*
