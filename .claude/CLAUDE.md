# Insider Threat Detection — MSc Thesis

Unsupervised anomaly detection on CMU-CERT r4.2. Four models, five seeds, one finding: the standard metric (AUC-ROC) gives the wrong answer at operational thresholds.

## Development Workflow

```bash
source .venv/bin/activate

# Quick test (1 model, 1 seed)
python run_experiments.py --model isolation_forest --quick

# Full suite (4 models × 5 seeds, ~4hrs CPU, ~1.5hrs GPU)
python run_experiments.py --all --seeds 5

# Generate figures
python generate_figures.py

# Compile paper (IEEE format)
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

```
src/
├── config.py                  # All hyperparameters (single source of truth)
├── models/
│   ├── isolation_forest.py    # 200 trees, auto contamination
│   ├── pca_anomaly.py         # PCA reconstruction error
│   ├── autoencoder.py         # Dense AE (encoder/decoder)
│   └── lstm_autoencoder.py    # 7-day temporal windows (stride 1)
├── experiments/
│   ├── runner.py              # Multi-seed experiment driver
│   ├── statistical_tests.py   # Wilcoxon, Cohen's d
│   └── visualizations.py      # Publication-ready figures
└── utils/                     # Logging, reproducibility
```

## Dataset: CMU-CERT r4.2

```
data/r4.2/
├── logon.csv     # Login/logout events
├── device.csv    # USB device activity
├── file.csv      # File operations
├── email.csv     # Email activity
└── http.csv      # Web browsing
data/answers/insiders.csv   # Ground truth labels
```

~1,000 employees, 18 months, 70 planted insider scenarios.
24 daily features (6 categories): Logon (6), Device (4), HTTP (5), Email (5), File (4).

Download from [KiltHub](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247).

## Key Results — Memorize These

| Model | AUC-ROC | Recall@5%FPR | Recall@10%FPR |
|-------|---------|-------------|---------------|
| Isolation Forest | **0.799** | 0.044 | 0.220 |
| LSTM Autoencoder | 0.770 | **0.149** | **0.254** |
| PCA Reconstruction | 0.612 | 0.049 | 0.129 |
| Dense Autoencoder | 0.659 | 0.048 | 0.118 |

Statistical significance: p=0.031 (Wilcoxon), Cohen's d=4.1.
85% of attack days evade all models ("boiling frog" pattern).

## Common Mistakes — Read This First

- **Never use 0.985 AUC-ROC.** That number was wrong. The correct best AUC-ROC is IF at 0.799. This error propagated across multiple files and has been eliminated. Do not reintroduce it.
- **IF wins on AUC-ROC. LSTM wins operationally.** The 3.4x gap at Recall@5%FPR is the thesis contribution, not the AUC-ROC ranking.
- **After-hours activity is NOT a strong predictor.** Ratio is 1.02x vs regular hours. USB device activity is the strongest signal (correlation 0.075).
- **The 85% evasion rate is a finding, not a failure.** It's a fundamental limitation of the unsupervised anomaly detection paradigm. Report it; don't hide it.
- **Temporal split, not random split.** Training: first 70% of timeline (normal data only). Testing: last 30%. This prevents data leakage.
- **5 random seeds minimum.** All results report mean ± std across seeds [42, 43, 44, 45, 46].

## Sibling Project

This thesis has a companion: `../threat-to-governance-pipeline/` which applies the same models to AI agent traces via a Unified Behavioural Feature Schema (UBFS). The CERT→TRAIL cross-domain transfer retains 97% detection power (IF: 0.731→0.711).

## Target Machine

**Ubuntu PC:** RTX 4060 (8GB VRAM), AMD Ryzen 5 5600X, 16GB RAM, 1TB SSD.
CMU-CERT dataset lives here. Run full experiments here.

**MacBook M4 Pro:** Development only. No CMU-CERT data (storage constraint).

## Code Style

- Google Python Style Guide
- Type annotations on public APIs
- Commit format: `<type>(<scope>): <description>`
