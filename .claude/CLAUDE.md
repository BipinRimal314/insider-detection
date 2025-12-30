# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Insider Threat Detection System using unsupervised machine learning for behavioral anomaly detection. The system analyzes enterprise log data (CMU-CERT dataset) using three complementary models: Isolation Forest, LSTM Autoencoder, and Deep Clustering.

## Common Commands

### Running the Pipeline

```bash
# Activate virtual environment
source .venv_tf/bin/activate

# Run complete pipeline (all 6 stages)
python main.py --full

# Run specific stages
python main.py --stages 1 2 3

# Skip stages (e.g., run full but skip preprocessing)
python main.py --full --skip 1 2

# Individual stage shortcuts
python main.py --preprocess      # Stage 1: Data preprocessing
python main.py --feature-eng     # Stage 2: Feature engineering
python main.py --train           # Stage 3: Model training
python main.py --evaluate        # Stage 4: Model evaluation
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/test_synthetic_attacks.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Direct Script Execution

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

## Architecture

### Pipeline Stages

```
Raw Logs → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
           Preprocess  Features  Train     Evaluate  Ensemble  Visualize
```

| Stage | Script | Output |
|-------|--------|--------|
| 1 | `data_preprocessing_polars.py` | `data/processed/processed_unified_logs.csv` |
| 2 | `feature_engineering_polars.py` | `data/processed/daily_features.parquet`, `data/processed/sequence_features.parquet` |
| 3 | Model scripts | `models/*.pkl`, `models/*.keras` |
| 4 | `model_evaluation.py` | `results/evaluation_metrics.csv` |
| 5 | `ensemble_system.py` | `results/ensemble_results.csv`, `results/alerts.csv` |
| 6 | `visualization.py` | `results/plots/` |

### Model Architecture

- **Isolation Forest**: Point anomaly detection using random trees (sklearn)
- **LSTM Autoencoder**: Sequence reconstruction for temporal pattern deviation (TensorFlow/Keras)
- **Deep Clustering**: Autoencoder + KMeans for behavioral profiling (TensorFlow/Keras)

### Key Files

- `config.py`: All hyperparameters, paths, and settings. Modify this for tuning.
- `utils.py`: Shared utilities (logging, metrics, data loading, pseudonymization)
- `main.py`: Pipeline orchestrator with `InsiderThreatDetectionPipeline` class

### Data Flow

1. Raw CMU-CERT logs in `data/all_data/r1/` (logon, device, file, email, http CSVs)
2. Stage 1 unifies logs → `processed_unified_logs.csv`
3. Stage 2 creates daily features with Z-scores + sequences for LSTM
4. Stage 3 trains models on normal behavior, detects anomalies
5. Stage 5 combines predictions using weighted/majority/cascade voting

## Code Style

- Follow Google Python Style Guide (see `conductor/code_styleguides/python.md`)
- Use type annotations for public APIs
- Max line length: 80 characters
- All executable files must have `if __name__ == '__main__':` with `main()` function
- Module constants in `ALL_CAPS_WITH_UNDERSCORES`

## Workflow Guidelines

- Track task progress in `conductor/tracks/` plan files
- Use TDD: write failing tests before implementation
- Follow commit message format: `<type>(<scope>): <description>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Key Configuration Parameters

Located in `config.py`:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `DATASET_SUBSET` | Which CMU-CERT subsets to process | `['r1', 'r2', 'r3.1', 'r3.2', 'r4.1']` |
| `SEQUENCE_LENGTH` | LSTM input timesteps | 15 |
| `MAX_SEQUENCE_SAMPLES` | Debug sample limit (`None` for full) | `None` |
| `ISOLATION_FOREST['n_estimators']` | Number of trees | 50 |
| `LSTM_AUTOENCODER['epochs']` | Training epochs | 20 |
| `ENSEMBLE['final_threshold']` | Anomaly threshold | 0.7 |

## Testing

Test files are in `tests/`:
- `test_synthetic_attacks.py`: Validates synthetic attack injection
- `test_hyperparameter_sensitivity.py`: Hyperparameter sensitivity analysis
- `test_feature_engineering_perturbation.py`: Data perturbation robustness

## Data Requirements

Download CMU-CERT dataset to `data/all_data/r1/`:
- `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`
- `LDAP/` directory with user metadata

## Logging

Logs written to `logs/insider_threat_detection.log`. Set debug level in `config.py`:
```python
LOGGING = {
    'level': 'DEBUG',  # or 'INFO'
    ...
}
```
