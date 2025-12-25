# Track Specification: Enhance Model Robustness and Hyperparameter Tuning

## 1. Goal
Systematically evaluate and optimize model hyperparameters (LSTM units, Isolation Forest estimators) and perform robustness testing (ablation studies, sensitivity analysis) to justify design choices and improve detection performance. This will transform ad-hoc configuration choices into empirically justified decisions, a critical requirement for high-quality research.

## 2. Core Features
- **Hyperparameter Sensitivity Analysis**:
    - SYSTEMATICALLY evaluate LSTM Autoencoder performance across varying architectures (e.g., [16], [32, 16], [64, 32]) and dropout rates.
    - SYSTEMATICALLY evaluate Isolation Forest performance across varying `n_estimators`.
    - GENERATE heatmaps and plots to visualize sensitivity.
- **Robustness Testing**:
    - IMPLEMENT "Boiling Frog" (slow evasion) synthetic attacks to test temporal detection limits.
    - IMPLEMENT data perturbation tests (randomly missing data) to evaluate system stability.
- **Dynamic Configuration**:
    - UPDATE `ensemble_system.py` to support dynamic weighting based on validation performance.
    - UPDATE `model_evaluation.py` to support automated threshold derivation (e.g., Youden's J statistic).

## 3. Success Metrics
- **Justification**: Empirical data (plots/tables) exists to justify every major hyperparameter choice in `config.py`.
- **Robustness**: The system's breaking point for slow-evasion attacks is quantified.
- **Performance**: Ensemble weights are optimized, potentially improving overall AUC-ROC.
