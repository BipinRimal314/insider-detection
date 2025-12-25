# Track Plan: Enhance Model Robustness and Hyperparameter Tuning

## Phase 1: Hyperparameter Sensitivity Infrastructure
- [x] 634a9b2 Task: Create `hyperparameter_sensitivity.py` script structure
- [x] e9ba018 Task: Implement LSTM sensitivity analysis logic (variable units/dropout)
- [x] 60c47dc Task: Implement Isolation Forest sensitivity analysis logic (variable estimators)
- [x] f2077f9 Task: Generate visualization plots for sensitivity results
- [~] Task: Conductor - User Manual Verification 'Hyperparameter Sensitivity Infrastructure' (Protocol in workflow.md)

## Phase 2: Robustness & Adversarial Testing
- [ ] Task: Implement "Boiling Frog" (Slow Evasion) attack generator in `synthetic_attacks.py`
- [ ] Task: Implement Data Perturbation simulation in `feature_engineering_polars.py`
- [ ] Task: Create `robustness_analysis.py` to execute and report on these tests
- [ ] Task: Conductor - User Manual Verification 'Robustness & Adversarial Testing' (Protocol in workflow.md)

## Phase 3: Dynamic Optimization & Thresholding
- [ ] Task: Implement `derive_dynamic_weights()` in `ensemble_system.py`
- [ ] Task: Update `EnsembleDetector` to use dynamic weights
- [ ] Task: Implement `optimize_threshold()` in `model_evaluation.py` (Youden's J)
- [ ] Task: Update alerting logic to use optimized thresholds
- [ ] Task: Conductor - User Manual Verification 'Dynamic Optimization & Thresholding' (Protocol in workflow.md)
