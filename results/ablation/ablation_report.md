# Ablation Study Report
## Feature Importance
Top 5 most important features:
- **daily_activity_count_peer_zscore**: +0.0313 AUC impact
- **daily_activity_count_self_zscore**: +0.0293 AUC impact
- **file_access_count_self_zscore**: +0.0149 AUC impact
- **email_count_self_zscore**: +0.0149 AUC impact
- **file_access_count_peer_zscore**: +0.0015 AUC impact

## Z-Score Feature Impact
- Without Z-scores: AUC = 0.5000
- With Z-scores: AUC = 0.8685
- **Improvement: +0.3685**
