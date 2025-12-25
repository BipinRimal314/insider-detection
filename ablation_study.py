"""
Ablation Study Module

Systematically evaluates the contribution of:
1. Individual features (feature importance)
2. Hyperparameters (sequence length, LSTM units, etc.)
3. Model components (with/without Z-scores, etc.)

Usage:
    python ablation_study.py

Output:
    results/ablation/feature_importance.csv
    results/ablation/hyperparameter_analysis.csv
    results/ablation/ablation_report.md
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure TensorFlow doesn't hang on Mac
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
import utils

logger = utils.logger


class AblationStudy:
    """
    Performs ablation studies to understand model behavior.
    """
    
    def __init__(self):
        self.results_dir = config.RESULTS_DIR / 'ablation'
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data
        self.daily_features = None
        self.sequences = None
        self.labels = None
        self._load_data()
        
    def _load_data(self):
        """Load preprocessed data."""
        try:
            daily_path = config.DAILY_FEATURES_FILE
            if daily_path.exists():
                self.daily_features = pl.read_parquet(daily_path).to_pandas()
                logger.info(f"Loaded daily features: {self.daily_features.shape}")
                
            seq_path = config.PROCESSED_DATA_DIR / 'sequences.npy'
            labels_path = config.SEQUENCE_LABELS_FILE
            if seq_path.exists():
                self.sequences = np.load(seq_path)
                self.labels = np.load(labels_path) if labels_path.exists() else None
                logger.info(f"Loaded sequences: {self.sequences.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    def run_feature_ablation(self) -> pd.DataFrame:
        """
        Evaluate importance of each feature by removing one at a time.
        
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ABLATION STUDY")
        logger.info("=" * 60)
        
        if self.daily_features is None:
            logger.error("No daily features loaded")
            return None
            
        # Get feature columns
        exclude_cols = ['user', 'day', 'is_anomaly']
        feature_cols = [c for c in self.daily_features.columns 
                       if c not in exclude_cols 
                       and self.daily_features[c].dtype in ['float64', 'int64', 'int32', 'float32']]
        
        if 'is_anomaly' not in self.daily_features.columns:
            logger.warning("No labels available for ablation study")
            return None
            
        X = self.daily_features[feature_cols].fillna(0).values
        y = self.daily_features['is_anomaly'].values
        
        if y.sum() == 0:
            logger.warning("No positive labels in data")
            return None
        
        # Train baseline Isolation Forest
        from sklearn.ensemble import IsolationForest
        baseline_model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        baseline_model.fit(X)
        baseline_scores = -baseline_model.decision_function(X)
        baseline_auc = roc_auc_score(y, baseline_scores)
        
        logger.info(f"Baseline AUC (all features): {baseline_auc:.4f}")
        
        # Ablate each feature
        results = []
        for i, feat in enumerate(feature_cols):
            # Remove feature i
            X_ablated = np.delete(X, i, axis=1)
            
            # Retrain
            model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
            model.fit(X_ablated)
            scores = -model.decision_function(X_ablated)
            auc = roc_auc_score(y, scores)
            
            # Importance = drop in AUC when removed
            importance = baseline_auc - auc
            results.append({
                'feature': feat,
                'auc_without': auc,
                'importance': importance
            })
            
            logger.info(f"  {feat}: AUC={auc:.4f}, Importance={importance:+.4f}")
        
        results_df = pd.DataFrame(results).sort_values('importance', ascending=False)
        results_df.to_csv(self.results_dir / 'feature_importance.csv', index=False)
        
        # Plot
        self._plot_feature_importance(results_df)
        
        return results_df
    
    def run_sequence_length_ablation(self) -> pd.DataFrame:
        """
        Evaluate impact of different sequence lengths on LSTM performance.
        
        Returns:
            DataFrame with sequence length vs AUC
        """
        logger.info("\n" + "=" * 60)
        logger.info("SEQUENCE LENGTH ABLATION")
        logger.info("=" * 60)
        
        if self.daily_features is None:
            return None
            
        # Get feature data for sequence creation
        exclude_cols = ['user', 'day', 'is_anomaly']
        feature_cols = [c for c in self.daily_features.columns 
                       if c not in exclude_cols 
                       and self.daily_features[c].dtype in ['float64', 'int64', 'int32', 'float32']]
        
        sequence_lengths = [5, 10, 15, 20, 30]
        results = []
        
        for seq_len in sequence_lengths:
            logger.info(f"\nTesting sequence length: {seq_len}")
            
            # Create sequences with this length
            X_seq, y_seq = self._create_sequences_with_length(
                self.daily_features, feature_cols, seq_len
            )
            
            if X_seq is None or len(X_seq) < 1000:
                logger.warning(f"Not enough sequences for length {seq_len}")
                continue
                
            # Quick LSTM evaluation (reduced epochs for speed)
            auc = self._quick_lstm_eval(X_seq, y_seq, epochs=5)
            
            results.append({
                'sequence_length': seq_len,
                'num_sequences': len(X_seq),
                'auc_roc': auc
            })
            logger.info(f"  Sequence Length {seq_len}: AUC={auc:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.results_dir / 'sequence_length_analysis.csv', index=False)
        
        return results_df
    
    def run_zscore_ablation(self) -> Dict:
        """
        Compare performance with and without Z-score features.
        
        Returns:
            Dict with comparison results
        """
        logger.info("\n" + "=" * 60)
        logger.info("Z-SCORE FEATURE ABLATION")
        logger.info("=" * 60)
        
        if self.daily_features is None:
            return None
            
        exclude_cols = ['user', 'day', 'is_anomaly']
        
        # Identify Z-score columns
        zscore_cols = [c for c in self.daily_features.columns if 'zscore' in c.lower()]
        non_zscore_cols = [c for c in self.daily_features.columns 
                         if c not in exclude_cols 
                         and c not in zscore_cols
                         and self.daily_features[c].dtype in ['float64', 'int64', 'int32', 'float32']]
        all_feature_cols = non_zscore_cols + zscore_cols
        
        if 'is_anomaly' not in self.daily_features.columns:
            logger.warning("No labels available")
            return None
            
        y = self.daily_features['is_anomaly'].values
        
        from sklearn.ensemble import IsolationForest
        
        results = {}
        
        # Without Z-scores
        X_no_zscore = self.daily_features[non_zscore_cols].fillna(0).values
        model_no_z = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        model_no_z.fit(X_no_zscore)
        scores_no_z = -model_no_z.decision_function(X_no_zscore)
        auc_no_z = roc_auc_score(y, scores_no_z) if y.sum() > 0 else 0
        
        results['without_zscore'] = {
            'num_features': len(non_zscore_cols),
            'auc_roc': auc_no_z
        }
        logger.info(f"Without Z-scores ({len(non_zscore_cols)} features): AUC={auc_no_z:.4f}")
        
        # With Z-scores
        if zscore_cols:
            X_with_zscore = self.daily_features[all_feature_cols].fillna(0).values
            model_with_z = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
            model_with_z.fit(X_with_zscore)
            scores_with_z = -model_with_z.decision_function(X_with_zscore)
            auc_with_z = roc_auc_score(y, scores_with_z) if y.sum() > 0 else 0
            
            results['with_zscore'] = {
                'num_features': len(all_feature_cols),
                'auc_roc': auc_with_z,
                'zscore_features': zscore_cols
            }
            logger.info(f"With Z-scores ({len(all_feature_cols)} features): AUC={auc_with_z:.4f}")
            logger.info(f"  Improvement: {auc_with_z - auc_no_z:+.4f}")
        else:
            logger.warning("No Z-score features found. Run feature engineering first.")
        
        return results
    
    def _create_sequences_with_length(self, df, feature_cols, seq_len, stride=None):
        """Create sequences with specified length."""
        if stride is None:
            stride = max(1, seq_len // 2)
            
        df_sorted = df.sort_values(['user', 'day'])
        
        X_list, y_list = [], []
        has_labels = 'is_anomaly' in df.columns
        
        for user, group in df_sorted.groupby('user'):
            if len(group) < seq_len:
                continue
                
            features = group[feature_cols].fillna(0).values
            labels = group['is_anomaly'].values if has_labels else np.zeros(len(group))
            
            for i in range(0, len(group) - seq_len + 1, stride):
                X_list.append(features[i:i+seq_len])
                y_list.append(labels[i+seq_len-1])  # Label of last day in sequence
        
        if not X_list:
            return None, None
            
        return np.array(X_list), np.array(y_list)
    
    def _quick_lstm_eval(self, X, y, epochs=5):
        """Quick LSTM training and evaluation."""
        import tensorflow as tf
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train on normal only
        X_train_normal = X_train[y_train == 0]
        
        if len(X_train_normal) < 100:
            return 0.5
        
        # Build simple autoencoder
        input_dim = X.shape[2]
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=False, input_shape=(X.shape[1], input_dim)),
            tf.keras.layers.RepeatVector(X.shape[1]),
            tf.keras.layers.LSTM(16, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim))
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_normal, X_train_normal, epochs=epochs, batch_size=256, verbose=0)
        
        # Evaluate
        reconstructed = model.predict(X_test, verbose=0)
        mse = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
        
        if y_test.sum() > 0:
            return roc_auc_score(y_test, mse)
        return 0.5
    
    def _plot_feature_importance(self, results_df):
        """Plot feature importance."""
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        sorted_df = results_df.sort_values('importance', ascending=True)
        
        colors = ['green' if x > 0 else 'red' for x in sorted_df['importance']]
        plt.barh(sorted_df['feature'], sorted_df['importance'], color=colors)
        plt.xlabel('Importance (AUC drop when removed)')
        plt.ylabel('Feature')
        plt.title('Feature Importance via Ablation')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=300)
        plt.close()
        logger.info(f"Saved feature importance plot to {self.results_dir / 'feature_importance.png'}")
    
    def generate_report(self):
        """Generate comprehensive ablation report."""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING ABLATION REPORT")
        logger.info("=" * 60)
        
        feature_results = self.run_feature_ablation()
        zscore_results = self.run_zscore_ablation()
        
        # Generate markdown report
        report = ["# Ablation Study Report\n"]
        
        if feature_results is not None:
            report.append("## Feature Importance\n")
            report.append("Top 5 most important features:\n")
            for _, row in feature_results.head(5).iterrows():
                report.append(f"- **{row['feature']}**: {row['importance']:+.4f} AUC impact\n")
        
        if zscore_results:
            report.append("\n## Z-Score Feature Impact\n")
            if 'without_zscore' in zscore_results:
                report.append(f"- Without Z-scores: AUC = {zscore_results['without_zscore']['auc_roc']:.4f}\n")
            if 'with_zscore' in zscore_results:
                report.append(f"- With Z-scores: AUC = {zscore_results['with_zscore']['auc_roc']:.4f}\n")
                improvement = zscore_results['with_zscore']['auc_roc'] - zscore_results['without_zscore']['auc_roc']
                report.append(f"- **Improvement: {improvement:+.4f}**\n")
        
        report_path = self.results_dir / 'ablation_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        logger.info(f"✓ Ablation report saved to {report_path}")


if __name__ == "__main__":
    study = AblationStudy()
    study.generate_report()
