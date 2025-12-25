"""
Explainability Module (XAI) for Insider Threat Detection

Provides interpretable explanations for model predictions using SHAP.
Critical for SOC analyst trust and regulatory compliance.

Features:
1. SHAP values for feature importance per prediction
2. Local explanations for individual alerts
3. Global feature importance across all predictions
4. Visualization of anomaly drivers

Usage:
    python explainability.py

Output:
    results/explanations/shap_summary.png
    results/explanations/alert_explanations.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from typing import Dict, List, Optional, Tuple

# Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
import utils

logger = utils.logger

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


class AlertExplainer:
    """
    Generates human-readable explanations for anomaly alerts.
    """
    
    def __init__(self):
        self.output_dir = config.RESULTS_DIR / 'explanations'
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.features = None
        self.feature_names = None
        self.explainer = None
        
    def load_model_and_data(self):
        """Load trained model and feature data."""
        # Load Isolation Forest (easiest to explain with SHAP)
        model_path = config.MODEL_PATHS.get('isolation_forest')
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
            logger.info(f"✓ Loaded Isolation Forest from {model_path}")
        else:
            logger.error("Isolation Forest model not found")
            return False
            
        # Load features
        daily_path = config.DAILY_FEATURES_FILE
        if daily_path.exists():
            df = pl.read_parquet(daily_path).to_pandas()
            
            # Get feature columns
            exclude_cols = ['user', 'day', 'is_anomaly']
            self.feature_names = [c for c in df.columns 
                                 if c not in exclude_cols 
                                 and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
            
            self.features = df[self.feature_names].fillna(0).values
            self.user_days = df[['user', 'day']].copy()
            self.labels = df['is_anomaly'].values if 'is_anomaly' in df.columns else None
            
            logger.info(f"✓ Loaded {len(self.features)} samples with {len(self.feature_names)} features")
            return True
        else:
            logger.error(f"Features not found at {daily_path}")
            return False
    
    def create_shap_explainer(self, sample_size=1000):
        """Create SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return False
            
        # Sample background data for SHAP
        if len(self.features) > sample_size:
            indices = np.random.choice(len(self.features), sample_size, replace=False)
            background = self.features[indices]
        else:
            background = self.features
            
        # Create TreeExplainer for Isolation Forest
        self.explainer = shap.TreeExplainer(self.model, background)
        logger.info("✓ Created SHAP TreeExplainer")
        return True
    
    def explain_predictions(self, indices: Optional[List[int]] = None, 
                           top_k: int = 100) -> pd.DataFrame:
        """
        Generate explanations for specified predictions.
        
        Args:
            indices: Sample indices to explain (default: top anomalies)
            top_k: Number of top anomalies to explain if indices not specified
            
        Returns:
            DataFrame with explanations
        """
        if self.explainer is None:
            if not self.create_shap_explainer():
                return None
        
        # Get anomaly scores
        scores = -self.model.decision_function(self.features)
        
        # Select samples to explain
        if indices is None:
            # Get top-k anomalies
            indices = np.argsort(scores)[-top_k:]
            
        logger.info(f"Explaining {len(indices)} predictions...")
        
        # Get SHAP values for selected samples
        X_explain = self.features[indices]
        shap_values = self.explainer.shap_values(X_explain, check_additivity=False)
        
        # Build explanation dataframe
        explanations = []
        for i, idx in enumerate(indices):
            # Get top contributing features
            feature_impacts = list(zip(self.feature_names, shap_values[i]))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_impacts[:5]
            explanation_text = self._format_explanation(
                top_features, 
                self.features[idx],
                scores[idx]
            )
            
            explanations.append({
                'sample_idx': idx,
                'user': self.user_days.iloc[idx]['user'],
                'day': self.user_days.iloc[idx]['day'],
                'anomaly_score': scores[idx],
                'is_insider': self.labels[idx] if self.labels is not None else None,
                'top_feature_1': top_features[0][0],
                'top_feature_1_impact': top_features[0][1],
                'top_feature_2': top_features[1][0] if len(top_features) > 1 else None,
                'top_feature_2_impact': top_features[1][1] if len(top_features) > 1 else None,
                'explanation': explanation_text
            })
        
        result_df = pd.DataFrame(explanations)
        
        # Save explanations
        output_path = self.output_dir / 'alert_explanations.csv'
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved explanations to {output_path}")
        
        return result_df
    
    def _format_explanation(self, feature_impacts: List[Tuple], 
                           feature_values: np.ndarray,
                           score: float) -> str:
        """Format human-readable explanation."""
        lines = [f"Anomaly Score: {score:.3f}"]
        lines.append("Key Factors:")
        
        for feat_name, impact in feature_impacts[:3]:
            feat_idx = self.feature_names.index(feat_name)
            value = feature_values[feat_idx]
            
            direction = "increased" if impact > 0 else "decreased"
            lines.append(f"  - {feat_name}: {value:.1f} ({direction} risk by {abs(impact):.3f})")
        
        return " | ".join(lines)
    
    def generate_global_importance(self):
        """Generate global feature importance plot."""
        if not SHAP_AVAILABLE:
            return self._fallback_importance()
            
        if self.explainer is None:
            if not self.create_shap_explainer():
                return None
        
        logger.info("Computing global feature importance...")
        
        # Sample for speed
        sample_size = min(5000, len(self.features))
        indices = np.random.choice(len(self.features), sample_size, replace=False)
        X_sample = self.features[indices]
        
        try:
            # Compute SHAP values (may fail on macOS with large models)
            shap_values = self.explainer.shap_values(X_sample, check_additivity=False)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            logger.info("Falling back to permutation importance...")
            return self._fallback_importance()
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                         show=False, max_display=20)
        plt.tight_layout()
        
        output_path = self.output_dir / 'shap_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved SHAP summary plot to {output_path}")
        
        # Also create bar plot of mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15][::-1], 
                importance_df['importance'][:15][::-1])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Global Feature Importance')
        plt.tight_layout()
        
        bar_path = self.output_dir / 'feature_importance_bar.png'
        plt.savefig(bar_path, dpi=300)
        plt.close()
        
        logger.info(f"✓ Saved feature importance bar plot to {bar_path}")
        
        return importance_df
    
    def _fallback_importance(self):
        """Fallback method when SHAP not available."""
        logger.info("Using permutation importance (SHAP not available)...")
        
        from sklearn.inspection import permutation_importance
        
        # Need labels for permutation importance
        if self.labels is None or self.labels.sum() == 0:
            logger.warning("No labels available for permutation importance")
            return None
            
        result = permutation_importance(self.model, self.features, self.labels, 
                                       n_repeats=10, random_state=42)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15][::-1], 
                importance_df['importance'][:15][::-1])
        plt.xlabel('Mean Decrease in AUC')
        plt.title('Feature Importance (Permutation)')
        plt.tight_layout()
        
        output_path = self.output_dir / 'permutation_importance.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"✓ Saved permutation importance to {output_path}")
        return importance_df
    
    def explain_single_alert(self, user: str, day: str) -> Dict:
        """
        Generate detailed explanation for a specific user-day.
        
        Args:
            user: User ID
            day: Day string (YYYY-MM-DD)
            
        Returns:
            Dictionary with explanation details
        """
        # Find the sample
        mask = (self.user_days['user'] == user) & (self.user_days['day'].astype(str) == day)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return {"error": f"No data found for user {user} on {day}"}
            
        idx = indices[0]
        
        # Get SHAP explanation
        explanations = self.explain_predictions([idx])
        if explanations is None or len(explanations) == 0:
            return {"error": "Failed to generate explanation"}
            
        return explanations.iloc[0].to_dict()
    
    def generate_report(self):
        """Generate comprehensive explainability report."""
        logger.info("=" * 80)
        logger.info("GENERATING EXPLAINABILITY REPORT")
        logger.info("=" * 80)
        
        if not self.load_model_and_data():
            return
        
        # Global importance
        importance_df = self.generate_global_importance()
        
        # Explain top anomalies
        explanations = self.explain_predictions(top_k=100)
        
        # Generate markdown report
        report_lines = [
            "# Model Explainability Report\n",
            "## Global Feature Importance\n"
        ]
        
        if importance_df is not None:
            report_lines.append("Top 10 most important features:\n")
            for _, row in importance_df.head(10).iterrows():
                report_lines.append(f"- **{row['feature']}**: {row['importance']:.4f}\n")
            report_lines.append("\n![SHAP Summary](shap_summary.png)\n")
        
        if explanations is not None:
            report_lines.append("\n## Sample Alert Explanations\n")
            for _, row in explanations.head(5).iterrows():
                report_lines.append(f"\n### Alert: {row['user']} on {row['day']}\n")
                report_lines.append(f"- Score: {row['anomaly_score']:.3f}\n")
                report_lines.append(f"- Key Factor: {row['top_feature_1']} (impact: {row['top_feature_1_impact']:.3f})\n")
                if row['is_insider']:
                    report_lines.append("- ⚠️ **TRUE INSIDER**\n")
        
        report_path = self.output_dir / 'explainability_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        logger.info(f"✓ Report saved to {report_path}")


if __name__ == "__main__":
    explainer = AlertExplainer()
    explainer.generate_report()
