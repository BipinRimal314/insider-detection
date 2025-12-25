"""
Cross-Dataset Generalization Evaluation

This module evaluates trained models on held-out datasets to test generalization.
Key for research: proves the model works on unseen attack patterns and users.

Usage:
    python cross_dataset_eval.py

Prerequisites:
    1. Models trained on train_datasets (run main.py --full first)
    2. Test datasets (r4.2, r5.1, r5.2) available in data/all_data/
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import joblib

# Ensure TensorFlow doesn't hang on Mac
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
import utils
from data_preprocessing_polars import DataPreprocessorPolars
from feature_engineering_polars import FeatureEngineerPolars

logger = utils.logger


class CrossDatasetEvaluator:
    """
    Evaluates trained models on held-out test datasets.
    """
    
    def __init__(self):
        self.cross_eval_config = getattr(config, 'CROSS_DATASET_EVAL', {})
        self.test_datasets = self.cross_eval_config.get('test_datasets', [])
        self.results_dir = config.RESULTS_DIR / 'cross_dataset'
        self.results_dir.mkdir(exist_ok=True)
        
    def run_evaluation(self):
        """Run full cross-dataset evaluation pipeline."""
        logger.info("=" * 80)
        logger.info("CROSS-DATASET GENERALIZATION EVALUATION")
        logger.info("=" * 80)
        
        if not self.test_datasets:
            logger.error("No test_datasets configured in CROSS_DATASET_EVAL")
            return
            
        logger.info(f"Test datasets: {self.test_datasets}")
        
        # Step 1: Process test datasets
        test_features = self._process_test_datasets()
        if test_features is None:
            return
            
        # Step 2: Load trained models
        models = self._load_trained_models()
        if not models:
            logger.error("No trained models found. Run main.py --full first.")
            return
            
        # Step 3: Evaluate each model on test data
        results = {}
        for model_name, model in models.items():
            logger.info(f"\nEvaluating {model_name} on test datasets...")
            metrics = self._evaluate_model(model_name, model, test_features)
            if metrics:
                results[model_name] = metrics
                
        # Step 4: Generate comparison report
        self._generate_report(results)
        
        return results
    
    def _process_test_datasets(self):
        """Process test datasets through preprocessing and feature engineering."""
        logger.info(f"\nProcessing test datasets: {self.test_datasets}")
        
        # Temporarily override DATASET_SUBSET to only process test datasets
        original_subset = config.DATASET_SUBSET
        config.DATASET_SUBSET = self.test_datasets
        
        try:
            # Create output paths for test data
            test_processed_dir = config.PROCESSED_DATA_DIR / 'test'
            test_processed_dir.mkdir(exist_ok=True)
            
            # Run preprocessing on test datasets
            preprocessor = DataPreprocessorPolars()
            preprocessor.processed_dir = test_processed_dir
            
            test_processed_file = test_processed_dir / 'processed_unified_logs.csv'
            
            # Only reprocess if not already done
            if not test_processed_file.exists():
                logger.info("Preprocessing test datasets...")
                preprocessor.run_pipeline()
            else:
                logger.info(f"Using cached test data: {test_processed_file}")
            
            # Run feature engineering on test data
            fe = FeatureEngineerPolars()
            fe.input_file = test_processed_file
            fe.daily_output = test_processed_dir / 'daily_features.parquet'
            fe.sequence_npy_output = test_processed_dir / 'sequences.npy'
            fe.sequence_labels_output = test_processed_dir / 'sequence_labels.npy'
            
            if not fe.daily_output.exists():
                logger.info("Generating test features...")
                fe.run_pipeline()
            else:
                logger.info(f"Using cached test features: {fe.daily_output}")
            
            # Load test features
            test_daily = pl.read_parquet(fe.daily_output)
            test_sequences = np.load(fe.sequence_npy_output) if fe.sequence_npy_output.exists() else None
            test_labels = np.load(fe.sequence_labels_output) if fe.sequence_labels_output.exists() else None
            
            return {
                'daily': test_daily,
                'sequences': test_sequences,
                'labels': test_labels
            }
            
        except Exception as e:
            logger.error(f"Failed to process test datasets: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Restore original subset
            config.DATASET_SUBSET = original_subset
    
    def _load_trained_models(self):
        """Load pre-trained models."""
        models = {}
        
        # Isolation Forest
        if_path = config.MODEL_PATHS.get('isolation_forest')
        if if_path and Path(if_path).exists():
            models['isolation_forest'] = joblib.load(if_path)
            logger.info(f"✓ Loaded Isolation Forest from {if_path}")
            
        # LSTM Autoencoder
        lstm_path = config.MODEL_PATHS.get('lstm_autoencoder')
        if lstm_path and Path(lstm_path).exists():
            import tensorflow as tf
            models['lstm_autoencoder'] = tf.keras.models.load_model(lstm_path)
            logger.info(f"✓ Loaded LSTM Autoencoder from {lstm_path}")
            
        # Deep Clustering (uses scaler + kmeans)
        dc_path = config.MODEL_PATHS.get('deep_clustering')
        if dc_path and Path(dc_path).exists():
            models['deep_clustering'] = joblib.load(dc_path)
            logger.info(f"✓ Loaded Deep Clustering from {dc_path}")
            
        return models
    
    def _evaluate_model(self, model_name, model, test_features):
        """Evaluate a single model on test features."""
        try:
            if model_name == 'isolation_forest':
                return self._evaluate_isolation_forest(model, test_features)
            elif model_name == 'lstm_autoencoder':
                return self._evaluate_lstm(model, test_features)
            elif model_name == 'deep_clustering':
                return self._evaluate_deep_clustering(model, test_features)
            else:
                logger.warning(f"Unknown model type: {model_name}")
                return None
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_isolation_forest(self, model, test_features):
        """Evaluate Isolation Forest on test daily features."""
        daily_df = test_features['daily'].to_pandas()
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['user', 'day', 'is_anomaly']
        feature_cols = [c for c in daily_df.columns if c not in exclude_cols and daily_df[c].dtype in ['float64', 'int64']]
        
        X_test = daily_df[feature_cols].fillna(0).values
        y_test = daily_df['is_anomaly'].values if 'is_anomaly' in daily_df.columns else None
        
        # Get predictions
        scores = -model.decision_function(X_test)  # Higher = more anomalous
        predictions = model.predict(X_test)
        predictions = (predictions == -1).astype(int)  # Convert to 0/1
        
        if y_test is not None:
            auc = roc_auc_score(y_test, scores) if y_test.sum() > 0 else 0
            return {
                'auc_roc': auc,
                'predictions': predictions.sum(),
                'true_positives': (predictions & y_test).sum(),
                'total_positives': y_test.sum()
            }
        return {'predictions': predictions.sum()}
    
    def _evaluate_lstm(self, model, test_features):
        """Evaluate LSTM Autoencoder on test sequences."""
        if test_features['sequences'] is None:
            logger.warning("No test sequences available for LSTM evaluation")
            return None
            
        X_test = test_features['sequences']
        y_test = test_features['labels']
        
        # Get reconstruction error
        reconstructed = model.predict(X_test, verbose=0)
        mse = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
        
        # Threshold at 95th percentile
        threshold = np.percentile(mse, 95)
        predictions = (mse > threshold).astype(int)
        
        if y_test is not None and y_test.sum() > 0:
            auc = roc_auc_score(y_test, mse)
            return {
                'auc_roc': auc,
                'predictions': predictions.sum(),
                'true_positives': (predictions & y_test).sum(),
                'total_positives': y_test.sum(),
                'mean_reconstruction_error': float(np.mean(mse))
            }
        return {'predictions': predictions.sum()}
    
    def _evaluate_deep_clustering(self, model, test_features):
        """Evaluate Deep Clustering on test features."""
        daily_df = test_features['daily'].to_pandas()
        
        exclude_cols = ['user', 'day', 'is_anomaly']
        feature_cols = [c for c in daily_df.columns if c not in exclude_cols and daily_df[c].dtype in ['float64', 'int64']]
        
        X_test = daily_df[feature_cols].fillna(0).values
        y_test = daily_df['is_anomaly'].values if 'is_anomaly' in daily_df.columns else None
        
        # Get cluster distances
        distances = model.transform(X_test).min(axis=1)  # Distance to nearest cluster
        threshold = np.percentile(distances, 95)
        predictions = (distances > threshold).astype(int)
        
        if y_test is not None and y_test.sum() > 0:
            auc = roc_auc_score(y_test, distances)
            return {
                'auc_roc': auc,
                'predictions': predictions.sum(),
                'true_positives': (predictions & y_test).sum(),
                'total_positives': y_test.sum()
            }
        return {'predictions': predictions.sum()}
    
    def _generate_report(self, results):
        """Generate comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-DATASET GENERALIZATION RESULTS")
        logger.info("=" * 80)
        
        report_lines = [
            "# Cross-Dataset Generalization Report",
            f"\nTest Datasets: {self.test_datasets}",
            "\n## Model Performance on Unseen Data\n",
            "| Model | AUC-ROC | Predictions | True Positives | Total Positives |",
            "|-------|---------|-------------|----------------|-----------------|"
        ]
        
        for model_name, metrics in results.items():
            auc = metrics.get('auc_roc', 'N/A')
            if isinstance(auc, float):
                auc = f"{auc:.4f}"
            preds = metrics.get('predictions', 'N/A')
            tp = metrics.get('true_positives', 'N/A')
            total = metrics.get('total_positives', 'N/A')
            
            report_lines.append(f"| {model_name} | {auc} | {preds} | {tp} | {total} |")
            logger.info(f"{model_name}: AUC={auc}, Predictions={preds}, TP={tp}/{total}")
        
        # Save report
        report_path = self.results_dir / 'generalization_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    evaluator = CrossDatasetEvaluator()
    evaluator.run_evaluation()
