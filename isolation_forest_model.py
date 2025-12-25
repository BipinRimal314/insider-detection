"""
Isolation Forest for Insider Threat Detection (Anomaly Detection)
Unsupervised anomaly detection based on daily behavioral profiles.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib # Standardized on joblib
import config
import utils

logger = utils.logger

class IsolationForestModel:
    """
    Isolation Forest wrapper for standardized training and evaluation.
    Supports dependency injection for hyperparameter tuning.
    """
    
    def __init__(self, n_estimators=None, contamination=None, max_samples=None, random_state=None):
        self.config = config.ISOLATION_FOREST.copy()
        # Override defaults if provided
        if n_estimators is not None: self.config['n_estimators'] = n_estimators
        if contamination is not None: self.config['contamination'] = contamination
        if max_samples is not None: self.config['max_samples'] = max_samples
        if random_state is not None: self.config['random_state'] = random_state
        
        self.model = None

    def train_and_evaluate(self, X_train, X_val, X_test, y_test, scaler=None):
        """
        Train the model and evaluate on test set.
        Args:
            X_train: Training data (usually flattened)
            X_val: Validation data (unused for IF but kept for API consistency)
            X_test: Test data
            y_test: Test labels
            scaler: Optional scaler (unused)
            
        Returns:
            history (None), auc, threshold, predictions, metrics
        """
        # Ensure input is 2D (flatten if 3D sequences passed by mistake)
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        self.model = IsolationForest(
            n_estimators=self.config['n_estimators'],
            max_samples=self.config['max_samples'],
            contamination=self.config['contamination'],
            max_features=self.config['max_features'],
            bootstrap=self.config['bootstrap'],
            n_jobs=self.config['n_jobs'],
            random_state=self.config['random_state'],
            verbose=self.config['verbose']
        )
        
        logger.info(f"Training Isolation Forest with params: n_estimators={self.config['n_estimators']}, contamination={self.config['contamination']}")
        
        # Train
        self.model.fit(X_train)
        
        # Predict / Score
        # decision_function: lower is more anomalous. We invert.
        raw_scores = self.model.decision_function(X_test)
        anomaly_scores = -raw_scores
        
        # Calculate threshold (e.g., 95th percentile) or use model's prediction
        # predict() returns -1 for outlier, 1 for inlier
        preds_sk = self.model.predict(X_test)
        predictions = np.where(preds_sk == -1, 1, 0)
        
        # Calculate simple threshold from scores for reporting
        threshold = np.percentile(anomaly_scores, 95)
        
        metrics = utils.calculate_metrics(y_test, predictions, anomaly_scores)
        
        return None, metrics['AUC-ROC'], threshold, predictions, metrics

# Legacy Compatibility
def load_data():
    """Load daily features for Isolation Forest"""
    file_path = config.DAILY_FEATURES_FILE
    
    if not file_path.exists():
        logger.error(f"Daily features not found at {file_path}")
        return None
    
    try:
        # Load Parquet
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded daily features: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading daily features: {e}")
        return None

def train_isolation_forest(X, **kwargs):
    """Legacy wrapper fitting the new Class logic"""
    # Extract known kwargs
    n_estimators = kwargs.get('n_estimators')
    contamination = kwargs.get('contamination')
    
    wrapper = IsolationForestModel(n_estimators=n_estimators, contamination=contamination)
    # Fit manual
    wrapper.train_and_evaluate(X, None, X, np.zeros(len(X))) # Dummy call to fit
    return wrapper.model

def main():
    logger.info(utils.generate_report_header("ISOLATION FOREST TRAINING"))
    
    # 1. Load Data
    df = load_data()
    if df is None:
        return None, None

    # 2. Preprocessing
    exclude_cols = ['user', 'day', 'is_anomaly', 'is_insider']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].fillna(0).values # Convert to numpy array
    
    true_labels = df['is_anomaly'].values if 'is_anomaly' in df.columns else np.zeros(len(X))

    # 3. Train & Evaluate using Wrapper
    model_wrapper = IsolationForestModel(**config.ISOLATION_FOREST)
    
    # For main run, we typically train on everything or split?
    # Original main() trained on X and evaluated on X (unsupervised assumption)
    # We will preserve that behavior
    _, auc, _, predictions, metrics = model_wrapper.train_and_evaluate(X, None, X, true_labels)
    
    clf = model_wrapper.model
    
    # Re-calculate scores for saving (train_and_evaluate does it internally on X_test=X)
    raw_scores = clf.decision_function(X)
    anomaly_scores = -raw_scores
    
    # 5. Save Model
    try:
        joblib.dump(clf, config.MODEL_PATHS['isolation_forest'])
        logger.info(f"Model saved to {config.MODEL_PATHS['isolation_forest']}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # 6. Save Predictions
    results_df = df[['user', 'day']].copy()
    results_df['true_label'] = true_labels
    results_df['prediction'] = predictions
    results_df['anomaly_score'] = anomaly_scores
    
    output_path = config.RESULTS_DIR / 'isolation_forest_predictions.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # 7. Metrics
    utils.print_metrics(metrics, "Isolation Forest")
    
    return clf, metrics

if __name__ == "__main__":
    main()