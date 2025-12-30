"""
Isolation Forest for anomaly detection.

Isolation Forest detects anomalies based on the principle that
anomalous points are easier to isolate through random partitioning.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import StaticDetector


class IsolationForestDetector(StaticDetector):
    """
    Isolation Forest anomaly detector.

    Uses sklearn's IsolationForest implementation. Anomaly scores
    are the negative of sklearn's decision_function (so higher = more anomalous).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        contamination: str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of trees in the forest.
            max_samples: Number of samples per tree ('auto' = min(256, n_samples)).
            contamination: Expected proportion of outliers ('auto' or float).
            max_features: Features per tree (1.0 = all features).
            bootstrap: Whether to use bootstrap sampling.
            seed: Random seed for reproducibility.
            n_jobs: Parallel jobs (-1 = all cores).
        """
        super().__init__(name="IsolationForest", seed=seed)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs

        self.model_ = None
        self.threshold_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        """
        Fit Isolation Forest on training data.

        Args:
            X: Training features (n_samples, n_features).
            y: Ignored (unsupervised method).

        Returns:
            self
        """
        X = self.validate_input(X)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )

        self.model_.fit(X)
        self.is_fitted = True

        # Compute threshold from training data
        train_scores = self.score(X)
        self.threshold_ = np.percentile(train_scores, 95)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Args:
            X: Test features (n_samples, n_features).

        Returns:
            Anomaly scores (n_samples,). Higher = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        X = self.validate_input(X)

        # sklearn returns negative scores (more negative = more anomalous)
        # We negate to make higher = more anomalous
        return -self.model_.decision_function(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": self.name,
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
        }


def create_isolation_forest(config: Optional[Dict] = None, seed: int = 42) -> IsolationForestDetector:
    """
    Factory function to create Isolation Forest with config.

    Args:
        config: Configuration dictionary (from src.config).
        seed: Random seed.

    Returns:
        Configured IsolationForestDetector.
    """
    if config is None:
        from ..config import default_config
        config = default_config.isolation_forest

    return IsolationForestDetector(
        n_estimators=getattr(config, "n_estimators", 100),
        max_samples=getattr(config, "max_samples", "auto"),
        contamination=getattr(config, "contamination", "auto"),
        max_features=getattr(config, "max_features", 1.0),
        bootstrap=getattr(config, "bootstrap", False),
        seed=seed,
    )


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Generate synthetic data
    X_normal = np.random.randn(1000, 10)
    X_anomaly = np.random.randn(50, 10) + 3  # Shifted anomalies

    X_train = X_normal[:800]
    X_test = np.vstack([X_normal[800:], X_anomaly])
    y_test = np.array([0] * 200 + [1] * 50)

    # Fit and score
    model = IsolationForestDetector(seed=42)
    result = model.fit_predict(X_train, X_test)

    # Evaluate
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, result.scores)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Train time: {result.train_time:.3f}s")
    print(f"Inference time: {result.inference_time:.3f}s")
