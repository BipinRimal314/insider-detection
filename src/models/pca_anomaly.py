"""
PCA-based anomaly detection.

Uses reconstruction error from PCA as anomaly score.
Points that cannot be well-reconstructed by the principal
components learned from normal data are considered anomalous.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import StaticDetector


class PCAAnomalyDetector(StaticDetector):
    """
    PCA Reconstruction Error anomaly detector.

    Fits PCA on normal data and uses reconstruction error
    as the anomaly score. Higher reconstruction error indicates
    the sample deviates from the learned normal patterns.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        whiten: bool = False,
        seed: int = 42,
    ):
        """
        Initialize PCA detector.

        Args:
            n_components: Number of components (None = use variance_threshold).
            variance_threshold: Cumulative variance to explain if n_components is None.
            whiten: Whether to whiten the components.
            seed: Random seed for reproducibility.
        """
        super().__init__(name="PCA_Reconstruction", seed=seed)

        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.whiten = whiten

        self.scaler_ = None
        self.pca_ = None
        self.threshold_ = None
        self.explained_variance_ratio_ = None
        self.n_components_fitted_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAAnomalyDetector":
        """
        Fit PCA on training data.

        Args:
            X: Training features (n_samples, n_features).
            y: Ignored (unsupervised method).

        Returns:
            self
        """
        X = self.validate_input(X)

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Determine number of components
        if self.n_components is None:
            # Use variance threshold
            pca_full = PCA(random_state=self.seed)
            pca_full.fit(X_scaled)

            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            n_components = max(1, min(n_components, X.shape[1]))
        else:
            n_components = self.n_components

        # Fit PCA with determined components
        self.pca_ = PCA(
            n_components=n_components,
            whiten=self.whiten,
            random_state=self.seed,
        )
        self.pca_.fit(X_scaled)

        self.n_components_fitted_ = n_components
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.is_fitted = True

        # Compute threshold from training data
        train_scores = self.score(X)
        self.threshold_ = np.percentile(train_scores, 95)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error as anomaly score.

        Args:
            X: Test features (n_samples, n_features).

        Returns:
            Reconstruction errors (n_samples,). Higher = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        X = self.validate_input(X)

        # Scale
        X_scaled = self.scaler_.transform(X)

        # Project and reconstruct
        X_transformed = self.pca_.transform(X_scaled)
        X_reconstructed = self.pca_.inverse_transform(X_transformed)

        # Compute reconstruction error (MSE per sample)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

        return errors

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": self.name,
            "seed": self.seed,
            "n_components": self.n_components,
            "variance_threshold": self.variance_threshold,
            "whiten": self.whiten,
            "n_components_fitted": self.n_components_fitted_,
            "explained_variance": (
                float(sum(self.explained_variance_ratio_))
                if self.explained_variance_ratio_ is not None
                else None
            ),
        }


def create_pca_detector(config: Optional[Dict] = None, seed: int = 42) -> PCAAnomalyDetector:
    """
    Factory function to create PCA detector with config.

    Args:
        config: Configuration dictionary (from src.config).
        seed: Random seed.

    Returns:
        Configured PCAAnomalyDetector.
    """
    if config is None:
        from ..config import default_config
        config = default_config.pca

    return PCAAnomalyDetector(
        n_components=getattr(config, "n_components", None),
        variance_threshold=getattr(config, "variance_threshold", 0.95),
        whiten=getattr(config, "whiten", False),
        seed=seed,
    )


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Generate synthetic data
    X_normal = np.random.randn(1000, 10)
    X_anomaly = np.random.randn(50, 10) + 3

    X_train = X_normal[:800]
    X_test = np.vstack([X_normal[800:], X_anomaly])
    y_test = np.array([0] * 200 + [1] * 50)

    # Fit and score
    model = PCAAnomalyDetector(variance_threshold=0.95, seed=42)
    result = model.fit_predict(X_train, X_test)

    print(f"Components fitted: {model.n_components_fitted_}")
    print(f"Variance explained: {sum(model.explained_variance_ratio_):.2%}")

    # Evaluate
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, result.scores)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Train time: {result.train_time:.3f}s")
    print(f"Inference time: {result.inference_time:.3f}s")
