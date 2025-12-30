"""
Base interface for anomaly detection models.

All models implement a common interface for training,
scoring, and evaluation, enabling fair comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ModelResult:
    """Container for model prediction results."""

    # Anomaly scores (higher = more anomalous)
    scores: np.ndarray

    # Binary predictions (optional, threshold-dependent)
    predictions: Optional[np.ndarray] = None

    # Threshold used for binary predictions
    threshold: Optional[float] = None

    # Training time in seconds
    train_time: float = 0.0

    # Inference time in seconds
    inference_time: float = 0.0

    # Additional model-specific information
    metadata: Optional[Dict[str, Any]] = None


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection models.

    All models must implement:
        - fit(): Train on normal data
        - score(): Compute anomaly scores
        - predict(): Binary predictions (optional)
    """

    def __init__(self, name: str, seed: int = 42, **kwargs):
        """
        Initialize detector.

        Args:
            name: Model name for logging/reporting.
            seed: Random seed for reproducibility.
            **kwargs: Model-specific parameters.
        """
        self.name = name
        self.seed = seed
        self.is_fitted = False
        self._set_seed()

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseAnomalyDetector":
        """
        Fit the model on training data.

        For unsupervised methods, y is ignored.
        Training should only use normal (non-anomalous) samples.

        Args:
            X: Training features (n_samples, n_features) or (n_samples, seq_len, n_features).
            y: Optional labels (ignored for unsupervised methods).

        Returns:
            self
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples.

        Args:
            X: Test features.

        Returns:
            Anomaly scores (n_samples,).
        """
        pass

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Test features.
            threshold: Decision threshold (default: use fitted threshold).

        Returns:
            Binary predictions (n_samples,): 1 = anomaly, 0 = normal.
        """
        scores = self.score(X)
        if threshold is None:
            threshold = getattr(self, "threshold_", np.percentile(scores, 95))
        return (scores > threshold).astype(int)

    def fit_predict(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> ModelResult:
        """
        Fit on training data and score test data.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Optional training labels.

        Returns:
            ModelResult with scores and timing.
        """
        import time

        # Fit
        start = time.time()
        self.fit(X_train, y_train)
        train_time = time.time() - start

        # Score
        start = time.time()
        scores = self.score(X_test)
        inference_time = time.time() - start

        return ModelResult(
            scores=scores,
            train_time=train_time,
            inference_time=inference_time,
            metadata={"model_name": self.name, "seed": self.seed},
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of parameter names and values.
        """
        return {"name": self.name, "seed": self.seed}

    def set_params(self, **params) -> "BaseAnomalyDetector":
        """
        Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"


class StaticDetector(BaseAnomalyDetector):
    """Base class for static (non-temporal) detectors."""

    def validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and reshape input.

        Args:
            X: Input array.

        Returns:
            2D array (n_samples, n_features).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            # Flatten sequence data
            X = X.reshape(X.shape[0], -1)
        return X


class TemporalDetector(BaseAnomalyDetector):
    """Base class for temporal (sequence-based) detectors."""

    def validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and reshape input.

        Args:
            X: Input array.

        Returns:
            3D array (n_samples, seq_len, n_features).
        """
        X = np.asarray(X)
        if X.ndim == 2:
            # Add sequence dimension
            X = X.reshape(X.shape[0], 1, X.shape[1])
        elif X.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {X.shape}")
        return X
