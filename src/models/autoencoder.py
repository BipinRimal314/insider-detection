"""
Dense Autoencoder for anomaly detection.

Uses reconstruction error from a feedforward autoencoder
as the anomaly score. The autoencoder learns to compress
and reconstruct normal patterns; anomalies have higher
reconstruction error.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .base import StaticDetector


class DenseAutoencoder(StaticDetector):
    """
    Dense (feedforward) Autoencoder for anomaly detection.

    Architecture:
        Encoder: input -> hidden layers -> latent
        Decoder: latent -> hidden layers (reversed) -> output

    Anomaly score is the Mean Squared Error between input and reconstruction.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        latent_dim: int = 16,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        seed: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize Dense Autoencoder.

        Args:
            hidden_layers: List of hidden layer sizes for encoder.
            latent_dim: Size of latent (bottleneck) layer.
            activation: Activation function ('relu', 'tanh', 'leaky_relu').
            dropout: Dropout rate between layers.
            batch_size: Training batch size.
            epochs: Maximum training epochs.
            learning_rate: Adam optimizer learning rate.
            early_stopping_patience: Epochs without improvement before stopping.
            seed: Random seed.
            verbose: Keras verbosity level.
        """
        super().__init__(name="DenseAutoencoder", seed=seed)

        self.hidden_layers = hidden_layers or [64, 32]
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.model_ = None
        self.encoder_ = None
        self.decoder_ = None
        self.history_ = None
        self.input_dim_ = None
        self.threshold_ = None

    def _set_seed(self) -> None:
        """Set seeds for TensorFlow reproducibility."""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _get_activation(self):
        """Get activation layer."""
        if self.activation == "leaky_relu":
            return layers.LeakyReLU(alpha=0.1)
        return self.activation

    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build autoencoder model.

        Args:
            input_dim: Number of input features.

        Returns:
            Compiled Keras model.
        """
        self.input_dim_ = input_dim

        # Encoder
        encoder_input = layers.Input(shape=(input_dim,), name="encoder_input")
        x = encoder_input

        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, name=f"encoder_dense_{i}")(x)
            x = layers.Activation(self._get_activation(), name=f"encoder_act_{i}")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name=f"encoder_dropout_{i}")(x)

        latent = layers.Dense(self.latent_dim, name="latent")(x)
        latent = layers.Activation(self._get_activation(), name="latent_act")(latent)

        self.encoder_ = keras.Model(encoder_input, latent, name="encoder")

        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        x = decoder_input

        for i, units in enumerate(reversed(self.hidden_layers)):
            x = layers.Dense(units, name=f"decoder_dense_{i}")(x)
            x = layers.Activation(self._get_activation(), name=f"decoder_act_{i}")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout, name=f"decoder_dropout_{i}")(x)

        decoder_output = layers.Dense(input_dim, name="decoder_output")(x)

        self.decoder_ = keras.Model(decoder_input, decoder_output, name="decoder")

        # Full autoencoder
        autoencoder_input = layers.Input(shape=(input_dim,), name="autoencoder_input")
        encoded = self.encoder_(autoencoder_input)
        decoded = self.decoder_(encoded)

        model = keras.Model(autoencoder_input, decoded, name="autoencoder")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DenseAutoencoder":
        """
        Fit autoencoder on training data.

        Args:
            X: Training features (n_samples, n_features).
            y: Ignored (unsupervised method).

        Returns:
            self
        """
        X = self.validate_input(X)
        self._set_seed()

        # Build model
        self.model_ = self._build_model(X.shape[1])

        # Callbacks
        cb = [
            callbacks.EarlyStopping(
                monitor="loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        # Train
        self.history_ = self.model_.fit(
            X, X,  # Autoencoder: input = target
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=cb,
            verbose=self.verbose,
        )

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

        # Reconstruct
        X_reconstructed = self.model_.predict(X, verbose=0)

        # Compute MSE per sample
        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        return errors

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representations.

        Args:
            X: Input features.

        Returns:
            Latent vectors (n_samples, latent_dim).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before encoding")

        X = self.validate_input(X)
        return self.encoder_.predict(X, verbose=0)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": self.name,
            "seed": self.seed,
            "hidden_layers": self.hidden_layers,
            "latent_dim": self.latent_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }

    def get_training_history(self) -> Optional[Dict]:
        """Get training history."""
        if self.history_ is None:
            return None
        return {
            "loss": self.history_.history.get("loss", []),
            "val_loss": self.history_.history.get("val_loss", []),
        }


def create_dense_autoencoder(config: Optional[Dict] = None, seed: int = 42) -> DenseAutoencoder:
    """
    Factory function to create Dense Autoencoder with config.

    Args:
        config: Configuration dictionary.
        seed: Random seed.

    Returns:
        Configured DenseAutoencoder.
    """
    if config is None:
        from ..config import default_config
        config = default_config.autoencoder

    return DenseAutoencoder(
        hidden_layers=getattr(config, "hidden_layers", [64, 32]),
        latent_dim=getattr(config, "latent_dim", 16),
        activation=getattr(config, "activation", "relu"),
        dropout=getattr(config, "dropout", 0.2),
        batch_size=getattr(config, "batch_size", 64),
        epochs=getattr(config, "epochs", 100),
        learning_rate=getattr(config, "learning_rate", 0.001),
        early_stopping_patience=getattr(config, "early_stopping_patience", 10),
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
    model = DenseAutoencoder(
        hidden_layers=[32, 16],
        latent_dim=8,
        epochs=50,
        seed=42,
        verbose=0,
    )
    result = model.fit_predict(X_train, X_test)

    # Evaluate
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, result.scores)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Train time: {result.train_time:.3f}s")
    print(f"Inference time: {result.inference_time:.3f}s")

    history = model.get_training_history()
    print(f"Final loss: {history['loss'][-1]:.4f}")
