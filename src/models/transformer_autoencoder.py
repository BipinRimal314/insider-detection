"""
Transformer Autoencoder for temporal anomaly detection.

Uses self-attention mechanism to capture temporal dependencies
in sequence data. Anomalies are detected by high reconstruction error.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from .base import TemporalDetector


class TransformerAutoencoder(TemporalDetector):
    """
    Transformer Autoencoder for sequence anomaly detection.

    Architecture:
        Encoder: Multi-head self-attention layers with positional encoding
        Decoder: Multi-head self-attention layers that reconstruct the sequence

    The model learns to reconstruct sequences of normal behavior.
    Anomalous sequences have higher reconstruction error.
    """

    def __init__(
        self,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        seed: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize Transformer Autoencoder.

        Args:
            num_heads: Number of attention heads.
            ff_dim: Hidden dimension in feedforward layers.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            batch_size: Training batch size.
            epochs: Maximum training epochs.
            learning_rate: Adam optimizer learning rate.
            early_stopping_patience: Epochs without improvement before stopping.
            seed: Random seed.
            verbose: Keras verbosity level.
        """
        super().__init__(name="TransformerAutoencoder", seed=seed)

        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.model_ = None
        self.history_ = None
        self.seq_length_ = None
        self.n_features_ = None
        self.threshold_ = None

    def _set_seed(self) -> None:
        """Set seeds for TensorFlow reproducibility."""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _positional_encoding(self, seq_length: int, d_model: int) -> np.ndarray:
        """Generate positional encoding."""
        positions = np.arange(seq_length)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return angles.astype(np.float32)

    def _transformer_encoder_block(self, x, head_size, num_heads, ff_dim, dropout):
        """Single transformer encoder block."""
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation="relu")(x1)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(x.shape[-1])(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        return layers.LayerNormalization(epsilon=1e-6)(x1 + ffn)

    def _build_model(self, seq_length: int, n_features: int) -> keras.Model:
        """
        Build Transformer autoencoder model.

        Args:
            seq_length: Length of input sequences.
            n_features: Number of features per timestep.

        Returns:
            Compiled Keras model.
        """
        self.seq_length_ = seq_length
        self.n_features_ = n_features

        # Input
        inputs = layers.Input(shape=(seq_length, n_features))

        # Project to model dimension
        x = layers.Dense(self.ff_dim)(inputs)

        # Add positional encoding
        pos_encoding = self._positional_encoding(seq_length, self.ff_dim)
        x = x + pos_encoding

        # Encoder: stack of transformer blocks
        head_size = self.ff_dim // self.num_heads
        for _ in range(self.num_layers):
            x = self._transformer_encoder_block(
                x, head_size, self.num_heads, self.ff_dim, self.dropout
            )

        # Decoder: stack of transformer blocks (reconstruction)
        for _ in range(self.num_layers):
            x = self._transformer_encoder_block(
                x, head_size, self.num_heads, self.ff_dim, self.dropout
            )

        # Output projection
        outputs = layers.Dense(n_features)(x)

        model = keras.Model(inputs, outputs, name="transformer_autoencoder")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        return model

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TransformerAutoencoder":
        """
        Fit Transformer autoencoder on training sequences.

        Args:
            X: Training sequences (n_samples, seq_length, n_features).
            y: Ignored (unsupervised method).

        Returns:
            self
        """
        X = self.validate_input(X)
        self._set_seed()

        n_samples, seq_length, n_features = X.shape

        # Build model
        self.model_ = self._build_model(seq_length, n_features)

        # Callbacks
        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
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
            X: Test sequences (n_samples, seq_length, n_features).

        Returns:
            Reconstruction errors (n_samples,). Higher = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        X = self.validate_input(X)

        # Reconstruct
        X_reconstructed = self.model_.predict(X, verbose=0)

        # Compute MSE per sequence (average over timesteps and features)
        errors = np.mean((X - X_reconstructed) ** 2, axis=(1, 2))

        return errors

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": self.name,
            "seed": self.seed,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
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


def create_transformer_autoencoder(config: Optional[Dict] = None, seed: int = 42) -> TransformerAutoencoder:
    """
    Factory function to create Transformer Autoencoder with config.

    Args:
        config: Configuration dictionary.
        seed: Random seed.

    Returns:
        Configured TransformerAutoencoder.
    """
    default_params = {
        "num_heads": 4,
        "ff_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "early_stopping_patience": 15,
    }

    if config:
        default_params.update(config)

    return TransformerAutoencoder(**default_params, seed=seed)


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Generate synthetic sequence data
    seq_length = 7
    n_features = 24

    # Normal sequences
    n_normal = 500
    X_normal = np.random.randn(n_normal, seq_length, n_features) * 0.5

    # Anomalous sequences: higher variance
    n_anomaly = 25
    X_anomaly = np.random.randn(n_anomaly, seq_length, n_features) * 2.0

    X_train = X_normal[:400]
    X_test = np.vstack([X_normal[400:], X_anomaly])
    y_test = np.array([0] * 100 + [1] * n_anomaly)

    # Fit and score
    model = TransformerAutoencoder(
        num_heads=4,
        ff_dim=32,
        num_layers=2,
        epochs=30,
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
