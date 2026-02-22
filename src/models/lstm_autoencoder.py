"""
LSTM Autoencoder for temporal anomaly detection.

Uses an encoder-decoder LSTM architecture to learn temporal
patterns from sequence data. Anomalies are detected by high
reconstruction error on sequences that deviate from learned patterns.
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


class LSTMAutoencoder(TemporalDetector):
    """
    LSTM Autoencoder for sequence anomaly detection.

    Architecture:
        Encoder: LSTM layers that compress sequence to fixed-size latent vector
        Decoder: LSTM layers that reconstruct sequence from latent vector

    The model learns to reconstruct sequences of normal behavior.
    Anomalous sequences have higher reconstruction error.
    """

    def __init__(
        self,
        encoder_units: List[int] = None,
        decoder_units: List[int] = None,
        latent_dim: int = 16,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        bidirectional: bool = False,
        seed: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize LSTM Autoencoder.

        Args:
            encoder_units: LSTM units for encoder layers.
            decoder_units: LSTM units for decoder layers.
            latent_dim: Size of latent representation.
            dropout: Dropout rate between layers.
            recurrent_dropout: Dropout within LSTM cells.
            batch_size: Training batch size.
            epochs: Maximum training epochs.
            learning_rate: Adam optimizer learning rate.
            early_stopping_patience: Epochs without improvement before stopping.
            bidirectional: Whether to use bidirectional LSTM.
            seed: Random seed.
            verbose: Keras verbosity level.
        """
        super().__init__(name="LSTMAutoencoder", seed=seed)

        self.encoder_units = encoder_units or [64, 32]
        self.decoder_units = decoder_units or [32, 64]
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.bidirectional = bidirectional
        self.verbose = verbose

        self.model_ = None
        self.encoder_ = None
        self.history_ = None
        self.seq_length_ = None
        self.n_features_ = None
        self.threshold_ = None

    def _set_seed(self) -> None:
        """Set seeds for TensorFlow reproducibility."""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _build_model(self, seq_length: int, n_features: int) -> keras.Model:
        """
        Build LSTM autoencoder model.

        Args:
            seq_length: Length of input sequences.
            n_features: Number of features per timestep.

        Returns:
            Compiled Keras model.
        """
        self.seq_length_ = seq_length
        self.n_features_ = n_features

        # Input
        inputs = layers.Input(shape=(seq_length, n_features), name="encoder_input")

        # Encoder
        x = inputs
        for i, units in enumerate(self.encoder_units[:-1]):
            lstm = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name=f"encoder_lstm_{i}",
            )
            if self.bidirectional:
                x = layers.Bidirectional(lstm, name=f"encoder_bidir_{i}")(x)
            else:
                x = lstm(x)

        # Final encoder LSTM (no return_sequences, outputs fixed-size vector)
        final_encoder = layers.LSTM(
            self.encoder_units[-1],
            return_sequences=False,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            name="encoder_lstm_final",
        )
        if self.bidirectional:
            encoded = layers.Bidirectional(final_encoder, name="encoder_bidir_final")(x)
        else:
            encoded = final_encoder(x)

        # Latent representation
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent")(encoded)

        # Repeat latent vector for each timestep
        repeated = layers.RepeatVector(seq_length, name="repeat")(latent)

        # Decoder
        x = repeated
        for i, units in enumerate(self.decoder_units):
            lstm = layers.LSTM(
                units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name=f"decoder_lstm_{i}",
            )
            if self.bidirectional:
                x = layers.Bidirectional(lstm, name=f"decoder_bidir_{i}")(x)
            else:
                x = lstm(x)

        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(n_features), name="output"
        )(x)

        model = keras.Model(inputs, outputs, name="lstm_autoencoder")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        # Create encoder model for latent space analysis
        self.encoder_ = keras.Model(inputs, latent, name="encoder")

        return model

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LSTMAutoencoder":
        """
        Fit LSTM autoencoder on training sequences.

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

    def score_per_timestep(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error per timestep.

        Useful for identifying which part of a sequence is anomalous.

        Args:
            X: Test sequences (n_samples, seq_length, n_features).

        Returns:
            Per-timestep errors (n_samples, seq_length).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        X = self.validate_input(X)
        X_reconstructed = self.model_.predict(X, verbose=0)

        # MSE per timestep (average over features only)
        errors = np.mean((X - X_reconstructed) ** 2, axis=2)

        return errors

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representations of sequences.

        Args:
            X: Input sequences.

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
            "encoder_units": self.encoder_units,
            "decoder_units": self.decoder_units,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "bidirectional": self.bidirectional,
        }

    def get_training_history(self) -> Optional[Dict]:
        """Get training history."""
        if self.history_ is None:
            return None
        return {
            "loss": self.history_.history.get("loss", []),
            "val_loss": self.history_.history.get("val_loss", []),
        }


def create_lstm_autoencoder(config: Optional[Dict] = None, seed: int = 42) -> LSTMAutoencoder:
    """
    Factory function to create LSTM Autoencoder with config.

    Args:
        config: Configuration dictionary.
        seed: Random seed.

    Returns:
        Configured LSTMAutoencoder.
    """
    if config is None:
        from ..config import default_config
        config = default_config.lstm_autoencoder

    return LSTMAutoencoder(
        encoder_units=getattr(config, "encoder_units", [64, 32]),
        decoder_units=getattr(config, "decoder_units", [32, 64]),
        latent_dim=getattr(config, "latent_dim", 16),
        dropout=getattr(config, "dropout", 0.2),
        recurrent_dropout=getattr(config, "recurrent_dropout", 0.0),
        batch_size=getattr(config, "batch_size", 32),
        epochs=getattr(config, "epochs", 100),
        learning_rate=getattr(config, "learning_rate", 0.001),
        early_stopping_patience=getattr(config, "early_stopping_patience", 15),
        bidirectional=getattr(config, "bidirectional", False),
        seed=seed,
    )


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Generate synthetic sequence data
    seq_length = 10
    n_features = 5

    # Normal sequences: random walk
    n_normal = 500
    X_normal = np.cumsum(np.random.randn(n_normal, seq_length, n_features) * 0.1, axis=1)

    # Anomalous sequences: sudden jump
    n_anomaly = 25
    X_anomaly = np.cumsum(np.random.randn(n_anomaly, seq_length, n_features) * 0.1, axis=1)
    X_anomaly[:, seq_length//2:, :] += 3  # Jump in middle

    X_train = X_normal[:400]
    X_test = np.vstack([X_normal[400:], X_anomaly])
    y_test = np.array([0] * 100 + [1] * n_anomaly)

    # Fit and score
    model = LSTMAutoencoder(
        encoder_units=[32, 16],
        decoder_units=[16, 32],
        latent_dim=8,
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

    history = model.get_training_history()
    print(f"Final loss: {history['loss'][-1]:.4f}")
