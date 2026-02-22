"""
Central configuration for all experiments.

This file contains all hyperparameters and settings used in the research.
Every parameter is documented with its purpose and rationale.

Usage:
    from src.config import Config
    config = Config()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Path to CMU-CERT dataset
    # The dataset contains: logon.csv, device.csv, file.csv, email.csv, http.csv
    data_dir: Path = Path("data")

    # Dataset version to use (r1 is smallest, good for development)
    dataset_version: str = "r4.2"

    # Ground truth file mapping users to insider threat scenarios
    ground_truth_file: str = "answers/insiders.csv"

    # Minimum number of activity days required for a user to be included
    # Rationale: Users with very few days cannot establish behavioral baseline
    min_user_days: int = 30

    # Date range for analysis (None = use all available data)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Temporal aggregation window (daily behavioral profiles)
    aggregation_window: str = "1D"

    # Features extracted from each data source
    # These are aggregated per user per day
    logon_features: List[str] = field(default_factory=lambda: [
        "logon_count",           # Total logon events
        "logoff_count",          # Total logoff events
        "after_hours_logons",    # Logons outside 9-5
        "unique_pcs",            # Number of different PCs used
    ])

    device_features: List[str] = field(default_factory=lambda: [
        "device_connects",       # USB device connections
        "device_disconnects",    # USB device disconnections
    ])

    file_features: List[str] = field(default_factory=lambda: [
        "file_copies",           # File copy operations
        "file_writes",           # File write operations
        "exe_access",            # Executable file access
    ])

    email_features: List[str] = field(default_factory=lambda: [
        "emails_sent",           # Outgoing emails
        "emails_received",       # Incoming emails
        "external_emails",       # Emails to external domains
        "attachment_count",      # Total attachments
    ])

    http_features: List[str] = field(default_factory=lambda: [
        "http_requests",         # Total web requests
        "unique_domains",        # Unique domains visited
        "upload_actions",        # Upload-related URLs
        "download_actions",      # Download-related URLs
    ])

    # Normalization method: 'zscore', 'minmax', or 'robust'
    # Z-score: (x - mean) / std - handles outliers reasonably well
    normalization: str = "zscore"

    # Whether to include day-of-week and hour features
    include_temporal_features: bool = True


@dataclass
class SequenceConfig:
    """Configuration for sequence generation (LSTM models)."""

    # Sliding window size in days
    # Rationale: 7 days captures weekly patterns; 14 days captures biweekly cycles
    window_size: int = 7

    # Stride between consecutive windows
    # stride=1 means maximum overlap (more training samples)
    # stride=window_size means no overlap
    stride: int = 1

    # Minimum sequence length (shorter sequences are padded)
    min_sequence_length: int = 3


@dataclass
class SplitConfig:
    """Configuration for train/test splitting."""

    # Temporal split: train on early data, test on later data
    # This prevents data leakage from future events
    train_ratio: float = 0.7

    # Validation set (carved from training data)
    val_ratio: float = 0.15

    # Ensure no insider's attack period appears in training
    # Attack periods are only in test set
    exclude_attack_from_train: bool = True


@dataclass
class IsolationForestConfig:
    """Isolation Forest hyperparameters."""

    # Number of trees in the forest
    # Higher = more stable but slower; 100-200 typically sufficient
    n_estimators: int = 100

    # Samples per tree (auto = min(256, n_samples))
    # Smaller subsample = faster training, works well for anomaly detection
    max_samples: str = "auto"

    # Expected contamination rate
    # 'auto' uses heuristic; can set to expected anomaly rate (e.g., 0.01)
    contamination: str = "auto"

    # Number of features per tree (None = all features)
    max_features: float = 1.0

    # Bootstrap sampling
    bootstrap: bool = False


@dataclass
class PCAConfig:
    """PCA Anomaly Detection hyperparameters."""

    # Number of components to retain
    # None = use variance_threshold to determine
    n_components: Optional[int] = None

    # Minimum cumulative variance to explain
    # Components are added until this threshold is reached
    variance_threshold: float = 0.95

    # Whether to whiten the data
    whiten: bool = False


@dataclass
class AutoencoderConfig:
    """Dense Autoencoder hyperparameters."""

    # Encoder architecture (input -> hidden layers -> latent)
    # Decoder mirrors this architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])

    # Latent space dimension
    # Should be << input dimension for compression
    latent_dim: int = 16

    # Activation function: 'relu', 'tanh', 'leaky_relu'
    activation: str = "relu"

    # Dropout rate for regularization
    dropout: float = 0.2

    # Training parameters
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001

    # Early stopping patience (epochs without improvement)
    early_stopping_patience: int = 10


@dataclass
class LSTMAutoencoderConfig:
    """LSTM Autoencoder hyperparameters."""

    # LSTM units per layer
    # Encoder: [128, 64] means 128-unit LSTM followed by 64-unit LSTM
    encoder_units: List[int] = field(default_factory=lambda: [64, 32])

    # Decoder architecture (should mirror encoder for symmetry)
    decoder_units: List[int] = field(default_factory=lambda: [32, 64])

    # Latent dimension (output of final encoder LSTM)
    latent_dim: int = 16

    # Dropout for LSTM layers (applied between layers)
    dropout: float = 0.2

    # Recurrent dropout (applied within LSTM cells)
    recurrent_dropout: float = 0.0

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001

    # Early stopping
    early_stopping_patience: int = 15

    # Whether to use bidirectional LSTM
    bidirectional: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""

    # Random seeds for reproducibility
    # Using 5 seeds as per statistical rigor requirements
    random_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    # Number of seeds to use (can be less than len(random_seeds) for quick tests)
    n_seeds: int = 5

    # Models to evaluate
    models: List[str] = field(default_factory=lambda: [
        "isolation_forest",
        "pca_anomaly",
        "dense_autoencoder",
        "lstm_autoencoder",
    ])

    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "auc_roc",
        "auc_pr",
        "recall_at_5fpr",
        "recall_at_10fpr",
        "precision_at_5fpr",
        "precision_at_10fpr",
    ])

    # Statistical significance threshold
    significance_level: float = 0.05

    # Output directories
    results_dir: Path = Path("results/clean_experiments")
    figures_dir: Path = Path("paper/figures")
    tables_dir: Path = Path("paper/tables")


@dataclass
class Config:
    """Master configuration combining all settings."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sequences: SequenceConfig = field(default_factory=SequenceConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    lstm_autoencoder: LSTMAutoencoderConfig = field(default_factory=LSTMAutoencoderConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Environment settings
    device: str = "cpu"  # 'cpu', 'cuda', or 'mps'
    n_jobs: int = -1     # Parallel jobs (-1 = all cores)
    verbose: bool = True

    def __post_init__(self):
        """Set device based on availability."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                self.device = "gpu"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from dictionary."""
        return cls(**d)


# Default configuration instance
default_config = Config()


if __name__ == "__main__":
    # Print configuration for verification
    config = Config()
    print("Configuration Summary:")
    print(f"  Device: {config.device}")
    print(f"  Data dir: {config.data.data_dir}")
    print(f"  Window size: {config.sequences.window_size}")
    print(f"  Random seeds: {config.experiment.random_seeds}")
    print(f"  Models: {config.experiment.models}")
