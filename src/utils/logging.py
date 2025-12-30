"""
Logging utilities for experiment tracking.

Provides structured logging with timestamps, log levels,
and optional file output for reproducibility.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Global logger registry
_loggers: dict = {}


def setup_logger(
    name: str = "insider_threat",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name (used to retrieve logger later).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        console: Whether to log to console.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger("experiment", log_file=Path("logs/exp.log"))
        >>> logger.info("Starting experiment")
    """
    # Return existing logger if already set up
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Formatter with timestamp
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "insider_threat") -> logging.Logger:
    """
    Get an existing logger or create a default one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class ExperimentLogger:
    """
    Structured experiment logger for tracking runs.

    Records experiment metadata, metrics, and timing information
    in a structured format for later analysis.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name for this experiment run.
            output_dir: Directory for log files.
            logger: Optional existing logger to use.
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"{experiment_name}_{timestamp}.log"

        self.logger = logger or setup_logger(
            experiment_name, log_file=log_file
        )

        self.metrics: dict = {}
        self.start_time: Optional[datetime] = None

    def start(self, config: Optional[dict] = None) -> None:
        """Log experiment start with configuration."""
        self.start_time = datetime.now()
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Timestamp: {self.start_time.isoformat()}")

        if config:
            self.logger.info("Configuration:")
            for key, value in config.items():
                self.logger.info(f"  {key}: {value}")

    def log_metric(self, name: str, value: float, seed: Optional[int] = None) -> None:
        """Log a metric value."""
        key = f"{name}_seed{seed}" if seed is not None else name
        self.metrics[key] = value
        self.logger.info(f"Metric {name}: {value:.4f}" + (f" (seed={seed})" if seed else ""))

    def log_summary(self, metrics: dict) -> None:
        """Log summary metrics (mean, std, etc.)."""
        self.logger.info("Summary Metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {name}: {value:.4f}")
            else:
                self.logger.info(f"  {name}: {value}")

    def end(self) -> None:
        """Log experiment end with duration."""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None
        self.logger.info(f"Experiment completed: {self.experiment_name}")
        if duration:
            self.logger.info(f"Duration: {duration}")
        self.logger.info(f"=" * 60)
