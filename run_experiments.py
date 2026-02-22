#!/usr/bin/env python
"""
Main experiment runner for insider threat detection research.

Runs all models with multiple seeds and generates results for the paper.

Usage:
    python run_experiments.py --all                    # Run all experiments
    python run_experiments.py --model isolation_forest # Run single model
    python run_experiments.py --quick                  # Quick test (1 seed)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_all
from src.data.features import compute_daily_features, add_labels
from src.data.splits import (
    temporal_train_test_split,
    prepare_train_test_arrays,
    prepare_sequence_train_test,
)
from src.experiments.runner import (
    run_multi_seed_experiment,
    save_results,
    generate_latex_table,
)
from src.utils.reproducibility import set_all_seeds


def load_data(dataset_version: str = "r4.2", min_user_days: int = 30):
    """Load and preprocess dataset."""
    print(f"Loading dataset {dataset_version}...")

    data = load_raw_data(Path("data"), dataset_version)
    preprocessed = preprocess_all(data, min_user_days=min_user_days)
    features = compute_daily_features(preprocessed)
    labeled = add_labels(features, preprocessed["ground_truth"])

    train_df, test_df, _ = temporal_train_test_split(
        labeled,
        train_ratio=0.7,
        exclude_attack_from_train=True,
        ground_truth=preprocessed["ground_truth"],
    )

    return train_df, test_df, preprocessed["ground_truth"]


def run_isolation_forest(X_train, X_test, y_test, seeds, verbose=True, save_dir=None):
    """Run Isolation Forest experiments."""
    from src.models.isolation_forest import IsolationForestDetector

    print("\n" + "=" * 60)
    print("ISOLATION FOREST")
    print("=" * 60)

    return run_multi_seed_experiment(
        IsolationForestDetector,
        {"n_estimators": 100},
        X_train, X_test, y_test,
        seeds=seeds,
        verbose=verbose,
        save_dir=save_dir,
    )


def run_pca(X_train, X_test, y_test, seeds, verbose=True, save_dir=None):
    """Run PCA Reconstruction experiments."""
    from src.models.pca_anomaly import PCAAnomalyDetector

    print("\n" + "=" * 60)
    print("PCA RECONSTRUCTION")
    print("=" * 60)

    return run_multi_seed_experiment(
        PCAAnomalyDetector,
        {"variance_threshold": 0.95},
        X_train, X_test, y_test,
        seeds=seeds,
        verbose=verbose,
        save_dir=save_dir,
    )


def run_dense_autoencoder(X_train, X_test, y_test, seeds, verbose=True, save_dir=None):
    """Run Dense Autoencoder experiments."""
    from src.models.autoencoder import DenseAutoencoder

    print("\n" + "=" * 60)
    print("DENSE AUTOENCODER")
    print("=" * 60)

    return run_multi_seed_experiment(
        DenseAutoencoder,
        {
            "hidden_layers": [64, 32],
            "latent_dim": 16,
            "epochs": 50,
            "verbose": 0,
        },
        X_train, X_test, y_test,
        seeds=seeds,
        verbose=verbose,
        save_dir=save_dir,
    )


def run_lstm_autoencoder(X_train_seq, X_test_seq, y_test_seq, seeds, verbose=True, save_dir=None):
    """Run LSTM Autoencoder experiments."""
    from src.models.lstm_autoencoder import LSTMAutoencoder

    print("\n" + "=" * 60)
    print("LSTM AUTOENCODER")
    print("=" * 60)

    return run_multi_seed_experiment(
        LSTMAutoencoder,
        {
            "encoder_units": [64, 32],
            "decoder_units": [32, 64],
            "latent_dim": 16,
            "epochs": 50,
            "verbose": 0,
        },
        X_train_seq, X_test_seq, y_test_seq,
        seeds=seeds,
        verbose=verbose,
        save_dir=save_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Run insider threat detection experiments")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--model", type=str, help="Run specific model")
    parser.add_argument("--quick", action="store_true", help="Quick test with 1 seed")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--dataset", type=str, default="r4.2", help="Dataset version")
    parser.add_argument("--output", type=str, default="results/clean_experiments", help="Output directory")
    parser.add_argument("--window", type=int, default=7, help="Sequence window size")
    args = parser.parse_args()

    # Determine seeds
    if args.quick:
        seeds = [42]
    else:
        seeds = list(range(42, 42 + args.seeds))

    print(f"Running experiments with seeds: {seeds}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")

    # Load data
    train_df, test_df, ground_truth = load_data(args.dataset)

    # Prepare static features
    X_train, X_test, y_train, y_test = prepare_train_test_arrays(train_df, test_df)
    print(f"\nStatic data: Train={X_train.shape}, Test={X_test.shape}")
    print(f"Test positives: {y_test.sum()} / {len(y_test)} ({100*y_test.mean():.2f}%)")

    # Prepare sequence features
    X_train_seq, X_test_seq, _, y_test_seq = prepare_sequence_train_test(
        train_df, test_df, window_size=args.window
    )
    print(f"Sequence data: Train={X_train_seq.shape}, Test={X_test_seq.shape}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run models
    models_to_run = []
    if args.all:
        models_to_run = ["isolation_forest", "pca", "dense_autoencoder", "lstm_autoencoder"]
    elif args.model:
        models_to_run = [args.model]
    else:
        print("Specify --all or --model <name>")
        return

    for model_name in models_to_run:
        if model_name == "isolation_forest":
            results = run_isolation_forest(X_train, X_test, y_test, seeds, save_dir=output_dir)
        elif model_name == "pca":
            results = run_pca(X_train, X_test, y_test, seeds, save_dir=output_dir)
        elif model_name == "dense_autoencoder":
            results = run_dense_autoencoder(X_train, X_test, y_test, seeds, save_dir=output_dir)
        elif model_name == "lstm_autoencoder":
            results = run_lstm_autoencoder(X_train_seq, X_test_seq, y_test_seq, seeds, save_dir=output_dir)
        else:
            print(f"Unknown model: {model_name}")
            continue

        all_results.append(results)
        save_results(results, output_dir)

        print(f"\n{results.model_name} Summary:")
        print(f"  AUC-ROC: {results.auc_roc_mean:.4f} ± {results.auc_roc_std:.4f}")
        print(f"  AUC-PR:  {results.auc_pr_mean:.4f} ± {results.auc_pr_std:.4f}")
        print(f"  Recall@5%FPR:  {results.recall_at_5fpr_mean:.4f} ± {results.recall_at_5fpr_std:.4f}")
        print(f"  Recall@10%FPR: {results.recall_at_10fpr_mean:.4f} ± {results.recall_at_10fpr_std:.4f}")

    # Generate summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Print comparison table
        print(f"\n{'Model':<20} {'AUC-ROC':<20} {'Recall@5%FPR':<20}")
        print("-" * 60)
        for r in all_results:
            print(f"{r.model_name:<20} {r.auc_roc_mean:.4f} ± {r.auc_roc_std:.4f}    {r.recall_at_5fpr_mean:.4f} ± {r.recall_at_5fpr_std:.4f}")

        # Save LaTeX table
        latex = generate_latex_table(all_results)
        latex_path = output_dir / "results_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {latex_path}")

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
