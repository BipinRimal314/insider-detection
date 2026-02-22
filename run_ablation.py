#!/usr/bin/env python
"""
Run LSTM Autoencoder ablation study with different sequence window sizes.

Tests 14-day and 30-day windows (7-day already completed in main experiments).
Results saved to results/clean_experiments/ablation_*.json

Usage:
    python run_ablation.py --window 14 --seeds 5
    python run_ablation.py --window 30 --seeds 5
    python run_ablation.py --all                    # Run both
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_all
from src.data.features import compute_daily_features, add_labels
from src.data.splits import temporal_train_test_split, prepare_sequence_train_test
from src.experiments.runner import run_multi_seed_experiment, save_results


def main():
    parser = argparse.ArgumentParser(description="LSTM window size ablation")
    parser.add_argument("--window", type=int, help="Window size (14 or 30)")
    parser.add_argument("--all", action="store_true", help="Run both 14 and 30")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--output", type=str, default="results/clean_experiments")
    args = parser.parse_args()

    windows = []
    if args.all:
        windows = [14, 30]
    elif args.window:
        windows = [args.window]
    else:
        print("Specify --window <size> or --all")
        return

    seeds = list(range(42, 42 + args.seeds))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data (only once)
    print("Loading dataset...")
    data = load_raw_data(Path("data"), "r4.2")
    preprocessed = preprocess_all(data, min_user_days=30)
    features = compute_daily_features(preprocessed)
    labeled = add_labels(features, preprocessed["ground_truth"])

    train_df, test_df, _ = temporal_train_test_split(
        labeled, train_ratio=0.7, exclude_attack_from_train=True,
        ground_truth=preprocessed["ground_truth"],
    )

    for window in windows:
        print(f"\n{'='*60}")
        print(f"LSTM AUTOENCODER — {window}-DAY WINDOW")
        print(f"{'='*60}")

        X_train_seq, X_test_seq, _, y_test_seq = prepare_sequence_train_test(
            train_df, test_df, window_size=window
        )
        print(f"Sequences: Train={X_train_seq.shape}, Test={X_test_seq.shape}")
        print(f"Test positives: {y_test_seq.sum()}")

        from src.models.lstm_autoencoder import LSTMAutoencoder

        results = run_multi_seed_experiment(
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
            verbose=True,
            # NOTE: do NOT pass save_dir here — it would overwrite the
            # original 7-day LSTMAutoencoder_results.json because the model's
            # .name is "LSTMAutoencoder" regardless of window size.
            # Instead, we save with window-specific filename below.
        )

        # Save with window-specific filename
        filepath = output_dir / f"LSTMAutoencoder_window{window}_results.json"
        data_out = {
            "model_name": f"LSTMAutoencoder_w{window}",
            "window_size": window,
            "n_seeds": results.n_seeds,
            "seeds": results.seeds,
            "metrics": {
                "auc_roc": {"mean": results.auc_roc_mean, "std": results.auc_roc_std},
                "auc_pr": {"mean": results.auc_pr_mean, "std": results.auc_pr_std},
                "recall_at_5fpr": {"mean": results.recall_at_5fpr_mean, "std": results.recall_at_5fpr_std},
                "recall_at_10fpr": {"mean": results.recall_at_10fpr_mean, "std": results.recall_at_10fpr_std},
            },
            "individual_results": [
                {
                    "seed": r.seed,
                    "auc_roc": r.auc_roc,
                    "auc_pr": r.auc_pr,
                    "recall_at_5fpr": r.recall_at_5fpr,
                    "recall_at_10fpr": r.recall_at_10fpr,
                    "train_time": r.train_time,
                    "inference_time": r.inference_time,
                }
                for r in results.results
            ],
            "timestamp": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data_out, f, indent=2)
        print(f"\nSaved: {filepath}")

        print(f"\n{window}-day window Summary:")
        print(f"  AUC-ROC:       {results.auc_roc_mean:.4f} ± {results.auc_roc_std:.4f}")
        print(f"  Recall@5%FPR:  {results.recall_at_5fpr_mean:.4f} ± {results.recall_at_5fpr_std:.4f}")
        print(f"  Recall@10%FPR: {results.recall_at_10fpr_mean:.4f} ± {results.recall_at_10fpr_std:.4f}")

    print("\nAblation study complete!")


if __name__ == "__main__":
    main()
