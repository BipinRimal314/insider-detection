#!/usr/bin/env python
"""
Run remaining GPU experiments sequentially:
1. Re-run 7-day LSTM (5 seeds) — results were overwritten by ablation bug
2. Run 30-day LSTM ablation (5 seeds)

Saves results incrementally after each seed to prevent data loss.
"""

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


def save_ablation_results(results, window, output_dir):
    """Save ablation results with window-specific filename."""
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
    print(f"Saved: {filepath}")


def main():
    seeds = [42, 43, 44, 45, 46]
    output_dir = Path("results/clean_experiments")
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

    from src.models.lstm_autoencoder import LSTMAutoencoder

    model_kwargs = {
        "encoder_units": [64, 32],
        "decoder_units": [32, 64],
        "latent_dim": 16,
        "epochs": 50,
        "verbose": 0,
    }

    # ── Part 1: Re-run 7-day LSTM (original results were overwritten) ──
    print(f"\n{'='*60}")
    print("PART 1: LSTM AUTOENCODER — 7-DAY WINDOW (re-run)")
    print(f"{'='*60}")

    X_train_7, X_test_7, _, y_test_7 = prepare_sequence_train_test(
        train_df, test_df, window_size=7
    )
    print(f"Sequences: Train={X_train_7.shape}, Test={X_test_7.shape}")
    print(f"Test positives: {y_test_7.sum()}")

    results_7 = run_multi_seed_experiment(
        LSTMAutoencoder, model_kwargs,
        X_train_7, X_test_7, y_test_7,
        seeds=seeds, verbose=True,
        save_dir=output_dir,  # Safe: saves as LSTMAutoencoder_results.json
    )

    print(f"\n7-day Summary:")
    print(f"  AUC-ROC:       {results_7.auc_roc_mean:.4f} ± {results_7.auc_roc_std:.4f}")
    print(f"  Recall@5%FPR:  {results_7.recall_at_5fpr_mean:.4f} ± {results_7.recall_at_5fpr_std:.4f}")
    print(f"  Recall@10%FPR: {results_7.recall_at_10fpr_mean:.4f} ± {results_7.recall_at_10fpr_std:.4f}")

    # ── Part 2: 30-day LSTM ablation ──
    print(f"\n{'='*60}")
    print("PART 2: LSTM AUTOENCODER — 30-DAY WINDOW (ablation)")
    print(f"{'='*60}")

    X_train_30, X_test_30, _, y_test_30 = prepare_sequence_train_test(
        train_df, test_df, window_size=30
    )
    print(f"Sequences: Train={X_train_30.shape}, Test={X_test_30.shape}")
    print(f"Test positives: {y_test_30.sum()}")

    results_30 = run_multi_seed_experiment(
        LSTMAutoencoder, model_kwargs,
        X_train_30, X_test_30, y_test_30,
        seeds=seeds, verbose=True,
        # Do NOT pass save_dir — would overwrite 7-day results
    )
    save_ablation_results(results_30, 30, output_dir)

    print(f"\n30-day Summary:")
    print(f"  AUC-ROC:       {results_30.auc_roc_mean:.4f} ± {results_30.auc_roc_std:.4f}")
    print(f"  Recall@5%FPR:  {results_30.recall_at_5fpr_mean:.4f} ± {results_30.recall_at_5fpr_std:.4f}")
    print(f"  Recall@10%FPR: {results_30.recall_at_10fpr_mean:.4f} ± {results_30.recall_at_10fpr_std:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"{'Window':<10} {'AUC-ROC':<20} {'Recall@5%FPR':<20} {'Recall@10%FPR':<20}")
    print("-" * 70)
    for label, r in [("7-day", results_7), ("30-day", results_30)]:
        print(f"{label:<10} {r.auc_roc_mean:.4f} ± {r.auc_roc_std:.4f}    "
              f"{r.recall_at_5fpr_mean:.4f} ± {r.recall_at_5fpr_std:.4f}    "
              f"{r.recall_at_10fpr_mean:.4f} ± {r.recall_at_10fpr_std:.4f}")


if __name__ == "__main__":
    main()
