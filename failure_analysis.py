#!/usr/bin/env python
"""
Failure analysis: compare detected vs missed attack features.

Trains an LSTM autoencoder (seed=42), identifies which attack sequences
are detected vs missed at 5% FPR, and computes per-feature statistics
(standard deviations from normal mean) for each group.

Usage:
    python failure_analysis.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_raw_data
from src.data.preprocessing import preprocess_all
from src.data.features import compute_daily_features, add_labels, FEATURE_COLUMNS
from src.data.splits import (
    temporal_train_test_split,
    prepare_train_test_arrays,
    prepare_sequence_train_test,
)
from src.models.lstm_autoencoder import LSTMAutoencoder

RESULTS_DIR = Path("results/clean_experiments")


def main():
    print("=" * 60)
    print("FAILURE ANALYSIS: Detected vs Missed Attack Features")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    data = load_raw_data(Path("data"), "r4.2")
    preprocessed = preprocess_all(data, min_user_days=30)
    features = compute_daily_features(preprocessed)
    labeled = add_labels(features, preprocessed["ground_truth"])

    train_df, test_df, split_day = temporal_train_test_split(
        labeled, train_ratio=0.7, exclude_attack_from_train=True
    )

    X_train, X_test, y_train, y_test = prepare_train_test_arrays(train_df, test_df)
    X_train_seq, X_test_seq, _, y_test_seq = prepare_sequence_train_test(
        train_df, test_df, window_size=7
    )

    print(f"\nTest set: {len(y_test)} user-days, {y_test.sum()} positive")
    print(f"Test sequences: {len(y_test_seq)} sequences, {y_test_seq.sum()} positive")

    # Train LSTM Autoencoder
    print("\nTraining LSTM Autoencoder (seed=42)...")
    model = LSTMAutoencoder(
        encoder_units=[64, 32],
        decoder_units=[32, 64],
        latent_dim=16,
        epochs=50,
        verbose=0,
        seed=42,
    )
    model.fit(X_train_seq)
    scores = model.score(X_test_seq)

    # Compute threshold at 5% FPR
    normal_scores = scores[y_test_seq == 0]
    threshold_5pct = np.percentile(normal_scores, 95)
    predictions = (scores >= threshold_5pct).astype(int)

    n_pos = y_test_seq.sum()
    detected = predictions & y_test_seq
    missed = (~predictions.astype(bool)) & y_test_seq.astype(bool)

    n_detected = detected.sum()
    n_missed = missed.sum()
    print(f"\nAt 5% FPR threshold ({threshold_5pct:.4f}):")
    print(f"  Detected: {n_detected} / {int(n_pos)} ({100*n_detected/max(int(n_pos),1):.1f}%)")
    print(f"  Missed:   {n_missed} / {int(n_pos)} ({100*n_missed/max(int(n_pos),1):.1f}%)")

    # Rebuild the per-user sliding windows to get the last-day index
    # for each sequence (matching what prepare_sequence_train_test does)
    window_size = 7
    test_features = test_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    test_users = test_df["user"].to_list()
    test_days = test_df["day"].to_list()

    seq_last_idx = []  # index into test_features for the last day of each window
    user_indices = {}
    for i, u in enumerate(test_users):
        user_indices.setdefault(u, []).append(i)
    for user, indices in user_indices.items():
        sorted_idx = sorted(indices, key=lambda i: test_days[i])
        if len(sorted_idx) < window_size:
            continue
        for start in range(len(sorted_idx) - window_size + 1):
            seq_last_idx.append(sorted_idx[start + window_size - 1])

    seq_last_idx = np.array(seq_last_idx)
    assert len(seq_last_idx) == len(y_test_seq), \
        f"Sequence count mismatch: {len(seq_last_idx)} vs {len(y_test_seq)}"
    seq_features = test_features[seq_last_idx]

    # Compute normal baseline stats (per-feature mean and std from training set)
    train_features = train_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    normal_mean = train_features.mean(axis=0)
    normal_std = train_features.std(axis=0)
    # Avoid division by zero
    normal_std[normal_std == 0] = 1.0

    # Standard deviations from normal mean
    z_scores = (seq_features - normal_mean) / normal_std

    # Feature stats for detected vs missed attacks
    detected_mask = detected.astype(bool)
    missed_mask = missed.astype(bool)
    normal_test_mask = (y_test_seq == 0)

    detected_z = z_scores[detected_mask]
    missed_z = z_scores[missed_mask]
    normal_z = z_scores[normal_test_mask]

    print(f"\n{'Feature':<25} {'Detected (SD)':>14} {'Missed (SD)':>14} {'Normal (SD)':>14} {'Gap':>8}")
    print("-" * 80)

    results = {}
    for i, feat in enumerate(FEATURE_COLUMNS):
        d_mean = detected_z[:, i].mean() if len(detected_z) > 0 else 0
        m_mean = missed_z[:, i].mean() if len(missed_z) > 0 else 0
        n_mean = normal_z[:, i].mean() if len(normal_z) > 0 else 0
        gap = d_mean - m_mean

        results[feat] = {
            "detected_z_mean": float(d_mean),
            "detected_z_std": float(detected_z[:, i].std()) if len(detected_z) > 0 else 0,
            "missed_z_mean": float(m_mean),
            "missed_z_std": float(missed_z[:, i].std()) if len(missed_z) > 0 else 0,
            "normal_z_mean": float(n_mean),
            "gap": float(gap),
        }

        print(f"{feat:<25} {d_mean:>+13.2f} {m_mean:>+13.2f} {n_mean:>+13.2f} {gap:>+7.2f}")

    # Top distinguishing features (by gap between detected and missed)
    sorted_feats = sorted(results.items(), key=lambda x: abs(x[1]["gap"]), reverse=True)
    print(f"\n{'='*60}")
    print("TOP DISTINGUISHING FEATURES (detected vs missed)")
    print(f"{'='*60}")
    for feat, stats in sorted_feats[:10]:
        print(f"  {feat:<25} detected: {stats['detected_z_mean']:>+.2f} SD, "
              f"missed: {stats['missed_z_mean']:>+.2f} SD, "
              f"gap: {stats['gap']:>+.2f}")

    # Per-scenario analysis
    print(f"\n{'='*60}")
    print("PER-SCENARIO DETECTION RATES")
    print(f"{'='*60}")

    test_labels = test_df.select("label").to_numpy().flatten().astype(int)

    # Get scenario info from ground truth
    gt = preprocessed["ground_truth"]
    # Build user -> scenario map
    if "scenario" in gt.columns:
        user_scenario = dict(zip(
            gt["user"].to_list(),
            gt["scenario"].to_list(),
        ))
    else:
        user_scenario = {}

    # Align test user IDs with sequences using the same index mapping
    seq_users = [test_users[i] for i in seq_last_idx]

    scenario_stats = {}
    for i in range(len(y_test_seq)):
        if y_test_seq[i] == 1:
            user = seq_users[i]
            scenario = user_scenario.get(user, "unknown")
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {"total": 0, "detected": 0}
            scenario_stats[scenario]["total"] += 1
            if detected[i]:
                scenario_stats[scenario]["detected"] += 1

    for scenario, stats in sorted(scenario_stats.items()):
        rate = stats["detected"] / max(stats["total"], 1)
        print(f"  Scenario {scenario}: {stats['detected']}/{stats['total']} "
              f"({100*rate:.0f}%)")

    # Save results
    output = {
        "threshold_5pct": float(threshold_5pct),
        "n_attack_sequences": int(n_pos),
        "n_detected": int(n_detected),
        "n_missed": int(n_missed),
        "detection_rate": float(n_detected / max(int(n_pos), 1)),
        "feature_stats": results,
        "per_scenario": scenario_stats,
    }

    out_path = RESULTS_DIR / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
