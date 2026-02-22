#!/usr/bin/env python
"""
Generate publication-quality figures for the insider threat detection paper.

Reads experiment results from results/clean_experiments/ and generates
figures for the IEEE paper.

Usage:
    python generate_figures.py              # Generate all figures
    python generate_figures.py --figure 1   # Generate specific figure
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("results/clean_experiments")
FIGURES_DIR = Path("paper/figures")

# Color scheme for models
COLORS = {
    "IsolationForest": "#1f77b4",
    "PCA_Reconstruction": "#ff7f0e",
    "DenseAutoencoder": "#2ca02c",
    "LSTMAutoencoder": "#d62728",
}
LABELS = {
    "IsolationForest": "Isolation Forest",
    "PCA_Reconstruction": "PCA Reconstruction",
    "DenseAutoencoder": "Dense Autoencoder",
    "LSTMAutoencoder": "LSTM Autoencoder",
}


def load_results():
    """Load all experiment results from JSON files."""
    results = {}
    for f in RESULTS_DIR.glob("*_results.json"):
        with open(f) as fp:
            data = json.load(fp)
        results[data["model_name"]] = data
    return results


def load_data_and_score():
    """Load dataset and compute scores for ROC curve generation."""
    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess_all
    from src.data.features import compute_daily_features, add_labels
    from src.data.splits import (
        temporal_train_test_split,
        prepare_train_test_arrays,
        prepare_sequence_train_test,
    )

    print("Loading dataset for figure generation...")
    data = load_raw_data(Path("data"), "r4.2")
    preprocessed = preprocess_all(data, min_user_days=30)
    features = compute_daily_features(preprocessed)
    labeled = add_labels(features, preprocessed["ground_truth"])

    train_df, test_df, _ = temporal_train_test_split(
        labeled, train_ratio=0.7, exclude_attack_from_train=True
    )

    X_train, X_test, y_train, y_test = prepare_train_test_arrays(train_df, test_df)
    X_train_seq, X_test_seq, _, y_test_seq = prepare_sequence_train_test(
        train_df, test_df, window_size=7
    )

    return X_train, X_test, y_test, X_train_seq, X_test_seq, y_test_seq, train_df, test_df


def get_model_scores(X_train, X_test, X_train_seq, X_test_seq, seed=42):
    """Train all models with one seed and return raw scores."""
    from src.models.isolation_forest import IsolationForestDetector
    from src.models.pca_anomaly import PCAAnomalyDetector
    from src.models.autoencoder import DenseAutoencoder
    from src.models.lstm_autoencoder import LSTMAutoencoder

    scores = {}

    print("  Scoring: Isolation Forest...")
    m = IsolationForestDetector(n_estimators=100, seed=seed)
    m.fit(X_train)
    scores["IsolationForest"] = m.score(X_test)

    print("  Scoring: PCA Reconstruction...")
    m = PCAAnomalyDetector(variance_threshold=0.95, seed=seed)
    m.fit(X_train)
    scores["PCA_Reconstruction"] = m.score(X_test)

    print("  Scoring: Dense Autoencoder...")
    m = DenseAutoencoder(hidden_layers=[64, 32], latent_dim=16, epochs=50, verbose=0, seed=seed)
    m.fit(X_train)
    scores["DenseAutoencoder"] = m.score(X_test)

    print("  Scoring: LSTM Autoencoder...")
    m = LSTMAutoencoder(
        encoder_units=[64, 32], decoder_units=[32, 64],
        latent_dim=16, epochs=50, verbose=0, seed=seed,
    )
    m.fit(X_train_seq)
    scores["LSTMAutoencoder"] = m.score(X_test_seq)

    return scores


def fig1_roc_curves(scores, y_test, y_test_seq):
    """Figure 1: ROC curves for all models with FPR threshold markers."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    model_order = ["IsolationForest", "PCA_Reconstruction", "DenseAutoencoder", "LSTMAutoencoder"]

    for name in model_order:
        s = scores[name]
        y = y_test_seq if name == "LSTMAutoencoder" else y_test
        fpr, tpr, _ = roc_curve(y, s)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2,
                label=f"{LABELS[name]} (AUC = {auc_val:.3f})")

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

    # FPR threshold markers
    for fpr_thresh, ls in [(0.05, ":"), (0.10, "--")]:
        ax.axvline(x=fpr_thresh, color="gray", linestyle=ls, linewidth=1.2, alpha=0.7)
        ax.text(fpr_thresh + 0.008, 0.02, f"{int(fpr_thresh*100)}% FPR",
                fontsize=9, color="gray", rotation=90, va="bottom")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Insider Threat Detection", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig2_model_comparison(results):
    """Figure 2: Bar chart comparing AUC-ROC vs Recall@5%FPR."""
    model_order = ["IsolationForest", "PCA_Reconstruction", "DenseAutoencoder", "LSTMAutoencoder"]
    available = [m for m in model_order if m in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = [LABELS[m] for m in available]
    colors = [COLORS[m] for m in available]

    # AUC-ROC
    means = [results[m]["metrics"]["auc_roc"]["mean"] for m in available]
    stds = [results[m]["metrics"]["auc_roc"]["std"] for m in available]
    bars = ax1.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("AUC-ROC", fontsize=11)
    ax1.set_title("Overall Ranking Performance", fontsize=12)
    ax1.set_ylim([0.5, 0.9])
    ax1.tick_params(axis="x", rotation=25, labelsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Recall@5%FPR
    means = [results[m]["metrics"]["recall_at_5fpr"]["mean"] for m in available]
    stds = [results[m]["metrics"]["recall_at_5fpr"]["std"] for m in available]
    bars = ax2.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Recall @ 5% FPR", fontsize=11)
    ax2.set_title("Detection at Low False Positive Rate", fontsize=12)
    ax2.set_ylim([0, max(means) * 1.4])
    ax2.tick_params(axis="x", rotation=25, labelsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig3_feature_importance(X_train, y_test, X_test, train_df, test_df):
    """Figure 3: Feature importance (IF splits) + Feature-label correlation."""
    from src.data.features import FEATURE_COLUMNS
    from src.models.isolation_forest import IsolationForestDetector

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Isolation Forest feature importance via split frequency
    print("  Computing feature importance...")
    m = IsolationForestDetector(n_estimators=100, seed=42)
    m.fit(X_train)

    # Get feature importances from the underlying sklearn model
    importances = np.zeros(len(FEATURE_COLUMNS))
    for tree in m.model_.estimators_:
        feature_idx = tree.tree_.feature
        for f in feature_idx:
            if f >= 0:  # -2 means leaf node
                importances[f] += 1
    importances = importances / importances.sum()

    # Sort by importance
    sorted_idx = np.argsort(importances)
    top_n = 15
    top_idx = sorted_idx[-top_n:]

    feat_names = [FEATURE_COLUMNS[i] for i in top_idx]
    feat_vals = importances[top_idx]

    # Color by category
    cat_colors = {
        "logon": "#1f77b4", "device": "#d62728", "file": "#2ca02c",
        "email": "#ff7f0e", "http": "#9467bd",
    }
    bar_colors = []
    for name in feat_names:
        if any(name.startswith(p) or name in ["logon_count", "logoff_count", "after_hours_logons",
                "unique_pcs", "first_logon_hour", "last_logoff_hour"]
               for p in ["logon"]):
            bar_colors.append(cat_colors["logon"])
        elif any(name.startswith(p) or name in ["device_connects", "device_disconnects",
                 "after_hours_connects", "device_activity"] for p in ["device"]):
            bar_colors.append(cat_colors["device"])
        elif any(name.startswith(p) or name in ["file_operations", "file_copies",
                 "exe_access", "after_hours_files"] for p in ["file"]):
            bar_colors.append(cat_colors["file"])
        elif any(name.startswith(p) or name in ["emails_sent", "total_recipients",
                 "attachment_count", "attachment_size", "after_hours_emails"] for p in ["email"]):
            bar_colors.append(cat_colors["email"])
        else:
            bar_colors.append(cat_colors["http"])

    ax1.barh(feat_names, feat_vals, color=bar_colors)
    ax1.set_xlabel("Importance Score", fontsize=10)
    ax1.set_title("Feature Importance (Isolation Forest)", fontsize=11)
    ax1.tick_params(axis="y", labelsize=8)

    # Feature-label correlation
    print("  Computing feature-label correlation...")
    test_features = test_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    test_labels = test_df["label"].to_numpy().astype(np.float32)

    correlations = np.array([
        np.corrcoef(test_features[:, i], test_labels)[0, 1]
        for i in range(len(FEATURE_COLUMNS))
    ])

    sorted_corr_idx = np.argsort(np.abs(correlations))[-top_n:]
    corr_names = [FEATURE_COLUMNS[i] for i in sorted_corr_idx]
    corr_vals = correlations[sorted_corr_idx]

    bar_colors2 = ["#d62728" if v > 0 else "#2ca02c" for v in corr_vals]
    ax2.barh(corr_names, corr_vals, color=bar_colors2)
    ax2.set_xlabel("Correlation with Insider Label", fontsize=10)
    ax2.set_title("Feature-Label Correlation", fontsize=11)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig4_seed_variance(results):
    """Figure 4: AUC-ROC variance across random seeds."""
    model_order = ["IsolationForest", "PCA_Reconstruction", "DenseAutoencoder", "LSTMAutoencoder"]
    available = [m for m in model_order if m in results]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for name in available:
        data = results[name]
        seeds = data["seeds"]
        auc_rocs = [r["auc_roc"] for r in data["individual_results"]]
        ax.plot(seeds, auc_rocs, "o-", color=COLORS[name], linewidth=2,
                markersize=6, label=LABELS[name])

    ax.set_xlabel("Random Seed", fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_title("Performance Variance Across Random Seeds", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([42, 43, 44, 45, 46])

    plt.tight_layout()
    path = FIGURES_DIR / "seed_variance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--figure", type=int, help="Generate specific figure (1-4)")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip model scoring (figures 2, 4 only)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load JSON results (always needed)
    results = load_results()
    print(f"Loaded results for: {list(results.keys())}")

    figures_to_gen = [args.figure] if args.figure else [1, 2, 3, 4]

    # Load data and score models if needed for ROC curves or feature importance
    need_scoring = (1 in figures_to_gen or 3 in figures_to_gen) and not args.skip_scoring
    if need_scoring:
        data = load_data_and_score()
        X_train, X_test, y_test, X_train_seq, X_test_seq, y_test_seq, train_df, test_df = data
        scores = get_model_scores(X_train, X_test, X_train_seq, X_test_seq, seed=42)

    for fig_num in figures_to_gen:
        print(f"\nGenerating Figure {fig_num}...")
        if fig_num == 1:
            fig1_roc_curves(scores, y_test, y_test_seq)
        elif fig_num == 2:
            fig2_model_comparison(results)
        elif fig_num == 3:
            fig3_feature_importance(X_train, y_test, X_test, train_df, test_df)
        elif fig_num == 4:
            fig4_seed_variance(results)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
