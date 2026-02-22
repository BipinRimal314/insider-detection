#!/usr/bin/env python
"""
Analyze experiment results and compute statistical comparisons.

Reads JSON results and produces:
1. Summary table with all metrics
2. Pairwise Wilcoxon tests (LSTM vs each other model on Recall@5%FPR)
3. Paper-ready numbers for updating main.tex

Usage:
    python analyze_results.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("results/clean_experiments")


def load_all_results():
    results = {}
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        with open(f) as fp:
            data = json.load(fp)
        results[data["model_name"]] = data
    return results


def pairwise_wilcoxon(results, metric_key, model_a, model_b):
    """Wilcoxon signed-rank test between two models on a given metric."""
    scores_a = [r[metric_key] for r in results[model_a]["individual_results"]]
    scores_b = [r[metric_key] for r in results[model_b]["individual_results"]]

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    try:
        stat, p = stats.wilcoxon(scores_a, scores_b)
    except ValueError:
        stat, p = 0.0, 1.0

    # Cohen's d
    diff = scores_a - scores_b
    if np.std(diff, ddof=1) > 0:
        d = np.mean(diff) / np.std(diff, ddof=1)
    else:
        d = 0.0

    return {
        "stat": stat,
        "p": p,
        "d": d,
        "mean_a": np.mean(scores_a),
        "mean_b": np.mean(scores_b),
        "ratio": np.mean(scores_a) / np.mean(scores_b) if np.mean(scores_b) > 0 else float("inf"),
    }


def main():
    results = load_all_results()
    print(f"Loaded results for: {list(results.keys())}")
    print()

    # ── Summary Table ──
    model_order = ["IsolationForest", "PCA_Reconstruction", "DenseAutoencoder", "LSTMAutoencoder"]
    available = [m for m in model_order if m in results]

    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<22} {'AUC-ROC':<20} {'AUC-PR':<20} {'Recall@5%FPR':<20} {'Recall@10%FPR':<20}")
    print("-" * 102)
    for name in available:
        m = results[name]["metrics"]
        print(f"{name:<22} "
              f"{m['auc_roc']['mean']:.4f}±{m['auc_roc']['std']:.4f}    "
              f"{m['auc_pr']['mean']:.4f}±{m['auc_pr']['std']:.4f}    "
              f"{m['recall_at_5fpr']['mean']:.4f}±{m['recall_at_5fpr']['std']:.4f}    "
              f"{m['recall_at_10fpr']['mean']:.4f}±{m['recall_at_10fpr']['std']:.4f}")
    print()

    # ── Pairwise Tests ──
    if "LSTMAutoencoder" in results:
        print("=" * 80)
        print("STATISTICAL COMPARISONS (LSTM vs others)")
        print("=" * 80)

        for metric in ["recall_at_5fpr", "recall_at_10fpr", "auc_roc"]:
            print(f"\nMetric: {metric}")
            print(f"{'Comparison':<40} {'p-value':<12} {'Cohen d':<12} {'Ratio':<10} {'Sig?':<6}")
            print("-" * 80)

            for other in available:
                if other == "LSTMAutoencoder":
                    continue
                comp = pairwise_wilcoxon(results, metric, "LSTMAutoencoder", other)
                sig = "*" if comp["p"] < 0.05 else ""
                print(f"LSTM vs {other:<30} "
                      f"{comp['p']:<12.4f} "
                      f"{comp['d']:<12.2f} "
                      f"{comp['ratio']:<10.2f} "
                      f"{sig}")

        # ── Paper Numbers ──
        print()
        print("=" * 80)
        print("PAPER-READY NUMBERS (for main.tex)")
        print("=" * 80)

        lstm = results["LSTMAutoencoder"]["metrics"]
        if_res = results["IsolationForest"]["metrics"]

        print(f"\nIsolation Forest AUC-ROC: {if_res['auc_roc']['mean']:.3f} ± {if_res['auc_roc']['std']:.3f}")
        print(f"LSTM AUC-ROC: {lstm['auc_roc']['mean']:.3f} ± {lstm['auc_roc']['std']:.3f}")
        print(f"LSTM Recall@5%FPR: {lstm['recall_at_5fpr']['mean']:.3f} ± {lstm['recall_at_5fpr']['std']:.3f}")
        print(f"IF Recall@5%FPR: {if_res['recall_at_5fpr']['mean']:.3f} ± {if_res['recall_at_5fpr']['std']:.3f}")

        ratio = lstm["recall_at_5fpr"]["mean"] / if_res["recall_at_5fpr"]["mean"]
        print(f"\nLSTM/IF Recall@5%FPR ratio: {ratio:.1f}x")

        comp = pairwise_wilcoxon(results, "recall_at_5fpr", "LSTMAutoencoder", "IsolationForest")
        print(f"Wilcoxon p-value (Recall@5%FPR, LSTM vs IF): {comp['p']:.3f}")
        print(f"Cohen's d: {comp['d']:.2f}")

        print(f"\n--- Table II values ---")
        for name in available:
            m = results[name]["metrics"]
            print(f"{name}: AUC-ROC={m['auc_roc']['mean']:.3f}±{m['auc_roc']['std']:.3f}  "
                  f"R@5%={m['recall_at_5fpr']['mean']:.3f}±{m['recall_at_5fpr']['std']:.3f}  "
                  f"R@10%={m['recall_at_10fpr']['mean']:.3f}±{m['recall_at_10fpr']['std']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
