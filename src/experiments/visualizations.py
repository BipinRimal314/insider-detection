"""
Visualization utilities for experiment results.

Generates publication-quality figures for the paper.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    title: str = "ROC Curves",
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        results: Dict mapping model_name to (y_true, scores).
        output_path: Path to save figure.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, (y_true, scores)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.axvline(x=0.05, color='gray', linestyle=':', lw=1, label='5% FPR')
    ax.axvline(x=0.10, color='gray', linestyle='-.', lw=1, label='10% FPR')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title(title)
    ax.legend(loc='lower right')

    if output_path:
        fig.savefig(output_path)
        print(f"Saved ROC curves to {output_path}")

    return fig


def plot_pr_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        results: Dict mapping model_name to (y_true, scores).
        output_path: Path to save figure.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, (y_true, scores)) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)

        ax.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})')

    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
               label=f'Baseline ({baseline:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='upper right')

    if output_path:
        fig.savefig(output_path)
        print(f"Saved PR curves to {output_path}")

    return fig


def plot_model_comparison_bars(
    results_dir: Path,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing models on multiple metrics.

    Args:
        results_dir: Directory containing result JSON files.
        output_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    # Load all results
    models = []
    auc_rocs = []
    auc_roc_stds = []
    recall_5s = []
    recall_5_stds = []

    for json_file in sorted(results_dir.glob("*_results.json")):
        with open(json_file) as f:
            data = json.load(f)

        models.append(data['model_name'].replace('_', ' '))
        auc_rocs.append(data['metrics']['auc_roc']['mean'])
        auc_roc_stds.append(data['metrics']['auc_roc']['std'])
        recall_5s.append(data['metrics']['recall_at_5fpr']['mean'])
        recall_5_stds.append(data['metrics']['recall_at_5fpr']['std'])

    # Create grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models))
    width = 0.6

    # AUC-ROC
    ax1 = axes[0]
    bars1 = ax1.bar(x, auc_rocs, width, yerr=auc_roc_stds, capsize=5,
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('Overall Ranking Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.set_ylim([0.5, 0.9])
    ax1.axhline(y=0.5, color='gray', linestyle='--', lw=1)

    # Recall@5%FPR
    ax2 = axes[1]
    bars2 = ax2.bar(x, recall_5s, width, yerr=recall_5_stds, capsize=5,
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Recall @ 5% FPR')
    ax2.set_title('Detection at Low False Positive Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.set_ylim([0, 0.2])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved comparison bars to {output_path}")

    return fig


def plot_seed_variance(
    results_dir: Path,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot variance across seeds for each model.

    Args:
        results_dir: Directory containing result JSON files.
        output_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, json_file in enumerate(sorted(results_dir.glob("*_results.json"))):
        with open(json_file) as f:
            data = json.load(f)

        name = data['model_name'].replace('_', ' ')
        aucs = [r['auc_roc'] for r in data['individual_results']]
        seeds = data['seeds']

        ax.plot(seeds, aucs, 'o-', color=colors[i % len(colors)],
                label=name, markersize=8, lw=2)

    ax.set_xlabel('Random Seed')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Performance Variance Across Random Seeds')
    ax.legend()
    ax.set_xticks([42, 43, 44, 45, 46])

    if output_path:
        fig.savefig(output_path)
        print(f"Saved seed variance plot to {output_path}")

    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    results_dir = Path("results/clean_experiments")
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison bar chart
    plot_model_comparison_bars(
        results_dir,
        figures_dir / "model_comparison.png"
    )

    # Generate seed variance plot
    plot_seed_variance(
        results_dir,
        figures_dir / "seed_variance.png"
    )

    print("\nVisualization complete!")
