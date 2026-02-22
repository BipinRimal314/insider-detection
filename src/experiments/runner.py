"""
Experiment runner for reproducible multi-seed experiments.

Executes models with multiple random seeds and aggregates results
for statistical analysis.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    model_name: str
    seed: int
    auc_roc: float
    auc_pr: float
    recall_at_5fpr: float
    recall_at_10fpr: float
    precision_at_5fpr: float
    precision_at_10fpr: float
    train_time: float
    inference_time: float
    threshold: float
    metadata: Dict[str, Any]


@dataclass
class AggregatedResults:
    """Aggregated results across multiple seeds."""

    model_name: str
    n_seeds: int
    seeds: List[int]

    # Mean and std for each metric
    auc_roc_mean: float
    auc_roc_std: float
    auc_pr_mean: float
    auc_pr_std: float
    recall_at_5fpr_mean: float
    recall_at_5fpr_std: float
    recall_at_10fpr_mean: float
    recall_at_10fpr_std: float

    # Individual results
    results: List[ExperimentResult]


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels (0/1).
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        Dictionary of metric names to values.
    """
    # AUC-ROC
    auc_roc = roc_auc_score(y_true, scores)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(recall, precision)

    # Recall at fixed FPR
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    def recall_at_fpr(target_fpr: float) -> float:
        idx = np.searchsorted(fpr, target_fpr)
        if idx >= len(tpr):
            return tpr[-1]
        return tpr[idx]

    def threshold_at_fpr(target_fpr: float) -> float:
        idx = np.searchsorted(fpr, target_fpr)
        if idx >= len(thresholds):
            return thresholds[-1]
        return thresholds[idx]

    recall_5fpr = recall_at_fpr(0.05)
    recall_10fpr = recall_at_fpr(0.10)

    # Precision at fixed FPR (using threshold)
    thresh_5 = threshold_at_fpr(0.05)
    thresh_10 = threshold_at_fpr(0.10)

    preds_5 = (scores >= thresh_5).astype(int)
    preds_10 = (scores >= thresh_10).astype(int)

    precision_5fpr = precision_score(y_true, preds_5, zero_division=0)
    precision_10fpr = precision_score(y_true, preds_10, zero_division=0)

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "recall_at_5fpr": recall_5fpr,
        "recall_at_10fpr": recall_10fpr,
        "precision_at_5fpr": precision_5fpr,
        "precision_at_10fpr": precision_10fpr,
        "threshold_5fpr": thresh_5,
        "threshold_10fpr": thresh_10,
    }


def compute_recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Compute recall at a specific false positive rate."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        return tpr[-1]
    return tpr[idx]


def run_experiment(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> ExperimentResult:
    """
    Run a single experiment with one model and seed.

    Args:
        model: Anomaly detector instance.
        X_train: Training features.
        X_test: Test features.
        y_test: Test labels.
        seed: Random seed used.

    Returns:
        ExperimentResult with all metrics.
    """
    # Fit and predict
    start = time.time()
    model.fit(X_train)
    train_time = time.time() - start

    start = time.time()
    scores = model.score(X_test)
    inference_time = time.time() - start

    # Compute metrics
    metrics = compute_metrics(y_test, scores)

    return ExperimentResult(
        model_name=model.name,
        seed=seed,
        auc_roc=metrics["auc_roc"],
        auc_pr=metrics["auc_pr"],
        recall_at_5fpr=metrics["recall_at_5fpr"],
        recall_at_10fpr=metrics["recall_at_10fpr"],
        precision_at_5fpr=metrics["precision_at_5fpr"],
        precision_at_10fpr=metrics["precision_at_10fpr"],
        train_time=train_time,
        inference_time=inference_time,
        threshold=metrics["threshold_5fpr"],
        metadata=model.get_params(),
    )


def _aggregate_results(results: List[ExperimentResult], seeds: List[int]) -> AggregatedResults:
    """Aggregate individual results into summary statistics."""
    auc_rocs = [r.auc_roc for r in results]
    auc_prs = [r.auc_pr for r in results]
    recall_5s = [r.recall_at_5fpr for r in results]
    recall_10s = [r.recall_at_10fpr for r in results]

    return AggregatedResults(
        model_name=results[0].model_name,
        n_seeds=len(results),
        seeds=seeds[:len(results)],
        auc_roc_mean=np.mean(auc_rocs),
        auc_roc_std=np.std(auc_rocs),
        auc_pr_mean=np.mean(auc_prs),
        auc_pr_std=np.std(auc_prs),
        recall_at_5fpr_mean=np.mean(recall_5s),
        recall_at_5fpr_std=np.std(recall_5s),
        recall_at_10fpr_mean=np.mean(recall_10s),
        recall_at_10fpr_std=np.std(recall_10s),
        results=results,
    )


def run_multi_seed_experiment(
    model_class,
    model_kwargs: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: List[int] = None,
    verbose: bool = True,
    save_dir: Optional[Path] = None,
) -> AggregatedResults:
    """
    Run experiments with multiple random seeds.

    Args:
        model_class: Model class to instantiate.
        model_kwargs: Arguments for model constructor (excluding seed).
        X_train: Training features.
        X_test: Test features.
        y_test: Test labels.
        seeds: List of random seeds.
        verbose: Print progress.
        save_dir: If provided, save results incrementally after each seed.

    Returns:
        AggregatedResults with mean/std across seeds.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    results = []

    for seed in seeds:
        if verbose:
            print(f"  Running with seed={seed}...", end=" ", flush=True)

        # Create model with this seed
        model = model_class(seed=seed, **model_kwargs)
        result = run_experiment(model, X_train, X_test, y_test, seed)
        results.append(result)

        if verbose:
            print(f"AUC-ROC={result.auc_roc:.4f}")

        # Save incrementally after each seed
        if save_dir is not None:
            agg = _aggregate_results(results, seeds)
            save_results(agg, save_dir)

    return _aggregate_results(results, seeds)


def save_results(results: AggregatedResults, output_dir: Path) -> None:
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{results.model_name}_results.json"

    # Convert to serializable dict
    data = {
        "model_name": results.model_name,
        "n_seeds": results.n_seeds,
        "seeds": results.seeds,
        "metrics": {
            "auc_roc": {"mean": results.auc_roc_mean, "std": results.auc_roc_std},
            "auc_pr": {"mean": results.auc_pr_mean, "std": results.auc_pr_std},
            "recall_at_5fpr": {"mean": results.recall_at_5fpr_mean, "std": results.recall_at_5fpr_std},
            "recall_at_10fpr": {"mean": results.recall_at_10fpr_mean, "std": results.recall_at_10fpr_std},
        },
        "individual_results": [asdict(r) for r in results.results],
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Results saved to: {filepath}")


def generate_latex_table(all_results: List[AggregatedResults]) -> str:
    """
    Generate LaTeX table from results.

    Args:
        all_results: List of aggregated results for each model.

    Returns:
        LaTeX table string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Model Comparison (" + f"{all_results[0].n_seeds} seeds" + r")}",
        r"\label{tab:main_results}",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{AUC-ROC} & \textbf{Recall@5\%FPR} & \textbf{Recall@10\%FPR} \\",
        r"\midrule",
    ]

    for r in all_results:
        line = f"{r.model_name} & "
        line += f"${r.auc_roc_mean:.3f} \\pm {r.auc_roc_std:.3f}$ & "
        line += f"${r.recall_at_5fpr_mean:.3f} \\pm {r.recall_at_5fpr_std:.3f}$ & "
        line += f"${r.recall_at_10fpr_mean:.3f} \\pm {r.recall_at_10fpr_std:.3f}$ \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
