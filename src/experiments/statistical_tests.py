"""
Statistical tests for comparing model performance.

Provides proper statistical analysis for multi-seed experiments:
- Wilcoxon signed-rank test for paired comparisons
- Cohen's d for effect size
- Confidence intervals
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric test comparing two related samples.
    Appropriate for comparing models across the same seeds.

    Args:
        scores_a: Performance scores for model A (one per seed).
        scores_b: Performance scores for model B (one per seed).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Tuple of (statistic, p-value).
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Arrays must have same length")

    if len(scores_a) < 5:
        # Wilcoxon needs at least 5 samples for meaningful results
        # Fall back to paired t-test warning
        print("Warning: <5 samples, Wilcoxon may not be reliable")

    try:
        statistic, p_value = stats.wilcoxon(
            scores_a, scores_b, alternative=alternative
        )
    except ValueError:
        # All differences are zero
        statistic, p_value = 0.0, 1.0

    return statistic, p_value


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform paired t-test for two related samples.

    Parametric alternative to Wilcoxon. Assumes normal distribution
    of differences.

    Args:
        scores_a: Performance scores for model A.
        scores_b: Performance scores for model B.

    Returns:
        Tuple of (t-statistic, p-value).
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    statistic, p_value = stats.ttest_rel(scores_a, scores_b)
    return statistic, p_value


def cohens_d(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Measures the standardized difference between two means.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        scores_a: Performance scores for model A.
        scores_b: Performance scores for model B.

    Returns:
        Cohen's d value (positive if A > B).
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    diff = scores_a - scores_b
    d = np.mean(diff) / np.std(diff, ddof=1)

    return d


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for mean.

    Args:
        scores: Performance scores.
        confidence: Confidence level (default: 0.95 for 95% CI).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    scores = np.asarray(scores)
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)

    # t-value for confidence level
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)

    margin = t_value * se
    return mean, mean - margin, mean + margin


def compare_models(
    results_a: Dict[str, List[float]],
    results_b: Dict[str, List[float]],
    model_name_a: str,
    model_name_b: str,
    metric: str = "auc_roc",
    alpha: float = 0.05,
) -> Dict:
    """
    Comprehensive comparison between two models.

    Args:
        results_a: Dict with metric -> list of scores for model A.
        results_b: Dict with metric -> list of scores for model B.
        model_name_a: Name of model A.
        model_name_b: Name of model B.
        metric: Which metric to compare.
        alpha: Significance level.

    Returns:
        Dict with comparison results.
    """
    scores_a = np.array(results_a[metric])
    scores_b = np.array(results_b[metric])

    # Wilcoxon test
    w_stat, w_pval = wilcoxon_test(scores_a, scores_b)

    # Paired t-test
    t_stat, t_pval = paired_ttest(scores_a, scores_b)

    # Effect size
    d = cohens_d(scores_a, scores_b)

    # Confidence intervals
    mean_a, ci_a_low, ci_a_high = confidence_interval(scores_a)
    mean_b, ci_b_low, ci_b_high = confidence_interval(scores_b)

    # Determine winner
    if w_pval < alpha:
        if mean_a > mean_b:
            winner = model_name_a
        else:
            winner = model_name_b
        significant = True
    else:
        winner = "tie"
        significant = False

    return {
        "model_a": model_name_a,
        "model_b": model_name_b,
        "metric": metric,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "std_a": np.std(scores_a),
        "std_b": np.std(scores_b),
        "ci_a": (ci_a_low, ci_a_high),
        "ci_b": (ci_b_low, ci_b_high),
        "wilcoxon_stat": w_stat,
        "wilcoxon_pval": w_pval,
        "ttest_stat": t_stat,
        "ttest_pval": t_pval,
        "cohens_d": d,
        "effect_size": effect_size_interpretation(d),
        "significant": significant,
        "winner": winner,
        "alpha": alpha,
    }


def generate_comparison_table(
    all_results: List[Dict],
    metric: str = "auc_roc",
) -> str:
    """
    Generate pairwise comparison table in LaTeX.

    Args:
        all_results: List of result dicts from runner.py.
        metric: Metric to compare.

    Returns:
        LaTeX table string.
    """
    n = len(all_results)
    model_names = [r["model_name"] for r in all_results]

    # Build comparison matrix
    comparisons = []
    for i in range(n):
        for j in range(i + 1, n):
            scores_a = [r[metric] for r in all_results[i]["individual_results"]]
            scores_b = [r[metric] for r in all_results[j]["individual_results"]]

            comp = compare_models(
                {metric: scores_a},
                {metric: scores_b},
                model_names[i],
                model_names[j],
                metric=metric,
            )
            comparisons.append(comp)

    # Generate table
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Pairwise Statistical Comparisons (Wilcoxon test)}",
        r"\label{tab:pairwise}",
        r"\centering",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Model A} & \textbf{Model B} & \textbf{p-value} & \textbf{Cohen's d} & \textbf{Winner} \\",
        r"\midrule",
    ]

    for c in comparisons:
        sig = "*" if c["significant"] else ""
        lines.append(
            f"{c['model_a']} & {c['model_b']} & "
            f"{c['wilcoxon_pval']:.4f}{sig} & "
            f"{c['cohens_d']:.2f} ({c['effect_size']}) & "
            f"{c['winner']} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{5}{l}{\small * p < 0.05}",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def significance_marker(p_value: float) -> str:
    """
    Return significance marker for p-value.

    Args:
        p_value: P-value from statistical test.

    Returns:
        Marker string: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), '' (ns)
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulated results from two models across 5 seeds
    model_a_scores = [0.85, 0.87, 0.84, 0.86, 0.88]
    model_b_scores = [0.80, 0.82, 0.79, 0.81, 0.83]

    # Compare
    result = compare_models(
        {"auc_roc": model_a_scores},
        {"auc_roc": model_b_scores},
        "Model A",
        "Model B",
    )

    print("Comparison Results:")
    print(f"  Model A: {result['mean_a']:.4f} ± {result['std_a']:.4f}")
    print(f"  Model B: {result['mean_b']:.4f} ± {result['std_b']:.4f}")
    print(f"  Wilcoxon p-value: {result['wilcoxon_pval']:.4f}")
    print(f"  Cohen's d: {result['cohens_d']:.2f} ({result['effect_size']})")
    print(f"  Winner: {result['winner']}")
