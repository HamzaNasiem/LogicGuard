"""
McNemar's Test for Statistical Significance.

Required for IEEE journal submission.
Tests whether AvicennaGuard accuracy improvement over baseline is
statistically significant (not due to random chance).

McNemar's test is appropriate here because:
    - We have paired observations (same queries, two conditions: baseline vs +LG)
    - Binary outcomes (correct / incorrect per query)
    - We want to test if the change in performance is significant

Usage:
    python experiments/statistical/mcnemar_test.py \
        --results data/results/phase1/all_model_results.json \
        --output  data/results/phase1/statistical_significance.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def mcnemar_test(n01: int, n10: int) -> tuple[float, float]:
    """
    McNemar's test for paired nominal data.

    Args:
        n01: Count where baseline=wrong, avicennaguard=correct
        n10: Count where baseline=correct, avicennaguard=wrong

    Returns:
        (chi2_statistic, p_value)

    With continuity correction (Yates):
        χ² = (|n01 - n10| - 1)² / (n01 + n10)
    """
    if (n01 + n10) == 0:
        return 0.0, 1.0
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_value)


def compute_confidence_interval(
    correct: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    More accurate than normal approximation for extreme proportions.
    """
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    p = correct / total
    n = total

    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)

    return max(0, center - margin), min(1, center + margin)


def run_significance_tests(results_path: Path) -> dict:
    """
    Run McNemar's tests on all model pairs (baseline vs +AvicennaGuard).

    Loads all_model_results.json and produces p-values and
    confidence intervals for all accuracy comparisons.
    """
    with open(results_path) as f:
        data = json.load(f)

    output = {}

    # Models to compare (baseline vs avicennaguard)
    model_pairs = [
        ("llama2_7b_baseline",  "llama2_7b_avicennaguard" if "llama2_7b_avicennaguard" in data.get("per_query_results", {}) else "llama2_7b_logicguard"),
        ("mistral_7b_baseline", "mistral_7b_avicennaguard" if "mistral_7b_avicennaguard" in data.get("per_query_results", {}) else "mistral_7b_logicguard"),
        ("llama32_3b_baseline", "llama32_3b_avicennaguard" if "llama32_3b_avicennaguard" in data.get("per_query_results", {}) else "llama32_3b_logicguard"),
    ]

    for base_key, lg_key in model_pairs:
        base_results = data.get("per_query_results", {}).get(base_key, [])
        lg_results   = data.get("per_query_results", {}).get(lg_key, [])

        if not base_results or not lg_results:
            logger.warning("Per-query results not found for %s / %s", base_key, lg_key)
            continue

        n01 = sum(1 for b, l in zip(base_results, lg_results) if not b and l)
        n10 = sum(1 for b, l in zip(base_results, lg_results) if b and not l)
        n   = len(base_results)

        chi2_stat, p_value = mcnemar_test(n01, n10)

        base_correct = sum(base_results)
        lg_correct   = sum(lg_results)

        ci_base = compute_confidence_interval(base_correct, n)
        ci_lg   = compute_confidence_interval(lg_correct, n)

        model_name = base_key.replace("_baseline", "")
        output[model_name] = {
            "n_queries":          n,
            "baseline_accuracy":  round(base_correct / n * 100, 1),
            "avicennaguard_accuracy": round(lg_correct  / n * 100, 1),
            "delta_pp":           round((lg_correct - base_correct) / n * 100, 1),
            "mcnemar": {
                "n01":        n01,
                "n10":        n10,
                "chi2":       round(chi2_stat, 4),
                "p_value":    round(p_value, 6),
                "significant": p_value < 0.05,
            },
            "confidence_intervals_95pct": {
                "baseline":   [round(ci_base[0] * 100, 1), round(ci_base[1] * 100, 1)],
                "avicennaguard": [round(ci_lg[0]   * 100, 1), round(ci_lg[1]   * 100, 1)],
            },
        }

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="McNemar's significance tests for AvicennaGuard")
    parser.add_argument("--results", default="data/results/phase1/all_model_results.json")
    parser.add_argument("--output",  default="data/results/phase1/statistical_significance.json")
    args = parser.parse_args()

    results = run_significance_tests(Path(args.results))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Statistical Significance Results ===")
    for model, stats in results.items():
        sig = "✓ SIGNIFICANT" if stats["mcnemar"]["significant"] else "✗ NOT significant"
        print(f"\n{model}:")
        print(f"  Accuracy: {stats['baseline_accuracy']}% → {stats.get('avicennaguard_accuracy', stats.get('logicguard_accuracy'))}% ({stats['delta_pp']:+.1f} pp)")
        print(f"  McNemar p-value: {stats['mcnemar']['p_value']:.6f}  {sig}")
        print(f"  95% CI baseline:    [{stats['confidence_intervals_95pct']['baseline'][0]}%, {stats['confidence_intervals_95pct']['baseline'][1]}%]")
        print(f"  95% CI +AvicennaGuard: [{stats['confidence_intervals_95pct'].get('avicennaguard', stats['confidence_intervals_95pct'].get('logicguard'))[0]}%, {stats['confidence_intervals_95pct'].get('avicennaguard', stats['confidence_intervals_95pct'].get('logicguard'))[1]}%]")

    print(f"\nSaved to: {args.output}")
