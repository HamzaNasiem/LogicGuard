"""
Statistical Analyzer for AvicennaGuard Evaluation.

Provides rigorous statistical significance testing and uncertainty quantification
for journal-grade IEEE publications:
  1. McNemar's paired test with Yates continuity correction and exact binomial p-values
  2. Wilson score confidence intervals for binomial proportions (robust for near 0/100%)
  3. Cohen's g effect size for paired binary outcomes
  4. Latency analysis across pipeline stages (Stage 1 semantic parsing, Stage 2 BFS graph validation)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import scipy.stats as stats


class StatisticalAnalyzer:
    """Statistical testing and confidence interval computation for AvicennaGuard."""

    @staticmethod
    def _extract_correctness(results: Sequence[Any]) -> List[bool]:
        """
        Extract boolean correctness flags from a list of result items.

        Supports:
          - dict with 'is_correct' or 'correct'
          - dict with 'predicted_answer' and 'ground_truth'
          - bool or numeric 0/1 values
        """
        correctness: List[bool] = []
        for r in results:
            if isinstance(r, dict):
                if "is_correct" in r:
                    correctness.append(bool(r["is_correct"]))
                elif "correct" in r:
                    correctness.append(bool(r["correct"]))
                elif "predicted_answer" in r and "ground_truth" in r:
                    pred = str(r["predicted_answer"]).strip().lower()
                    gt = str(r["ground_truth"]).strip().lower()
                    correctness.append(pred == gt)
                elif "final_answer" in r and "ground_truth" in r:
                    pred = str(r["final_answer"]).strip().lower()
                    gt = str(r["ground_truth"]).strip().lower()
                    correctness.append(pred == gt)
                else:
                    raise KeyError(f"Cannot determine correctness from dictionary: {r.keys()}")
            elif isinstance(r, (bool, np.bool_)):
                correctness.append(bool(r))
            elif isinstance(r, (int, float)):
                correctness.append(bool(r != 0))
            else:
                raise TypeError(f"Unsupported result type for correctness extraction: {type(r)}")
        return correctness

    @classmethod
    def mcnemar_test(
        cls,
        results_baseline: Sequence[Any] | None = None,
        results_guard: Sequence[Any] | None = None,
        *,
        n01: int | None = None,
        n10: int | None = None,
        n11: int | None = None,
        n00: int | None = None,
    ) -> Dict[str, Any]:
        """
        Compute McNemar's test for paired binary classifications.

        Can be called with paired results sequences or directly with 2x2 contingency counts:
          - n01: Baseline WRONG, Guard CORRECT (Guard helped)
          - n10: Baseline CORRECT, Guard WRONG (Guard hurt / false alarm)
          - n11: Both CORRECT
          - n00: Both WRONG

        Continuity-corrected Chi-squared:
            χ² = (|n01 - n10| - 1)² / (n01 + n10)  (when |n01 - n10| >= 1 and n01 + n10 > 0)

        Returns:
            Dictionary containing contingency matrix, chi2 statistic, asymptotic and exact
            p-values, Cohen's g effect size, odds ratio, and significance flags.
        """
        if results_baseline is not None and results_guard is not None:
            base_corr = cls._extract_correctness(results_baseline)
            guard_corr = cls._extract_correctness(results_guard)

            if len(base_corr) != len(guard_corr):
                raise ValueError(
                    f"Paired evaluation mismatch: baseline has {len(base_corr)} items, "
                    f"guard has {len(guard_corr)} items"
                )

            n_total = len(base_corr)
            c_01 = sum(1 for b, g in zip(base_corr, guard_corr) if (not b) and g)
            c_10 = sum(1 for b, g in zip(base_corr, guard_corr) if b and (not g))
            c_11 = sum(1 for b, g in zip(base_corr, guard_corr) if b and g)
            c_00 = sum(1 for b, g in zip(base_corr, guard_corr) if (not b) and (not g))
        elif n01 is not None and n10 is not None:
            c_01 = int(n01)
            c_10 = int(n10)
            c_11 = int(n11 if n11 is not None else 0)
            c_00 = int(n00 if n00 is not None else 0)
            n_total = c_01 + c_10 + c_11 + c_00
        else:
            raise ValueError("Must provide either (results_baseline, results_guard) or (n01, n10).")

        discordant_total = c_01 + c_10

        # Chi-squared with Yates continuity correction
        if discordant_total == 0:
            chi2_stat = 0.0
            p_val_asym = 1.0
            p_val_exact = 1.0
        else:
            diff = abs(c_01 - c_10)
            if diff > 1:
                chi2_stat = float((diff - 1.0) ** 2 / discordant_total)
            else:
                chi2_stat = 0.0

            # Asymptotic p-value with df=1
            p_val_asym = float(stats.chi2.sf(chi2_stat, df=1))

            # Exact 2-sided binomial p-value
            # Under H0, p = 0.5 for discordant pairs
            k_min = min(c_01, c_10)
            binom_res = stats.binomtest(k_min, discordant_total, 0.5, alternative="two-sided")
            p_val_exact = float(binom_res.pvalue)

        # Cohen's g effect size: g = |P(01) - 0.5|
        if discordant_total == 0:
            effect_size_g = 0.0
        else:
            effect_size_g = abs((c_01 / discordant_total) - 0.5)

        if effect_size_g > 0.3:
            magnitude = "large"
        elif effect_size_g > 0.1:
            magnitude = "medium"
        elif effect_size_g > 0.0:
            magnitude = "small"
        else:
            magnitude = "negligible"

        # Odds ratio
        if c_10 == 0:
            odds_ratio = float("inf") if c_01 > 0 else 1.0
        else:
            odds_ratio = float(c_01 / c_10)

        # Accuracies
        base_acc = float((c_11 + c_10) / n_total) if n_total > 0 else 0.0
        guard_acc = float((c_11 + c_01) / n_total) if n_total > 0 else 0.0
        delta_acc = guard_acc - base_acc

        # Primary p_value is the asymptotic chi2 p-value (or exact if discordant < 25)
        primary_p = p_val_asym

        return {
            "n_queries": n_total,
            "n01": c_01,  # baseline wrong, guard correct
            "n10": c_10,  # baseline correct, guard wrong
            "n11": c_11,  # both correct
            "n00": c_00,  # both wrong
            "discordant_total": discordant_total,
            "contingency_matrix": [[c_11, c_10], [c_01, c_00]],
            "contingency_dict": {"n11": c_11, "n10": c_10, "n01": c_01, "n00": c_00},
            "chi2": round(chi2_stat, 4),
            "chi2_stat": float(chi2_stat),
            "p_value": float(primary_p),
            "p_value_asymptotic": float(p_val_asym),
            "p_value_exact": float(p_val_exact),
            "significant_p05": bool(primary_p < 0.05),
            "significant_p01": bool(primary_p < 0.01),
            "significant_p001": bool(primary_p < 0.001),
            "effect_size_g": round(effect_size_g, 4),
            "effect_magnitude": magnitude,
            "odds_ratio": odds_ratio,
            "baseline_accuracy": round(base_acc, 4),
            "guard_accuracy": round(guard_acc, 4),
            "delta_accuracy": round(delta_acc, 4),
            "delta_pp": round(delta_acc * 100, 2),
        }

    @staticmethod
    def wilson_confidence_interval(
        k: int,
        n: int,
        confidence: float = 0.95,
        as_percentage: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute the Wilson score confidence interval for a binomial proportion p = k/n.

        The Wilson score interval provides superior coverage probability compared to the
        Wald normal approximation, especially near boundary values (p -> 0 or p -> 1)
        and with finite sample sizes.

        Formula:
            center = (p + z^2 / (2n)) / (1 + z^2 / n)
            margin = (z * sqrt(p*(1-p)/n + z^2 / (4n^2))) / (1 + z^2 / n)
            interval = [center - margin, center + margin]

        Args:
            k: Number of successes (0 <= k <= n)
            n: Total number of trials (n >= 0)
            confidence: Confidence level in (0, 1), default 0.95
            as_percentage: If True, returns interval as percentages in [0, 100]

        Returns:
            (lower_bound, upper_bound)
        """
        if n <= 0:
            return (0.0, 100.0) if as_percentage else (0.0, 1.0)

        k = max(0, min(k, n))
        p = k / n

        # Two-sided standard normal critical value
        alpha = 1.0 - confidence
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z2 = z * z

        denom = 1.0 + (z2 / n)
        center = (p + (z2 / (2.0 * n))) / denom
        radicand = (p * (1.0 - p) / n) + (z2 / (4.0 * (n ** 2)))
        margin = (z * math.sqrt(max(0.0, radicand))) / denom

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        # Exact boundary clamping for zero and full successes
        if k == 0 or lower < 1e-15:
            lower = 0.0
        if k == n or (1.0 - upper) < 1e-15:
            upper = 1.0

        if as_percentage:
            return (round(lower * 100.0, 2), round(upper * 100.0, 2))
        return (float(lower), float(upper))

    @staticmethod
    def latency_stats(values: Sequence[float | int | None]) -> Dict[str, Any]:
        """
        Compute robust summary statistics for latency measurements in milliseconds.

        Returns:
            mean, median, p95, p99, min, max, std, and sample count n.
        """
        valid_vals = [float(v) for v in values if v is not None and not math.isnan(float(v)) and float(v) >= 0]
        if not valid_vals:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "n": 0,
            }

        s = sorted(valid_vals)
        n = len(s)
        mean_val = float(np.mean(s))
        med_val = float(np.median(s))
        p95_val = float(np.percentile(s, 95))
        p99_val = float(np.percentile(s, 99))
        min_val = float(s[0])
        max_val = float(s[-1])
        std_val = float(np.std(s)) if n > 1 else 0.0

        return {
            "mean": round(mean_val, 3),
            "median": round(med_val, 3),
            "p95": round(p95_val, 3),
            "p99": round(p99_val, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "std": round(std_val, 3),
            "n": n,
        }

    @classmethod
    def compute_paired_comparison(
        cls,
        results_baseline: Sequence[Any],
        results_guard: Sequence[Any],
        model_name: str = "",
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Comprehensive paired model evaluation: McNemar significance, Wilson CIs, and latency.
        """
        base_corr = cls._extract_correctness(results_baseline)
        guard_corr = cls._extract_correctness(results_guard)
        n = len(base_corr)

        mcnemar = cls.mcnemar_test(results_baseline, results_guard)

        base_k = sum(base_corr)
        guard_k = sum(guard_corr)

        base_ci = cls.wilson_confidence_interval(base_k, n, confidence=confidence, as_percentage=True)
        guard_ci = cls.wilson_confidence_interval(guard_k, n, confidence=confidence, as_percentage=True)

        base_acc = round(base_k / n * 100.0, 2) if n > 0 else 0.0
        guard_acc = round(guard_k / n * 100.0, 2) if n > 0 else 0.0
        delta_pp = round(guard_acc - base_acc, 2)

        # Latencies if present in dicts
        llm_lats = [r.get("latency", {}).get("llm_ms") for r in results_guard if isinstance(r, dict) and "latency" in r]
        s1_lats = [r.get("latency", {}).get("stage1_ms") for r in results_guard if isinstance(r, dict) and "latency" in r]
        s2_lats = [r.get("latency", {}).get("stage2_ms") for r in results_guard if isinstance(r, dict) and "latency" in r]
        oh_lats = [r.get("latency", {}).get("total_overhead_ms") for r in results_guard if isinstance(r, dict) and "latency" in r]

        llm_stat = cls.latency_stats(llm_lats)
        s1_stat = cls.latency_stats(s1_lats)
        s2_stat = cls.latency_stats(s2_lats)
        oh_stat = cls.latency_stats(oh_lats)

        oh_pct = (
            round((oh_stat["mean"] / llm_stat["mean"]) * 100.0, 2)
            if llm_stat["mean"] > 0
            else 0.0
        )

        return {
            "model_name": model_name,
            "n_queries": n,
            "baseline": {
                "correct_count": base_k,
                "accuracy_pct": base_acc,
                "ci_95_pct": list(base_ci),
            },
            "avicennaguard": {
                "correct_count": guard_k,
                "accuracy_pct": guard_acc,
                "ci_95_pct": list(guard_ci),
            },
            "delta_pp": delta_pp,
            "mcnemar": mcnemar,
            "latency": {
                "llm_call_ms": llm_stat,
                "stage1_ms": s1_stat,
                "stage2_ms": s2_stat,
                "total_overhead_ms": oh_stat,
                "overhead_pct_of_llm": oh_pct,
            },
        }

    @classmethod
    def evaluate_experiment_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full statistical evaluation on an experiment results dictionary
        (e.g., loaded from `all_model_results.json`).
        """
        summaries = data.get("summaries", {})
        all_results = data.get("results", {})
        metadata = data.get("metadata", {})

        model_pairs = [
            ("llama2_7b_baseline", "llama2_7b_avicennaguard" if "llama2_7b_avicennaguard" in all_results else "llama2_7b_logicguard", "LLaMA2-7B"),
            ("mistral_7b_baseline", "mistral_7b_avicennaguard" if "mistral_7b_avicennaguard" in all_results else "mistral_7b_logicguard", "Mistral-7B"),
            ("llama32_3b_baseline", "llama32_3b_avicennaguard" if "llama32_3b_avicennaguard" in all_results else "llama32_3b_logicguard", "LLaMA3.2-3B"),
        ]

        mcnemar_tests: Dict[str, Any] = {}
        confidence_intervals: Dict[str, Any] = {}
        latency_analysis: Dict[str, Any] = {}
        summary_table: List[Dict[str, Any]] = []

        for base_key, guard_key, display_name in model_pairs:
            base_res = all_results.get(base_key, [])
            guard_res = all_results.get(guard_key, [])

            if not base_res or not guard_res:
                continue

            comparison = cls.compute_paired_comparison(
                base_res, guard_res, model_name=display_name
            )

            mcnemar_tests[display_name] = comparison["mcnemar"]
            confidence_intervals[display_name] = {
                "baseline": comparison["baseline"],
                "avicennaguard": comparison["avicennaguard"],
                "delta_pp": comparison["delta_pp"],
            }
            latency_analysis[display_name] = comparison["latency"]

            summary_table.append({
                "model": display_name,
                "base_acc": comparison["baseline"]["accuracy_pct"],
                "guard_acc": comparison["avicennaguard"]["accuracy_pct"],
                "delta_pp": comparison["delta_pp"],
                "base_ci": f"[{comparison['baseline']['ci_95_pct'][0]}, {comparison['baseline']['ci_95_pct'][1]}]",
                "guard_ci": f"[{comparison['avicennaguard']['ci_95_pct'][0]}, {comparison['avicennaguard']['ci_95_pct'][1]}]",
                "chi2": comparison["mcnemar"]["chi2"],
                "p_value": comparison["mcnemar"]["p_value"],
                "significant": comparison["mcnemar"]["significant_p05"],
                "effect_size_g": comparison["mcnemar"]["effect_size_g"],
            })

        return {
            "source_filter": metadata.get("source_filter", "original"),
            "n_queries": metadata.get("n_queries_used", len(summary_table)),
            "mcnemar_tests": mcnemar_tests,
            "confidence_intervals": confidence_intervals,
            "latency_analysis": latency_analysis,
            "summary_table": summary_table,
        }
