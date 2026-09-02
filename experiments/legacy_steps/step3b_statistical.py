"""
STEP 3b: Statistical Significance Testing  (IEEE journal requirement)
======================================================================
Reads all_model_results.json (from step2 v2) and computes:

  1. McNemar's test — is AvicennaGuard improvement statistically significant?
  2. Wilson score 95% confidence intervals for all accuracy values
  3. Effect size (Cohen's g for paired proportions)
  4. Latency analysis — Stage 1, Stage 2, and total overhead vs baseline LLM

Output:
  statistical_significance.json   machine-readable
  statistical_report.txt          paste into paper

Why McNemar's test?
  - Paired binary observations (same query answered by baseline AND +LG)
  - Binary outcome per query (correct / incorrect)
  - Tests if the change in performance is significant (not random chance)
  - Standard for comparing two classifiers on same test set

Usage:
    python step3b_statistical.py
    python step3b_statistical.py --results all_model_results.json
"""

import json
import math
import argparse
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# STATISTICAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def mcnemar_test(n01: int, n10: int) -> tuple:
    """
    McNemar's test with Yates continuity correction.

    n01 = queries where baseline WRONG, +LogicGuard CORRECT  (LG helped)
    n10 = queries where baseline CORRECT, +LogicGuard WRONG  (LG hurt)

    H0: P(baseline wrong, LG correct) = P(baseline correct, LG wrong)
    H1: The two conditions differ significantly

    Chi-squared with continuity correction:
        χ² = (|n01 - n10| - 1)² / (n01 + n10)

    Returns: (chi2_stat, p_value)
    """
    if (n01 + n10) == 0:
        return 0.0, 1.0

    chi2 = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    # Chi-squared CDF with 1 degree of freedom
    p_value = _chi2_sf(chi2, df=1)
    return round(chi2, 4), round(p_value, 6)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function P(X > x) for chi-squared distribution (df=1)."""
    # For df=1: chi2_sf(x) = erfc(sqrt(x/2))
    # erfc(z) = 1 - erf(z)
    return math.erfc(math.sqrt(x / 2))


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple:
    """
    Wilson score confidence interval for a proportion p = k/n.

    More accurate than normal approximation for extreme proportions
    (e.g., p near 0 or 1, which is common with 100% precision claims).

    Returns: (lower_pct, upper_pct) as percentages
    """
    if n == 0:
        return (0.0, 100.0)
    # z for 95% CI
    z = 1.959964  # scipy.stats.norm.ppf(0.975)
    p = k / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n ** 2))) / (1 + z2 / n)
    lower  = max(0.0, center - margin) * 100
    upper  = min(1.0, center + margin) * 100
    return round(lower, 1), round(upper, 1)


def cohens_g(n01: int, n10: int) -> float:
    """
    Effect size (Cohen's g) for McNemar's test.
    g = |p01 - 0.5|  where p01 = n01/(n01+n10)

    Interpretation:
        g < 0.1  — small
        0.1–0.3  — medium
        > 0.3    — large
    """
    total = n01 + n10
    if total == 0:
        return 0.0
    p01 = n01 / total
    return round(abs(p01 - 0.5), 4)


def latency_stats(values: list) -> dict:
    """Compute mean, median, p95 for a list of latency values (ms)."""
    if not values:
        return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'n': 0}
    s = sorted(v for v in values if v is not None and v > 0)
    if not s:
        return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'n': 0}
    n    = len(s)
    mean = sum(s) / n
    med  = s[n // 2]
    p95  = s[min(int(n * 0.95), n - 1)]
    return {'mean': round(mean, 2), 'median': round(med, 2), 'p95': round(p95, 2), 'n': n}


# ─────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def run_analysis(results_path: str) -> dict:
    with open(results_path, 'r') as f:
        data = json.load(f)

    summaries   = data.get('summaries',   {})
    all_results = data.get('results',     {})

    output = {
        'source_filter': data.get('metadata', {}).get('source_filter', 'unknown'),
        'n_queries':     data.get('metadata', {}).get('n_queries_used', '?'),
        'mcnemar_tests': {},
        'confidence_intervals': {},
        'latency_analysis': {},
        'summary_table': [],
    }

    model_pairs = [
        ('llama2_7b_baseline',   'llama2_7b_avicennaguard' if 'llama2_7b_avicennaguard' in all_results else 'llama2_7b_logicguard',   'LLaMA2-7B'),
        ('mistral_7b_baseline',  'mistral_7b_avicennaguard' if 'mistral_7b_avicennaguard' in all_results else 'mistral_7b_logicguard',  'Mistral-7B'),
        ('llama32_3b_baseline',  'llama32_3b_avicennaguard' if 'llama32_3b_avicennaguard' in all_results else 'llama32_3b_logicguard',  'LLaMA3.2-3B'),
    ]

    for base_key, lg_key, display_name in model_pairs:
        base_sum = summaries.get(base_key, {})
        lg_sum   = summaries.get(lg_key,   {})
        base_res = all_results.get(base_key, [])
        lg_res   = all_results.get(lg_key,   [])

        if not base_res or not lg_res:
            print(f"  ⚠️  No per-query results for {base_key} / {lg_key}")
            continue

        # ── McNemar's test ────────────────────────────────────────────
        base_correct = base_sum.get('per_query_correct', [r['is_correct'] for r in base_res])
        lg_correct   = lg_sum.get('per_query_correct',   [r['is_correct'] for r in lg_res])

        n01 = sum(1 for b, l in zip(base_correct, lg_correct) if not b and l)
        n10 = sum(1 for b, l in zip(base_correct, lg_correct) if b     and not l)
        n00 = sum(1 for b, l in zip(base_correct, lg_correct) if not b and not l)
        n11 = sum(1 for b, l in zip(base_correct, lg_correct) if b     and l)
        n   = len(base_correct)

        chi2, p_val = mcnemar_test(n01, n10)
        g           = cohens_g(n01, n10)

        # ── Confidence intervals ──────────────────────────────────────
        base_k  = sum(base_correct)
        lg_k    = sum(lg_correct)
        ci_base = wilson_ci(base_k, n)
        ci_lg   = wilson_ci(lg_k, n)

        base_acc = round(base_k / n * 100, 1) if n > 0 else 0
        lg_acc   = round(lg_k   / n * 100, 1) if n > 0 else 0
        delta    = round(lg_acc - base_acc, 1)

        mcnemar_result = {
            'display_name':       display_name,
            'n_queries':          n,
            'n01':                n01,   # baseline wrong, AG correct
            'n10':                n10,   # baseline correct, AG wrong
            'n11':                n11,   # both correct
            'n00':                n00,   # both wrong
            'chi2':               chi2,
            'p_value':            p_val,
            'significant_p05':    p_val < 0.05,
            'significant_p001':   p_val < 0.001,
            'effect_size_g':      g,
            'effect_magnitude':   'large' if g > 0.3 else 'medium' if g > 0.1 else 'small',
        }
        ci_result = {
            'baseline':       {'accuracy': base_acc, 'ci_95': list(ci_base)},
            'avicennaguard':  {'accuracy': lg_acc,   'ci_95': list(ci_lg)},
            'logicguard':     {'accuracy': lg_acc,   'ci_95': list(ci_lg)},
            'delta_pp':       delta,
        }

        output['mcnemar_tests'][display_name]        = mcnemar_result
        output['confidence_intervals'][display_name] = ci_result

        # ── Latency analysis ──────────────────────────────────────────
        llm_lats    = [r['latency']['llm_ms']           for r in lg_res if 'latency' in r]
        s1_lats     = [r['latency']['stage1_ms']        for r in lg_res if 'latency' in r]
        s2_lats     = [r['latency']['stage2_ms']        for r in lg_res if 'latency' in r]
        overhead_lats = [r['latency']['total_overhead_ms'] for r in lg_res if 'latency' in r]

        llm_mean    = latency_stats(llm_lats)['mean']
        overhead_mean = latency_stats(overhead_lats)['mean']
        overhead_pct = round(overhead_mean / llm_mean * 100, 1) if llm_mean > 0 else 0

        output['latency_analysis'][display_name] = {
            'llm_call_ms':           latency_stats(llm_lats),
            'stage1_ms':             latency_stats(s1_lats),
            'stage2_ms':             latency_stats(s2_lats),
            'total_overhead_ms':     latency_stats(overhead_lats),
            'overhead_pct_of_llm':   overhead_pct,
        }

        output['summary_table'].append({
            'model':         display_name,
            'base_acc':      base_acc,
            'lg_acc':        lg_acc,
            'delta_pp':      delta,
            'base_ci':       f"[{ci_base[0]}, {ci_base[1]}]",
            'lg_ci':         f"[{ci_lg[0]}, {ci_lg[1]}]",
            'p_value':       p_val,
            'significant':   p_val < 0.05,
            'effect_size_g': g,
        })

    return output


# ─────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────

def format_report(analysis: dict) -> str:
    lines = [
        "=" * 70,
        "  AvicennaGuard — Statistical Significance Report  (Step 3b)",
        f"  Source filter: {analysis['source_filter'].upper()}  |  "
        f"N queries: {analysis['n_queries']}",
        "=" * 70,
        "",
        "─" * 70,
        "  TABLE VII — Statistical Significance (McNemar's Test)",
        "─" * 70,
        f"  {'Model':<16} {'Base':>7} {'+AG':>7} {'Δ(pp)':>7}  {'p-value':>10}  {'Sig?':>6}  {'Effect':>8}",
        f"  {'─'*16} {'─'*7} {'─'*7} {'─'*7}  {'─'*10}  {'─'*6}  {'─'*8}",
    ]

    for row in analysis.get('summary_table', []):
        sig_str = '✓ p<.05' if row['significant'] else '—'
        if row['p_value'] < 0.001:
            sig_str = '✓ p<.001'
        lines.append(
            f"  {row['model']:<16} {row['base_acc']:>6.1f}% {row['lg_acc']:>6.1f}% "
            f"{row['delta_pp']:>+6.1f}%  {row['p_value']:>10.6f}  {sig_str:>6}  "
            f"{row['effect_size_g']:>8.4f}"
        )

    lines += [
        "",
        "  Note: McNemar's test with Yates continuity correction (df=1).",
        "  Effect size: Cohen's g (large >0.3, medium 0.1-0.3, small <0.1).",
        "",
        "─" * 70,
        "  TABLE VIII — 95% Confidence Intervals (Wilson Score)",
        "─" * 70,
        f"  {'Model':<16} {'Base Acc':>10}  {'95% CI':>14}    {'AG Acc':>8}  {'95% CI':>14}",
        f"  {'─'*16} {'─'*10}  {'─'*14}    {'─'*8}  {'─'*14}",
    ]

    for model_name, ci_data in analysis.get('confidence_intervals', {}).items():
        b   = ci_data['baseline']
        lg  = ci_data.get('avicennaguard', ci_data.get('logicguard', {}))
        lines.append(
            f"  {model_name:<16} {b['accuracy']:>9.1f}%  {str(b['ci_95']):>14}    "
            f"{lg['accuracy']:>7.1f}%  {str(lg['ci_95']):>14}"
        )

    lines += [
        "",
        "─" * 70,
        "  TABLE IX — Latency Analysis (AvicennaGuard overhead)",
        "─" * 70,
        f"  {'Model':<16}  {'LLM (ms)':>10}  {'Stage1 (ms)':>13}  "
        f"{'Stage2 (ms)':>13}  {'Overhead':>10}  {'Overhead %':>11}",
        f"  {'─'*16}  {'─'*10}  {'─'*13}  {'─'*13}  {'─'*10}  {'─'*11}",
    ]

    for model_name, lat in analysis.get('latency_analysis', {}).items():
        llm_m    = lat['llm_call_ms']['mean']
        s1_m     = lat['stage1_ms']['mean']
        s2_m     = lat['stage2_ms']['mean']
        oh_m     = lat['total_overhead_ms']['mean']
        oh_pct   = lat['overhead_pct_of_llm']
        lines.append(
            f"  {model_name:<16}  {llm_m:>10.2f}  {s1_m:>13.3f}  "
            f"{s2_m:>13.3f}  {oh_m:>10.3f}  {oh_pct:>10.1f}%"
        )

    lines += [
        "",
        "  Note: Stage 2 (BFS) is typically < 1ms — deterministic graph",
        "  traversal adds negligible overhead over baseline LLM call.",
        "",
        "─" * 70,
        "  McNemar's Test Detail",
        "─" * 70,
    ]

    for model_name, mc in analysis.get('mcnemar_tests', {}).items():
        lines += [
            f"\n  {model_name}:",
            f"    n01 (baseline wrong, AG correct) : {mc['n01']:4}  ← AG helped",
            f"    n10 (baseline correct, AG wrong) : {mc['n10']:4}  ← AG hurt (false alarms)",
            f"    n11 (both correct)               : {mc['n11']:4}",
            f"    n00 (both wrong)                 : {mc['n00']:4}",
            f"    χ²                               : {mc['chi2']:.4f}",
            f"    p-value                          : {mc['p_value']:.6f}",
            f"    Significant (p < 0.05)?          : {'YES' if mc['significant_p05'] else 'NO'}",
            f"    Significant (p < 0.001)?         : {'YES' if mc['significant_p001'] else 'NO'}",
            f"    Effect size (Cohen's g)          : {mc['effect_size_g']:.4f} ({mc['effect_magnitude']})",
        ]

    lines += [
        "",
        "─" * 70,
        "  PAPER CITATION (paste into Section V)",
        "─" * 70,
    ]

    for row in analysis.get('summary_table', []):
        p_str = f"p < 0.001" if row['p_value'] < 0.001 else f"p = {row['p_value']:.4f}"
        sig_sentence = (
            f"statistically significant ({p_str}, McNemar's test, effect size g={row['effect_size_g']:.3f})"
            if row['significant']
            else f"not statistically significant ({p_str})"
        )
        lines.append(
            f"\n  {row['model']}: The improvement from {row['base_acc']}% "
            f"(95% CI: {row['base_ci']}) to {row['lg_acc']}% "
            f"(95% CI: {row['lg_ci']}) is {sig_sentence}."
        )

    lines += ["", "=" * 70, "  END OF STATISTICAL REPORT", "=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',     default='all_model_results.json')
    parser.add_argument('--output_json', default='statistical_significance.json')
    parser.add_argument('--output_txt',  default='statistical_report.txt')
    args = parser.parse_args()

    print("=" * 70)
    print("  AvicennaGuard — Step 3b: Statistical Significance Tests")
    print("=" * 70)

    if not Path(args.results).exists():
        print(f"ERROR: {args.results} not found. Run step2 first.")
        sys.exit(1)

    print(f"\nLoading: {args.results}")
    analysis = run_analysis(args.results)

    report = format_report(analysis)
    print("\n" + report)

    # Save JSON
    with open(args.output_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  💾 JSON saved  : {args.output_json}")

    # Save TXT
    with open(args.output_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  💾 Report saved: {args.output_txt}")

    print(f"\n{'=' * 70}")
    print("  STEP 3b COMPLETE")
    print("  Tables VII, VIII, IX ready for IEEE paper.")
    print("  Next: python step5_generate_paper_tables.py")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
