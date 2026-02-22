# metrics.py

"""
STEP 3: Metrics Calculator
===========================
Reads all_model_results.json from Step 2 and produces:
  - Precision / Recall / F1 per model per query type
  - Confusion matrix per model
  - Hallucination interception breakdown
  - Paper-ready ASCII tables
  - metrics_report.json  (machine-readable)
  - metrics_report.txt   (human-readable, paste into paper)

Usage:
    python step3_metrics.py --results all_model_results.json
"""

import json
import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
# CORE METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(results: List[Dict]) -> Dict:
    """
    Binary confusion matrix for logical validation.
    Positive class  = ground_truth is TRUE  (valid logical claim)
    Negative class  = ground_truth is FALSE (invalid logical claim)

    TP: model said YES,   truth is TRUE
    TN: model said NO,    truth is FALSE
    FP: model said YES,   truth is FALSE  (hallucination)
    FN: model said NO,    truth is TRUE   (missed valid claim)
    """
    TP = TN = FP = FN = 0

    for r in results:
        gt     = r['ground_truth']            # bool
        pred   = r['final_answer']            # bool or None
        if pred is None:
            pred = False  # unclear → treat as negative prediction

        if gt is True  and pred is True:  TP += 1
        elif gt is False and pred is False: TN += 1
        elif gt is False and pred is True:  FP += 1
        elif gt is True  and pred is False: FN += 1

    total = TP + TN + FP + FN
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'total': total}


def compute_prf1(cm: Dict) -> Dict:
    """Compute Precision, Recall, F1, Accuracy from confusion matrix."""
    TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    total = cm['total']

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / total if total > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return {
        'precision':   round(precision * 100, 1),
        'recall':      round(recall    * 100, 1),
        'f1':          round(f1        * 100, 1),
        'accuracy':    round(accuracy  * 100, 1),
        'specificity': round(specificity * 100, 1),
    }


def compute_per_type_metrics(results: List[Dict]) -> Dict:
    """Compute confusion matrix + P/R/F1 broken down by query type."""
    by_type = defaultdict(list)
    for r in results:
        by_type[r['type']].append(r)

    per_type = {}
    for qtype, type_results in by_type.items():
        cm   = compute_confusion_matrix(type_results)
        prf1 = compute_prf1(cm)
        per_type[qtype] = {
            'n':       len(type_results),
            'cm':      cm,
            'metrics': prf1,
        }
    return per_type


def compute_hallucination_analysis(
    baseline_results: List[Dict],
    logicguard_results: List[Dict]
) -> Dict:
    """
    Detailed hallucination interception analysis.
    Compares baseline vs LogicGuard on same queries.
    """
    # Index baseline by question
    baseline_by_q = {r['question']: r for r in baseline_results}

    intercepted      = []   # LLM wrong → LogicGuard correct
    false_alarms     = []   # LLM correct → LogicGuard wrong
    both_correct     = []
    both_wrong       = []

    for lg_r in logicguard_results:
        q       = lg_r['question']
        bl_r    = baseline_by_q.get(q)
        if not bl_r:
            continue

        bl_correct = bl_r['is_correct']
        lg_correct = lg_r['is_correct']

        if not bl_correct and lg_correct:
            intercepted.append(q)
        elif bl_correct and not lg_correct:
            false_alarms.append(q)
        elif bl_correct and lg_correct:
            both_correct.append(q)
        else:
            both_wrong.append(q)

    total_llm_errors = len(intercepted) + len(both_wrong)

    return {
        'intercepted':         len(intercepted),
        'intercepted_qs':      intercepted,
        'false_alarms':        len(false_alarms),
        'false_alarm_qs':      false_alarms,
        'both_correct':        len(both_correct),
        'both_wrong':          len(both_wrong),
        'total_llm_errors':    total_llm_errors,
        'interception_rate':   round(len(intercepted) / total_llm_errors * 100, 1)
                               if total_llm_errors > 0 else 100.0,
    }


# ─────────────────────────────────────────────────────────────────────
# REPORT FORMATTERS
# ─────────────────────────────────────────────────────────────────────

def format_confusion_matrix(cm: Dict, model_name: str) -> str:
    """Format confusion matrix as ASCII art."""
    TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    lines = [
        f"\n  Confusion Matrix — {model_name}",
        f"  {'':25} Predicted YES   Predicted NO",
        f"  {'Actual YES (valid claim)':25}  TP = {TP:4}         FN = {FN:4}",
        f"  {'Actual NO  (invalid claim)':25}  FP = {FP:4}         TN = {TN:4}",
    ]
    return '\n'.join(lines)


def format_prf1_table(all_metrics: Dict) -> str:
    """
    Format Precision/Recall/F1 comparison table for all models.
    all_metrics: {run_key: {'overall': prf1, 'per_type': {...}}}
    """
    header = (
        f"\n  {'Model':<28} {'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'Acc':>6} {'Spec':>6}"
    )
    sep    = "  " + "─" * 62
    rows   = [header, sep]

    for run_key, data in all_metrics.items():
        m    = data['overall']
        tag  = ' [+LG]' if 'logicguard' in run_key else '      '
        name = run_key.replace('_baseline', '').replace('_logicguard', '') + tag
        rows.append(
            f"  {name:<28} {m['precision']:>5.1f}% {m['recall']:>5.1f}% "
            f"{m['f1']:>5.1f}% {m['accuracy']:>5.1f}% {m['specificity']:>5.1f}%"
        )

    return '\n'.join(rows)


def format_per_type_table(all_metrics: Dict) -> str:
    """Per query-type accuracy breakdown."""
    qtypes = ['taxonomic', 'categorical', 'hypothetical']

    header = (
        f"\n  {'Model':<28} {'Taxonomic':>12} {'Categorical':>12} "
        f"{'Hypothetical':>14} {'Overall':>9}"
    )
    sep    = "  " + "─" * 72
    rows   = [header, sep]

    for run_key, data in all_metrics.items():
        tag  = ' [+LG]' if 'logicguard' in run_key else '      '
        name = run_key.replace('_baseline', '').replace('_logicguard', '') + tag
        per  = data['per_type']
        vals = []
        for qt in qtypes:
            if qt in per:
                vals.append(f"{per[qt]['metrics']['accuracy']:>10.1f}%")
            else:
                vals.append(f"{'N/A':>11}")
        overall = data['overall']['accuracy']
        rows.append(
            f"  {name:<28} {vals[0]} {vals[1]} {vals[2]} {overall:>8.1f}%"
        )

    return '\n'.join(rows)


def format_hallucination_table(hall_data: Dict) -> str:
    """Hallucination interception table per model."""
    header = (
        f"\n  {'Model':<25} {'LLM Errors':>12} {'Intercepted':>12} "
        f"{'Interc. Rate':>14} {'False Alarms':>13}"
    )
    sep    = "  " + "─" * 78
    rows   = [header, sep]

    for model_key, h in hall_data.items():
        rows.append(
            f"  {model_key:<25} {h['total_llm_errors']:>12} "
            f"{h['intercepted']:>12} {h['interception_rate']:>12.1f}% "
            f"{h['false_alarms']:>13}"
        )

    return '\n'.join(rows)


def format_paper_numbers(all_metrics: Dict, hall_data: Dict) -> str:
    """
    Print the exact numbers to use in the paper.
    """
    lines = [
        "",
        "  ╔══════════════════════════════════════════════════════════════╗",
        "  ║              NUMBERS TO USE IN IEEE PAPER                   ║",
        "  ╚══════════════════════════════════════════════════════════════╝",
        "",
        "  TABLE II — Multi-Model Comparison:",
        "",
        f"  {'Method':<30} {'Logic Q':>9} {'Categorical':>12} {'Hypothet.':>11} {'Overall':>9} {'Hall.Caught':>12}",
        "  " + "─" * 78,
    ]

    for run_key, data in all_metrics.items():
        tag     = '+LogicGuard' if 'logicguard' in run_key else 'Baseline  '
        base    = run_key.replace('_baseline', '').replace('_logicguard', '')
        name    = f"{base} ({tag})"
        pt      = data['per_type']
        tax_acc = pt.get('taxonomic', {}).get('metrics', {}).get('accuracy', 0)
        cat_acc = pt.get('categorical', {}).get('metrics', {}).get('accuracy', 0)
        hyp_acc = pt.get('hypothetical', {}).get('metrics', {}).get('accuracy', 0)
        overall = data['overall']['accuracy']
        # Hallucinations
        hkey   = run_key.replace('_baseline', '').replace('_logicguard', '')
        if 'logicguard' in run_key and hkey in hall_data:
            caught = hall_data[hkey]['intercepted']
            total  = hall_data[hkey]['total_llm_errors']
            hall_str = f"{caught}/{total}"
        else:
            hall_str = "0/0 (none)"

        lines.append(
            f"  {name:<30} {tax_acc:>8.1f}% {cat_acc:>11.1f}% "
            f"{hyp_acc:>10.1f}% {overall:>8.1f}% {hall_str:>12}"
        )

    lines += [
        "",
        "  F1 SCORES (for Precision/Recall/F1 table in paper):",
        "",
    ]
    for run_key, data in all_metrics.items():
        tag  = '+LG' if 'logicguard' in run_key else 'base'
        base = run_key.replace('_baseline', '').replace('_logicguard', '')
        m    = data['overall']
        lines.append(
            f"    {base} ({tag}):  "
            f"Precision={m['precision']}%  "
            f"Recall={m['recall']}%  "
            f"F1={m['f1']}%  "
            f"Accuracy={m['accuracy']}%"
        )

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='all_model_results.json',
                        help='Output from step2_multi_model_runner.py')
    parser.add_argument('--report_txt',  type=str, default='metrics_report.txt')
    parser.add_argument('--report_json', type=str, default='metrics_report.json')
    args = parser.parse_args()

    print("=" * 65)
    print("  LogicGuard — Step 3: Metrics & Confusion Matrices")
    print("=" * 65)

    # ── Load results ──────────────────────────────────────────────────
    print(f"\nLoading: {args.results}")
    try:
        with open(args.results, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {args.results} not found. Run step2 first.")
        sys.exit(1)

    all_run_results  = data.get('results', {})
    if not all_run_results:
        print("ERROR: No results found in file.")
        sys.exit(1)

    run_keys = list(all_run_results.keys())
    print(f"  Run keys found: {run_keys}")

    # ── Compute all metrics ───────────────────────────────────────────
    print("\nComputing metrics...")
    all_metrics  = {}
    cms          = {}
    hall_data    = {}

    for run_key, results in all_run_results.items():
        if not results:
            continue
        cm       = compute_confusion_matrix(results)
        prf1     = compute_prf1(cm)
        per_type = compute_per_type_metrics(results)

        all_metrics[run_key] = {
            'overall':  prf1,
            'per_type': per_type,
        }
        cms[run_key] = cm
        print(f"  ✓ {run_key}: Acc={prf1['accuracy']}%  F1={prf1['f1']}%")

    # ── Hallucination analysis ─────────────────────────────────────────
    print("\nComputing hallucination interception...")
    for run_key in run_keys:
        if '_logicguard' not in run_key:
            continue
        base_key = run_key.replace('_logicguard', '_baseline')
        model_key = run_key.replace('_logicguard', '')
        if base_key in all_run_results and run_key in all_run_results:
            h = compute_hallucination_analysis(
                all_run_results[base_key],
                all_run_results[run_key]
            )
            hall_data[model_key] = h
            print(f"  ✓ {model_key}: {h['intercepted']}/{h['total_llm_errors']} caught "
                  f"({h['interception_rate']}%)")

    # ── Build report ───────────────────────────────────────────────────
    print("\nBuilding report...")
    report_lines = [
        "=" * 65,
        "  LogicGuard — Complete Metrics Report",
        "  (Generated by step3_metrics.py)",
        "=" * 65,
        "",
        "─" * 65,
        "  1. PRECISION / RECALL / F1 — ALL MODELS",
        "─" * 65,
        format_prf1_table(all_metrics),
        "",
        "─" * 65,
        "  2. ACCURACY BY QUERY TYPE",
        "─" * 65,
        format_per_type_table(all_metrics),
        "",
        "─" * 65,
        "  3. CONFUSION MATRICES",
        "─" * 65,
    ]

    for run_key, cm in cms.items():
        report_lines.append(format_confusion_matrix(cm, run_key))
        prf1 = all_metrics[run_key]['overall']
        report_lines.append(
            f"    Precision={prf1['precision']}%  "
            f"Recall={prf1['recall']}%  "
            f"F1={prf1['f1']}%  "
            f"Specificity={prf1['specificity']}%\n"
        )

    report_lines += [
        "─" * 65,
        "  4. HALLUCINATION INTERCEPTION ANALYSIS",
        "─" * 65,
        format_hallucination_table(hall_data),
        "",
    ]

    # Per model detail
    for model_key, h in hall_data.items():
        report_lines += [
            f"  {model_key} — Intercepted hallucinations:",
        ]
        for i, q in enumerate(h['intercepted_qs'][:10], 1):
            report_lines.append(f"    {i:2}. {q}")
        if h['false_alarms'] > 0:
            report_lines.append(f"  {model_key} — False alarms (LG wrongly overrode):")
            for i, q in enumerate(h['false_alarm_qs'][:5], 1):
                report_lines.append(f"    {i:2}. {q}")
        report_lines.append("")

    report_lines += [
        "─" * 65,
        "  5. NUMBERS TO PASTE INTO IEEE PAPER",
        "─" * 65,
        format_paper_numbers(all_metrics, hall_data),
        "",
        "=" * 65,
        "  END OF REPORT",
        "=" * 65,
    ]

    report_text = '\n'.join(report_lines)

    # ── Print to console ──────────────────────────────────────────────
    print("\n" + report_text)

    # ── Save TXT ──────────────────────────────────────────────────────
    with open(args.report_txt, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  💾 Report saved: {args.report_txt}")

    # ── Save JSON ─────────────────────────────────────────────────────
    json_report = {
        'metrics':    all_metrics,
        'confusion_matrices': {k: v for k, v in cms.items()},
        'hallucination_analysis': hall_data,
    }
    with open(args.report_json, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"  💾 JSON report saved: {args.report_json}")

    print(f"\n{'=' * 65}")
    print(f"  STEP 3 COMPLETE — All metrics computed")
    print(f"  Open metrics_report.txt to see paper numbers")
    print(f"{'=' * 65}\n")


if __name__ == '__main__':
    main()