"""
Baseline Evaluation Metrics for AvicennaGuard.

Provides standard binary classification metrics (Accuracy, Precision, Recall,
F1, Specificity, Confusion Matrix) and per-group breakdowns for baseline methods
(SelfCheckGPT, Dense RAG, Logic-LM) on the AvicennaGuard benchmark dataset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def parse_bool_answer(val: Any) -> Optional[bool]:
    """
    Parse various truth/boolean representations to standard bool or None.

    Args:
        val: Input truth value (bool, str, int, etc.).

    Returns:
        True, False, or None if uncertain / OOD / unparseable.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "yes", "1", "t", "y", "entailed", "sat", "proven_true", "valid"):
            return True
        if v in ("false", "no", "0", "f", "n", "refuted", "unsat", "proven_false", "invalid"):
            return False
        if v in ("ood", "unknown", "shakk", "none", "null", "uncertain"):
            return None
    return None


def compute_classification_metrics(
    predictions: List[Any],
    ground_truths: List[Any],
    include_ood: bool = False,
) -> Dict[str, Any]:
    """
    Compute standard binary classification metrics and confusion matrix.

    Positive class = Ground truth is True (valid logical relation / true statement).
    Negative class = Ground truth is False (invalid logical relation / false statement).

    Args:
        predictions: List of predictions (bool, str, or None).
        ground_truths: List of ground truths (bool, str, or None).
        include_ood: If True, OOD ground truths are treated as negative/rejection target.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, specificity,
        and confusion_matrix dict with (tp, fp, tn, fn, total).
    """
    tp = tn = fp = fn = 0
    ood_count = 0
    unresolved_count = 0

    for pred_raw, gt_raw in zip(predictions, ground_truths):
        gt_bool = parse_bool_answer(gt_raw)
        pred_bool = parse_bool_answer(pred_raw)

        if gt_raw == "OOD" or (isinstance(gt_raw, str) and gt_raw.upper() == "OOD"):
            ood_count += 1
            if not include_ood:
                continue

        if gt_bool is None:
            # Non-boolean ground truth skipped if not include_ood
            continue

        if pred_bool is None:
            # Baseline could not decide (abstention/unknown) -> treat as False for binary claim
            pred_bool = False
            unresolved_count += 1

        if gt_bool is True and pred_bool is True:
            tp += 1
        elif gt_bool is False and pred_bool is False:
            tn += 1
        elif gt_bool is False and pred_bool is True:
            fp += 1
        elif gt_bool is True and pred_bool is False:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total": total,
        },
        "ood_count": ood_count,
        "unresolved_count": unresolved_count,
    }


def compute_group_metrics(
    results: List[Dict[str, Any]],
    group_key: str = "query_type",
) -> Dict[str, Dict[str, Any]]:
    """
    Compute classification metrics grouped by a query attribute (e.g. query_type, source).

    Args:
        results: List of per-query result dictionaries.
        group_key: Key name to group by.

    Returns:
        Dictionary mapping group names to their respective classification metrics.
    """
    grouped = defaultdict(list)
    for r in results:
        key = r.get(group_key, "unknown")
        grouped[key].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for group_name, items in grouped.items():
        preds = [item.get("prediction") for item in items]
        gts = [item.get("ground_truth") for item in items]
        metrics = compute_classification_metrics(preds, gts)
        metrics["count"] = len(items)
        out[group_name] = metrics

    return out


def format_metrics_summary(
    method_name: str,
    metrics: Dict[str, Any],
    by_type: Optional[Dict[str, Dict[str, Any]]] = None,
    by_source: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Format evaluation metrics as a clean, human-readable terminal table.

    Args:
        method_name: Name of evaluated baseline or pipeline.
        metrics: Aggregate classification metrics dictionary.
        by_type: Optional per-query-type breakdown dictionary.
        by_source: Optional per-source-dataset breakdown dictionary.

    Returns:
        Formatted multi-line summary string.
    """
    cm = metrics.get("confusion_matrix", {})
    tp, fp, tn, fn = cm.get("tp", 0), cm.get("fp", 0), cm.get("tn", 0), cm.get("fn", 0)
    total = cm.get("total", 0)

    lines = [
        "=" * 68,
        f"  EVALUATION SUMMARY: {method_name.upper()}",
        "=" * 68,
        f"  Total Evaluated: {total:<6} | Accuracy:    {metrics.get('accuracy', 0.0):.1%}",
        f"  Precision:       {metrics.get('precision', 0.0):.1%} | Recall:      {metrics.get('recall', 0.0):.1%}",
        f"  F1 Score:        {metrics.get('f1', 0.0):.1%} | Specificity: {metrics.get('specificity', 0.0):.1%}",
        "-" * 68,
        f"  Confusion Matrix: TP={tp:<4} FP={fp:<4} TN={tn:<4} FN={fn:<4}",
        "-" * 68,
    ]

    if by_type:
        lines.append("  Breakdown by Query Type:")
        lines.append(f"    {'Type':<15} {'N':<6} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        lines.append("    " + "-" * 55)
        for qtype, m in sorted(by_type.items()):
            cnt = m.get("count", m.get("confusion_matrix", {}).get("total", 0))
            lines.append(
                f"    {qtype:<15} {cnt:<6} "
                f"{m.get('accuracy', 0.0):>6.1%}  "
                f"{m.get('precision', 0.0):>6.1%}  "
                f"{m.get('recall', 0.0):>6.1%}  "
                f"{m.get('f1', 0.0):>6.1%}"
            )
        lines.append("-" * 68)

    if by_source:
        lines.append("  Breakdown by Source Dataset:")
        lines.append(f"    {'Source':<18} {'N':<6} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        lines.append("    " + "-" * 58)
        for src, m in sorted(by_source.items()):
            cnt = m.get("count", m.get("confusion_matrix", {}).get("total", 0))
            lines.append(
                f"    {src:<18} {cnt:<6} "
                f"{m.get('accuracy', 0.0):>6.1%}  "
                f"{m.get('precision', 0.0):>6.1%}  "
                f"{m.get('recall', 0.0):>6.1%}  "
                f"{m.get('f1', 0.0):>6.1%}"
            )
        lines.append("=" * 68)

    return "\n".join(lines)
