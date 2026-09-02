#!/usr/bin/env python3
"""
CLI Script: Run AvicennaGuard Multi-Model Benchmark Evaluation.

Executes baseline LLM versus +AvicennaGuard across 500-query benchmark
(or subsets) with full epistemic state tracking, latency profiling,
and hallucination interception metrics.

Usage Examples:
    # Quick offline mock mode across all models on 10 queries
    python scripts/run_benchmark_eval.py --mock --limit 10

    # Mock mode on full benchmark for specific model
    python scripts/run_benchmark_eval.py --mock --models llama3.2:3b

    # Live evaluation on Ollama models
    python scripts/run_benchmark_eval.py --models llama3.2:3b,mistral:7b --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

# Reconfigure stdout to UTF-8 or safe replacement for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src/ is in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from avicennaguard.eval.benchmark_runner import (  # noqa: E402
    DEFAULT_MODELS,
    MODEL_ALIASES,
    BenchmarkRunner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_cli")


def format_summary_table(comparison_table: list[dict]) -> str:
    """Format comparison summary into an ASCII table."""
    header = (
        f"\n  {'Model':<18} {'Base Acc':>9} {'Base F1':>8} {'+AG Acc':>8} {'+AG F1':>8} "
        f"{'Gain Acc':>9} {'Hall. Caught':>13} {'Interc.%':>9} {'Overhead':>9}"
    )
    sep = "  " + "-" * 98
    lines = [header, sep]

    for row in comparison_table:
        m = row["model"]
        b_acc = f"{row['baseline_acc']:.1f}%"
        b_f1 = f"{row['baseline_f1']:.1f}%"
        ag_acc = f"{row['guard_acc']:.1f}%"
        ag_f1 = f"{row['guard_f1']:.1f}%"
        d_acc = f"+{row['accuracy_gain']:.1f}%" if row["accuracy_gain"] >= 0 else f"{row['accuracy_gain']:.1f}%"
        hall = f"{row['hallucinations_intercepted']}"
        rate = f"{row['interception_rate']:.1f}%"
        ovh = f"{row['avg_overhead_ms']:.2f}ms"

        lines.append(
            f"  {m:<18} {b_acc:>9} {b_f1:>8} {ag_acc:>8} {ag_f1:>8} "
            f"{d_acc:>9} {hall:>13} {rate:>9} {ovh:>9}"
        )

    return "\n".join(lines)


def format_detailed_report(results: dict) -> str:
    """Generate comprehensive human-readable report."""
    meta = results.get("metadata", {})
    comp = results.get("comparison_summary", [])
    models_data = results.get("models", {})

    lines = [
        "=" * 85,
        "  AVICENNAGUARD -- MULTI-MODEL BENCHMARK EVALUATION REPORT",
        "=" * 85,
        f"  Benchmark File   : {meta.get('benchmark_file', 'N/A')}",
        f"  Mock Mode        : {meta.get('mock_mode', False)}",
        f"  Parser Mode      : {meta.get('parser_mode', 'N/A')}",
        f"  Models Evaluated : {', '.join(meta.get('models_evaluated', []))}",
        "",
        "-" * 85,
        "  1. MULTI-MODEL PERFORMANCE SUMMARY",
        "-" * 85,
        format_summary_table(comp),
        "",
    ]

    for model_name, m_data in models_data.items():
        ag = m_data.get("avicennaguard", {})
        bl = m_data.get("baseline", {})
        m_comp = m_data.get("comparison", {})
        m_lats = ag.get("latency_ms", {})
        ep_states = ag.get("epistemic_states", {})

        lines += [
            "-" * 85,
            f"  MODEL: {model_name}",
            "-" * 85,
            f"  Baseline Accuracy   : {bl.get('metrics', {}).get('accuracy', 0.0):.2f}%",
            f"  +AvicennaGuard Acc  : {ag.get('metrics', {}).get('accuracy', 0.0):.2f}%",
            f"  Accuracy Gain       : +{m_comp.get('accuracy_delta', 0.0):.2f}%",
            f"  Precision           : {ag.get('metrics', {}).get('precision', 0.0):.2f}% (Baseline: {bl.get('metrics', {}).get('precision', 0.0):.2f}%)",
            f"  Recall              : {ag.get('metrics', {}).get('recall', 0.0):.2f}% (Baseline: {bl.get('metrics', {}).get('recall', 0.0):.2f}%)",
            f"  F1-Score            : {ag.get('metrics', {}).get('f1', 0.0):.2f}% (Baseline: {bl.get('metrics', {}).get('f1', 0.0):.2f}%)",
            f"  Specificity         : {ag.get('metrics', {}).get('specificity', 0.0):.2f}%",
            f"  False Positive Rate : {ag.get('metrics', {}).get('fpr', 0.0):.2f}%",
            "",
            "  Epistemic State Distribution:",
        ]
        for st, count in ep_states.items():
            lines.append(f"    - {st:<10}: {count:4d}")

        lines += [
            "",
            "  Latency Profile (Mean / Median / P95):",
            f"    - Stage 1 Parser   : {m_lats.get('stage1', {}).get('mean', 0.0):.2f}ms / {m_lats.get('stage1', {}).get('median', 0.0):.2f}ms / {m_lats.get('stage1', {}).get('p95', 0.0):.2f}ms",
            f"    - Stage 2 BFS      : {m_lats.get('stage2', {}).get('mean', 0.0):.2f}ms / {m_lats.get('stage2', {}).get('median', 0.0):.2f}ms / {m_lats.get('stage2', {}).get('p95', 0.0):.2f}ms",
            f"    - Total Overhead   : {m_lats.get('total_overhead', {}).get('mean', 0.0):.2f}ms / {m_lats.get('total_overhead', {}).get('median', 0.0):.2f}ms / {m_lats.get('total_overhead', {}).get('p95', 0.0):.2f}ms",
            "",
        ]

    lines += [
        "=" * 85,
        "  END OF BENCHMARK EVALUATION REPORT",
        "=" * 85,
    ]
    return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AvicennaGuard Multi-Model Benchmark Runner & Evaluation Orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark JSON dataset (default: auto-resolved avicenna_benchmark_500.json)",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default=None,
        help="Path to KnowledgeBase JSON file (default: auto-resolved knowledge_base_extended.json)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names/tags to evaluate, or 'all' for standard 5-model suite",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of queries to evaluate per model",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode for fast offline testing and CI without Ollama server",
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["llm", "regex", "both"],
        default="llm",
        help="Stage 1 parsing strategy",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="Filter queries by dataset source (FOLIO, ProofWriter, Curated_Gold, TruthfulQA_OOD, all)",
    )
    parser.add_argument(
        "--query_type",
        type=str,
        default="all",
        help="Filter queries by logical type (taxonomic, categorical, hypothetical, ood, all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/benchmark_eval_results.json",
        help="File path to save JSON evaluation results",
    )
    parser.add_argument(
        "--report_txt",
        type=str,
        default="data/results/benchmark_eval_report.txt",
        help="File path to save human-readable TXT evaluation report",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between LLM calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("\n" + "=" * 80)
    print("  AvicennaGuard -- Multi-Model Benchmark Runner")
    print("=" * 80)

    # Determine models
    if args.models.strip().lower() == "all":
        target_models = DEFAULT_MODELS
    else:
        target_models = [MODEL_ALIASES.get(m.strip(), m.strip()) for m in args.models.split(",") if m.strip()]

    print(f"  Target Models : {', '.join(target_models)}")
    print(f"  Mock Mode     : {args.mock}")
    print(f"  Query Limit   : {args.limit if args.limit is not None else 'Full Dataset (500)'}")
    print(f"  Filter Source : {args.source}")
    print(f"  Filter Type   : {args.query_type}")
    print("=" * 80 + "\n")

    t_start = time.perf_counter()

    runner = BenchmarkRunner(
        kb=args.kb,
        benchmark_path=args.benchmark,
        models=target_models,
        mock_mode=args.mock,
        parser_mode=args.parser,
        delay=args.delay,
        seed=args.seed,
    )

    results = runner.run_all(
        limit=args.limit,
        models=target_models,
        filter_source=args.source if args.source != "all" else None,
        filter_type=args.query_type if args.query_type != "all" else None,
    )

    total_time = time.perf_counter() - t_start

    # Save JSON results
    out_json = Path(args.output)
    runner.save_results(results, out_json)

    # Generate and save TXT report
    report_text = format_detailed_report(results)
    out_txt = Path(args.report_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  [OK] JSON Results Saved : {out_json.resolve()}")
    print(f"  [OK] TXT Report Saved   : {out_txt.resolve()}")
    print(f"  [OK] Total Elapsed Time : {total_time:.2f}s\n")


if __name__ == "__main__":
    main()
