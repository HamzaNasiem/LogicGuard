#!/usr/bin/env python3
"""
Standalone Runner for SelfCheckGPT Baseline.
============================================
Evaluates SelfCheckGPT (Manakul et al., EMNLP 2023) stochastic consistency
hallucination detection on the AvicennaGuard benchmark dataset.

Usage:
    python scripts/run_baseline_selfcheck.py --mock
    python scripts/run_baseline_selfcheck.py --mock --subset 50
    python scripts/run_baseline_selfcheck.py --benchmark data/benchmarks/avicenna_benchmark_500.json --output results/baselines/selfcheck_results.json --mock
    python scripts/run_baseline_selfcheck.py --model llama3.2:3b --n_samples 5
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from avicennaguard.baselines.selfcheckgpt import SelfCheckGPTBaseline
from avicennaguard.data.benchmark_loader import BenchmarkLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SelfCheckGPT Stochastic Consistency Baseline on AvicennaGuard Benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/benchmarks/avicenna_benchmark_500.json",
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baselines/selfcheck_results.json",
        help="Path to save evaluation results JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="LLM model identifier for Ollama.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of stochastic samples to generate per query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for LLM.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Confidence threshold below which to flag hallucination.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Limit evaluation to first N queries.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run in deterministic mock/offline mode without calling Ollama.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-query progress output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_path = REPO_ROOT / args.benchmark if not Path(args.benchmark).is_absolute() else Path(args.benchmark)
    output_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    print(f"\n========================================================")
    print(f"  AvicennaGuard Baseline: SelfCheckGPT (EMNLP 2023)")
    print(f"========================================================")
    print(f"  Benchmark : {benchmark_path}")
    print(f"  Output    : {output_path}")
    print(f"  Model     : {args.model}")
    print(f"  N Samples : {args.n_samples}")
    print(f"  Temp      : {args.temperature}")
    print(f"  Mode      : {'MOCK / OFFLINE' if args.mock else 'OLLAMA'}")
    if args.subset:
        print(f"  Subset    : {args.subset} queries")
    print(f"========================================================\n")

    loader = BenchmarkLoader(benchmark_path)
    queries = loader.get_all_queries()
    if args.subset:
        queries = queries[:args.subset]

    baseline = SelfCheckGPTBaseline(
        model=args.model,
        n_samples=args.n_samples,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        mock=args.mock,
    )

    t0 = time.perf_counter()

    def progress(cur, total):
        if not args.quiet and (cur % 25 == 0 or cur == total):
            pct = (cur / total) * 100
            print(f"  [Progress] {cur:4d}/{total:4d} ({pct:5.1f}%) queries evaluated...")

    results = baseline.evaluate_dataset(queries, progress_callback=progress)
    elapsed = time.perf_counter() - t0

    # Ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation completed in {elapsed:.2f}s.")
    print(f"Results saved to: {output_path}\n")
    print(results.get("summary_text", ""))


if __name__ == "__main__":
    main()
