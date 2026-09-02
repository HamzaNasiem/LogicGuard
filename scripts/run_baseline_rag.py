#!/usr/bin/env python3
"""
Standalone Runner for Dense RAG Baseline.
=========================================
Evaluates Dense RAG (Lewis et al., NeurIPS 2020) retriever-generator baseline
against the 1,500-node Knowledge Base on the AvicennaGuard benchmark dataset.

Usage:
    python scripts/run_baseline_rag.py --mock
    python scripts/run_baseline_rag.py --mock --subset 50
    python scripts/run_baseline_rag.py --benchmark data/benchmarks/avicenna_benchmark_500.json --output results/baselines/rag_results.json --mock
    python scripts/run_baseline_rag.py --model llama3.2:3b --top_k 5
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

from avicennaguard.baselines.dense_rag import DenseRAGBaseline
from avicennaguard.data.benchmark_loader import BenchmarkLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Dense RAG Knowledge Retrieval Baseline on AvicennaGuard Benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/benchmarks/avicenna_benchmark_500.json",
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--kb",
        type=str,
        default="data/knowledge_bases/knowledge_base_extended.json",
        help="Path to knowledge base JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baselines/rag_results.json",
        help="Path to save evaluation results JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="LLM generation model identifier for Ollama.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved KB facts per query.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=False,
        help="Force TF-IDF sparse retriever instead of dense embeddings.",
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
    kb_path = REPO_ROOT / args.kb if not Path(args.kb).is_absolute() else Path(args.kb)
    output_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    print(f"\n========================================================")
    print(f"  AvicennaGuard Baseline: Dense RAG (NeurIPS 2020)")
    print(f"========================================================")
    print(f"  Benchmark : {benchmark_path}")
    print(f"  KB Path   : {kb_path}")
    print(f"  Output    : {output_path}")
    print(f"  Model     : {args.model}")
    print(f"  Top-K     : {args.top_k}")
    print(f"  Dense     : {'SPARSE TF-IDF' if args.sparse else args.embedding_model}")
    print(f"  Mode      : {'MOCK / OFFLINE' if args.mock else 'OLLAMA'}")
    if args.subset:
        print(f"  Subset    : {args.subset} queries")
    print(f"========================================================\n")

    loader = BenchmarkLoader(benchmark_path)
    queries = loader.get_all_queries()
    if args.subset:
        queries = queries[:args.subset]

    baseline = DenseRAGBaseline(
        kb_path=kb_path,
        model=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        use_dense=not args.sparse,
        mock=args.mock,
    )

    t0 = time.perf_counter()

    def progress(cur, total):
        if not args.quiet and (cur % 25 == 0 or cur == total):
            pct = (cur / total) * 100
            print(f"  [Progress] {cur:4d}/{total:4d} ({pct:5.1f}%) queries evaluated...")

    results = baseline.evaluate_dataset(queries, top_k=args.top_k, progress_callback=progress)
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
