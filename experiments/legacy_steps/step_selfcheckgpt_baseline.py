#!/usr/bin/env python3
"""
AvicennaGuard — SelfCheckGPT Baseline Comparison
==============================================
Implements SelfCheckGPT (Manakul et al., EMNLP 2023) on the same dataset
as AvicennaGuard for direct comparison.

SelfCheckGPT strategy:
  1. Sample N responses from the LLM for the same question (T > 0.0)
  2. Check consistency: if all N samples agree → confident
  3. If samples disagree → uncertain (likely hallucination)
  4. Vote: majority answer is the final answer

This is a SAMPLING-BASED approach with no KB, no formal reasoning.
It relies on variance in LLM responses to detect hallucinations.

Comparison points:
  - SelfCheckGPT: probabilistic, depends on LLM calibration
  - AvicennaGuard: deterministic, FP=0 by construction

Reference:
  Manakul P, Liusie A, Gales M. SelfCheckGPT: Zero-Resource Black-Box
  Hallucination Detection for Generative Large Language Models. EMNLP 2023.
  https://arxiv.org/abs/2303.08896

Usage:
    python step_selfcheckgpt_baseline.py --model llama3.2:3b --n_samples 5
    python step_selfcheckgpt_baseline.py --model llama3.2:3b --n_samples 5 --subset 100
"""

import json
import time
import argparse
import sys
import re
from collections import Counter
from pathlib import Path

import ollama

# ── Helpers ──────────────────────────────────────────────────────────────────

def call_llm(model: str, question: str, temperature: float = 0.7) -> str:
    """Single LLM call. Temperature > 0 for sampling diversity."""
    system = (
        "You are a factual assistant. Answer the question with only YES or NO. "
        "Do not explain."
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": question},
            ],
            options={"temperature": temperature, "num_predict": 10},
        )
        text = resp["message"]["content"].strip().upper()
        # Extract YES/NO from response
        if re.search(r"\bYES\b", text):
            return "yes"
        if re.search(r"\bNO\b", text):
            return "no"
        return "yes" if "TRUE" in text else "no"
    except Exception as e:
        return "error"


def selfcheck_predict(model: str, question: str, n_samples: int = 5) -> tuple[str, float, list[str]]:
    """
    SelfCheckGPT: sample N responses and vote.

    Returns:
        (final_answer, confidence, all_samples)
        confidence = fraction of samples agreeing with majority vote
    """
    samples = []
    for _ in range(n_samples):
        ans = call_llm(model, question, temperature=0.7)
        samples.append(ans)
        time.sleep(0.1)

    # Majority vote
    counts = Counter(s for s in samples if s != "error")
    if not counts:
        return "no", 0.0, samples

    majority = counts.most_common(1)[0][0]
    confidence = counts[majority] / len([s for s in samples if s != "error"])
    return majority, confidence, samples


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: str, queries: list[dict], n_samples: int, max_queries: int) -> dict:
    """Run SelfCheckGPT on all queries and compute accuracy."""
    subset = queries[:max_queries]
    total  = len(subset)
    correct = 0
    tp = fp = tn = fn = 0
    high_conf_correct = high_conf_wrong = 0
    results = []

    print(f"\n  Running SelfCheckGPT (N={n_samples} samples per query)...")
    print(f"  Model: {model} | Queries: {total}\n")

    for i, q in enumerate(subset, 1):
        sys.stdout.write(f"\r  [{i:4d}/{total}] {q['question'][:55]:<55}")
        sys.stdout.flush()

        t0 = time.perf_counter()
        pred_ans, confidence, samples = selfcheck_predict(model, q["question"], n_samples)
        latency = (time.perf_counter() - t0) * 1000

        gt = "yes" if q.get("ground_truth", True) else "no"
        is_correct = (pred_ans == gt)
        if is_correct:
            correct += 1

        # Confusion matrix (treating "yes" as positive)
        if gt == "yes" and pred_ans == "yes":   tp += 1
        elif gt == "no"  and pred_ans == "yes":  fp += 1
        elif gt == "yes" and pred_ans == "no":   fn += 1
        elif gt == "no"  and pred_ans == "no":   tn += 1

        # High-confidence accuracy
        if confidence >= 0.8:
            if is_correct: high_conf_correct += 1
            else:          high_conf_wrong += 1

        results.append({
            "question":   q["question"],
            "ground_truth": gt,
            "prediction": pred_ans,
            "confidence": round(confidence, 3),
            "samples":    samples,
            "correct":    is_correct,
            "latency_ms": round(latency, 1),
        })

    accuracy  = correct / total
    precision = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp)  if (tn + fp) > 0 else 0.0

    high_conf_acc = high_conf_correct / (high_conf_correct + high_conf_wrong) \
                    if (high_conf_correct + high_conf_wrong) > 0 else 0.0
    high_conf_n   = high_conf_correct + high_conf_wrong

    return {
        "model":       model,
        "n_samples":   n_samples,
        "total":       total,
        "correct":     correct,
        "accuracy":    round(accuracy,   4),
        "precision":   round(precision,  4),
        "recall":      round(recall,     4),
        "f1":          round(f1,         4),
        "specificity": round(specificity, 4),
        "false_positives": fp,
        "false_negatives": fn,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "high_confidence": {
            "n":        high_conf_n,
            "accuracy": round(high_conf_acc, 4),
        },
        "results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SelfCheckGPT Baseline for AvicennaGuard")
    parser.add_argument("--model",      default="llama3.2:3b",   help="Ollama model")
    parser.add_argument("--queries",    default="queries_1200.json")
    parser.add_argument("--n_samples",  type=int, default=5,      help="Samples per query")
    parser.add_argument("--subset",     type=int, default=200,    help="Max queries to evaluate (0=all)")
    parser.add_argument("--output",     default="selfcheck_results.json")
    parser.add_argument("--query_type", default="all",
                        choices=["all", "taxonomic", "categorical", "hypothetical"],
                        help="Filter by query type")
    args = parser.parse_args()

    print("=" * 65)
    print("  SelfCheckGPT Baseline Evaluation")
    print("=" * 65)

    # Load queries
    with open(args.queries, encoding="utf-8") as f:
        dataset = json.load(f)

    queries = dataset.get("kb_queries", [])
    if args.query_type != "all":
        queries = [q for q in queries if q.get("type") == args.query_type]

    max_q = args.subset if args.subset > 0 else len(queries)
    print(f"\n  Queries file : {args.queries}")
    print(f"  Query type   : {args.query_type}")
    print(f"  Total loaded : {len(queries)}")
    print(f"  Evaluating   : {min(max_q, len(queries))}")
    print(f"  N samples    : {args.n_samples}")
    print(f"  Model        : {args.model}")
    print(f"\n  Estimated time: {min(max_q, len(queries)) * args.n_samples * 0.5 / 60:.1f} minutes")

    # Shuffle with seed for reproducibility
    import random; random.seed(42)
    random.shuffle(queries)

    # Run evaluation
    t0 = time.time()
    results = evaluate(args.model, queries, args.n_samples, max_q)
    elapsed = time.time() - t0

    # Print report
    print(f"\n\n{'=' * 65}")
    print(f"  SELFCHECKGPT RESULTS  (elapsed: {elapsed:.0f}s)")
    print(f"{'=' * 65}")
    print(f"  Model     : {args.model}")
    print(f"  N samples : {args.n_samples}")
    print(f"  Queries   : {results['total']}")
    print(f"  Accuracy  : {results['accuracy']:.1%}")
    print(f"  Precision : {results['precision']:.1%}  (FP={results['false_positives']})")
    print(f"  Recall    : {results['recall']:.1%}    (FN={results['false_negatives']})")
    print(f"  F1-Score  : {results['f1']:.1%}")
    print(f"  Specificity: {results['specificity']:.1%}")
    print(f"  High-conf accuracy: {results['high_confidence']['accuracy']:.1%} (n={results['high_confidence']['n']})")
    print(f"\n  Confusion Matrix:")
    c = results['confusion']
    print(f"    TP={c['tp']}  FP={c['fp']}")
    print(f"    FN={c['fn']}  TN={c['tn']}")
    print(f"\n  Note: SelfCheckGPT FP={results['false_positives']} vs AvicennaGuard FP=0 (architectural guarantee)")

    results["elapsed_s"] = round(elapsed, 1)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
