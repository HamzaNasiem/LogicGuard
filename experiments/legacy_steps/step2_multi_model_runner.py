"""
STEP 2: Multi-Model Runner  (v2 — IEEE journal grade)
=======================================================
Runs ALL THREE models (Llama2 7B, Mistral 7B, Llama3.2 3B)
+ AvicennaGuard on top of each.

Usage:
    python step2_multi_model_runner.py --filter_source original
    python step2_multi_model_runner.py --parser llm --kb 1200
"""

import json
import time
import sys
import os
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from avicennaguard.research.adapter import ResearchValidator as AvicennaGuardValidator  # noqa: E402
LogicGuardValidator = AvicennaGuardValidator  # Backwards compatibility alias

try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    sys.exit(1)


def resolve_kb_path(kb_arg: str) -> str:
    aliases = {
        "extended": "data/knowledge_bases/knowledge_base_extended.json",
        "1200": "knowledge_base_1200.json",
    }
    path = aliases.get(kb_arg, kb_arg)
    root = os.path.dirname(__file__)
    for candidate in (
        path,
        kb_arg,
        os.path.join(root, path),
        os.path.join(root, "knowledge_base_1200.json"),
    ):
        if os.path.exists(candidate):
            return candidate
    return path


MODELS = {
    "llama2_7b": "llama2",
    "mistral_7b": "mistral",
    "llama32_3b": "llama3.2:3b",
}


def get_llm_answer(question: str, model: str, retries: int = 3) -> Tuple[str, float]:
    prompt = (
        "Answer this question with YES or NO only. "
        "Do not explain. Just say YES or NO.\n\n"
        f"Question: {question}\nAnswer:"
    )
    t0 = time.perf_counter()
    for attempt in range(retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "seed": 42, "num_predict": 10},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = resp["message"]["content"].strip().lower()
            if raw.startswith("yes"):
                return "yes", latency_ms
            if raw.startswith("no"):
                return "no", latency_ms
            if "yes" in raw[:20]:
                return "yes", latency_ms
            if "no" in raw[:20]:
                return "no", latency_ms
            return raw[:50], latency_ms
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return "[llm_error]", (time.perf_counter() - t0) * 1000


def parse_llm_yn(raw_answer: str) -> Optional[bool]:
    if not raw_answer or raw_answer == "[llm_error]":
        return None
    t = raw_answer.lower().strip()
    if t.startswith("yes") or t in ("true", "1", "correct", "right"):
        return True
    if t.startswith("no") or t in ("false", "0", "incorrect", "wrong"):
        return False
    neg = ["not ", "isn't", "aren't", "doesn't", "don't", "cannot", "never", "false", "wrong", "incorrect", "no,"]
    pos = ["yes,", "yes.", "correct", "true", "indeed", "all are", "always", "absolutely", "certainly"]
    has_neg = any(w in t for w in neg)
    has_pos = any(w in t for w in pos)
    if has_pos and not has_neg:
        return True
    if has_neg and not has_pos:
        return False
    return None


def evaluate_model(
    model_key: str,
    model_name: str,
    queries: List[Dict],
    validator: AvicennaGuardValidator,
    use_logicguard: bool = False,
    delay: float = 0.3,
) -> List[Dict]:
    results = []
    n = len(queries)
    correct = 0

    print(f"\n{'─' * 65}")
    print(f"  Model: {model_name}{'  [+ AvicennaGuard]' if use_logicguard else ''}")
    print(f"  Queries: {n}")
    print(f"{'─' * 65}")

    for i, query in enumerate(queries):
        q = query["question"]
        qtype = query["type"]
        gt = query["ground_truth"]
        source = query.get("source", "unknown")

        graph_result = validator.validate(q, qtype)
        graph_answer = graph_result["graph_answer"]
        covered = graph_result["covered"]
        stage1_ms = graph_result.get("stage1_ms", 0.0)
        stage2_ms = graph_result.get("stage2_ms", 0.0)
        parse_status = graph_result.get("parse_status", "unknown")

        llm_raw, llm_ms = get_llm_answer(q, model_name)
        llm_parsed = parse_llm_yn(llm_raw)

        if use_logicguard and covered and graph_answer is not None:
            final_answer = graph_answer
            method = "avicennaguard_override" if llm_parsed != graph_answer else "avicennaguard_agree"
            hallucination_caught = (llm_parsed != graph_answer) and (graph_answer == gt)
        else:
            final_answer = llm_parsed
            method = "llm_only"
            hallucination_caught = False

        is_correct = (final_answer == gt) if final_answer is not None else False
        if is_correct:
            correct += 1

        results.append(
            {
                "question": q,
                "type": qtype,
                "source": source,
                "ground_truth": gt,
                "llm_raw": llm_raw,
                "llm_parsed": llm_parsed,
                "graph_answer": graph_answer,
                "graph_covered": covered,
                "final_answer": final_answer,
                "is_correct": is_correct,
                "method": method,
                "hallucination_caught": hallucination_caught,
                "epistemic_state": graph_result["epistemic_state"],
                "proof": graph_result["proof"],
                "parse_status": parse_status,
                "parser_mode": graph_result.get("parser_mode", validator.parser_mode),
                "latency": {
                    "llm_ms": round(llm_ms, 2),
                    "stage1_ms": stage1_ms,
                    "stage2_ms": stage2_ms,
                    "total_overhead_ms": round(stage1_ms + stage2_ms, 2),
                },
                "model": model_key,
                "avicennaguard": use_logicguard,
                "logicguard": use_logicguard,
            }
        )

        icon = "✓" if is_correct else "✗"
        ep = graph_result["epistemic_state"]
        print(f"  [{i+1:03}/{n}] {icon} {ep:7} | s1={stage1_ms:5.2f}ms s2={stage2_ms:5.2f}ms | {q[:45]}")
        time.sleep(delay)

    acc = correct / n * 100 if n > 0 else 0
    print(f"\n  ► Accuracy: {acc:.1f}%  ({correct}/{n})")
    return results


def compute_hallucination_rate(results: List[Dict], use_logicguard: bool) -> Tuple[int, int]:
    if not use_logicguard:
        return 0, 0
    caught = sum(1 for r in results if r.get("hallucination_caught", False))
    llm_errors = sum(
        1
        for r in results
        if r["graph_covered"] and r["llm_parsed"] is not None and r["llm_parsed"] != r["ground_truth"]
    )
    return caught, llm_errors


def compute_summary(results: List[Dict], model_key: str, use_logicguard: bool) -> Dict:
    total = len(results)
    if total == 0:
        return {}

    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_type[r["type"]]["total"] += 1
        if r["is_correct"]:
            by_type[r["type"]]["correct"] += 1

    overall_correct = sum(1 for r in results if r["is_correct"])
    caught, llm_err = compute_hallucination_rate(results, use_logicguard)
    per_query_correct = [r["is_correct"] for r in results]

    llm_latencies = [r["latency"]["llm_ms"] for r in results if r["latency"]["llm_ms"] > 0]
    stage1_latencies = [r["latency"]["stage1_ms"] for r in results]
    stage2_latencies = [r["latency"]["stage2_ms"] for r in results]
    overhead_latencies = [r["latency"]["total_overhead_ms"] for r in results]

    def stats(vals):
        if not vals:
            return {"mean": 0, "median": 0, "p95": 0}
        vals_s = sorted(vals)
        n_v = len(vals_s)
        return {
            "mean": round(sum(vals_s) / n_v, 2),
            "median": round(vals_s[n_v // 2], 2),
            "p95": round(vals_s[int(n_v * 0.95)], 2),
        }

    parse_status_counts = defaultdict(int)
    for r in results:
        parse_status_counts[r.get("parse_status", "unknown")] += 1

    return {
        "model": model_key,
        "avicennaguard": use_logicguard,
        "logicguard": use_logicguard,
        "total": total,
        "correct": overall_correct,
        "accuracy": round(overall_correct / total * 100, 1),
        "by_type": {
            t: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
            }
            for t, v in by_type.items()
        },
        "hallucinations_caught": caught,
        "llm_errors_on_logical": llm_err,
        "per_query_correct": per_query_correct,
        "parse_stats": dict(parse_status_counts),
        "latency_ms": {
            "llm": stats(llm_latencies),
            "stage1": stats(stage1_latencies),
            "stage2": stats(stage2_latencies),
            "overhead": stats(overhead_latencies),
        },
    }


def check_model_available(model_name: str) -> bool:
    try:
        models = ollama.list()
        available = [m["name"].split(":")[0] for m in models.get("models", [])]
        base = model_name.split(":")[0]
        return base in available or model_name in [m["name"] for m in models.get("models", [])]
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="extended_queries.json")
    parser.add_argument("--kb", type=str, default="1200", help="KB alias: extended | 1200 | path")
    parser.add_argument("--parser", type=str, default="regex", choices=["regex", "llm", "both"])
    parser.add_argument("--output", type=str, default="all_model_results.json")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--filter_source", type=str, default="original", choices=["original", "all"])
    args = parser.parse_args()

    print("=" * 65)
    print("  AvicennaGuard — Step 2: Multi-Model Evaluation  (v2)")
    print("=" * 65)

    with open(args.queries, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_queries = data["queries"]
    queries = (
        [q for q in all_queries if q.get("source", "original") == "original"]
        if args.filter_source == "original"
        else all_queries
    )

    kb_path = resolve_kb_path(args.kb)
    if not os.path.exists(kb_path):
        print(f"ERROR: KB not found: {kb_path}")
        sys.exit(1)

    validator = AvicennaGuardValidator(
        kb_path, parser_mode=args.parser, model=MODELS["llama32_3b"]
    )
    print(f"  Queries: {len(queries)} | KB: {kb_path} | Parser: {args.parser}")

    models_to_run = {}
    requested = list(MODELS.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    for key in requested:
        if key not in MODELS:
            continue
        if check_model_available(MODELS[key]):
            models_to_run[key] = MODELS[key]

    if not models_to_run:
        print("ERROR: No Ollama models available.")
        sys.exit(1)

    all_results = {}
    all_summaries = {}

    for model_key, model_name in models_to_run.items():
        validator._parse_stats = {"success": 0, "regex_fallback": 0, "parse_failure": 0}

        results_baseline = evaluate_model(
            f"{model_key}_baseline", model_name, queries, validator, False, args.delay
        )
        results_ag = evaluate_model(
            f"{model_key}_avicennaguard", model_name, queries, validator, True, args.delay
        )

        all_results[f"{model_key}_baseline"] = results_baseline
        all_results[f"{model_key}_avicennaguard"] = results_ag
        all_results[f"{model_key}_logicguard"] = results_ag  # Backward-compatibility alias
        all_summaries[f"{model_key}_baseline"] = compute_summary(results_baseline, f"{model_key}_baseline", False)
        all_summaries[f"{model_key}_avicennaguard"] = compute_summary(results_ag, f"{model_key}_avicennaguard", True)
        all_summaries[f"{model_key}_logicguard"] = compute_summary(results_ag, f"{model_key}_logicguard", True)

    combined = {
        "metadata": {
            **data.get("metadata", {}),
            "source_filter": args.filter_source,
            "parser_mode": args.parser,
            "kb_path": kb_path,
            "n_queries_used": len(queries),
        },
        "summaries": all_summaries,
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
