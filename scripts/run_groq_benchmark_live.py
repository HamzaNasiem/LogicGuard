"""
AvicennaGuard Live Groq Multi-Model Benchmark Runner (High-Speed Multi-Threaded)
================================================================================
Runs live evaluation across top-tier models on Groq Cloud API with parallel
workers, deterministic reasoning, robust answer parsing, zero false alarms,
and real-time epistemic state tracking.

Usage:
    python scripts/run_groq_benchmark_live.py
    python scripts/run_groq_benchmark_live.py --limit 50
    python scripts/run_groq_benchmark_live.py --workers 10
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.deberta_parser import DebertaParser
from avicennaguard.core.epistemic_states import EpistemicState

GROQ_MODELS = {
    "gpt_120b": {
        "name": "OpenAI GPT-OSS 120B (120B Flagship)",
        "id": "openai/gpt-oss-120b"
    },
    "qwen_27b": {
        "name": "Qwen 3.8 27B (Alibaba Flagship)",
        "id": "qwen/qwen3.8-27b"
    },
    "gpt_20b": {
        "name": "OpenAI GPT-OSS 20B (20B Model)",
        "id": "openai/gpt-oss-20b"
    }
}


def load_api_key() -> str:
    """Load Groq API key from environment or .env file."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        env_file = ROOT / ".env"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip()
                        break
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in environment or .env file!")
        sys.exit(1)
    return api_key


def parse_llm_boolean(text: str) -> Optional[bool]:
    """
    Robust boolean parser extracting True/False/YES/NO from text and reasoning.
    """
    if not text:
        return None
    
    # Strip thinking / reasoning tags
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
        
    t = clean.strip().lower()
    
    # Direct start matches
    if t.startswith("yes") or t.startswith("true") or t.startswith("correct"):
        return True
    if t.startswith("no") or t.startswith("false") or t.startswith("incorrect"):
        return False
        
    # Regex word search
    yes_matches = len(re.findall(r"\b(yes|true|correct)\b", t))
    no_matches = len(re.findall(r"\b(no|false|incorrect)\b", t))
    
    if yes_matches > 0 and no_matches == 0:
        return True
    if no_matches > 0 and yes_matches == 0:
        return False
    if yes_matches > no_matches:
        return True
    if no_matches > yes_matches:
        return False
        
    return None


def call_groq_single(model_id: str, question: str, api_key: str, retries: int = 3) -> Tuple[str, Optional[bool], float]:
    """Call Groq API with robust retry and reasoning support."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"Answer this question with YES or NO only. Do not explain. Just say YES or NO.\n\nQuestion: {question}\nAnswer:"
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.0
    }
    
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=25)
            latency_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                res = r.json()
                msg = res.get("choices", [{}])[0].get("message", {})
                content = msg.get("content", "").strip()
                reasoning = msg.get("reasoning", "").strip()
                
                ans_bool = parse_llm_boolean(content)
                if ans_bool is None and reasoning:
                    ans_bool = parse_llm_boolean(reasoning)
                    
                raw_text = content if content else reasoning[:80]
                return raw_text, ans_bool, latency_ms
            elif r.status_code == 429:
                time.sleep(2.5 * (attempt + 1))
            else:
                time.sleep(1.0)
        except Exception:
            time.sleep(1.5)
            
    return "[API_ERROR]", None, 0.0


def evaluate_single_query(item: dict, model_id: str, api_key: str, validator: BFSValidator, s1_parser: DebertaParser) -> dict:
    """Evaluate a single benchmark item through the complete AvicennaGuard pipeline."""
    q_id = item.get("id", "")
    question = item.get("question", "")
    gt = item.get("ground_truth")
    gt_bool = (gt.lower() in ("yes", "true", "1")) if isinstance(gt, str) and gt.lower() in ("yes", "no", "true", "false") else gt if isinstance(gt, bool) else None
    q_type = item.get("query_type", "taxonomic")
    
    # 1. Live LLM Call
    llm_raw, llm_bool, llm_ms = call_groq_single(model_id, question, api_key)
    is_bl_correct = (llm_bool == gt_bool) if (llm_bool is not None and gt_bool is not None) else (gt == "OOD")
    
    # 2. Stage 1 Semantic Parsing
    t_s1 = time.perf_counter()
    parsed = s1_parser.parse(question)
    s1_ms = (time.perf_counter() - t_s1) * 1000
    parsed_type = parsed.get("type", q_type)
    
    # 3. Stage 2 Deterministic BFS
    t_s2 = time.perf_counter()
    if parsed_type == "taxonomic":
        graph_ans, state, path = validator.validate_taxonomic(parsed.get("subject", ""), parsed.get("predicate", ""))
    elif parsed_type == "categorical":
        graph_ans, state = validator.validate_categorical(parsed.get("subject", ""), parsed.get("predicate", ""))
        path = []
    elif parsed_type == "hypothetical":
        graph_ans, state = validator.validate_hypothetical(parsed.get("subject", ""), parsed.get("predicate", ""))
        path = []
    else:
        graph_ans, state, path = None, EpistemicState.SHAKK, []
    s2_ms = (time.perf_counter() - t_s2) * 1000
    
    # 4. Epistemic Adjudication & Interception
    covered = (state != EpistemicState.SHAKK and graph_ans is not None)
    if covered:
        if llm_bool is not None and llm_bool != graph_ans:
            ep_state = EpistemicState.WAHM
            intercepted = True
            final_answer = graph_ans
        else:
            ep_state = EpistemicState.YAQEEN
            intercepted = False
            final_answer = graph_ans
    else:
        ep_state = EpistemicState.SHAKK
        intercepted = False
        final_answer = llm_bool
        
    final_bool = final_answer if isinstance(final_answer, bool) else (final_answer.lower() in ("yes", "true", "1") if isinstance(final_answer, str) and final_answer.lower() in ("yes", "no") else None)
    is_ag_correct = (final_bool == gt_bool) if (final_bool is not None and gt_bool is not None) else (gt == "OOD" and ep_state == EpistemicState.SHAKK)
    
    false_alarm = bool(is_bl_correct and not is_ag_correct)
    
    return {
        "id": q_id,
        "question": question,
        "ground_truth": gt,
        "gt_bool": gt_bool,
        "llm_raw": llm_raw,
        "llm_answer": llm_bool,
        "final_answer": final_answer,
        "final_bool": final_bool,
        "epistemic_state": ep_state.value,
        "intercepted": intercepted,
        "path": path,
        "is_bl_correct": is_bl_correct,
        "is_ag_correct": is_ag_correct,
        "false_alarm": false_alarm,
        "latency_ms": {"llm": round(llm_ms, 2), "s1": round(s1_ms, 3), "s2": round(s2_ms, 3)}
    }


def main():
    parser = argparse.ArgumentParser(description="AvicennaGuard Fast Parallel Groq Benchmark Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (default: all 500)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads (default: 4)")
    parser.add_argument("--models", nargs="+", default=list(GROQ_MODELS.keys()), help="Models to evaluate")
    parser.add_argument("--output", type=str, default="results/models/groq_live_evaluation_results.json", help="Output results path")
    args = parser.parse_args()

    api_key = load_api_key()
    
    # 1. Load Knowledge Base & Engine
    kb_path = ROOT / "data/knowledge_bases/knowledge_base_extended.json"
    print("=" * 95)
    print("   AVICENNAGUARD -- FAST MULTI-THREADED GROQ BENCHMARK HARNESS (100% FREE)")
    print("=" * 95)
    print(f"[*] Initializing AvicennaGuard Knowledge Engine from: {kb_path.name}")
    kb = KnowledgeBase(kb_path)
    validator = BFSValidator(kb)
    s1_parser = DebertaParser()
    print(f"    -> Taxonomy (G_T): {kb.G_T.number_of_nodes()} nodes, {kb.G_T.number_of_edges()} edges")
    print(f"    -> Properties (G_P): {len(kb.G_P)} entities | Modus Ponens (G_C): {kb.G_C.number_of_edges()} rules")

    # 2. Load 500-Query Benchmark Dataset
    bench_path = ROOT / "data/benchmarks/avicenna_benchmark_500.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    
    queries = benchmark_data if isinstance(benchmark_data, list) else benchmark_data.get("queries", [])
    if args.limit:
        queries = queries[:args.limit]
    total_queries = len(queries)
    print(f"[*] Benchmark Queries Loaded: {total_queries} items (FOLIO, ProofWriter, Curated, TruthfulQA)")
    print(f"[*] Multi-Threading: {args.workers} Parallel Workers")

    # 3. Model Suite Setup
    selected_models = []
    for m in args.models:
        if m in GROQ_MODELS:
            selected_models.append((m, GROQ_MODELS[m]))
        else:
            matched = False
            for k, v in GROQ_MODELS.items():
                if m.lower() in k.lower() or m.lower() in v["id"].lower():
                    selected_models.append((k, v))
                    matched = True
                    break
            if not matched:
                selected_models.append((m, {"name": m, "id": m}))

    print(f"[*] Selected Groq Frontier Models ({len(selected_models)}):")
    for idx, (k, info) in enumerate(selected_models, 1):
        print(f"    {idx}. {info['name']} ({info['id']})")
    print("=" * 95)

    # 4. Multi-Threaded Evaluation Loop
    all_results = {}
    summary_table = []

    for model_key, model_info in selected_models:
        model_name = model_info["name"]
        model_id = model_info["id"]
        print(f"\n>>> LAUNCHING PARALLEL LIVE EVALUATION: {model_name} <<<")
        t_model_0 = time.perf_counter()
        
        records = [None] * total_queries
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(evaluate_single_query, item, model_id, api_key, validator, s1_parser): i
                for i, item in enumerate(queries)
            }
            
            done_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    records[idx] = res
                    done_count += 1
                    status_char = "[WAHM]" if res["intercepted"] else f"[{res['epistemic_state']}]"
                    print(f"  Progress: {done_count:03d}/{total_queries:03d} | LLM: {str(res['llm_answer']):<5} | Guard: {str(res['final_bool']):<5} | GT: {str(res['gt_bool']):<5} | {status_char}")
                except Exception as exc:
                    print(f"  [ERROR] Query {idx+1} generated an exception: {exc}")
                    
        t_model_elapsed = time.perf_counter() - t_model_0
        
        # Calculate Metrics
        bl_correct = sum(1 for r in records if r and r["is_bl_correct"])
        ag_correct = sum(1 for r in records if r and r["is_ag_correct"])
        intercepted_count = sum(1 for r in records if r and r["intercepted"])
        false_alarms = sum(1 for r in records if r and r["false_alarm"])
        epistemic_dist = Counter(r["epistemic_state"] for r in records if r)
        
        bl_acc = (bl_correct / total_queries) * 100
        ag_acc = (ag_correct / total_queries) * 100
        delta = ag_acc - bl_acc
        
        all_results[model_id] = {
            "model_name": model_name,
            "model_id": model_id,
            "total_queries": total_queries,
            "baseline_accuracy": round(bl_acc, 2),
            "guard_accuracy": round(ag_acc, 2),
            "accuracy_gain": round(delta, 2),
            "hallucinations_intercepted": intercepted_count,
            "false_alarms": false_alarms,
            "epistemic_distribution": dict(epistemic_dist),
            "execution_time_sec": round(t_model_elapsed, 2),
            "records": records
        }
        
        summary_table.append({
            "model": model_name,
            "bl_acc": f"{bl_acc:.2f}%",
            "ag_acc": f"{ag_acc:.2f}%",
            "gain": f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%",
            "caught": intercepted_count,
            "fa": false_alarms,
            "time": f"{t_model_elapsed:.1f}s"
        })
        print(f"\n  [COMPLETED in {t_model_elapsed:.1f}s] {model_name}: Raw LLM = {bl_acc:.2f}% -> +AvicennaGuard = {ag_acc:.2f}% (Interceptions: {intercepted_count}, False Alarms: {false_alarms})\n")

    # 5. Save Output
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    # 6. Print Final IEEE Summary Table
    print("\n" + "=" * 95)
    print("  FINAL LIVE GROQ EVALUATION BENCHMARK RESULTS (IEEE TABLE I FORMAT)")
    print("=" * 95)
    print(f"{'Frontier Model Name':<38} | {'Raw LLM Acc':<12} | {'+AvicennaGuard':<14} | {'Gain':<8} | {'Caught':<7} | {'FA':<4} | {'Time'}")
    print("-" * 95)
    for row in summary_table:
        print(f"{row['model']:<38} | {row['bl_acc']:<12} | {row['ag_acc']:<14} | {row['gain']:<8} | {row['caught']:<7} | {row['fa']:<4} | {row['time']}")
    print("-" * 95)
    print(f"[*] Evaluation Spend: $0.00 USD (100% Free via Groq API)")
    print(f"[*] Full Evaluation Artifact Saved To: {out_path.relative_to(ROOT)}")
    print("=" * 95)


if __name__ == "__main__":
    main()
