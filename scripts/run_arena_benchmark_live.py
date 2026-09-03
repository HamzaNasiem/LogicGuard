"""
AvicennaGuard Live Multi-Model Benchmark Runner (Arena Leaderboard SOTA Suite)
==============================================================================
Runs live evaluation across 6 top-ranked models from LMSYS Arena Leaderboard
via OpenRouter API, recording real model outputs, Stage 1 parsing, Stage 2 BFS
graph adjudication, hallucination interceptions, and metrics.

Usage:
    python scripts/run_arena_benchmark_live.py
    python scripts/run_arena_benchmark_live.py --limit 50
    python scripts/run_arena_benchmark_live.py --models claude_sonnet_5 gpt_5_6_luna llama_3_3_70b
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter
import requests

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.deberta_parser import DebertaParser
from avicennaguard.core.epistemic_states import EpistemicState

# Arena Leaderboard Model Registry
ARENA_MODELS = {
    "claude_sonnet_5": {
        "name": "Claude Sonnet 5 (Anthropic #8)",
        "id": "anthropic/claude-sonnet-5"
    },
    "grok_4_5": {
        "name": "Grok 4.5 (xAI #12)",
        "id": "x-ai/grok-4.5"
    },
    "gpt_5_6_luna": {
        "name": "GPT 5.6 Luna (OpenAI #27)",
        "id": "openai/gpt-5.6-luna"
    },
    "deepseek_v3": {
        "name": "DeepSeek-V3 671B (DeepSeek)",
        "id": "deepseek/deepseek-chat"
    },
    "llama_3_3_70b": {
        "name": "Llama 3.3 70B (Meta AI)",
        "id": "meta-llama/llama-3.3-70b-instruct"
    },
    "glm_5_3": {
        "name": "GLM 5.3 (Zhipu AI #21)",
        "id": "z-ai/glm-5.3"
    }
}


def load_api_key() -> str:
    """Load OpenRouter API key from environment or .env file."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_file = ROOT / ".env"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip()
                        break
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not found in environment or .env file!")
        sys.exit(1)
    return api_key


def parse_llm_boolean(text: str) -> Optional[bool]:
    """Parse raw LLM response text into boolean True/False/None."""
    if not text:
        return None
    t = text.strip().lower()
    if t.startswith("yes") or t in ("true", "1", "correct"):
        return True
    if t.startswith("no") or t in ("false", "0", "incorrect"):
        return False
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def call_openrouter(model_id: str, question: str, api_key: str, retries: int = 3) -> tuple[str, Optional[bool], float, float]:
    """
    Call OpenRouter chat completions API with zero temperature for deterministic reasoning.
    Returns: (raw_text, parsed_bool, latency_ms, cost_usd)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HamzaNasiem/LogicGuard",
        "X-Title": "AvicennaGuard Arena Live Benchmark"
    }
    
    prompt = f"Answer this question with YES or NO only. Do not explain. Just say YES or NO.\n\nQuestion: {question}\nAnswer:"
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0
    }
    
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=25)
            latency_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                res = r.json()
                raw_text = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                cost = res.get("usage", {}).get("cost", 0.0)
                parsed_bool = parse_llm_boolean(raw_text)
                return raw_text, parsed_bool, latency_ms, cost
            elif r.status_code == 402:
                err_msg = r.json().get("error", {}).get("message", "Insufficient credits")
                print(f"\n[OPENROUTER ERROR 402]: {err_msg}")
                print("[ACTION REQUIRED]: Please add $1 to $2 credit balance at https://openrouter.ai/settings/credits to run live frontier models.\n")
                return "[INSUFFICIENT_CREDITS]", None, 0.0, 0.0
            elif r.status_code == 401:
                print(f"\n[OPENROUTER ERROR 401]: Invalid API Key or Unauthorized.\n")
                return "[UNAUTHORIZED]", None, 0.0, 0.0
            elif r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
            else:
                time.sleep(1.0)
        except Exception as e:
            time.sleep(1.5)
            
    return "[API_ERROR]", None, 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="AvicennaGuard Live Arena Benchmark Evaluator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (default: all 500)")
    parser.add_argument("--models", nargs="+", default=list(ARENA_MODELS.keys()), help="Models to evaluate")
    parser.add_argument("--output", type=str, default="results/models/arena_live_evaluation_results.json", help="Output results path")
    args = parser.parse_args()

    api_key = load_api_key()
    
    # 1. Load Knowledge Base & Engine
    kb_path = ROOT / "data/knowledge_bases/knowledge_base_extended.json"
    print("=" * 95)
    print("   AVICENNAGUARD -- LIVE LMSYS ARENA LEADERBOARD BENCHMARK HARNESS")
    print("=" * 95)
    print(f"[*] Initializing AvicennaGuard Knowledge Engine from: {kb_path.name}")
    t_kb_0 = time.perf_counter()
    kb = KnowledgeBase(kb_path)
    validator = BFSValidator(kb)
    s1_parser = DebertaParser()
    t_kb_1 = time.perf_counter()
    print(f"    -> Taxonomy (G_T): {kb.G_T.number_of_nodes()} nodes, {kb.G_T.number_of_edges()} edges")
    print(f"    -> Properties (G_P): {len(kb.G_P)} entities | Modus Ponens (G_C): {kb.G_C.number_of_edges()} rules")
    print(f"    -> Engine Loaded in {(t_kb_1 - t_kb_0)*1000:.2f} ms")

    # 2. Load 500-Query Benchmark Dataset
    bench_path = ROOT / "data/benchmarks/avicenna_benchmark_500.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    
    queries = benchmark_data if isinstance(benchmark_data, list) else benchmark_data.get("queries", [])
    if args.limit:
        queries = queries[:args.limit]
    total_queries = len(queries)
    print(f"[*] Benchmark Queries Loaded: {total_queries} items (FOLIO, ProofWriter, Curated, TruthfulQA)")

    # 3. Model Suite Setup
    selected_models = []
    for m in args.models:
        if m in ARENA_MODELS:
            selected_models.append((m, ARENA_MODELS[m]))
        else:
            matched = False
            for k, v in ARENA_MODELS.items():
                if m.lower() in k.lower() or m.lower() in v["id"].lower():
                    selected_models.append((k, v))
                    matched = True
                    break
            if not matched:
                selected_models.append((m, {"name": m, "id": m}))

    print(f"[*] Selected Arena Frontier Models ({len(selected_models)}):")
    for idx, (k, info) in enumerate(selected_models, 1):
        print(f"    {idx}. {info['name']} ({info['id']})")
    print("=" * 95)

    # 4. Evaluation Loop
    all_results = {}
    summary_table = []
    total_spend_usd = 0.0

    for model_key, model_info in selected_models:
        model_name = model_info["name"]
        model_id = model_info["id"]
        print(f"\n>>> LAUNCHING LIVE EVALUATION: {model_name} <<<")
        
        bl_correct = 0
        ag_correct = 0
        intercepted_count = 0
        false_alarms = 0
        epistemic_counter = Counter()
        model_records = []
        model_cost = 0.0
        
        for q_idx, item in enumerate(queries, 1):
            q_id = item.get("id", f"q_{q_idx}")
            question = item.get("question", "")
            gt = item.get("ground_truth")
            gt_bool = (gt.lower() in ("yes", "true", "1")) if isinstance(gt, str) and gt.lower() in ("yes", "no", "true", "false") else gt if isinstance(gt, bool) else None
            q_type = item.get("query_type", "taxonomic")
            
            # 1. Live LLM Call
            llm_raw, llm_bool, llm_ms, cost = call_openrouter(model_id, question, api_key)
            model_cost += cost
            total_spend_usd += cost
            
            is_bl_correct = (llm_bool == gt_bool) if (llm_bool is not None and gt_bool is not None) else (gt == "OOD")
            if is_bl_correct:
                bl_correct += 1
                
            # 2. AvicennaGuard Stage 1 Parsing
            t_s1 = time.perf_counter()
            parsed = s1_parser.parse(question)
            s1_ms = (time.perf_counter() - t_s1) * 1000
            parsed_type = parsed.get("type", q_type)
            
            # 3. AvicennaGuard Stage 2 Graph BFS
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
                    # Contradiction -> WAHM (Hallucination Intercepted)
                    ep_state = EpistemicState.WAHM
                    intercepted = True
                    final_answer = graph_ans
                    intercepted_count += 1
                else:
                    ep_state = EpistemicState.YAQEEN
                    intercepted = False
                    final_answer = graph_ans
            else:
                ep_state = EpistemicState.SHAKK
                intercepted = False
                final_answer = llm_bool

            epistemic_counter[ep_state.value] += 1
            
            final_bool = final_answer if isinstance(final_answer, bool) else (final_answer.lower() in ("yes", "true", "1") if isinstance(final_answer, str) and final_answer.lower() in ("yes", "no") else None)
            is_ag_correct = (final_bool == gt_bool) if (final_bool is not None and gt_bool is not None) else (gt == "OOD" and ep_state == EpistemicState.SHAKK)
            if is_ag_correct:
                ag_correct += 1

            if is_bl_correct and not is_ag_correct:
                false_alarms += 1
                
            model_records.append({
                "id": q_id,
                "question": question,
                "ground_truth": gt,
                "llm_raw": llm_raw,
                "llm_answer": llm_bool,
                "final_answer": final_answer,
                "epistemic_state": ep_state.value,
                "intercepted": intercepted,
                "path": path,
                "is_bl_correct": is_bl_correct,
                "is_ag_correct": is_ag_correct,
                "latency_ms": {"llm": round(llm_ms, 2), "s1": round(s1_ms, 3), "s2": round(s2_ms, 3)}
            })

            action_tag = "[WAHM: INTERCEPTED]" if intercepted else f"[{ep_state.value}]"
            print(f"  [{q_idx:03d}/{total_queries:03d}] Raw LLM: {str(llm_bool):<5} | Guard: {str(final_bool):<5} | GT: {str(gt_bool):<5} | Action: {action_tag}")
            
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
            "epistemic_distribution": dict(epistemic_counter),
            "total_cost_usd": round(model_cost, 6),
            "records": model_records
        }
        
        summary_table.append({
            "model": model_name,
            "bl_acc": f"{bl_acc:.2f}%",
            "ag_acc": f"{ag_acc:.2f}%",
            "gain": f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%",
            "caught": intercepted_count,
            "fa": false_alarms,
            "cost": f"${model_cost:.4f}"
        })
        print(f"\n  [COMPLETED] {model_name}: Raw LLM = {bl_acc:.2f}% -> +AvicennaGuard = {ag_acc:.2f}% (Interceptions: {intercepted_count}, False Alarms: {false_alarms})\n")

    # 5. Save Output
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    # 6. Print Final IEEE Summary Table
    print("\n" + "=" * 95)
    print("  FINAL LIVE ARENA EVALUATION BENCHMARK RESULTS (IEEE TABLE I FORMAT)")
    print("=" * 95)
    print(f"{'Frontier Model Name':<32} | {'Raw LLM Acc':<12} | {'+AvicennaGuard':<14} | {'Gain':<8} | {'Caught':<7} | {'FA':<4} | {'Cost'}")
    print("-" * 95)
    for row in summary_table:
        print(f"{row['model']:<32} | {row['bl_acc']:<12} | {row['ag_acc']:<14} | {row['gain']:<8} | {row['caught']:<7} | {row['fa']:<4} | {row['cost']}")
    print("-" * 95)
    print(f"[*] Total Benchmark Spend: ${total_spend_usd:.4f} USD")
    print(f"[*] Full Evaluation Artifact Saved To: {out_path.relative_to(ROOT)}")
    print("=" * 95)


if __name__ == "__main__":
    main()
