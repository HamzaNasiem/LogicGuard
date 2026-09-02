#!/usr/bin/env python3
"""
AvicennaGuard — Step 2b: Modern Model Evaluation via OpenRouter
=============================================================
Tests AvicennaGuard against GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro
using the OpenRouter unified API (one key for all providers).

Free models (no billing required):
  - meta-llama/llama-3.3-70b-instruct:free
  - mistralai/mistral-small-3.1-24b-instruct:free
  - google/gemma-3-27b-it:free

Premium models (need paid account):
  - openai/gpt-4o
  - anthropic/claude-3.5-sonnet
  - google/gemini-1.5-pro

Usage:
    # Free tier (works now)
    python step2b_openrouter_eval.py --api_key sk-or-v1-xxx

    # Premium tier (after upgrade)
    python step2b_openrouter_eval.py --api_key sk-or-v1-xxx --tier premium

    # Specific models only
    python step2b_openrouter_eval.py --api_key sk-or-v1-xxx --models gpt4o,claude35

    # With expanded KB
    python step2b_openrouter_eval.py --api_key sk-or-v1-xxx --kb knowledge_base_1200.json
"""

import os
import sys
import json
import time
import re
import argparse
import urllib.request
import urllib.error
from typing import Optional

# ── Model registry ────────────────────────────────────────────────────────────

FREE_MODELS = {
    "mistral_small_24b": {
        "id":          "mistralai/mistral-small-3.1-24b-instruct:free",
        "display":     "Mistral Small 3.1-24B (Mistral AI, 2025)",
        "tier":        "free",
    },
    "llama33_70b": {
        "id":          "meta-llama/llama-3.3-70b-instruct:free",
        "display":     "LLaMA 3.3-70B (Meta, 2024)",
        "tier":        "free",
    },
    "gemma3_27b": {
        "id":          "google/gemma-3-27b-it:free",
        "display":     "Gemma 3-27B (Google, 2025)",
        "tier":        "free",
    },
    # Fallback models (use if main models are rate-limited)
    "arcee_trinity": {
        "id":          "arcee-ai/trinity-large-preview:free",
        "display":     "Arcee Trinity Large (Arcee AI, 2025)",
        "tier":        "free",
    },
    "lfm_1b": {
        "id":          "liquid/lfm-2.5-1.2b-instruct:free",
        "display":     "LFM 2.5-1.2B (Liquid AI, 2025)",
        "tier":        "free",
    },
}

PREMIUM_MODELS = {
    "gpt4o": {
        "id":          "openai/gpt-4o",
        "display":     "GPT-4o (OpenAI, 2024)",
        "tier":        "premium",
    },
    "claude35": {
        "id":          "anthropic/claude-3.5-sonnet",
        "display":     "Claude 3.5 Sonnet (Anthropic, 2024)",
        "tier":        "premium",
    },
    "gemini15pro": {
        "id":          "google/gemini-1.5-pro",
        "display":     "Gemini 1.5 Pro (Google, 2024)",
        "tier":        "premium",
    },
    "llama31_70b": {
        "id":          "meta-llama/llama-3.1-70b-instruct",
        "display":     "LLaMA 3.1-70B (Meta, 2024)",
        "tier":        "premium",
    },
    "qwen25_72b": {
        "id":          "qwen/qwen-2.5-72b-instruct",
        "display":     "Qwen 2.5-72B (Alibaba, 2024)",
        "tier":        "premium",
    },
}

ALL_MODELS = {**FREE_MODELS, **PREMIUM_MODELS}

# ── OpenRouter client ─────────────────────────────────────────────────────────

class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization":  f"Bearer {api_key}",
            "Content-Type":   "application/json",
            "HTTP-Referer":   "https://avicennaguard.research",
            "X-Title":        "AvicennaGuard IEEE Research",
        }

    def chat(self, model_id: str, prompt: str,
             max_tokens: int = 20, temperature: float = 0.0) -> tuple[str, float]:
        """
        Call OpenRouter chat completion.
        Returns (answer_text, latency_ms).
        """
        payload = json.dumps({
            "model":       model_id,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.BASE_URL}/chat/completions",
            data=payload,
            headers=self.headers,
            method="POST",
        )

        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    data = json.loads(r.read())
                    text = data["choices"][0]["message"]["content"]
                    if text is None:
                        text = ""
                    return text.strip(), latency_ms
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                if e.code == 429 and attempt < 4:
                    # Exponential backoff: 30s, 60s, 90s, 120s
                    wait = 30 * (attempt + 1)
                    print(f"\n    Rate limited (429), waiting {wait}s (attempt {attempt+1}/5)...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {body[:200]}")
        raise RuntimeError("Max retries exceeded (rate limited)")

    def test_connection(self, model_id: str) -> bool:
        try:
            text, _ = self.chat(model_id, "Reply with YES", max_tokens=5)
            return len(text) > 0
        except Exception as e:
            print(f"    Connection test failed: {e}")
            return False


# ── AvicennaGuard Validator (same BFS logic as step2) ───────────────────────────

class KBValidator:
    def __init__(self, kb_path: str):
        with open(kb_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        # Build ancestor graph via BFS closure
        raw = self.kb["taxonomies"]
        self.graph: dict[str, set] = {k: set() for k in raw}
        for entity, parents in raw.items():
            visited, queue = set(), list(parents)
            while queue:
                p = queue.pop(0)
                if p not in visited:
                    visited.add(p)
                    self.graph[entity].add(p)
                    queue.extend(raw.get(p, []))

    def normalize(self, word: str) -> str:
        w = word.lower().strip()
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        return w

    def is_ancestor(self, child: str, ancestor: str) -> bool:
        c, a = self.normalize(child), self.normalize(ancestor)
        return c == a or a in self.graph.get(c, set())

    def has_property(self, entity: str, prop: str) -> bool:
        e = self.normalize(entity)
        p = prop.lower().strip().replace(" ", "_")
        for ent in [e] + list(self.graph.get(e, set())):
            if ent in self.kb["properties"]:
                if p in self.kb["properties"][ent] or \
                   p.replace("_", " ") in self.kb["properties"][ent]:
                    return True
        return False

    def check_conditional(self, condition: str, consequence: str) -> tuple[bool, bool]:
        cond = condition.lower().strip()
        cons = consequence.lower().strip()
        cons_vars = {
            cons, cons.replace(" ", "_"),
            cons.split()[-1] if cons else "",
            cons.replace("do we need an ", "").replace("do we need ", "").strip(),
            cons.replace("does it ", "").replace("is it ", "").strip(),
            "_".join(w for w in cons.split()
                     if w not in ("is","are","the","a","an","do","does","we")),
        }
        cond_vars = {
            cond, cond.replace(" ", "_"),
            re.sub(r"\b(is|are|the|it|there)\b", "", cond).strip(),
            cond.replace("it is ", "").strip(),
            cond.replace("there is ", "").strip(),
            cond.replace("if ", "").strip(),
        }
        cond_vars.discard(""); cons_vars.discard("")
        for cv in cond_vars:
            if cv in self.kb["conditionals"]:
                for cnv in cons_vars:
                    if cnv in self.kb["conditionals"][cv]:
                        return True, True
                return False, True
        return False, False

    def validate(self, question: str, qtype: str) -> dict:
        q = question.strip()
        t = q.lower().rstrip("?")

        if qtype == "taxonomic":
            m = re.match(r"are all (\w+)\s+(?:a\s+|an\s+)?([\w ]+)", t)
            if not m:
                return {"epistemic_state": "UNKNOWN", "covered": False}
            subj = self.normalize(m.group(1))
            pred = self.normalize(m.group(2).strip())
            if subj not in self.graph:
                return {"epistemic_state": "UNKNOWN", "covered": False}
            result = self.is_ancestor(subj, pred)
            return {"epistemic_state": "YAQEEN" if result else "WAHM",
                    "covered": True, "graph_answer": result}

        elif qtype == "categorical":
            # have/need pattern
            m = re.match(r"do all (\w+)\s+(?:have|need)\s+(?:a\s+|an\s+|the\s+)?(.+)", t)
            if not m:
                # lay/give/grow/live pattern
                m2 = re.match(r"do all (\w+)\s+(lay|give|grow)\s+(?:a\s+|an\s+)?(\w.*)", t)
                m3 = re.match(r"do all (\w+)\s+live\s+in\s+(\w+)", t)
                if m2:
                    entity = m2.group(1)
                    prop   = f"{m2.group(2)}_{m2.group(3).replace(' ','_')}"
                elif m3:
                    entity = m3.group(1)
                    prop   = f"live_in_{m3.group(2)}"
                else:
                    return {"epistemic_state": "UNKNOWN", "covered": False}
            else:
                entity = m.group(1)
                prop   = m.group(2).strip()

            e_norm = self.normalize(entity)
            if e_norm not in self.graph and \
               not any(a in self.kb.get("properties", {}) for a in self.graph.get(e_norm, set())):
                return {"epistemic_state": "UNKNOWN", "covered": False}
            result = self.has_property(entity, prop)
            return {"epistemic_state": "YAQEEN" if result else "WAHM",
                    "covered": True, "graph_answer": result}

        elif qtype == "hypothetical":
            if "if" not in t:
                return {"epistemic_state": "UNKNOWN", "covered": False}
            after_if = t.split("if", 1)[1].strip()
            parts    = re.split(r",\s*|\bthen\b|\?", after_if)
            cond = parts[0].strip() if parts else ""
            cons = parts[1].strip() if len(parts) > 1 else ""
            if not cond or not cons:
                return {"epistemic_state": "UNKNOWN", "covered": False}
            is_neg = any(w in cons for w in (" not ", " no ", "doesn't", "don't"))
            if is_neg:
                cons = re.sub(r"\b(not|no|doesn't|don't)\b", "", cons).strip()
            result, covered = self.check_conditional(cond, cons)
            if not covered:
                return {"epistemic_state": "UNKNOWN", "covered": False}
            if is_neg:
                result = not result
            return {"epistemic_state": "YAQEEN" if result else "WAHM",
                    "covered": True, "graph_answer": result}

        return {"epistemic_state": "UNKNOWN", "covered": False}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    return (
        f"Answer this question with only YES or NO (one word only).\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def parse_answer(text: str) -> Optional[bool]:
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    if "YES" in t:
        return True
    if "NO" in t:
        return False
    return None


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_model(client: OpenRouterClient, model_key: str, model_info: dict,
                   queries: list, validator: KBValidator,
                   filter_source: str, rate_limit_delay: float) -> dict:

    model_id   = model_info["id"]
    model_name = model_info["display"]

    filtered = [q for q in queries
                if filter_source == "all" or q.get("source") == "original"]

    print(f"\n{'='*65}")
    print(f"  EVALUATING: {model_key}  ({model_name})")
    print(f"  Model ID: {model_id}")
    print(f"  Queries:  {len(filtered)}")
    print(f"{'='*65}")

    baseline_results = []
    avicennaguard_results = []

    # ── Phase A: Pure LLM baseline ────────────────────────────────
    print(f"\n  Phase A: Pure LLM baseline\n")
    for i, q in enumerate(filtered, 1):
        question     = q["question"]
        ground_truth = q["ground_truth"]
        qtype        = q.get("type", "unknown")

        try:
            answer_text, latency_ms = client.chat(
                model_id, build_prompt(question), max_tokens=10, temperature=0.0)
            llm_answer = parse_answer(answer_text)
        except Exception as e:
            print(f"\n    [Error Q{i}] {e}")
            llm_answer  = None
            latency_ms  = 0.0

        is_correct = (llm_answer == ground_truth) if llm_answer is not None else False

        baseline_results.append({
            "question":      question,
            "ground_truth":  ground_truth,
            "type":          qtype,
            "llm_answer":    llm_answer,
            "is_correct":    is_correct,
            "llm_ms":        round(latency_ms, 1),
        })

        icon = "✓" if is_correct else "✗"
        print(f"  [{i:3d}/{len(filtered)}] {icon}  {question[:55]}")
        time.sleep(rate_limit_delay)

    base_correct = sum(1 for r in baseline_results if r["is_correct"])
    print(f"\n  Baseline accuracy: {base_correct}/{len(filtered)} = {base_correct/len(filtered)*100:.1f}%")

    # ── Phase B: + AvicennaGuard override ───────────────────────────
    print(f"\n  Phase B: + AvicennaGuard override\n")
    for i, (q, base_r) in enumerate(zip(filtered, baseline_results), 1):
        question     = q["question"]
        ground_truth = q["ground_truth"]
        qtype        = q.get("type", "unknown")

        # Run validator (no LLM call needed — reuse baseline answer)
        t1 = time.perf_counter()
        val_result = validator.validate(question, qtype)
        stage2_ms  = (time.perf_counter() - t1) * 1000

        epistemic  = val_result["epistemic_state"]
        covered    = val_result.get("covered", False)
        graph_ans  = val_result.get("graph_answer", None)

        # Override decision
        if covered and graph_ans is not None:
            final_answer = graph_ans      # AvicennaGuard overrides
        else:
            final_answer = base_r["llm_answer"]  # defer to LLM

        is_correct = (final_answer == ground_truth) if final_answer is not None else False

        avicennaguard_results.append({
            "question":       question,
            "ground_truth":   ground_truth,
            "type":           qtype,
            "epistemic_state": epistemic,
            "llm_answer":     base_r["llm_answer"],
            "final_answer":   final_answer,
            "is_correct":     is_correct,
            "covered":        covered,
            "llm_ms":         base_r["llm_ms"],
            "stage2_ms":      round(stage2_ms, 3),
        })

        icon = "✓" if is_correct else "✗"
        print(f"  [{i:3d}/{len(filtered)}] {icon}  {epistemic:7s}  {question[:50]}")

    lg_correct = sum(1 for r in avicennaguard_results if r["is_correct"])

    # ── Summaries ────────────────────────────────────────────────
    caught = sum(1 for b, l in zip(baseline_results, avicennaguard_results)
                 if not b["is_correct"] and l["is_correct"])
    false_alarms = sum(1 for b, l in zip(baseline_results, avicennaguard_results)
                       if b["is_correct"] and not l["is_correct"])

    base_by_type = {}
    lg_by_type   = {}
    for qtype_ in ["taxonomic", "categorical", "hypothetical"]:
        bt = [r for r in baseline_results   if r["type"] == qtype_]
        lt = [r for r in avicennaguard_results if r["type"] == qtype_]
        if bt:
            base_by_type[qtype_] = sum(1 for r in bt if r["is_correct"]) / len(bt)
        if lt:
            lg_by_type[qtype_]   = sum(1 for r in lt if r["is_correct"]) / len(lt)

    print(f"\n  +AvicennaGuard accuracy: {lg_correct}/{len(filtered)} = {lg_correct/len(filtered)*100:.1f}%")
    print(f"  Hallucinations caught: {caught}  |  False alarms: {false_alarms}")

    return {
        "model_key":      model_key,
        "model_id":       model_id,
        "model_display":  model_name,
        "n_queries":      len(filtered),
        "baseline": {
            "results": baseline_results,
            "summary": {
                "correct":         base_correct,
                "total":           len(filtered),
                "by_type":         base_by_type,
                "per_query_correct": [r["is_correct"] for r in baseline_results],
            },
        },
        "avicennaguard": {
            "results": avicennaguard_results,
            "summary": {
                "correct":              lg_correct,
                "total":                len(filtered),
                "by_type":              lg_by_type,
                "hallucinations_caught": caught,
                "false_alarms":         false_alarms,
                "per_query_correct":    [r["is_correct"] for r in avicennaguard_results],
            },
        },
        "logicguard": {
            "results": avicennaguard_results,
            "summary": {
                "correct":              lg_correct,
                "total":                len(filtered),
                "by_type":              lg_by_type,
                "hallucinations_caught": caught,
                "false_alarms":         false_alarms,
                "per_query_correct":    [r["is_correct"] for r in avicennaguard_results],
            },
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AvicennaGuard — Modern Model Evaluation via OpenRouter")
    parser.add_argument("--api_key",       default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--queries",       default="extended_queries.json")
    parser.add_argument("--kb",            default="knowledge_base_1200.json",
                        help="KB to use (default: expanded 1200-node KB)")
    parser.add_argument("--output",        default="openrouter_results.json")
    parser.add_argument("--models",        default="free",
                        help="'free', 'premium', 'all', or comma-separated keys "
                             "(llama31_8b,gemma2_9b,gpt4o,claude35,gemini15pro,...)")
    parser.add_argument("--filter_source", default="original",
                        choices=["original", "all"])
    parser.add_argument("--delay",         type=float, default=1.5,
                        help="Seconds between API calls (free tier: 1.5s)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: OpenRouter API key required.")
        print("  Set OPENROUTER_API_KEY env var or use --api_key sk-or-v1-xxx")
        sys.exit(1)

    print("=" * 65)
    print("  AvicennaGuard — Modern Model Evaluation via OpenRouter")
    print("=" * 65)

    # Resolve model list
    if args.models == "free":
        selected = FREE_MODELS
    elif args.models == "premium":
        selected = PREMIUM_MODELS
    elif args.models == "all":
        selected = ALL_MODELS
    else:
        keys = [k.strip() for k in args.models.split(",")]
        selected = {k: ALL_MODELS[k] for k in keys if k in ALL_MODELS}
        if not selected:
            print(f"Unknown model keys: {keys}")
            print(f"Available: {list(ALL_MODELS.keys())}")
            sys.exit(1)

    print(f"\n  API key   : {args.api_key[:20]}...")
    print(f"  Models    : {list(selected.keys())}")
    print(f"  KB        : {args.kb}")
    print(f"  Filter    : {args.filter_source}")
    print(f"  Delay     : {args.delay}s")

    # Load queries
    with open(args.queries, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"] if isinstance(data, dict) else data
    print(f"  Queries   : {len(queries)} total")

    # Load KB validator
    print(f"\n  Loading KB: {args.kb}...")
    validator = KBValidator(args.kb)
    print(f"  KB nodes  : {len(validator.graph)}")

    # Verify API key is valid (a 429 still means key is good — just rate limited)
    client = OpenRouterClient(args.api_key)
    print(f"\n  Verifying API key...")
    import urllib.request as _ur
    try:
        _req = _ur.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {args.api_key}"}
        )
        with _ur.urlopen(_req, timeout=10) as _r:
            _d = json.loads(_r.read())
            print(f"  API key: OK  ({len(_d['data'])} models available)\n")
    except Exception as _e:
        print(f"  ERROR verifying API key: {_e}")
        sys.exit(1)

    # Run evaluations
    all_results = {}
    start_time  = time.time()

    for model_key, model_info in selected.items():
        try:
            result = evaluate_model(
                client, model_key, model_info, queries,
                validator, args.filter_source, args.delay)
            all_results[model_key] = result
        except KeyboardInterrupt:
            print("\n\n  Interrupted — saving partial results...")
            break
        except Exception as e:
            print(f"\n  ERROR evaluating {model_key}: {e}")
            continue

    elapsed = time.time() - start_time

    # Save
    output = {
        "metadata": {
            "source":        "OpenRouter API",
            "kb_path":       args.kb,
            "kb_nodes":      len(validator.graph),
            "filter_source": args.filter_source,
            "models_tested": list(all_results.keys()),
            "elapsed_min":   round(elapsed / 60, 1),
        },
        "results": all_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE  ({elapsed/60:.1f} min)")
    print(f"{'='*65}")
    print(f"  {'Model':<22}  {'Base':>6}  {'+ AG':>6}  {'Δ':>6}  {'Caught':>7}  {'FA':>4}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*4}")
    for key, r in all_results.items():
        bs  = r["baseline"]["summary"]
        lgs = r.get("avicennaguard", r.get("logicguard", {}))["summary"]
        n   = bs["total"]
        ba  = bs["correct"] / n * 100
        la  = lgs["correct"] / n * 100
        print(f"  {r['model_display'][:22]:<22}  {ba:5.1f}%  {la:5.1f}%  "
              f"{la-ba:+5.1f}%  {lgs['hallucinations_caught']:>7}  "
              f"{lgs['false_alarms']:>4}")

    print(f"\n  Saved: {args.output}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
