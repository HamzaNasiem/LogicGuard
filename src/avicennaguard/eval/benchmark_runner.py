"""
AvicennaGuard Central Evaluation Orchestrator & Benchmark Runner.

Runs multi-model evaluations across logic reasoning benchmarks
(e.g., FOLIO, ProofWriter, Curated Gold, TruthfulQA OOD) for baseline LLMs
versus +AvicennaGuard, logging epistemic states, latency profiling,
hallucination interceptions, and full confusion matrix metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

from avicennaguard.core.epistemic_states import EpistemicState
from avicennaguard.data.benchmark_loader import BenchmarkLoader
from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.llm_parser import LLMParser
from avicennaguard.parsers.regex_parser import RegexParser
from avicennaguard.parsers.typed_regex import (
    extract_categorical,
    extract_hypothetical,
    extract_taxonomic,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS: list[str] = [
    "llama3.2:3b",
    "mistral:7b",
    "llama2:7b",
    "deepseek-r1:7b",
    "phi4:latest",
]

MODEL_ALIASES: dict[str, str] = {
    "llama3.2": "llama3.2:3b",
    "llama32": "llama3.2:3b",
    "llama32_3b": "llama3.2:3b",
    "llama3.2:3b": "llama3.2:3b",
    "mistral": "mistral:7b",
    "mistral_7b": "mistral:7b",
    "mistral:7b": "mistral:7b",
    "llama2": "llama2:7b",
    "llama2_7b": "llama2:7b",
    "llama2:7b": "llama2:7b",
    "deepseek": "deepseek-r1:7b",
    "deepseek-r1": "deepseek-r1:7b",
    "deepseek_r1": "deepseek-r1:7b",
    "deepseek-r1:7b": "deepseek-r1:7b",
    "phi4": "phi4:latest",
    "phi-4": "phi4:latest",
    "phi4:latest": "phi4:latest",
}

# Baseline model simulated accuracy priors for mock mode (for deterministic offline testing)
MOCK_ACCURACY_PRIORS: dict[str, float] = {
    "llama2:7b": 0.60,
    "mistral:7b": 0.72,
    "llama3.2:3b": 0.84,
    "deepseek-r1:7b": 0.88,
    "phi4:latest": 0.85,
}


def to_bool(val: Any) -> Optional[bool]:
    """Convert boolean or string truth representation to Python bool."""
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "yes", "1", "t", "y", "correct", "right"):
        return True
    if s in ("false", "no", "0", "f", "n", "incorrect", "wrong"):
        return False
    return None


def parse_llm_yn(raw_answer: str) -> Optional[bool]:
    """Parse raw LLM textual answer into boolean YES/NO decision."""
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


def compute_stats(vals: list[float]) -> dict[str, float]:
    """Calculate descriptive statistics (mean, median, p95, min, max)."""
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(vals)
    n = len(s)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "mean": round(sum(s) / n, 3),
        "median": round(s[n // 2], 3),
        "p95": round(s[p95_idx], 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


class BenchmarkRunner:
    """
    Central evaluation orchestrator for running multi-model benchmarks.

    Evaluates baseline LLM performance versus +AvicennaGuard on logic benchmarks
    such as `avicenna_benchmark_500.json`. Profiles latency, records Ibn Sina
    epistemic states (YAQEEN, WAHM, SHAKK, ZANN), tracks hallucination interceptions,
    and computes full confusion matrix metrics.

    Args:
        kb: Optional KnowledgeBase instance or path to KB JSON file.
        benchmark_path: Optional path to the benchmark JSON dataset file.
        models: Optional list of model names / tags to evaluate.
        mock_mode: If True, uses offline deterministic simulation for CI and tests.
        parser_mode: Stage 1 parsing mode ('llm', 'regex', or 'both').
        delay: Artificial delay (in seconds) between LLM calls to prevent rate limits.
        seed: Random seed for deterministic reproducibility in mock mode.
    """

    def __init__(
        self,
        kb: Optional[Union[KnowledgeBase, str, Path]] = None,
        benchmark_path: Optional[Union[str, Path]] = None,
        models: Optional[list[str]] = None,
        mock_mode: bool = False,
        parser_mode: str = "llm",
        delay: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.mock_mode = mock_mode
        self.parser_mode = parser_mode
        self.delay = delay
        self.seed = seed

        # Resolve Knowledge Base
        self.kb = self._resolve_kb(kb)
        self._validator = BFSValidator(self.kb)

        # Resolve Benchmark Loader
        self.benchmark_loader = BenchmarkLoader(benchmark_path)
        self.benchmark_path = self.benchmark_loader.benchmark_path

        # Resolve Models
        resolved_models = []
        models_input = models or DEFAULT_MODELS
        for m in models_input:
            resolved_models.append(MODEL_ALIASES.get(m, m))
        self.models = resolved_models

        # Initialize Parsers
        self._regex_parser = RegexParser()
        self._llm_parsers: dict[str, LLMParser] = {}

    @staticmethod
    def _resolve_kb(kb: Optional[Union[KnowledgeBase, str, Path]]) -> KnowledgeBase:
        """Resolve or load a KnowledgeBase instance."""
        if isinstance(kb, KnowledgeBase):
            return kb
        if kb is not None:
            p = Path(kb)
            if p.exists():
                return KnowledgeBase(p)
            raise FileNotFoundError(f"Knowledge base file not found: {p}")

        # Search default paths
        candidates = [
            Path("data/knowledge_bases/knowledge_base_extended.json"),
            Path("data/knowledge_bases/knowledge_base.json"),
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "knowledge_bases"
            / "knowledge_base_extended.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return KnowledgeBase(candidate)

        raise FileNotFoundError("Could not resolve default knowledge base file.")

    def _get_llm_parser(self, model_name: str) -> LLMParser:
        """Retrieve or instantiate an LLMParser for a specific model."""
        if model_name not in self._llm_parsers:
            self._llm_parsers[model_name] = LLMParser(model=model_name)
        return self._llm_parsers[model_name]

    def _call_llm_baseline(
        self,
        question: str,
        model_name: str,
        ground_truth: Any,
        query_idx: int = 0,
    ) -> tuple[str, Optional[bool], float]:
        """
        Call LLM for baseline question answering (or simulate in mock mode).

        Returns:
            Tuple of (raw_answer_string, parsed_bool, latency_ms).
        """
        if self.mock_mode:
            # Deterministic pseudo-random simulation based on question and model
            hash_int = int(hashlib.md5(f"{model_name}:{question}:{self.seed}:{query_idx}".encode()).hexdigest(), 16)
            latency_ms = round(12.0 + (hash_int % 1500) / 100.0, 2)

            gt_bool = to_bool(ground_truth)
            if gt_bool is None:
                # OOD query
                return "unknown", None, latency_ms

            prior = MOCK_ACCURACY_PRIORS.get(model_name, 0.80)
            roll = (hash_int % 1000) / 1000.0
            is_correct = roll < prior

            pred_bool = gt_bool if is_correct else not gt_bool
            raw_str = "yes" if pred_bool else "no"
            return raw_str, pred_bool, latency_ms

        # Live Ollama call
        prompt = (
            "Answer this question with YES or NO only. "
            "Do not explain. Just say YES or NO.\n\n"
            f"Question: {question}\nAnswer:"
        )
        t0 = time.perf_counter()
        try:
            import ollama

            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "seed": self.seed, "num_predict": 10},
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = resp["message"]["content"].strip()
            parsed_bool = parse_llm_yn(raw)
            return raw, parsed_bool, latency_ms
        except Exception as e:
            logger.warning("Live LLM call to %s failed: %s", model_name, e)
            latency_ms = (time.perf_counter() - t0) * 1000
            return "[llm_error]", None, latency_ms

    def _parse_stage1(
        self,
        question: str,
        query_type_hint: str,
        model_name: str,
    ) -> tuple[dict, bool, str, float]:
        """
        Execute Stage 1 Semantic Parsing (probabilistic form extraction).

        Returns:
            Tuple of (parsed_dict, used_fallback, parse_status, stage1_latency_ms).
        """
        t0 = time.perf_counter()

        if self.mock_mode:
            # Deterministic fast offline parser using typed regex / regex parser
            parsed = self._regex_parser.parse(question)
            used_fallback = False
            parse_status = "success"

            if parsed.get("type") == "non-logical" and query_type_hint in ("taxonomic", "categorical", "hypothetical"):
                # Try typed regex extraction as fallback
                if query_type_hint == "taxonomic":
                    s, p, st = extract_taxonomic(question)
                    if s and p:
                        parsed = {"type": "taxonomic", "subject": s, "predicate": p}
                        used_fallback = True
                        parse_status = st
                elif query_type_hint == "categorical":
                    e, pr, st = extract_categorical(question)
                    if e and pr:
                        parsed = {"type": "categorical", "entity": e, "property": pr}
                        used_fallback = True
                        parse_status = st
                elif query_type_hint == "hypothetical":
                    c, cq, is_neg, st = extract_hypothetical(question)
                    if c and cq:
                        parsed = {"type": "hypothetical", "condition": c, "consequence": cq}
                        if is_neg:
                            parsed["_negate"] = True
                        used_fallback = True
                        parse_status = st

            if parsed.get("type") == "non-logical":
                parse_status = "parse_failure"

            stage1_ms = round((time.perf_counter() - t0) * 1000 + 0.8, 3)
            return parsed, used_fallback, parse_status, stage1_ms

        # Live parser
        if self.parser_mode == "regex":
            parsed = self._regex_parser.parse(question)
            used_fallback = True
            parse_status = "regex_only" if parsed.get("type") != "non-logical" else "parse_failure"
        else:
            parser = self._get_llm_parser(model_name)
            parsed, used_fallback = parser.parse(question)
            parse_status = "regex_fallback" if used_fallback else "success"
            if parsed.get("type") == "non-logical":
                parse_status = "parse_failure"

        stage1_ms = (time.perf_counter() - t0) * 1000
        return parsed, used_fallback, parse_status, round(stage1_ms, 3)

    def _validate_stage2(
        self,
        parsed: dict,
        query_type_hint: str,
    ) -> tuple[Optional[bool], EpistemicState, list[str], str, float]:
        """
        Execute Stage 2 Deterministic BFS Graph Validation.

        Returns:
            Tuple of (graph_answer, epistemic_state, path, proof, stage2_latency_ms).
        """
        t0 = time.perf_counter()
        qtype = parsed.get("type", query_type_hint)
        negate = parsed.get("_negate", False)

        if qtype == "taxonomic":
            s = parsed.get("subject", "")
            p = parsed.get("predicate", "")
            ans, state, path = self._validator.validate_taxonomic(s, p)
            if ans is not None and negate:
                ans = not ans
            if path:
                proof = f"BFS: {' -> '.join(path)} = {ans}"
            elif ans is not None:
                proof = f"BFS: {s} ⊬ {p} = {ans}"
            else:
                proof = f"Entity '{s}' or '{p}' not in KB (SHAKK)"
            stage2_ms = (time.perf_counter() - t0) * 1000
            return ans, state, path, proof, round(stage2_ms, 3)

        elif qtype == "categorical":
            e = parsed.get("entity", "")
            pr = parsed.get("property", "")
            ans, state = self._validator.validate_categorical(e, pr)
            if ans is not None and negate:
                ans = not ans
            if ans is not None:
                proof = f"Property: {e}.{pr} = {ans}"
            else:
                proof = f"Entity '{e}' or property '{pr}' not in KB (SHAKK)"
            stage2_ms = (time.perf_counter() - t0) * 1000
            return ans, state, [], proof, round(stage2_ms, 3)

        elif qtype == "hypothetical":
            c = parsed.get("condition", "")
            cq = parsed.get("consequence", "")
            ans, state = self._validator.validate_hypothetical(c, cq)
            if ans is not None and negate:
                ans = not ans
            if ans is not None:
                proof = f"Modus Ponens: {c} -> {cq} = {ans}"
            else:
                proof = f"Condition '{c}' not in KB (SHAKK)"
            stage2_ms = (time.perf_counter() - t0) * 1000
            return ans, state, [], proof, round(stage2_ms, 3)

        stage2_ms = (time.perf_counter() - t0) * 1000
        return None, EpistemicState.SHAKK, [], "Non-logical / Out of KB scope (SHAKK)", round(stage2_ms, 3)

    @staticmethod
    def compute_confusion_matrix(records: list[dict]) -> dict[str, int]:
        """
        Compute binary confusion matrix for logical queries.

        Positive class = ground_truth is TRUE (valid claim)
        Negative class = ground_truth is FALSE (invalid claim)
        """
        tp = tn = fp = fn = 0
        for r in records:
            gt = to_bool(r.get("ground_truth"))
            pred = to_bool(r.get("final_answer"))
            if gt is None:
                continue
            if pred is None:
                pred = False

            if gt is True and pred is True:
                tp += 1
            elif gt is False and pred is False:
                tn += 1
            elif gt is False and pred is True:
                fp += 1
            elif gt is True and pred is False:
                fn += 1

        total = tp + tn + fp + fn
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "total": total}

    @staticmethod
    def compute_prf1(cm: dict[str, int]) -> dict[str, float]:
        """Compute Accuracy, Precision, Recall, F1, Specificity, FPR from confusion matrix."""
        tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
        total = cm["total"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "specificity": round(specificity * 100, 2),
            "fpr": round(fpr * 100, 2),
        }

    def compute_metrics(self, records: list[dict]) -> dict[str, Any]:
        """Calculate complete metrics bundle: Confusion Matrix + PRF1 + Specificity + FPR."""
        cm = self.compute_confusion_matrix(records)
        prf1 = self.compute_prf1(cm)
        return {**cm, **prf1}

    @staticmethod
    def compute_hallucination_analysis(
        baseline_records: list[dict],
        ag_records: list[dict],
    ) -> dict[str, Any]:
        """
        Analyze hallucination interception dynamics between Baseline and +AvicennaGuard.
        """
        base_by_q = {r["question"]: r for r in baseline_records}
        intercepted: list[str] = []
        false_alarms: list[str] = []
        both_correct: list[str] = []
        both_wrong: list[str] = []

        for ag_r in ag_records:
            q = ag_r["question"]
            bl_r = base_by_q.get(q)
            if not bl_r:
                continue

            bl_correct = bl_r.get("is_correct", False)
            ag_correct = ag_r.get("is_correct", False)

            if not bl_correct and ag_correct:
                intercepted.append(q)
            elif bl_correct and not ag_correct:
                false_alarms.append(q)
            elif bl_correct and ag_correct:
                both_correct.append(q)
            else:
                both_wrong.append(q)

        total_llm_errors = len(intercepted) + len(both_wrong)
        rate = round(len(intercepted) / total_llm_errors * 100, 2) if total_llm_errors > 0 else 100.0

        return {
            "intercepted": len(intercepted),
            "intercepted_questions": intercepted,
            "false_alarms": len(false_alarms),
            "false_alarm_questions": false_alarms,
            "both_correct": len(both_correct),
            "both_wrong": len(both_wrong),
            "total_llm_errors": total_llm_errors,
            "interception_rate": rate,
        }

    def run_evaluation(
        self,
        model_name: str,
        limit: Optional[int] = None,
        filter_source: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run complete evaluation loop on a single model (Baseline vs +AvicennaGuard).

        Args:
            model_name: Model identifier (e.g., 'llama3.2:3b', 'mistral:7b').
            limit: Maximum number of queries to evaluate.
            filter_source: Optional benchmark source filter ('FOLIO', 'ProofWriter', etc.).
            filter_type: Optional logical query type filter ('taxonomic', etc.).

        Returns:
            Dictionary with per-query logs, baseline metrics, AvicennaGuard metrics,
            latency breakdowns, and comparative performance.
        """
        resolved_model = MODEL_ALIASES.get(model_name, model_name)

        # Retrieve queries
        all_queries = self.benchmark_loader.get_all_queries()
        if filter_source and filter_source.lower() != "all":
            all_queries = [q for q in all_queries if q.get("source", "").lower() == filter_source.lower()]
        if filter_type and filter_type.lower() != "all":
            all_queries = [q for q in all_queries if q.get("query_type", "").lower() == filter_type.lower()]

        if limit is not None and limit > 0:
            all_queries = all_queries[:limit]

        baseline_records: list[dict] = []
        ag_records: list[dict] = []

        epistemic_counts: Counter[str] = Counter()
        hallucinations_caught = 0

        logger.info("Evaluating model %s on %d queries...", resolved_model, len(all_queries))

        for idx, item in enumerate(all_queries):
            qid = item.get("id", f"q_{idx}")
            q = item.get("question", "")
            gt = item.get("ground_truth")
            qtype = item.get("query_type", "unknown")
            source = item.get("source", "unknown")
            diff = item.get("difficulty", "medium")

            gt_bool = to_bool(gt)

            # 1. Baseline LLM invocation
            llm_raw, llm_bool, llm_ms = self._call_llm_baseline(q, resolved_model, gt, query_idx=idx)

            bl_correct = (llm_bool == gt_bool) if (llm_bool is not None and gt_bool is not None) else False
            if gt == "OOD":
                bl_correct = True  # Baseline handles open domain directly

            baseline_record = {
                "id": qid,
                "question": q,
                "source": source,
                "query_type": qtype,
                "difficulty": diff,
                "ground_truth": gt,
                "llm_raw": llm_raw,
                "llm_answer": llm_bool,
                "final_answer": llm_bool,
                "epistemic_state": None,
                "intercepted": False,
                "proof": "LLM baseline inference",
                "is_correct": bl_correct,
                "latency_ms": {
                    "llm_ms": round(llm_ms, 3),
                    "stage1_ms": 0.0,
                    "stage2_ms": 0.0,
                    "total_overhead_ms": 0.0,
                },
            }
            baseline_records.append(baseline_record)

            # 2. AvicennaGuard Stage 1 Parsing
            parsed, used_fb, parse_status, stage1_ms = self._parse_stage1(q, qtype, resolved_model)

            # 3. AvicennaGuard Stage 2 BFS Validation
            graph_ans, ep_state, path, proof, stage2_ms = self._validate_stage2(parsed, qtype)

            covered = ep_state != EpistemicState.SHAKK and graph_ans is not None

            # 4. Epistemic State Resolution and Interception
            if covered:
                if llm_bool is not None and llm_bool != graph_ans:
                    # Contradiction detected -> Hallucination caught (WAHM)
                    epistemic_state = EpistemicState.WAHM
                    intercepted = True
                    final_answer = graph_ans
                    hallucinations_caught += 1
                else:
                    # Logical agreement
                    epistemic_state = EpistemicState.ZANN if used_fb else EpistemicState.YAQEEN
                    intercepted = False
                    final_answer = graph_ans
            else:
                # Out of KB scope -> Defer entirely to LLM (SHAKK)
                epistemic_state = EpistemicState.SHAKK
                intercepted = False
                final_answer = llm_bool

            epistemic_counts[epistemic_state.value] += 1

            final_bool = to_bool(final_answer)
            if gt_bool is not None:
                ag_correct = (final_bool == gt_bool) if final_bool is not None else False
            elif gt == "OOD":
                ag_correct = (epistemic_state == EpistemicState.SHAKK)
            else:
                ag_correct = False

            total_overhead_ms = round(stage1_ms + stage2_ms, 3)

            ag_record = {
                "id": qid,
                "question": q,
                "source": source,
                "query_type": qtype,
                "difficulty": diff,
                "ground_truth": gt,
                "llm_raw": llm_raw,
                "llm_answer": llm_bool,
                "final_answer": final_answer,
                "epistemic_state": epistemic_state.value,
                "intercepted": intercepted,
                "proof": proof,
                "covered": covered,
                "is_correct": ag_correct,
                "latency_ms": {
                    "llm_ms": round(llm_ms, 3),
                    "stage1_ms": stage1_ms,
                    "stage2_ms": stage2_ms,
                    "total_overhead_ms": total_overhead_ms,
                },
            }
            ag_records.append(ag_record)

            if self.delay > 0:
                time.sleep(self.delay)

        # ── Aggregated Metrics ──────────────────────────────────────────
        bl_metrics = self.compute_metrics(baseline_records)
        ag_metrics = self.compute_metrics(ag_records)

        # ── Latency Summaries ───────────────────────────────────────────
        bl_llm_lats = [r["latency_ms"]["llm_ms"] for r in baseline_records]
        ag_s1_lats = [r["latency_ms"]["stage1_ms"] for r in ag_records]
        ag_s2_lats = [r["latency_ms"]["stage2_ms"] for r in ag_records]
        ag_tot_lats = [r["latency_ms"]["total_overhead_ms"] for r in ag_records]

        # ── Per-Type Breakdown ──────────────────────────────────────────
        by_type_ag: dict[str, dict[str, Any]] = {}
        for qt in ("taxonomic", "categorical", "hypothetical", "ood"):
            type_records = [r for r in ag_records if r.get("query_type", "").lower() == qt]
            if type_records:
                type_metrics = self.compute_metrics(type_records)
                by_type_ag[qt] = {
                    "total": len(type_records),
                    "correct": sum(1 for r in type_records if r.get("is_correct", False)),
                    "metrics": type_metrics,
                }

        # ── Hallucination Interception Analysis ─────────────────────────
        hall_analysis = self.compute_hallucination_analysis(baseline_records, ag_records)

        comparison = {
            "accuracy_delta": round(ag_metrics["accuracy"] - bl_metrics["accuracy"], 2),
            "f1_delta": round(ag_metrics["f1"] - bl_metrics["f1"], 2),
            "precision_delta": round(ag_metrics["precision"] - bl_metrics["precision"], 2),
            "recall_delta": round(ag_metrics["recall"] - bl_metrics["recall"], 2),
            "hallucinations_intercepted": hall_analysis["intercepted"],
            "false_alarms": hall_analysis["false_alarms"],
            "interception_rate": hall_analysis["interception_rate"],
        }

        return {
            "model": resolved_model,
            "total_queries": len(all_queries),
            "mock_mode": self.mock_mode,
            "baseline": {
                "metrics": bl_metrics,
                "latency_ms": {
                    "llm": compute_stats(bl_llm_lats),
                },
                "results": baseline_records,
            },
            "avicennaguard": {
                "metrics": ag_metrics,
                "latency_ms": {
                    "stage1": compute_stats(ag_s1_lats),
                    "stage2": compute_stats(ag_s2_lats),
                    "total_overhead": compute_stats(ag_tot_lats),
                },
                "by_type": by_type_ag,
                "epistemic_states": dict(epistemic_counts),
                "hallucination_analysis": hall_analysis,
                "results": ag_records,
            },
            "comparison": comparison,
        }

    def run_all(
        self,
        limit: Optional[int] = None,
        models: Optional[list[str]] = None,
        filter_source: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run benchmark evaluation across all configured models.

        Returns:
            Dictionary containing metadata, summaries per model, and full result logs.
        """
        target_models = models or self.models
        all_evaluations: dict[str, Any] = {}
        comparison_table: list[dict[str, Any]] = []

        logger.info("Starting multi-model evaluation across %d models...", len(target_models))

        for model_id in target_models:
            resolved = MODEL_ALIASES.get(model_id, model_id)
            eval_res = self.run_evaluation(
                model_name=resolved,
                limit=limit,
                filter_source=filter_source,
                filter_type=filter_type,
            )
            all_evaluations[resolved] = eval_res

            comparison_table.append(
                {
                    "model": resolved,
                    "baseline_acc": eval_res["baseline"]["metrics"]["accuracy"],
                    "baseline_f1": eval_res["baseline"]["metrics"]["f1"],
                    "guard_acc": eval_res["avicennaguard"]["metrics"]["accuracy"],
                    "guard_f1": eval_res["avicennaguard"]["metrics"]["f1"],
                    "accuracy_gain": eval_res["comparison"]["accuracy_delta"],
                    "hallucinations_intercepted": eval_res["comparison"]["hallucinations_intercepted"],
                    "interception_rate": eval_res["comparison"]["interception_rate"],
                    "avg_overhead_ms": eval_res["avicennaguard"]["latency_ms"]["total_overhead"]["mean"],
                }
            )

        return {
            "metadata": {
                "benchmark_file": self.benchmark_path.name,
                "total_benchmark_queries": len(self.benchmark_loader),
                "mock_mode": self.mock_mode,
                "parser_mode": self.parser_mode,
                "models_evaluated": [MODEL_ALIASES.get(m, m) for m in target_models],
            },
            "comparison_summary": comparison_table,
            "models": all_evaluations,
        }

    @staticmethod
    def save_results(results: dict[str, Any], output_path: Union[str, Path]) -> None:
        """Serialize benchmark results to a formatted JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Evaluation results saved to %s", p)
