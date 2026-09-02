"""
Research evaluation adapter — unifies legacy step scripts with src/AvicennaGuard.

Stage 2 always uses BFSValidator (package). Stage 1 supports regex, llm, or both.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from avicennaguard.core.epistemic_states import EpistemicState
from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.llm_parser import LLMParser
from avicennaguard.parsers.typed_regex import (
    extract_categorical,
    extract_hypothetical,
    extract_taxonomic,
)


def _epistemic_label(state: EpistemicState, used_fallback: bool, covered: bool) -> str:
    """Resolve epistemic state string representation for research evaluation logs."""
    if not covered:
        return EpistemicState.SHAKK.value
    if state == EpistemicState.WAHM:
        return EpistemicState.WAHM.value
    if used_fallback and state == EpistemicState.YAQEEN:
        return EpistemicState.ZANN.value
    return state.value


class ResearchValidator:
    """Drop-in replacement for legacy AvicennaGuardValidator in step2/step4."""

    def __init__(
        self,
        kb_path: str | Path,
        parser_mode: str = "regex",
        model: str = "llama3.2:3b",
    ) -> None:
        """
        Initialize ResearchValidator.

        Args:
            kb_path: Path to knowledge base JSON file.
            parser_mode: Stage 1 parsing mode ('regex', 'llm', or 'both').
            model: Ollama model name for LLM parser.
        """
        self.kb_path = str(kb_path)
        self.kb = KnowledgeBase(kb_path)
        self._validator = BFSValidator(self.kb)
        self.parser_mode = parser_mode
        self._llm_parser = LLMParser(model=model) if parser_mode in ("llm", "both") else None
        self._parse_stats: Dict[str, int] = {
            "success": 0,
            "regex_fallback": 0,
            "parse_failure": 0,
        }

    @property
    def graph(self) -> Dict[str, Set[str]]:
        """Legacy compat: taxonomy adjacency as dict-of-sets."""
        out: Dict[str, Set[str]] = {}
        for child, ancestors in self.kb._raw.get("taxonomies", {}).items():
            out[child] = set(ancestors)
        return out

    def _parse_regex(self, question: str, qtype: str) -> Tuple[Dict[str, Any], str, bool]:
        """Extract logical form using typed regex extraction."""
        q = question.strip()
        used_fallback = False
        if qtype == "taxonomic":
            subj, pred, status = extract_taxonomic(q)
            if not subj:
                return {"type": "non-logical"}, status, False
            return {"type": "taxonomic", "subject": subj, "predicate": pred}, status, used_fallback
        if qtype == "categorical":
            entity, prop, status = extract_categorical(q)
            if not entity:
                return {"type": "non-logical"}, status, False
            return {"type": "categorical", "entity": entity, "property": prop}, status, used_fallback
        if qtype == "hypothetical":
            cond, cons, is_neg, status = extract_hypothetical(q)
            if not cond:
                return {"type": "non-logical"}, status, False
            parsed = {"type": "hypothetical", "condition": cond, "consequence": cons}
            if is_neg:
                parsed["_negate"] = True
            return parsed, status, used_fallback
        return {"type": "non-logical"}, "parse_failure", False

    def _parse_llm(self, question: str) -> Tuple[Dict[str, Any], str, bool]:
        """Extract logical form using LLM parser."""
        assert self._llm_parser is not None
        parsed, used_fallback = self._llm_parser.parse(question)
        status = "regex_fallback" if used_fallback else "success"
        if parsed.get("type") == "non-logical":
            status = "parse_failure"
        return parsed, status, used_fallback

    def validate(self, question: str, qtype: str) -> Dict[str, Any]:
        """
        Validate question for research step evaluation.

        Args:
            question: Natural language query.
            qtype: Query type string.

        Returns:
            Dictionary with graph_answer, epistemic_state, proof, covered, latency, etc.
        """
        t1_start = time.perf_counter()
        used_fallback = False

        if qtype in ("other", "unknown", "non-logical"):
            stage1_ms = (time.perf_counter() - t1_start) * 1000
            return self._shakk_result("Non-logical query type", "parse_failure", stage1_ms, 0.0)

        if self.parser_mode == "llm" and self._llm_parser:
            parsed, parse_status, used_fallback = self._parse_llm(question)
        elif self.parser_mode == "both" and self._llm_parser:
            regex_parsed, regex_status, _ = self._parse_regex(question, qtype)
            llm_parsed, llm_status, llm_fb = self._parse_llm(question)
            parsed = llm_parsed if llm_parsed.get("type") != "non-logical" else regex_parsed
            parse_status = llm_status if llm_parsed.get("type") != "non-logical" else regex_status
            used_fallback = llm_fb or parse_status == "regex_fallback"
        else:
            parsed, parse_status, used_fallback = self._parse_regex(question, qtype)

        stage1_ms = (time.perf_counter() - t1_start) * 1000
        self._parse_stats[parse_status if parse_status in self._parse_stats else "parse_failure"] += 1

        if parsed.get("type") == "non-logical" or parse_status == "parse_failure":
            return self._shakk_result(f"Parse failed: {question[:60]}", parse_status, stage1_ms, 0.0)

        t2_start = time.perf_counter()
        result = self._run_stage2(parsed, qtype)
        stage2_ms = (time.perf_counter() - t2_start) * 1000

        graph_answer = result["graph_answer"]
        ep_state = result["epistemic_state"]
        covered = result["covered"]
        label = _epistemic_label(ep_state, used_fallback, covered)

        return {
            "graph_answer": graph_answer,
            "epistemic_state": label,
            "proof": result["proof"],
            "covered": covered,
            "parse_status": parse_status,
            "parser_mode": self.parser_mode,
            "used_fallback": used_fallback,
            "stage1_ms": round(stage1_ms, 3),
            "stage2_ms": round(stage2_ms, 3),
        }

    def _run_stage2(self, parsed: Dict[str, Any], qtype: str) -> Dict[str, Any]:
        """Dispatch Stage 2 BFS verification."""
        effective_type = parsed.get("type", qtype)
        negate = parsed.pop("_negate", False)

        if effective_type == "taxonomic" or qtype == "taxonomic":
            ans, state, path = self._validator.validate_taxonomic(
                parsed.get("subject", ""), parsed.get("predicate", "")
            )
            covered = state != EpistemicState.SHAKK
            if ans is not None and negate:
                ans = not ans
            proof = f"BFS: {' → '.join(path)} = {ans}" if path else f"BFS taxonomic = {ans}"
            ep = EpistemicState.WAHM if covered and ans is False else state
            if covered and ans is True:
                ep = EpistemicState.YAQEEN
            return {"graph_answer": ans, "epistemic_state": ep, "covered": covered, "proof": proof}

        if effective_type == "categorical" or qtype == "categorical":
            ans, state = self._validator.validate_categorical(
                parsed.get("entity", ""), parsed.get("property", "")
            )
            covered = state != EpistemicState.SHAKK
            proof = f"Property: {parsed.get('entity')}.{parsed.get('property')} = {ans}"
            ep = EpistemicState.WAHM if covered and ans is False else state
            if covered and ans is True:
                ep = EpistemicState.YAQEEN
            return {"graph_answer": ans, "epistemic_state": ep, "covered": covered, "proof": proof}

        if effective_type == "hypothetical" or qtype == "hypothetical":
            ans, state = self._validator.validate_hypothetical(
                parsed.get("condition", ""), parsed.get("consequence", "")
            )
            covered = state != EpistemicState.SHAKK
            if ans is not None and negate:
                ans = not ans
            proof = f"Modus Ponens: {parsed.get('condition')} → {parsed.get('consequence')} = {ans}"
            ep = EpistemicState.WAHM if covered and ans is False else state
            if covered and ans is True:
                ep = EpistemicState.YAQEEN
            return {"graph_answer": ans, "epistemic_state": ep, "covered": covered, "proof": proof}

        return {"graph_answer": None, "epistemic_state": EpistemicState.SHAKK, "covered": False, "proof": "Unknown type"}

    def _shakk_result(self, proof: str, parse_status: str, stage1_ms: float, stage2_ms: float) -> Dict[str, Any]:
        """Construct SHAKK default result."""
        return {
            "graph_answer": None,
            "epistemic_state": EpistemicState.SHAKK.value,
            "proof": proof,
            "covered": False,
            "parse_status": parse_status,
            "parser_mode": self.parser_mode,
            "used_fallback": False,
            "stage1_ms": round(stage1_ms, 3),
            "stage2_ms": round(stage2_ms, 3),
        }

    @property
    def parse_stats(self) -> Dict[str, Any]:
        """
        Stage 1 reliability statistics dictionary.

        Returns:
            Dictionary with counts and percentage rates.
        """
        total = sum(self._parse_stats.values())
        if total == 0:
            return {**self._parse_stats, "total": 0, "success_rate": 0.0, "fallback_rate": 0.0, "failure_rate": 0.0}
        return {
            **self._parse_stats,
            "total": total,
            "success_rate": round(self._parse_stats["success"] / total * 100, 1),
            "fallback_rate": round(self._parse_stats["regex_fallback"] / total * 100, 1),
            "failure_rate": round(self._parse_stats["parse_failure"] / total * 100, 1),
        }
