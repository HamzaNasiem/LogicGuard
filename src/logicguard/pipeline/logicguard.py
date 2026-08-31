"""
LogicGuard Two-Stage Pipeline — Main Entry Point.

This class orchestrates the full pipeline:
    Stage 1: Neural Semantic Parser (probabilistic — LLM constrained to JSON)
    Stage 2: BFS Graph Validator    (deterministic — pure graph traversal)

Design constraint (immutable):
    No probabilistic computation enters Stage 2.
    No symbolic computation occurs in Stage 1.
    The boundary between stages is a structured JSON proposition.

Usage:
    from logicguard.pipeline.logicguard import LogicGuard
    from logicguard.kb.loader import KnowledgeBase

    kb = KnowledgeBase("data/knowledge_bases/knowledge_base_extended.json")
    lg = LogicGuard(kb, model="llama3.2:3b")

    result = lg.validate("Are all dogs mammals?", llm_answer="yes")
    print(result.epistemic_state)   # EpistemicState.YAQEEN
    print(result.intercepted)       # False (LLM was correct)
    print(result.path)              # ['dog', 'canine', 'mammal']
"""

import logging
import time
from typing import Optional

from logicguard.core.epistemic_states import EpistemicState, QueryType, ValidatorResult
from logicguard.kb.loader import KnowledgeBase
from logicguard.kb.validator import BFSValidator
from logicguard.parsers.llm_parser import LLMParser

logger = logging.getLogger(__name__)


class LogicGuard:
    """
    Main LogicGuard pipeline. Thread-safe, stateless per-call.

    Args:
        kb:     Pre-loaded KnowledgeBase instance.
        model:  Ollama model name for Stage 1 parser.
    """

    def __init__(self, kb: KnowledgeBase, model: str = "llama3.2:3b"):
        self.kb        = kb
        self._parser   = LLMParser(model=model)
        self._validator = BFSValidator(kb)

    def validate(
        self,
        question: str,
        llm_answer: Optional[str] = None,
    ) -> ValidatorResult:
        """
        Run the full two-stage pipeline on a question.

        Args:
            question:   Natural language question.
            llm_answer: Raw LLM answer ("yes"/"no") before LogicGuard.
                        If None, no interception comparison is made.

        Returns:
            ValidatorResult with epistemic_state, final_answer, path, latency.
        """
        # ── Stage 1: Neural Semantic Parsing ─────────────────────────────
        t1_start = time.perf_counter()
        parsed, used_fallback = self._parser.parse(question)
        latency_stage1 = (time.perf_counter() - t1_start) * 1000

        query_type = QueryType(parsed.get("type", "non-logical"))

        if query_type == QueryType.NON_LOGICAL:
            # No logical structure — defer entirely to LLM (SHAKK)
            return ValidatorResult(
                epistemic_state=EpistemicState.SHAKK,
                graph_answer=None,
                llm_answer=llm_answer or "",
                final_answer=llm_answer or "",
                covered=False,
                intercepted=False,
                query_type=query_type,
                latency_stage1=latency_stage1,
                latency_stage2=0.0,
                used_fallback=used_fallback,
            )

        # ── Stage 2: Deterministic BFS Validation ────────────────────────
        t2_start = time.perf_counter()
        graph_answer, epistemic_state, path = self._run_stage2(parsed, query_type)
        latency_stage2 = (time.perf_counter() - t2_start) * 1000

        covered = epistemic_state != EpistemicState.SHAKK

        # ── Hallucination Interception Logic ─────────────────────────────
        # graph_answer overrides llm_answer when covered=True
        # An interception occurs when LLM was wrong and KB is correct
        llm_bool = self._parse_bool(llm_answer)
        intercepted = (
            covered
            and llm_bool is not None
            and graph_answer is not None
            and llm_bool != graph_answer
        )

        if intercepted:
            epistemic_state = EpistemicState.WAHM
            final_answer = "yes" if graph_answer else "no"
            logger.info("WAHM intercept — Q: %s | LLM: %s | KB: %s", question, llm_answer, final_answer)
        elif covered and graph_answer is not None:
            if used_fallback:
                epistemic_state = EpistemicState.ZANN
            else:
                epistemic_state = EpistemicState.YAQEEN
            final_answer = "yes" if graph_answer else "no"
        else:
            final_answer = llm_answer or ""

        subject   = parsed.get("subject") or parsed.get("entity") or parsed.get("condition", "")
        predicate = parsed.get("predicate") or parsed.get("property") or parsed.get("consequence", "")

        return ValidatorResult(
            epistemic_state=epistemic_state,
            graph_answer=graph_answer,
            llm_answer=llm_answer or "",
            final_answer=final_answer,
            covered=covered,
            intercepted=intercepted,
            query_type=query_type,
            subject=subject,
            predicate=predicate,
            path=path,
            latency_stage1=latency_stage1,
            latency_stage2=latency_stage2,
            used_fallback=used_fallback,
        )

    def _run_stage2(
        self, parsed: dict, query_type: QueryType
    ) -> tuple[Optional[bool], EpistemicState, list[str]]:
        """Dispatch to the appropriate BFS validator method."""
        if query_type == QueryType.TAXONOMIC:
            result, state, path = self._validator.validate_taxonomic(
                subject=parsed.get("subject", ""),
                predicate=parsed.get("predicate", ""),
            )
            return result, state, path

        elif query_type == QueryType.CATEGORICAL:
            result, state = self._validator.validate_categorical(
                entity=parsed.get("entity", ""),
                prop=parsed.get("property", ""),
            )
            return result, state, []

        elif query_type == QueryType.HYPOTHETICAL:
            result, state = self._validator.validate_hypothetical(
                condition=parsed.get("condition", ""),
                consequence=parsed.get("consequence", ""),
            )
            return result, state, []

        return None, EpistemicState.SHAKK, []

    @staticmethod
    def _parse_bool(answer: Optional[str]) -> Optional[bool]:
        if answer is None:
            return None
        a = answer.strip().lower()
        if a in ("yes", "true", "1"):
            return True
        if a in ("no", "false", "0"):
            return False
        return None
