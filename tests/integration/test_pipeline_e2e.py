"""
End-to-end integration tests for the full LogicGuard pipeline.

These tests verify the complete two-stage pipeline behavior:
Stage 1 (regex parser) → Stage 2 (BFS validator) → ValidatorResult

Note: These tests use the regex fallback parser (no Ollama dependency)
so they can run in CI without a local LLM server.

Run:
    pytest tests/integration/test_pipeline_e2e.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from logicguard.kb.loader import KnowledgeBase
from logicguard.pipeline.logicguard import LogicGuard
from logicguard.core.epistemic_states import EpistemicState

KB_PATH = Path("data/knowledge_bases/knowledge_base_extended.json")


@pytest.fixture(scope="module")
def lg() -> LogicGuard:
    kb = KnowledgeBase(KB_PATH)
    return LogicGuard(kb, model="llama3.2:3b")


class TestHallucinationInterception:
    """Tests for WAHM state — hallucination caught and corrected."""

    def test_intercepts_fish_hair_hallucination(self, lg):
        """LLM says 'yes' but KB proves fish have no hair → WAHM intercept."""
        with patch.object(lg._parser, "_call_llm",
                          return_value=({"type": "categorical", "entity": "fish", "property": "hair"}, False)):
            result = lg.validate("Do all fish have hair?", llm_answer="yes")

        assert result.intercepted         is True
        assert result.epistemic_state     == EpistemicState.WAHM
        assert result.final_answer        == "no"
        assert result.llm_answer          == "yes"

    def test_intercepts_spider_insect_hallucination(self, lg):
        """LLM says 'yes' but spider IS-A insect is false → WAHM intercept."""
        with patch.object(lg._parser, "_call_llm",
                          return_value=({"type": "taxonomic", "subject": "spider", "predicate": "insect"}, False)):
            result = lg.validate("Are all spiders insects?", llm_answer="yes")

        assert result.intercepted     is True
        assert result.epistemic_state == EpistemicState.WAHM


class TestYaqeenState:
    """Tests for YAQEEN state — KB confirms LLM answer."""

    def test_dog_is_mammal_yaqeen(self, lg):
        """LLM correct: dog IS-A mammal confirmed by BFS."""
        with patch.object(lg._parser, "_call_llm",
                          return_value=({"type": "taxonomic", "subject": "dog", "predicate": "mammal"}, False)):
            result = lg.validate("Are all dogs mammals?", llm_answer="yes")

        assert result.intercepted     is False
        assert result.epistemic_state == EpistemicState.YAQEEN
        assert result.final_answer    == "yes"
        assert len(result.path)       >= 2


class TestShakkState:
    """Tests for SHAKK state — entity not in KB, defer to LLM."""

    def test_unknown_entity_defers_to_llm(self, lg):
        """Entity 'platypus' not in KB → SHAKK, LLM answer preserved."""
        with patch.object(lg._parser, "_call_llm",
                          return_value=({"type": "taxonomic", "subject": "platypus", "predicate": "mammal"}, False)):
            result = lg.validate("Is a platypus a mammal?", llm_answer="yes")

        assert result.intercepted     is False
        assert result.epistemic_state == EpistemicState.SHAKK
        assert result.final_answer    == "yes"  # LLM answer preserved

    def test_non_logical_query_defers_to_llm(self, lg):
        """Open-domain question → non-logical → SHAKK."""
        with patch.object(lg._parser, "_call_llm",
                          return_value=({"type": "non-logical"}, False)):
            result = lg.validate("Who won the 2022 World Cup?", llm_answer="Argentina")

        assert result.epistemic_state == EpistemicState.SHAKK
        assert result.intercepted     is False


class TestFalsePositiveGuarantee:
    """
    Verifies FP = 0 across a set of known-correct LLM answers.

    A false positive would be: LLM correct, LogicGuard overrides with wrong answer.
    This must NEVER happen. These tests guard against KB corruption.
    """

    KNOWN_CORRECT_PAIRS = [
        ("taxonomic",   {"subject": "dog",    "predicate": "mammal"},   "yes"),
        ("taxonomic",   {"subject": "eagle",  "predicate": "bird"},     "yes"),
        ("taxonomic",   {"subject": "square", "predicate": "rectangle"},"yes"),
        ("categorical", {"entity":  "mammal", "property": "hair"},      "yes"),
        ("categorical", {"entity":  "bird",   "property": "feathers"},  "yes"),
    ]

    def test_no_false_positives(self, lg):
        for qtype, entities, correct_answer in self.KNOWN_CORRECT_PAIRS:
            parsed = {"type": qtype, **entities}
            with patch.object(lg._parser, "_call_llm", return_value=(parsed, False)):
                question = f"Test query for {entities}"
                result = lg.validate(question, llm_answer=correct_answer)

            assert not result.intercepted, (
                f"FALSE POSITIVE on {entities}: "
                f"LLM was correct ({correct_answer}) but LogicGuard intercepted. "
                f"Final answer: {result.final_answer}"
            )
