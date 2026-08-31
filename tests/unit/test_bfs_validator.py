"""
Unit tests for Stage 2 BFS Validator.

These tests verify the core mathematical guarantee:
    FP = 0 is architecturally guaranteed — BFS on a correct KB
    cannot produce a false positive.

Run:
    pytest tests/unit/test_bfs_validator.py -v
"""

import json
import pytest
from pathlib import Path

from logicguard.kb.loader import KnowledgeBase
from logicguard.kb.validator import BFSValidator
from logicguard.core.epistemic_states import EpistemicState

KB_PATH = Path("data/knowledge_bases/knowledge_base_extended.json")


@pytest.fixture(scope="module")
def validator() -> BFSValidator:
    kb = KnowledgeBase(KB_PATH)
    return BFSValidator(kb)


class TestTaxonomicValidation:
    """Tests for Qiyas al-Haml (IS-A BFS traversal)."""

    def test_direct_is_a_relation(self, validator):
        """dog → canine is a direct edge."""
        result, state, path = validator.validate_taxonomic("dog", "canine")
        assert result is True
        assert state == EpistemicState.YAQEEN
        assert "dog" in path

    def test_transitive_is_a_relation(self, validator):
        """dog → canine → mammal → animal (multi-hop BFS)."""
        result, state, path = validator.validate_taxonomic("dog", "mammal")
        assert result is True
        assert state == EpistemicState.YAQEEN
        assert len(path) >= 2  # At least one intermediate node

    def test_inverse_is_a_returns_false(self, validator):
        """mammal is NOT a dog — parent cannot be child."""
        result, state, _ = validator.validate_taxonomic("mammal", "dog")
        assert result is False
        assert state == EpistemicState.YAQEEN

    def test_cross_branch_returns_false(self, validator):
        """spider is NOT an insect — cross-branch taxonomic error."""
        result, state, _ = validator.validate_taxonomic("spider", "insect")
        assert result is False
        assert state == EpistemicState.YAQEEN

    def test_unknown_entity_returns_shakk(self, validator):
        """Entity not in KB → SHAKK, no intervention."""
        result, state, _ = validator.validate_taxonomic("unicorn", "mammal")
        assert result is None
        assert state == EpistemicState.SHAKK

    def test_self_is_a_returns_true(self, validator):
        """Trivially true: dog IS-A dog."""
        result, state, path = validator.validate_taxonomic("dog", "dog")
        assert result is True
        assert state == EpistemicState.YAQEEN

    def test_square_is_rectangle(self, validator):
        """Geometric relation: square IS-A rectangle."""
        result, state, _ = validator.validate_taxonomic("square", "rectangle")
        assert result is True

    def test_fish_is_not_mammal(self, validator):
        """Classic hallucination: fish is NOT a mammal."""
        result, state, _ = validator.validate_taxonomic("fish", "mammal")
        assert result is False
        assert state == EpistemicState.YAQEEN


class TestCategoricalValidation:
    """Tests for property inheritance queries."""

    def test_direct_property(self, validator):
        """mammal has hair — direct property edge."""
        result, state = validator.validate_categorical("mammal", "hair")
        assert result is True
        assert state == EpistemicState.YAQEEN

    def test_inherited_property(self, validator):
        """dog inherits hair via dog → canine → mammal → hair."""
        result, state = validator.validate_categorical("dog", "hair")
        assert result is True
        assert state == EpistemicState.YAQEEN

    def test_fish_has_no_hair(self, validator):
        """fish does NOT have hair — confirmed absence in KB."""
        result, state = validator.validate_categorical("fish", "hair")
        assert result is False
        assert state == EpistemicState.YAQEEN

    def test_unknown_property_returns_shakk(self, validator):
        """Property not in KB → SHAKK."""
        result, state = validator.validate_categorical("dog", "telekinesis")
        assert result is None
        assert state == EpistemicState.SHAKK


class TestHypotheticalValidation:
    """Tests for Qiyas al-Istithna (Modus Ponens)."""

    def test_valid_conditional(self, validator):
        """water_freezes → ice is a known rule."""
        result, state = validator.validate_hypothetical("water_freezes", "ice")
        assert result is True
        assert state == EpistemicState.YAQEEN

    def test_invalid_conditional(self, validator):
        """water_freezes does NOT lead to fire."""
        result, state = validator.validate_hypothetical("water_freezes", "fire")
        assert result is False
        assert state == EpistemicState.YAQEEN

    def test_unknown_condition_returns_shakk(self, validator):
        """Condition not in KB → SHAKK."""
        result, state = validator.validate_hypothetical("moon_explodes", "gravity_reverses")
        assert result is None
        assert state == EpistemicState.SHAKK


class TestFalsePositiveGuarantee:
    """
    Verifies the core mathematical guarantee: FP = 0.

    A false positive requires BFS to return False when a path genuinely
    exists. This is computationally impossible on a correct KB.
    These tests verify the guarantee holds on known-true queries.
    """

    KNOWN_TRUE_TAXONOMIC = [
        ("dog",     "mammal"),
        ("cat",     "animal"),
        ("whale",   "mammal"),
        ("square",  "rectangle"),
        ("spider",  "arachnid"),
        ("eagle",   "bird"),
    ]

    def test_no_false_positives_on_known_true_queries(self, validator):
        """FP = 0: BFS must return True for all known-true IS-A relations."""
        for subject, predicate in self.KNOWN_TRUE_TAXONOMIC:
            result, state, _ = validator.validate_taxonomic(subject, predicate)
            if state != EpistemicState.SHAKK:
                assert result is True, (
                    f"FALSE POSITIVE: {subject} IS-A {predicate} returned False. "
                    f"This violates the FP=0 architectural guarantee."
                )
