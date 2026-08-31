"""Unit tests for epistemic state definitions and ValidatorResult."""

import pytest
from logicguard.core.epistemic_states import EpistemicState, QueryType, ValidatorResult


def test_epistemic_state_values():
    assert EpistemicState.YAQEEN == "YAQEEN"
    assert EpistemicState.WAHM   == "WAHM"
    assert EpistemicState.SHAKK  == "SHAKK"
    assert EpistemicState.ZANN   == "ZANN"


def test_query_type_values():
    assert QueryType.TAXONOMIC    == "taxonomic"
    assert QueryType.CATEGORICAL  == "categorical"
    assert QueryType.HYPOTHETICAL == "hypothetical"
    assert QueryType.NON_LOGICAL  == "non-logical"


def test_validator_result_to_dict():
    result = ValidatorResult(
        epistemic_state=EpistemicState.YAQEEN,
        graph_answer=True,
        llm_answer="no",
        final_answer="yes",
        covered=True,
        intercepted=True,
        query_type=QueryType.TAXONOMIC,
        subject="dog",
        predicate="mammal",
        path=["dog", "canine", "mammal"],
        latency_stage1=12.5,
        latency_stage2=0.3,
    )
    d = result.to_dict()

    assert d["epistemic_state"]   == "YAQEEN"
    assert d["intercepted"]       is True
    assert d["path"]              == ["dog", "canine", "mammal"]
    assert d["latency_ms"]["total"] == pytest.approx(12.8, abs=0.1)
