"""
Unit tests for Dynamic ContextGraphBuilder and Contextual Syllogistic Validation.
"""

import pytest
import networkx as nx
from avicennaguard.core.epistemic_states import EpistemicState
from avicennaguard.kb.context_builder import ContextGraphBuilder
from avicennaguard.kb.validator import BFSValidator


def test_taxonomic_chain_deduction():
    builder = ContextGraphBuilder()
    premises = [
        "All felines are mammals.",
        "Every tiger is a feline.",
        "Tony is a tiger."
    ]
    kb = builder.build_unified_kb(premises)
    v = BFSValidator(kb)
    
    # Positive deduction (Tony -> tiger -> feline -> mammal)
    ans, state, path = v.validate_taxonomic("tony", "mammal")
    assert ans is True
    assert state == EpistemicState.YAQEEN
    assert "tony" in path and "mammal" in path


def test_disjointness_contradiction():
    builder = ContextGraphBuilder()
    premises = [
        "All mammals are warm-blooded animals.",
        "No mammal is a fish.",
        "A dolphin is a mammal."
    ]
    kb = builder.build_unified_kb(premises)
    v = BFSValidator(kb)
    
    # Contradiction: dolphin cannot be a fish
    ans, state, path = v.validate_taxonomic("dolphin", "fish")
    assert ans is False
    assert state == EpistemicState.YAQEEN


def test_property_inheritance():
    builder = ContextGraphBuilder()
    premises = [
        "All birds have feathers.",
        "Every eagle is a bird.",
        "Aquila is an eagle."
    ]
    kb = builder.build_unified_kb(premises)
    v = BFSValidator(kb)
    
    # Property inheritance (Aquila inherits feathers from bird)
    ans, state = v.validate_categorical("aquila", "feathers")
    assert ans is True
    assert state == EpistemicState.YAQEEN


def test_hypothetical_modus_ponens():
    builder = ContextGraphBuilder()
    premises = [
        "If water freezes, then it expands.",
        "If ice melts, then liquid forms."
    ]
    kb = builder.build_unified_kb(premises)
    v = BFSValidator(kb)
    
    ans, state = v.validate_hypothetical("water_freezes", "it_expands")
    assert ans is True
    assert state == EpistemicState.YAQEEN


def test_open_world_safe_deferral():
    builder = ContextGraphBuilder()
    premises = [
        "All dogs are loyal.",
        "Fido is a dog."
    ]
    kb = builder.build_unified_kb(premises)
    v = BFSValidator(kb)
    
    # Unmentioned entity / property outside scope -> SHAKK (0 False Alarms)
    ans, state, _ = v.validate_taxonomic("fido", "galaxy")
    assert ans is None
    assert state == EpistemicState.SHAKK
