"""Unit tests for deterministic regex fallback parser."""

import pytest
from avicennaguard.parsers.regex_parser import RegexParser


@pytest.fixture
def parser() -> RegexParser:
    return RegexParser()


class TestTaxonomicPatterns:
    def test_are_all_x_a_y(self, parser):
        r = parser.parse("Are all dogs mammals?")
        assert r["type"]      == "taxonomic"
        assert r["subject"]   == "dogs"
        assert r["predicate"] == "mammals"

    def test_is_x_a_y(self, parser):
        r = parser.parse("Is a spider an insect?")
        assert r["type"] == "taxonomic"

    def test_every_x_is_y(self, parser):
        r = parser.parse("Every square is a rectangle?")
        assert r["type"] == "taxonomic"


class TestCategoricalPatterns:
    def test_do_all_x_have_y(self, parser):
        r = parser.parse("Do all birds have feathers?")
        assert r["type"]     == "categorical"
        assert r["entity"]   == "birds"
        assert r["property"] == "feathers"

    def test_does_x_have_y(self, parser):
        r = parser.parse("Does a fish have hair?")
        assert r["type"] == "categorical"


class TestHypotheticalPatterns:
    def test_if_x_then_y(self, parser):
        r = parser.parse("If water freezes, does it become ice?")
        assert r["type"] == "hypothetical"

    def test_when_x_y(self, parser):
        r = parser.parse("When metal is heated, does it expand?")
        assert r["type"] == "hypothetical"


class TestNonLogical:
    def test_general_question_returns_non_logical(self, parser):
        r = parser.parse("What is the capital of France?")
        assert r["type"] == "non-logical"

    def test_empty_string(self, parser):
        r = parser.parse("")
        assert r["type"] == "non-logical"
