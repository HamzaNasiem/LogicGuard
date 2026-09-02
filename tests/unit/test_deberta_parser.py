"""Unit tests for Stage 1 DeBERTa Fast Parser and Regex Fallback."""

import time
from unittest.mock import MagicMock
import pytest
import torch

from avicennaguard.parsers.deberta_parser import DebertaParser, LABEL_MAP


@pytest.fixture
def fallback_parser() -> DebertaParser:
    """Fixture providing DebertaParser operating in pure regex fallback mode."""
    return DebertaParser(model_path=None)


class TestDebertaFallbackTaxonomic:
    def test_are_all_x_a_y(self, fallback_parser):
        r = fallback_parser.parse("Are all dogs mammals?")
        assert r["type"] == "taxonomic"
        assert r["subject"] == "dogs"
        assert r["predicate"] == "mammals"
        assert r["confidence"] == 1.0
        assert r["method"] == "regex_fallback"

    def test_is_x_a_y(self, fallback_parser):
        r = fallback_parser.parse("Is a spider an insect?")
        assert r["type"] == "taxonomic"
        assert r["subject"] == "spider"
        assert r["predicate"] == "insect"
        assert r["method"] == "regex_fallback"

    def test_every_x_is_y(self, fallback_parser):
        r = fallback_parser.parse("Every square is a rectangle?")
        assert r["type"] == "taxonomic"
        assert r["subject"] == "square"
        assert r["predicate"] == "rectangle"

    def test_do_all_x_belong_to_y(self, fallback_parser):
        r = fallback_parser.parse("Do all dolphins belong to mammals?")
        assert r["type"] == "taxonomic"
        assert r["subject"] == "dolphins"
        assert r["predicate"] == "mammals"


class TestDebertaFallbackCategorical:
    def test_do_all_x_have_y(self, fallback_parser):
        r = fallback_parser.parse("Do all birds have feathers?")
        assert r["type"] == "categorical"
        assert r["subject"] == "birds"
        assert r["predicate"] == "feathers"
        assert r["confidence"] == 1.0
        assert r["method"] == "regex_fallback"

    def test_does_x_have_y(self, fallback_parser):
        r = fallback_parser.parse("Does a fish have hair?")
        assert r["type"] == "categorical"
        assert r["subject"] == "fish"
        assert r["predicate"] == "hair"
        assert r["method"] == "regex_fallback"

    def test_do_x_possess_y(self, fallback_parser):
        r = fallback_parser.parse("Do insects possess six_legs?")
        assert r["type"] == "categorical"
        assert r["subject"] == "insects"
        assert r["predicate"] == "six_legs"


class TestDebertaFallbackHypothetical:
    def test_if_x_then_y(self, fallback_parser):
        r = fallback_parser.parse("If water freezes, does it become ice?")
        assert r["type"] == "hypothetical"
        assert r["condition"] == "water_freezes"
        assert r["consequence"] == "become_ice"
        assert r["confidence"] == 1.0
        assert r["method"] == "regex_fallback"

    def test_when_x_y(self, fallback_parser):
        r = fallback_parser.parse("When metal is heated, does it expand?")
        assert r["type"] == "hypothetical"
        assert r["condition"] == "metal_is_heated"
        assert r["consequence"] == "expand"
        assert r["method"] == "regex_fallback"


class TestDebertaFallbackNonLogical:
    def test_general_open_domain_question(self, fallback_parser):
        r = fallback_parser.parse("What is the capital of France?")
        assert r["type"] == "non-logical"
        assert r["subject"] == ""
        assert r["predicate"] == ""
        assert r["condition"] == ""
        assert r["consequence"] == ""
        assert r["confidence"] == 1.0
        assert r["method"] == "regex_fallback"

    def test_conversational_query(self, fallback_parser):
        r = fallback_parser.parse("Tell me a funny joke about robots.")
        assert r["type"] == "non-logical"
        assert r["method"] == "regex_fallback"

    def test_empty_string(self, fallback_parser):
        r = fallback_parser.parse("")
        assert r["type"] == "non-logical"
        assert r["method"] == "regex_fallback"

    def test_none_and_whitespace(self, fallback_parser):
        r1 = fallback_parser.parse("   ")
        assert r1["type"] == "non-logical"
        r2 = fallback_parser.parse(None)
        assert r2["type"] == "non-logical"


class TestDebertaOutputSchema:
    def test_all_schema_keys_present(self, fallback_parser):
        queries = [
            "Are all dogs mammals?",
            "Do all birds have feathers?",
            "If water freezes, does it become ice?",
            "What is quantum entanglement?",
        ]
        required_keys = {"type", "subject", "predicate", "condition", "consequence", "confidence", "method"}

        for q in queries:
            res = fallback_parser.parse(q)
            assert isinstance(res, dict)
            assert set(res.keys()) == required_keys
            assert res["type"] in ("taxonomic", "categorical", "hypothetical", "non-logical")
            assert isinstance(res["subject"], str)
            assert isinstance(res["predicate"], str)
            assert isinstance(res["condition"], str)
            assert isinstance(res["consequence"], str)
            assert isinstance(res["confidence"], float)
            assert res["method"] in ("deberta", "regex_fallback")


class TestDebertaMissingWeightsAndRobustness:
    def test_missing_model_path_graceful_fallback(self):
        # Initializing with non-existent path should not raise an error
        parser = DebertaParser(model_path="non_existent/directory/or/checkpoint/model")
        assert parser.model is None

        # Parsing should seamlessly succeed via regex fallback
        r = parser.parse("Are all dogs mammals?")
        assert r["type"] == "taxonomic"
        assert r["subject"] == "dogs"
        assert r["predicate"] == "mammals"
        assert r["method"] == "regex_fallback"

    def test_sub_millisecond_fallback_latency(self, fallback_parser):
        # Warmup
        fallback_parser.parse("Are all dogs mammals?")

        # Measure 1000 parses
        n_iters = 1000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fallback_parser.parse("Are all dogs mammals?")
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / n_iters

        # Sub-millisecond guarantee (should be well below 0.1ms per query)
        assert avg_ms < 1.0, f"Average latency {avg_ms:.4f}ms exceeded 1.0ms limit"

    def test_parse_stats_tracking(self, fallback_parser):
        fallback_parser.parse("Are all dogs mammals?")
        fallback_parser.parse("Do all birds have feathers?")
        fallback_parser.parse("What is the capital of France?")

        stats = fallback_parser.parse_stats
        assert stats["total"] >= 3
        assert stats["regex_fallback"] >= 3
        assert stats["fallback_rate"] == 100.0


class TestDebertaNeuralMockPath:
    def test_mocked_deberta_inference_taxonomic(self):
        parser = DebertaParser(model_path=None)
        parser.device = "cpu"

        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        # Logits predicting class 0 (taxonomic) with high confidence
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[5.0, 0.1, 0.1, 0.1]])
        mock_model.return_value = mock_outputs

        parser.tokenizer = mock_tokenizer
        parser.model = mock_model

        r = parser.parse("Is a wolf classified as a canine?")
        assert r["type"] == "taxonomic"
        assert r["subject"] == "wolf"
        assert r["predicate"] == "canine"
        assert r["method"] == "deberta"
        assert r["confidence"] > 0.9

    def test_mocked_deberta_low_confidence_fallback(self):
        parser = DebertaParser(model_path=None, confidence_threshold=0.8)
        parser.device = "cpu"

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        mock_model = MagicMock()
        # Uniform logits -> ~0.25 confidence (below 0.8 threshold)
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        mock_model.return_value = mock_outputs

        parser.tokenizer = mock_tokenizer
        parser.model = mock_model

        r = parser.parse("Are all dogs mammals?")
        # Should fall back to regex due to low confidence
        assert r["type"] == "taxonomic"
        assert r["subject"] == "dogs"
        assert r["predicate"] == "mammals"
        assert r["method"] == "regex_fallback"


class TestDebertaTrainedSklearnModel:
    """Integration and unit tests for DebertaParser using trained joblib artifact."""

    MODEL_PATH = "models/stage1_classifier.joblib"

    def test_load_trained_joblib_model(self):
        import os
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip(f"Model artifact not found at {self.MODEL_PATH}")

        parser = DebertaParser(model_path=self.MODEL_PATH)
        assert parser.model is not None
        assert parser.model_backend == "sklearn"

    def test_trained_model_auto_discovery(self):
        import os
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip(f"Model artifact not found at {self.MODEL_PATH}")

        parser = DebertaParser(model_path="auto")
        assert parser.model is not None
        assert parser.model_backend == "sklearn"

    def test_trained_model_classification_and_slots(self):
        import os
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip(f"Model artifact not found at {self.MODEL_PATH}")

        parser = DebertaParser(model_path=self.MODEL_PATH, confidence_threshold=0.5)

        # Categorical query
        r1 = parser.parse("Do all birds have feathers?")
        assert r1["type"] == "categorical"
        assert r1["subject"] == "birds"
        assert r1["predicate"] == "feathers"
        assert r1["confidence"] > 0.9

        # Hypothetical query
        r2 = parser.parse("If water freezes, does it become ice?")
        assert r2["type"] == "hypothetical"
        assert r2["condition"] == "water_freezes"
        assert r2["consequence"] == "become_ice"
        assert r2["confidence"] > 0.8

        # Non-logical query
        r3 = parser.parse("What is the capital of France?")
        assert r3["type"] == "non-logical"
        assert r3["confidence"] > 0.9

    def test_trained_model_throughput_latency(self):
        import os
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip(f"Model artifact not found at {self.MODEL_PATH}")

        parser = DebertaParser(model_path=self.MODEL_PATH)

        # Warmup
        for _ in range(20):
            parser.parse("Do all birds have feathers?")

        # Measure 200 parses
        n_iters = 200
        t0 = time.perf_counter()
        for _ in range(n_iters):
            parser.parse("Do all birds have feathers?")
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / n_iters

        # Sub-30 millisecond Stage 1 SLA guarantee for neural/sklearn classifier
        assert avg_ms < 30.0, f"Average latency {avg_ms:.4f}ms exceeded 30.0ms threshold"

