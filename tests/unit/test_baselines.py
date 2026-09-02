"""
Unit Tests for SOTA Baselines (SelfCheckGPT, Dense RAG, Logic-LM).
==================================================================
Verifies deterministic mock execution, metrics computation, and output schema integrity.
"""

import json
from pathlib import Path
import tempfile
import pytest

from avicennaguard.baselines.dense_rag import (
    DenseRAGBaseline,
    DenseRAGResult,
    SparseTFIDFRetriever,
    kb_to_facts,
)
from avicennaguard.baselines.logic_lm import (
    LogicLMBaseline,
    LogicLMResult,
    RuleBasedLogicTranslator,
    SymbolicLogicSolver,
)
from avicennaguard.baselines.metrics import (
    compute_classification_metrics,
    compute_group_metrics,
    format_metrics_summary,
    parse_bool_answer,
)
from avicennaguard.baselines.selfcheckgpt import (
    SelfCheckGPTBaseline,
    SelfCheckGPTResult,
)
from avicennaguard.data.benchmark_loader import BenchmarkLoader
from avicennaguard.kb.loader import KnowledgeBase


# ── Sample Fixtures ───────────────────────────────────────────────────────────

SAMPLE_BENCHMARK_QUERIES = [
    {
        "id": "folio_001",
        "source": "FOLIO",
        "question": "Is it true that if a dog is an animal, then a dog is a living thing?",
        "ground_truth": True,
        "query_type": "taxonomic",
        "difficulty": "easy",
    },
    {
        "id": "curated_002",
        "source": "Curated_Gold",
        "question": "Are all spiders insects?",
        "ground_truth": False,
        "query_type": "taxonomic",
        "difficulty": "easy",
    },
    {
        "id": "proofwriter_003",
        "source": "ProofWriter",
        "question": "Do all dogs have hair?",
        "ground_truth": True,
        "query_type": "categorical",
        "difficulty": "medium",
    },
    {
        "id": "curated_004",
        "source": "Curated_Gold",
        "question": "If water freezes, does it become ice?",
        "ground_truth": True,
        "query_type": "hypothetical",
        "difficulty": "easy",
    },
    {
        "id": "ood_005",
        "source": "TruthfulQA_OOD",
        "question": "What happens if you smash a mirror?",
        "ground_truth": "OOD",
        "query_type": "ood",
        "difficulty": "hard",
    },
]


# ── 1. Metrics & Evaluation Unit Tests ────────────────────────────────────────

class TestBaselineMetrics:
    """Test suite for metrics calculation and formatting."""

    def test_parse_bool_answer_variants(self):
        assert parse_bool_answer(True) is True
        assert parse_bool_answer("yes") is True
        assert parse_bool_answer("YES") is True
        assert parse_bool_answer("true") is True
        assert parse_bool_answer("PROVEN_TRUE") is True
        assert parse_bool_answer(False) is False
        assert parse_bool_answer("no") is False
        assert parse_bool_answer("false") is False
        assert parse_bool_answer("PROVEN_FALSE") is False
        assert parse_bool_answer("OOD") is None
        assert parse_bool_answer("UNKNOWN") is None
        assert parse_bool_answer(None) is None

    def test_compute_metrics_perfect_predictions(self):
        preds = [True, False, True, False]
        gts = [True, False, True, False]
        m = compute_classification_metrics(preds, gts)

        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0, "total": 4}

    def test_compute_metrics_with_errors(self):
        preds = [True, True, False, False]
        gts = [True, False, True, False]
        m = compute_classification_metrics(preds, gts)

        assert m["confusion_matrix"]["tp"] == 1
        assert m["confusion_matrix"]["fp"] == 1
        assert m["confusion_matrix"]["tn"] == 1
        assert m["confusion_matrix"]["fn"] == 1
        assert m["accuracy"] == 0.5
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["f1"] == 0.5

    def test_compute_metrics_handles_zero_division(self):
        preds = [False, False]
        gts = [False, False]
        m = compute_classification_metrics(preds, gts)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["accuracy"] == 1.0
        assert m["confusion_matrix"]["tn"] == 2

    def test_compute_group_metrics(self):
        group_m = compute_group_metrics(SAMPLE_BENCHMARK_QUERIES, group_key="query_type")
        assert "taxonomic" in group_m
        assert "categorical" in group_m
        assert "hypothetical" in group_m
        assert "ood" in group_m
        assert group_m["taxonomic"]["count"] == 2

    def test_format_metrics_summary_string(self):
        m = compute_classification_metrics([True, False], [True, False])
        summary = format_metrics_summary("TestBaseline", m)
        assert "EVALUATION SUMMARY: TESTBASELINE" in summary
        assert "Accuracy:" in summary
        assert "Confusion Matrix:" in summary


# ── 2. SelfCheckGPT Unit Tests ────────────────────────────────────────────────

class TestSelfCheckGPTBaseline:
    """Test suite for SelfCheckGPT stochastic consistency baseline."""

    def test_initialization_and_mock_mode(self):
        baseline = SelfCheckGPTBaseline(n_samples=5, mock=True, seed=42)
        assert baseline.mock is True
        assert baseline.n_samples == 5

    def test_predict_single_query(self):
        baseline = SelfCheckGPTBaseline(n_samples=5, mock=True, seed=42)
        res = baseline.predict(
            question="Are all dogs mammals?",
            query_id="test_001",
            ground_truth=True,
            query_type="taxonomic",
        )
        assert isinstance(res, SelfCheckGPTResult)
        assert res.query_id == "test_001"
        assert len(res.samples) == 5
        assert 0.0 <= res.confidence <= 1.0
        assert 0.0 <= res.consistency_score <= 1.0
        assert isinstance(res.prediction, bool)
        assert res.final_answer in ("yes", "no")

    def test_deterministic_reproducibility(self):
        b1 = SelfCheckGPTBaseline(n_samples=5, mock=True, seed=1234)
        b2 = SelfCheckGPTBaseline(n_samples=5, mock=True, seed=1234)

        q = "Is a fish a mammal?"
        r1 = b1.predict(q)
        r2 = b2.predict(q)

        assert r1.samples == r2.samples
        assert r1.confidence == r2.confidence
        assert r1.prediction == r2.prediction

    def test_confidence_threshold_hallucination_flag(self):
        baseline_strict = SelfCheckGPTBaseline(n_samples=5, mock=True, confidence_threshold=0.99, seed=42)
        res = baseline_strict.predict("Are some foxes deceivers?")
        # High threshold flags even moderate consistency as hallucination
        if res.confidence < 0.99:
            assert res.is_hallucination is True

    def test_evaluate_dataset(self):
        baseline = SelfCheckGPTBaseline(n_samples=3, mock=True, seed=42)
        output = baseline.evaluate_dataset(SAMPLE_BENCHMARK_QUERIES)

        assert output["method"] == "SelfCheckGPT"
        assert output["total_queries"] == 5
        assert "metrics" in output
        assert "accuracy" in output["metrics"]
        assert "precision" in output["metrics"]
        assert "recall" in output["metrics"]
        assert "f1" in output["metrics"]
        assert "confusion_matrix" in output["metrics"]
        assert "per_query_type" in output
        assert "per_source" in output
        assert len(output["results"]) == 5


# ── 3. Dense RAG Unit Tests ───────────────────────────────────────────────────

class TestDenseRAGBaseline:
    """Test suite for Dense RAG retrieval and generation baseline."""

    def test_kb_to_facts_extraction(self):
        facts = kb_to_facts(Path("data/knowledge_bases/knowledge_base_extended.json"))
        assert len(facts) >= 1000
        # Check sample fact formats
        assert any("is a" in f or "is a type of" in f for f in facts)
        assert any("has" in f or "have" in f for f in facts)
        assert any("If" in f and "then" in f for f in facts)

    def test_sparse_tfidf_retriever(self):
        sample_facts = [
            "A dog is a mammal.",
            "All mammals are animals.",
            "A fish has gills.",
            "If water freezes, then it becomes ice.",
        ]
        retriever = SparseTFIDFRetriever(sample_facts)
        results = retriever.retrieve("Does a dog belong to mammal?", top_k=2)

        assert len(results) > 0
        top_fact, top_score = results[0]
        assert "dog" in top_fact.lower() or "mammal" in top_fact.lower()
        assert top_score >= 0.0

    def test_predict_single_query(self):
        baseline = DenseRAGBaseline(mock=True, top_k=3)
        res = baseline.predict(
            question="Do all dogs have hair?",
            query_id="rag_001",
            ground_truth=True,
            query_type="categorical",
        )
        assert isinstance(res, DenseRAGResult)
        assert res.query_id == "rag_001"
        assert len(res.retrieved_facts) <= 3
        assert len(res.similarity_scores) == len(res.retrieved_facts)
        assert isinstance(res.prediction, bool)
        assert res.latency_retrieval_ms >= 0.0
        assert res.latency_generation_ms >= 0.0

    def test_evaluate_dataset(self):
        baseline = DenseRAGBaseline(mock=True, top_k=3)
        output = baseline.evaluate_dataset(SAMPLE_BENCHMARK_QUERIES)

        assert output["method"] == "Dense RAG"
        assert output["total_queries"] == 5
        assert "metrics" in output
        assert "accuracy" in output["metrics"]
        assert "precision" in output["metrics"]
        assert "recall" in output["metrics"]
        assert "f1" in output["metrics"]
        assert "confusion_matrix" in output["metrics"]
        assert "mean_latency_retrieval_ms" in output
        assert len(output["results"]) == 5


# ── 4. Logic-LM Unit Tests ────────────────────────────────────────────────────

class TestLogicLMBaseline:
    """Test suite for Logic-LM formal translation and symbolic solver baseline."""

    def test_rule_based_logic_translator(self):
        translator = RuleBasedLogicTranslator()

        # Taxonomic translation
        res_tax = translator.translate("Are all dogs mammals?", hint_type="taxonomic")
        assert res_tax["formula_type"] == "taxonomic"
        assert "∀x" in res_tax["formula"]
        assert res_tax["subject"] == "dog"
        assert res_tax["target"] == "mammal"

        # Categorical translation
        res_cat = translator.translate("Do all dogs have hair?", hint_type="categorical")
        assert res_cat["formula_type"] == "categorical"
        assert "HasProperty" in res_cat["formula"]
        assert res_cat["entity"] == "dog"

        # Hypothetical translation
        res_hyp = translator.translate("If water freezes, then it becomes ice.", hint_type="hypothetical")
        assert res_hyp["formula_type"] == "hypothetical"
        assert "→" in res_hyp["formula"]

    def test_symbolic_logic_solver_truth_values(self):
        kb_path = Path("data/knowledge_bases/knowledge_base_extended.json")
        kb = KnowledgeBase(kb_path)
        solver = SymbolicLogicSolver(kb)

        # 1. Known true taxonomic relation
        status, pred, proof = solver.solve({
            "formula_type": "taxonomic",
            "subject": "dog",
            "target": "animal",
            "formula": "∀x (Dog(x) → Animal(x))",
        })
        assert status == "PROVEN_TRUE"
        assert pred is True
        assert len(proof) > 0

        # 2. Known false / inverted relation
        status, pred, proof = solver.solve({
            "formula_type": "taxonomic",
            "subject": "animal",
            "target": "dog",
            "formula": "∀x (Animal(x) → Dog(x))",
        })
        assert status == "PROVEN_FALSE"
        assert pred is False

        # 3. Unknown entity (outside KB scope)
        status, pred, proof = solver.solve({
            "formula_type": "taxonomic",
            "subject": "xenomorph_alien",
            "target": "animal",
            "formula": "∀x (Xenomorph(x) → Animal(x))",
        })
        assert status == "UNKNOWN"
        assert pred is False

    def test_predict_single_query(self):
        baseline = LogicLMBaseline(mock=True)
        res = baseline.predict(
            question="Are all cats felines?",
            query_id="logic_001",
            query_type="taxonomic",
            ground_truth=True,
        )
        assert isinstance(res, LogicLMResult)
        assert res.query_id == "logic_001"
        assert "∀x" in res.logical_formula or "Cat" in res.logical_formula or res.formula_type == "taxonomic"
        assert res.solver_status in ("PROVEN_TRUE", "PROVEN_FALSE", "UNKNOWN", "SAT", "UNSAT")
        assert isinstance(res.prediction, bool)
        assert len(res.proof_steps) > 0

    def test_evaluate_dataset(self):
        baseline = LogicLMBaseline(mock=True)
        output = baseline.evaluate_dataset(SAMPLE_BENCHMARK_QUERIES)

        assert output["method"] == "Logic-LM"
        assert output["total_queries"] == 5
        assert "metrics" in output
        assert "accuracy" in output["metrics"]
        assert "precision" in output["metrics"]
        assert "recall" in output["metrics"]
        assert "f1" in output["metrics"]
        assert "confusion_matrix" in output["metrics"]
        assert "solver_status_counts" in output
        assert len(output["results"]) == 5


# ── 5. Integration with Benchmark Dataset ─────────────────────────────────────

class TestBaselinesBenchmarkIntegration:
    """End-to-end integration test with the real avicenna_benchmark_500 dataset."""

    def test_all_baselines_run_on_benchmark_slice(self):
        loader = BenchmarkLoader("data/benchmarks/avicenna_benchmark_500.json")
        benchmark_slice = loader.get_all_queries()[:10]

        # 1. SelfCheckGPT
        sc_baseline = SelfCheckGPTBaseline(n_samples=3, mock=True, seed=42)
        sc_out = sc_baseline.evaluate_dataset(benchmark_slice)
        assert sc_out["total_queries"] == 10
        assert sc_out["metrics"]["confusion_matrix"]["total"] > 0

        # 2. Dense RAG
        rag_baseline = DenseRAGBaseline(mock=True, top_k=3)
        rag_out = rag_baseline.evaluate_dataset(benchmark_slice)
        assert rag_out["total_queries"] == 10
        assert rag_out["metrics"]["confusion_matrix"]["total"] > 0

        # 3. Logic-LM
        logic_baseline = LogicLMBaseline(mock=True)
        logic_out = logic_baseline.evaluate_dataset(benchmark_slice)
        assert logic_out["total_queries"] == 10
        assert logic_out["metrics"]["confusion_matrix"]["total"] > 0
