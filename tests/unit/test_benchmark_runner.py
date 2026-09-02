"""
Unit tests for BenchmarkRunner module.

Verifies initialization, offline mock evaluation loop, per-query epistemic logging,
latency profiling, confusion matrix and PRF1 metric computations, multi-model execution,
and result export.

Run:
    pytest tests/unit/test_benchmark_runner.py -v
"""

import json
from pathlib import Path
import pytest

from avicennaguard.eval.benchmark_runner import (
    DEFAULT_MODELS,
    MODEL_ALIASES,
    BenchmarkRunner,
    compute_stats,
    parse_llm_yn,
    to_bool,
)
from avicennaguard.kb.loader import KnowledgeBase

KB_PATH = Path("data/knowledge_bases/knowledge_base_extended.json")
BENCHMARK_PATH = Path("data/benchmarks/avicenna_benchmark_500.json")


@pytest.fixture(scope="module")
def shared_kb() -> KnowledgeBase:
    """Fixture providing an initialized KnowledgeBase."""
    return KnowledgeBase(KB_PATH)


@pytest.fixture(scope="module")
def mock_runner(shared_kb: KnowledgeBase) -> BenchmarkRunner:
    """Fixture providing a mock-mode BenchmarkRunner."""
    return BenchmarkRunner(
        kb=shared_kb,
        benchmark_path=BENCHMARK_PATH,
        mock_mode=True,
        seed=42,
    )


class TestBenchmarkRunnerInit:
    """Test initialization and configuration of BenchmarkRunner."""

    def test_default_init(self, mock_runner: BenchmarkRunner):
        """Verify initialization with default models and resolved paths."""
        assert mock_runner.mock_mode is True
        assert len(mock_runner.models) == len(DEFAULT_MODELS)
        assert mock_runner.benchmark_path.name == "avicenna_benchmark_500.json"
        assert len(mock_runner.benchmark_loader) == 500

    def test_custom_models_and_aliases(self, shared_kb: KnowledgeBase):
        """Verify model alias resolution during initialization."""
        custom_models = ["llama3.2", "mistral", "llama2", "deepseek-r1", "phi4"]
        runner = BenchmarkRunner(
            kb=shared_kb,
            models=custom_models,
            mock_mode=True,
        )
        assert runner.models == [
            "llama3.2:3b",
            "mistral:7b",
            "llama2:7b",
            "deepseek-r1:7b",
            "phi4:latest",
        ]

    def test_invalid_kb_path_raises_error(self, tmp_path: Path):
        """Verify FileNotFoundError on non-existent KB path."""
        fake_kb = tmp_path / "non_existent_kb.json"
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(kb=fake_kb, mock_mode=True)


class TestEvaluationLoopMockMode:
    """Test full evaluation loop for a single model in mock mode."""

    def test_run_evaluation_structure(self, mock_runner: BenchmarkRunner):
        """Verify return dictionary structure from run_evaluation."""
        limit = 10
        res = mock_runner.run_evaluation(model_name="llama3.2:3b", limit=limit)

        assert res["model"] == "llama3.2:3b"
        assert res["total_queries"] == limit
        assert res["mock_mode"] is True

        # Check top-level sections
        assert "baseline" in res
        assert "avicennaguard" in res
        assert "comparison" in res

        # Check baseline section
        assert "metrics" in res["baseline"]
        assert "latency_ms" in res["baseline"]
        assert "results" in res["baseline"]
        assert len(res["baseline"]["results"]) == limit

        # Check AvicennaGuard section
        assert "metrics" in res["avicennaguard"]
        assert "latency_ms" in res["avicennaguard"]
        assert "by_type" in res["avicennaguard"]
        assert "epistemic_states" in res["avicennaguard"]
        assert "hallucination_analysis" in res["avicennaguard"]
        assert "results" in res["avicennaguard"]
        assert len(res["avicennaguard"]["results"]) == limit

    def test_per_query_record_fields(self, mock_runner: BenchmarkRunner):
        """Verify all required per-query fields are present and typed correctly."""
        res = mock_runner.run_evaluation(model_name="llama3.2:3b", limit=5)
        ag_results = res["avicennaguard"]["results"]

        required_query_keys = {
            "id",
            "question",
            "ground_truth",
            "llm_answer",
            "final_answer",
            "epistemic_state",
            "intercepted",
            "proof",
            "latency_ms",
        }

        for record in ag_results:
            for key in required_query_keys:
                assert key in record, f"Missing required key '{key}' in query record"

            assert isinstance(record["question"], str) and record["question"]
            assert record["epistemic_state"] in ("YAQEEN", "WAHM", "SHAKK", "ZANN")
            assert isinstance(record["intercepted"], bool)
            assert isinstance(record["proof"], str) and record["proof"]

            # Latency breakdown validation
            lats = record["latency_ms"]
            assert "llm_ms" in lats
            assert "stage1_ms" in lats
            assert "stage2_ms" in lats
            assert "total_overhead_ms" in lats

            assert lats["stage1_ms"] >= 0.0
            assert lats["stage2_ms"] >= 0.0
            assert lats["total_overhead_ms"] == pytest.approx(
                lats["stage1_ms"] + lats["stage2_ms"], rel=1e-2, abs=1e-3
            )

    def test_latency_summary_profiling(self, mock_runner: BenchmarkRunner):
        """Verify latency statistics (mean, median, p95) are calculated properly."""
        res = mock_runner.run_evaluation(model_name="mistral:7b", limit=10)
        ag_lats = res["avicennaguard"]["latency_ms"]

        for comp in ("stage1", "stage2", "total_overhead"):
            assert comp in ag_lats
            stats = ag_lats[comp]
            assert "mean" in stats
            assert "median" in stats
            assert "p95" in stats
            assert stats["mean"] >= 0.0
            assert stats["median"] >= 0.0
            assert stats["p95"] >= stats["median"]


class TestConfusionMatrixAndMetrics:
    """Test confusion matrix, PRF1, Specificity, and FPR calculations."""

    def test_compute_confusion_matrix_synthetic(self, mock_runner: BenchmarkRunner):
        """Verify binary confusion matrix calculation on controlled data."""
        synthetic_records = [
            {"ground_truth": True, "final_answer": True},    # TP
            {"ground_truth": True, "final_answer": True},    # TP
            {"ground_truth": False, "final_answer": False},  # TN
            {"ground_truth": False, "final_answer": False},  # TN
            {"ground_truth": False, "final_answer": True},   # FP
            {"ground_truth": True, "final_answer": False},   # FN
            {"ground_truth": "OOD", "final_answer": False},  # Ignored for binary CM
        ]

        cm = mock_runner.compute_confusion_matrix(synthetic_records)
        assert cm["TP"] == 2
        assert cm["TN"] == 2
        assert cm["FP"] == 1
        assert cm["FN"] == 1
        assert cm["total"] == 6

    def test_compute_prf1_synthetic(self, mock_runner: BenchmarkRunner):
        """Verify PRF1, accuracy, specificity, and FPR formulas on synthetic CM."""
        cm = {"TP": 40, "FP": 10, "TN": 40, "FN": 10, "total": 100}
        metrics = mock_runner.compute_prf1(cm)

        # Accuracy = (40 + 40) / 100 = 80.0%
        assert metrics["accuracy"] == 80.0
        # Precision = 40 / (40 + 10) = 80.0%
        assert metrics["precision"] == 80.0
        # Recall = 40 / (40 + 10) = 80.0%
        assert metrics["recall"] == 80.0
        # F1 = 80.0%
        assert metrics["f1"] == 80.0
        # Specificity = 40 / (40 + 10) = 80.0%
        assert metrics["specificity"] == 80.0
        # FPR = 10 / (40 + 10) = 20.0%
        assert metrics["fpr"] == 20.0
        # Complementarity: Specificity + FPR = 100%
        assert metrics["specificity"] + metrics["fpr"] == pytest.approx(100.0)

    def test_metrics_empty_records(self, mock_runner: BenchmarkRunner):
        """Verify zero division handling on empty record sets."""
        cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "total": 0}
        metrics = mock_runner.compute_prf1(cm)
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["specificity"] == 0.0
        assert metrics["fpr"] == 0.0


class TestHallucinationAnalysis:
    """Test hallucination interception tracking and calculations."""

    def test_hallucination_analysis_logic(self, mock_runner: BenchmarkRunner):
        """Verify calculation of caught hallucinations vs false alarms."""
        baseline_records = [
            {"question": "Q1", "is_correct": False},
            {"question": "Q2", "is_correct": False},
            {"question": "Q3", "is_correct": True},
            {"question": "Q4", "is_correct": True},
        ]
        ag_records = [
            {"question": "Q1", "is_correct": True},   # Intercepted!
            {"question": "Q2", "is_correct": False},  # Both wrong
            {"question": "Q3", "is_correct": True},   # Both correct
            {"question": "Q4", "is_correct": False},  # False alarm
        ]

        analysis = mock_runner.compute_hallucination_analysis(baseline_records, ag_records)

        assert analysis["intercepted"] == 1
        assert analysis["intercepted_questions"] == ["Q1"]
        assert analysis["false_alarms"] == 1
        assert analysis["false_alarm_questions"] == ["Q4"]
        assert analysis["both_correct"] == 1
        assert analysis["both_wrong"] == 1
        assert analysis["total_llm_errors"] == 2
        # Interception rate = 1 / 2 = 50.0%
        assert analysis["interception_rate"] == 50.0


class TestMultiModelExecution:
    """Test running evaluation across all models in suite."""

    def test_run_all_mock_mode(self, mock_runner: BenchmarkRunner):
        """Verify run_all executes across multiple models and outputs comparisons."""
        models_to_run = ["llama3.2:3b", "mistral:7b"]
        all_results = mock_runner.run_all(limit=5, models=models_to_run)

        assert "metadata" in all_results
        assert "comparison_summary" in all_results
        assert "models" in all_results

        summary = all_results["comparison_summary"]
        assert len(summary) == 2
        assert summary[0]["model"] == "llama3.2:3b"
        assert summary[1]["model"] == "mistral:7b"

        for row in summary:
            assert "baseline_acc" in row
            assert "guard_acc" in row
            assert "accuracy_gain" in row
            assert "avg_overhead_ms" in row

    def test_filtering_by_source_and_type(self, mock_runner: BenchmarkRunner):
        """Verify filtering options work correctly during evaluation."""
        # Test source filter
        folio_res = mock_runner.run_evaluation(
            model_name="llama3.2:3b",
            limit=5,
            filter_source="FOLIO",
        )
        assert all(r["source"] == "FOLIO" for r in folio_res["avicennaguard"]["results"])

        # Test query type filter
        tax_res = mock_runner.run_evaluation(
            model_name="llama3.2:3b",
            limit=5,
            filter_type="taxonomic",
        )
        assert all(r["query_type"] == "taxonomic" for r in tax_res["avicennaguard"]["results"])


class TestSaveAndExport:
    """Test saving and serializing evaluation results to JSON."""

    def test_save_results_to_file(self, mock_runner: BenchmarkRunner, tmp_path: Path):
        """Verify saving results to JSON and reloading preserves data structure."""
        res = mock_runner.run_evaluation(model_name="llama3.2:3b", limit=3)
        out_file = tmp_path / "test_eval_output.json"

        mock_runner.save_results(res, out_file)
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["model"] == "llama3.2:3b"
        assert len(loaded["avicennaguard"]["results"]) == 3
        assert "metrics" in loaded["baseline"]


class TestHelperFunctions:
    """Test utility helper functions (to_bool, parse_llm_yn, compute_stats)."""

    def test_to_bool_conversion(self):
        """Verify boolean normalization helper."""
        assert to_bool(True) is True
        assert to_bool(False) is False
        assert to_bool("yes") is True
        assert to_bool("YES") is True
        assert to_bool("no") is False
        assert to_bool("NO") is False
        assert to_bool("true") is True
        assert to_bool("false") is False
        assert to_bool("OOD") is None
        assert to_bool(None) is None

    def test_parse_llm_yn(self):
        """Verify LLM YES/NO string parser."""
        assert parse_llm_yn("YES") is True
        assert parse_llm_yn("no.") is False
        assert parse_llm_yn("Yes, all dogs are mammals.") is True
        assert parse_llm_yn("No, spiders are arachnids.") is False
        assert parse_llm_yn("[llm_error]") is None
        assert parse_llm_yn("") is None

    def test_compute_stats_calculation(self):
        """Verify numerical descriptive statistics computation."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_stats(data)
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["p95"] == 50.0

        empty_stats = compute_stats([])
        assert empty_stats["mean"] == 0.0
