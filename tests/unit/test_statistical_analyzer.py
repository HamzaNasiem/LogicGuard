"""
Unit tests for AvicennaGuard Statistical Analyzer and LaTeX Table Generator.

Verifies:
  1. Contingency matrix generation (n01, n10, n11, n00)
  2. McNemar's test with Yates continuity correction and exact binomial p-values
  3. Wilson score confidence interval mathematical bounds and edge cases
  4. Cohen's g effect size calculation and magnitude classification
  5. Latency statistical summaries
  6. LaTeX publication table formatting, environments, and syntax
"""

import math
from pathlib import Path
import pytest

from avicennaguard.eval.statistical_analyzer import StatisticalAnalyzer
from avicennaguard.eval.latex_generator import (
    generate_main_results_table,
    generate_baseline_comparison_table,
    generate_mcnemar_table,
    generate_latency_table,
    generate_ablation_table,
    generate_all_tables,
    export_tables_to_files,
    wrap_standalone_document,
)


# =====================================================================
# 1. CONTINGENCY MATRIX & MATH TESTS
# =====================================================================

class TestContingencyMatrixAndMath:
    def test_extract_correctness_from_dicts(self):
        records = [
            {"is_correct": True},
            {"is_correct": False},
            {"correct": True},
            {"predicted_answer": "yes", "ground_truth": "yes"},
            {"predicted_answer": "no", "ground_truth": "yes"},
        ]
        flags = StatisticalAnalyzer._extract_correctness(records)
        assert flags == [True, False, True, True, False]

    def test_extract_correctness_from_booleans_and_ints(self):
        records = [True, False, 1, 0, True]
        flags = StatisticalAnalyzer._extract_correctness(records)
        assert flags == [True, False, True, False, True]

    def test_extract_correctness_invalid_type_raises(self):
        with pytest.raises(TypeError):
            StatisticalAnalyzer._extract_correctness(["invalid_string_item"])

    def test_paired_contingency_matrix_computation(self):
        # 4 queries:
        # q0: Base=False, Guard=True  -> n01 (1)
        # q1: Base=True,  Guard=False -> n10 (1)
        # q2: Base=True,  Guard=True  -> n11 (1)
        # q3: Base=False, Guard=False -> n00 (1)
        base = [False, True, True, False]
        guard = [True, False, True, False]

        res = StatisticalAnalyzer.mcnemar_test(base, guard)
        assert res["n01"] == 1
        assert res["n10"] == 1
        assert res["n11"] == 1
        assert res["n00"] == 1
        assert res["n_queries"] == 4
        assert res["contingency_matrix"] == [[1, 1], [1, 1]]

    def test_mismatched_sequence_lengths_raises(self):
        base = [{"is_correct": True}, {"is_correct": False}]
        guard = [{"is_correct": True}]
        with pytest.raises(ValueError, match="Paired evaluation mismatch"):
            StatisticalAnalyzer.mcnemar_test(base, guard)


# =====================================================================
# 2. MCNEMAR TEST CALCULATIONS & P-VALUES
# =====================================================================

class TestMcNemarCalculations:
    def test_llama2_7b_exact_counts(self):
        """Verify LLaMA2-7B empirical experiment numbers (n01=36, n10=0, n11=63, n00=1)."""
        res = StatisticalAnalyzer.mcnemar_test(n01=36, n10=0, n11=63, n00=1)

        # chi2 = (|36 - 0| - 1)^2 / 36 = 35^2 / 36 = 1225 / 36 = 34.02777...
        assert res["chi2"] == pytest.approx(34.0278, abs=1e-4)
        assert res["p_value"] < 1e-6
        assert res["significant_p001"] is True
        assert res["effect_size_g"] == pytest.approx(0.50, abs=1e-4)
        assert res["effect_magnitude"] == "large"
        assert res["odds_ratio"] == float("inf")

    def test_mistral_7b_counts(self):
        """Verify Mistral-7B empirical numbers (n01=5, n10=0, n11=95, n00=0)."""
        res = StatisticalAnalyzer.mcnemar_test(n01=5, n10=0, n11=95, n00=0)

        # chi2 = (5 - 1)^2 / 5 = 16 / 5 = 3.2000
        assert res["chi2"] == pytest.approx(3.2000, abs=1e-4)
        assert res["p_value"] == pytest.approx(0.073638, abs=1e-4)
        # Exact binomial p-value for 5 discordants (2 * 0.5^5 = 2/32 = 0.0625)
        assert res["p_value_exact"] == pytest.approx(0.0625, abs=1e-4)
        assert res["significant_p05"] is False  # p = 0.0736 > 0.05

    def test_llama32_3b_counts(self):
        """Verify LLaMA3.2-3B empirical numbers (n01=15, n10=0, n11=85, n00=0)."""
        res = StatisticalAnalyzer.mcnemar_test(n01=15, n10=0, n11=85, n00=0)

        # chi2 = (15 - 1)^2 / 15 = 196 / 15 = 13.0667
        assert res["chi2"] == pytest.approx(13.0667, abs=1e-4)
        assert res["p_value"] < 0.001
        assert res["significant_p001"] is True

    def test_symmetric_discordant_pairs(self):
        """When n01 == n10, chi2 should be 0 and p-value 1.0."""
        res = StatisticalAnalyzer.mcnemar_test(n01=10, n10=10, n11=50, n00=30)
        assert res["chi2"] == 0.0
        assert res["p_value"] == 1.0
        assert res["p_value_exact"] == 1.0
        assert res["effect_size_g"] == 0.0
        assert res["effect_magnitude"] == "negligible"
        assert res["significant_p05"] is False

    def test_zero_discordant_pairs(self):
        """When n01 == 0 and n10 == 0 (identical models), chi2 is 0 and p-value 1.0."""
        res = StatisticalAnalyzer.mcnemar_test(n01=0, n10=0, n11=100, n00=0)
        assert res["chi2"] == 0.0
        assert res["p_value"] == 1.0
        assert res["p_value_exact"] == 1.0
        assert res["effect_size_g"] == 0.0


# =====================================================================
# 3. WILSON SCORE CONFIDENCE INTERVAL TESTS
# =====================================================================

class TestWilsonScoreConfidenceInterval:
    def test_wilson_ci_bounds_validity(self):
        """Test that lower <= p <= upper and both are in [0, 1]."""
        for k in [0, 10, 50, 90, 100]:
            lower, upper = StatisticalAnalyzer.wilson_confidence_interval(k, 100, confidence=0.95)
            assert 0.0 <= lower <= upper <= 1.0
            p = k / 100
            if k > 0 and k < 100:
                assert lower < p < upper

    def test_wilson_ci_percentages(self):
        lower_pct, upper_pct = StatisticalAnalyzer.wilson_confidence_interval(
            63, 100, confidence=0.95, as_percentage=True
        )
        assert lower_pct == pytest.approx(53.2, abs=0.2)
        assert upper_pct == pytest.approx(71.8, abs=0.2)

    def test_wilson_ci_boundary_k_zero(self):
        lower, upper = StatisticalAnalyzer.wilson_confidence_interval(0, 100, confidence=0.95)
        assert lower == 0.0
        assert upper > 0.0
        assert upper < 0.05

    def test_wilson_ci_boundary_k_equals_n(self):
        lower, upper = StatisticalAnalyzer.wilson_confidence_interval(100, 100, confidence=0.95)
        assert upper == 1.0
        assert lower > 0.95
        assert lower < 1.0

    def test_wilson_ci_zero_sample_size(self):
        lower, upper = StatisticalAnalyzer.wilson_confidence_interval(0, 0, confidence=0.95)
        assert lower == 0.0
        assert upper == 1.0

    def test_higher_confidence_widens_interval(self):
        low_95, up_95 = StatisticalAnalyzer.wilson_confidence_interval(50, 100, confidence=0.95)
        low_99, up_99 = StatisticalAnalyzer.wilson_confidence_interval(50, 100, confidence=0.99)
        width_95 = up_95 - low_95
        width_99 = up_99 - low_99
        assert width_99 > width_95


# =====================================================================
# 4. COHEN'S G & LATENCY STATS TESTS
# =====================================================================

class TestCohenGAndLatencyStats:
    def test_cohens_g_effect_size(self):
        # 100% one-sided discordance
        res = StatisticalAnalyzer.mcnemar_test(n01=20, n10=0)
        assert res["effect_size_g"] == 0.50
        assert res["effect_magnitude"] == "large"

        # 75% vs 25% discordance: p01 = 15/20 = 0.75 -> g = 0.25
        res2 = StatisticalAnalyzer.mcnemar_test(n01=15, n10=5)
        assert res2["effect_size_g"] == 0.25
        assert res2["effect_magnitude"] == "medium"

    def test_latency_statistics_calculation(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = StatisticalAnalyzer.latency_stats(latencies)
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["n"] == 5
        assert stats["std"] > 0

    def test_latency_statistics_handles_none_and_empty(self):
        stats = StatisticalAnalyzer.latency_stats([None, -1.0, None])
        assert stats["n"] == 0
        assert stats["mean"] == 0.0


# =====================================================================
# 5. LATEX TABLE GENERATOR TESTS
# =====================================================================

class TestLatexGenerator:
    @pytest.fixture
    def mock_experiment_data(self):
        return {
            "summaries": {
                "llama2_7b_baseline": {
                    "accuracy": 63.0,
                    "by_type": {
                        "taxonomic": {"accuracy": 56.0},
                        "categorical": {"accuracy": 62.9},
                        "hypothetical": {"accuracy": 86.7},
                    },
                },
                "llama2_7b_avicennaguard": {
                    "accuracy": 99.0,
                    "hallucinations_caught": 36,
                    "llm_errors_on_logical": 36,
                    "false_alarms": 0,
                    "by_type": {
                        "taxonomic": {"accuracy": 98.0},
                        "categorical": {"accuracy": 100.0},
                        "hypothetical": {"accuracy": 100.0},
                    },
                },
            }
        }

    def test_generate_main_results_table_syntax(self, mock_experiment_data):
        latex = generate_main_results_table(mock_experiment_data)
        assert "\\begin{table}[t]" in latex
        assert "\\caption{\\textsc{Multi-Model Accuracy Comparison Across Logical Reasoning Types}}" in latex
        assert "\\label{tab:main_results}" in latex
        assert "\\begin{tabular}" in latex
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex
        assert "\\end{tabular}" in latex
        assert "\\end{table}" in latex
        assert "LLaMA2-7B" in latex
        assert "\\textsc{AvicennaGuard}" in latex

    def test_generate_baseline_comparison_table_syntax(self):
        latex = generate_baseline_comparison_table({})
        assert "\\begin{table}[t]" in latex
        assert "\\label{tab:baseline_comparison}" in latex
        assert "SelfCheckGPT" in latex
        assert "RAG-Sparse" in latex
        assert "RAG-Dense" in latex
        assert "\\textbf{\\textsc{AvicennaGuard} (Ours)}" in latex
        assert "\\bottomrule" in latex
        assert "\\end{table}" in latex

    def test_generate_mcnemar_table_syntax(self):
        sig_data = {
            "mcnemar_tests": {
                "LLaMA2-7B": {"n01": 36, "n10": 0, "chi2": 34.03, "p_value": 0.000001, "effect_size_g": 0.5},
            },
            "confidence_intervals": {
                "LLaMA2-7B": {
                    "baseline": {"accuracy": 63.0, "ci_95": [53.2, 71.8]},
                    "avicennaguard": {"accuracy": 99.0, "ci_95": [94.5, 99.8]},
                }
            },
        }
        latex = generate_mcnemar_table(sig_data)
        assert "\\label{tab:mcnemar_significance}" in latex
        assert "34.03" in latex
        assert "$p < 0.001$" in latex
        assert "0.50" in latex
        assert "\\end{table}" in latex

    def test_generate_latency_table_syntax(self):
        lat_data = {
            "LLaMA2-7B": {
                "llm_call_ms": {"mean": 11012.5},
                "stage1_ms": {"mean": 0.067},
                "stage2_ms": {"mean": 0.067},
                "total_overhead_ms": {"mean": 0.134},
                "overhead_pct_of_llm": 0.001,
            }
        }
        latex = generate_latency_table(lat_data)
        assert "\\label{tab:latency_breakdown}" in latex
        assert "Stage 1: Parser" in latex
        assert "Stage 2: BFS Graph" in latex
        assert "\\end{table}" in latex

    def test_generate_ablation_table_syntax(self):
        latex = generate_ablation_table()
        assert "\\begin{table*}[t]" in latex
        assert "\\label{tab:ablation_study}" in latex
        assert "Stage 2 BFS" in latex
        assert "4-State Epistemics" in latex
        assert "\\end{table*}" in latex

    def test_export_tables_and_standalone_doc(self, tmp_path):
        tables = {
            "table1_main_results": "\\begin{table}Test1\\end{table}",
            "table2_baseline_comparison": "\\begin{table}Test2\\end{table}",
        }
        saved = export_tables_to_files(tables, tmp_path)
        assert len(saved) == 3  # table1, table2, all_tables.tex
        assert (tmp_path / "table1_main_results.tex").exists()
        assert (tmp_path / "all_tables.tex").exists()

        standalone = wrap_standalone_document(tables)
        assert "\\documentclass[journal]{IEEEtran}" in standalone
        assert "\\begin{document}" in standalone
        assert "\\end{document}" in standalone
