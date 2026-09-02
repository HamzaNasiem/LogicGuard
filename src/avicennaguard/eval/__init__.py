"""
AvicennaGuard Evaluation & Benchmark Package.

Provides evaluation orchestrators, statistical analyzers, and LaTeX table generators:
  - BenchmarkRunner: Multi-model evaluation orchestrator
  - StatisticalAnalyzer: McNemar test, Wilson score CIs, Cohen's g, latency stats
  - LaTeX table generators: Publication-ready IEEE table generators
"""

from avicennaguard.eval.benchmark_runner import (
    DEFAULT_MODELS,
    MODEL_ALIASES,
    BenchmarkRunner,
)
from avicennaguard.eval.latex_generator import (
    export_tables_to_files,
    generate_ablation_table,
    generate_all_tables,
    generate_baseline_comparison_table,
    generate_latency_table,
    generate_main_results_table,
    generate_mcnemar_table,
    wrap_standalone_document,
)
from avicennaguard.eval.statistical_analyzer import StatisticalAnalyzer

__all__ = [
    "DEFAULT_MODELS",
    "MODEL_ALIASES",
    "BenchmarkRunner",
    "StatisticalAnalyzer",
    "export_tables_to_files",
    "generate_ablation_table",
    "generate_all_tables",
    "generate_baseline_comparison_table",
    "generate_latency_table",
    "generate_main_results_table",
    "generate_mcnemar_table",
    "wrap_standalone_document",
]
