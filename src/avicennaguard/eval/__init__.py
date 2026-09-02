"""
AvicennaGuard Evaluation & Multi-Model Benchmark Runner.

Provides orchestrator utilities for running standard logic benchmarks
across multiple LLMs with epistemic state logging and latency profiling.
"""

from avicennaguard.eval.benchmark_runner import (
    BenchmarkRunner,
    DEFAULT_MODELS,
    MODEL_ALIASES,
)

__all__ = [
    "BenchmarkRunner",
    "DEFAULT_MODELS",
    "MODEL_ALIASES",
]
