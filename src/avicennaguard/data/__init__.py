"""
Data loading and benchmark dataset utilities for AvicennaGuard.

Exports BenchmarkLoader for loading, validating, splitting, and filtering
the 500-query AvicennaGuard multi-source benchmark dataset.
"""

from avicennaguard.data.benchmark_loader import BenchmarkLoader

__all__ = [
    "BenchmarkLoader",
]
