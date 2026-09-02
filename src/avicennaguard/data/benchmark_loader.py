"""
Benchmark Dataset Loader for AvicennaGuard.

Provides structured access, filtering, splitting, and summary statistics
for the 500-query AvicennaGuard benchmark dataset (FOLIO, ProofWriter,
Curated Gold, and TruthfulQA OOD).
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_FILENAME = "avicenna_benchmark_500.json"
REQUIRED_FIELDS = ("id", "source", "question", "ground_truth", "query_type", "difficulty")


class BenchmarkLoader:
    """
    Loader and manager for AvicennaGuard evaluation benchmark datasets.

    Loads the benchmark JSON dataset, validates record integrity, and provides
    query retrieval, source/type filtering, deterministic dataset splitting,
    and statistical summaries.

    Args:
        benchmark_path: Optional path to the benchmark JSON file. If None,
            automatically resolves to the standard repo location.
    """

    def __init__(self, benchmark_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the BenchmarkLoader.

        Args:
            benchmark_path: Path to benchmark JSON file (or auto-discover).
        """
        self.benchmark_path = self._resolve_path(benchmark_path)
        self._queries: list[dict[str, Any]] = []
        self._id_map: dict[str, dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _resolve_path(benchmark_path: Optional[Union[str, Path]]) -> Path:
        """Resolve benchmark path from argument or default search locations."""
        if benchmark_path is not None:
            p = Path(benchmark_path)
            if not p.exists():
                raise FileNotFoundError(f"Benchmark file not found: {p}")
            return p

        # Default search paths
        candidate_paths = [
            Path("data/benchmarks") / DEFAULT_BENCHMARK_FILENAME,
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "benchmarks" / DEFAULT_BENCHMARK_FILENAME,
            Path.cwd() / "data" / "benchmarks" / DEFAULT_BENCHMARK_FILENAME,
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not locate '{DEFAULT_BENCHMARK_FILENAME}' in default search paths: "
            f"{[str(c) for c in candidate_paths]}"
        )

    def _load(self) -> None:
        """Load and validate the benchmark dataset JSON."""
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected benchmark JSON to contain a list of queries, got {type(data).__name__}"
            )

        validated_queries: list[dict[str, Any]] = []
        id_map: dict[str, dict[str, Any]] = {}

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {idx} in benchmark is not a dictionary: {entry}")

            missing_fields = [f for f in REQUIRED_FIELDS if f not in entry]
            if missing_fields:
                raise ValueError(
                    f"Entry {idx} (id={entry.get('id', 'unknown')}) missing required fields: {missing_fields}"
                )

            # Validate string non-emptiness for essential text fields
            for str_field in ("id", "source", "question", "query_type", "difficulty"):
                val = entry.get(str_field)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(
                        f"Entry {idx} field '{str_field}' must be a non-empty string, got {val!r}"
                    )

            query_id = entry["id"].strip()
            if query_id in id_map:
                logger.warning("Duplicate query ID detected in benchmark: %s", query_id)

            validated_queries.append(entry)
            id_map[query_id] = entry

        self._queries = validated_queries
        self._id_map = id_map
        logger.info(
            "Loaded %d benchmark queries from %s",
            len(self._queries),
            self.benchmark_path.name,
        )

    def get_all_queries(self) -> list[dict[str, Any]]:
        """
        Return a copy of all loaded benchmark queries.

        Returns:
            List of query dictionaries.
        """
        return [dict(q) for q in self._queries]

    def get_by_source(self, source: str) -> list[dict[str, Any]]:
        """
        Filter queries by source dataset (case-insensitive).

        Args:
            source: Source dataset name (e.g. 'FOLIO', 'ProofWriter', 'Curated_Gold', 'TruthfulQA_OOD').

        Returns:
            List of matching query dictionaries.
        """
        target = source.strip().lower()
        return [dict(q) for q in self._queries if q.get("source", "").strip().lower() == target]

    def get_by_type(self, query_type: str) -> list[dict[str, Any]]:
        """
        Filter queries by logical query type (case-insensitive).

        Args:
            query_type: Logical query type (e.g. 'taxonomic', 'categorical', 'hypothetical', 'ood').

        Returns:
            List of matching query dictionaries.
        """
        target = query_type.strip().lower()
        return [dict(q) for q in self._queries if q.get("query_type", "").strip().lower() == target]

    def get_by_difficulty(self, difficulty: str) -> list[dict[str, Any]]:
        """
        Filter queries by difficulty level (case-insensitive).

        Args:
            difficulty: Difficulty level (e.g. 'easy', 'medium', 'hard').

        Returns:
            List of matching query dictionaries.
        """
        target = difficulty.strip().lower()
        return [dict(q) for q in self._queries if q.get("difficulty", "").strip().lower() == target]

    def get_by_id(self, query_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve a single query by its unique ID.

        Args:
            query_id: Unique query identifier (e.g. 'folio_001').

        Returns:
            Query dictionary if found, None otherwise.
        """
        entry = self._id_map.get(query_id.strip())
        return dict(entry) if entry is not None else None

    def get_splits(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Deterministically split benchmark queries into train and test/validation subsets.

        Args:
            train_ratio: Fraction of queries allocated to the first split (0.0 to 1.0).
            seed: Random seed for deterministic shuffling.

        Returns:
            Tuple of (train_queries, test_queries).
        """
        if not (0.0 <= train_ratio <= 1.0):
            raise ValueError(f"train_ratio must be between 0.0 and 1.0 inclusive, got {train_ratio}")

        indices = list(range(len(self._queries)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_queries = [dict(self._queries[i]) for i in train_indices]
        test_queries = [dict(self._queries[i]) for i in test_indices]

        return train_queries, test_queries

    def summary_stats(self) -> dict[str, Any]:
        """
        Compute summary statistics for the benchmark dataset.

        Returns:
            Dictionary containing total count, breakdowns by source, query type,
            difficulty, ground truth distribution, and the source file path.
        """
        sources = dict(Counter(q["source"] for q in self._queries))
        query_types = dict(Counter(q["query_type"] for q in self._queries))
        difficulties = dict(Counter(q["difficulty"] for q in self._queries))
        ground_truth_dist = dict(Counter(str(q["ground_truth"]) for q in self._queries))

        return {
            "total_queries": len(self._queries),
            "sources": sources,
            "query_types": query_types,
            "difficulties": difficulties,
            "ground_truth_distribution": ground_truth_dist,
            "benchmark_file": self.benchmark_path.name,
        }

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self._queries[idx])

    def __iter__(self):
        for q in self._queries:
            yield dict(q)
