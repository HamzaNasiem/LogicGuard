"""
Unit tests for BenchmarkLoader module.

Tests loading, validation, filtering, splitting, and statistics calculation
for the 500-query AvicennaGuard benchmark dataset.

Run:
    pytest tests/unit/test_benchmark_loader.py -v
"""

from pathlib import Path
import pytest

from avicennaguard.data.benchmark_loader import BenchmarkLoader


BENCHMARK_PATH = Path("data/benchmarks/avicenna_benchmark_500.json")


@pytest.fixture(scope="module")
def loader() -> BenchmarkLoader:
    """Fixture providing an initialized BenchmarkLoader with default path."""
    return BenchmarkLoader(BENCHMARK_PATH)


class TestBenchmarkLoading:
    """Test loading and general container behavior."""

    def test_load_all_queries(self, loader: BenchmarkLoader):
        """Verify all 500 queries are loaded."""
        queries = loader.get_all_queries()
        assert len(queries) == 500
        assert len(loader) == 500

    def test_default_path_initialization(self):
        """Verify loader initializes properly without explicit path argument."""
        auto_loader = BenchmarkLoader()
        assert len(auto_loader) == 500
        assert auto_loader.benchmark_path.name == "avicenna_benchmark_500.json"

    def test_missing_file_raises_error(self, tmp_path):
        """Verify FileNotFoundError on non-existent path."""
        fake_path = tmp_path / "non_existent.json"
        with pytest.raises(FileNotFoundError):
            BenchmarkLoader(fake_path)

    def test_required_fields_present_in_all_records(self, loader: BenchmarkLoader):
        """Verify every record has all required fields with non-empty string values."""
        required_fields = ("id", "source", "question", "ground_truth", "query_type", "difficulty")
        for q in loader:
            for field in required_fields:
                assert field in q, f"Missing field '{field}' in query {q.get('id')}"
            assert isinstance(q["id"], str) and q["id"].strip()
            assert isinstance(q["source"], str) and q["source"].strip()
            assert isinstance(q["question"], str) and q["question"].strip()
            assert isinstance(q["query_type"], str) and q["query_type"].strip()
            assert isinstance(q["difficulty"], str) and q["difficulty"].strip()

    def test_unique_query_ids(self, loader: BenchmarkLoader):
        """Verify all 500 query IDs are distinct."""
        ids = [q["id"] for q in loader]
        assert len(ids) == 500
        assert len(set(ids)) == 500

    def test_indexing_and_iteration(self, loader: BenchmarkLoader):
        """Verify list indexing and iteration behave consistently."""
        first = loader[0]
        assert isinstance(first, dict)
        assert first["id"] == "folio_001"
        iterated = list(loader)
        assert len(iterated) == 500
        assert iterated[0] == first


class TestFilterBySource:
    """Test filtering queries by benchmark source dataset."""

    def test_filter_folio(self, loader: BenchmarkLoader):
        """Verify FOLIO queries (200 expected)."""
        folio = loader.get_by_source("FOLIO")
        assert len(folio) == 200
        assert all(q["source"] == "FOLIO" for q in folio)

    def test_filter_proofwriter(self, loader: BenchmarkLoader):
        """Verify ProofWriter queries (150 expected)."""
        pw = loader.get_by_source("ProofWriter")
        assert len(pw) == 150
        assert all(q["source"] == "ProofWriter" for q in pw)

    def test_filter_curated_gold(self, loader: BenchmarkLoader):
        """Verify Curated Gold queries (100 expected)."""
        curated = loader.get_by_source("Curated_Gold")
        assert len(curated) == 100
        assert all(q["source"] == "Curated_Gold" for q in curated)

    def test_filter_truthfulqa_ood(self, loader: BenchmarkLoader):
        """Verify TruthfulQA OOD queries (50 expected)."""
        ood = loader.get_by_source("TruthfulQA_OOD")
        assert len(ood) == 50
        assert all(q["source"] == "TruthfulQA_OOD" for q in ood)

    def test_filter_source_case_insensitivity(self, loader: BenchmarkLoader):
        """Verify filtering by source is case-insensitive."""
        folio_lower = loader.get_by_source("folio")
        assert len(folio_lower) == 200

    def test_filter_nonexistent_source_returns_empty(self, loader: BenchmarkLoader):
        """Verify filtering for unknown source returns empty list."""
        unknown = loader.get_by_source("NonExistentSource")
        assert unknown == []


class TestFilterByQueryType:
    """Test filtering queries by logical query type."""

    def test_filter_taxonomic(self, loader: BenchmarkLoader):
        """Verify taxonomic queries (250 expected)."""
        tax = loader.get_by_type("taxonomic")
        assert len(tax) == 250
        assert all(q["query_type"] == "taxonomic" for q in tax)

    def test_filter_hypothetical(self, loader: BenchmarkLoader):
        """Verify hypothetical queries (157 expected)."""
        hypo = loader.get_by_type("hypothetical")
        assert len(hypo) == 157
        assert all(q["query_type"] == "hypothetical" for q in hypo)

    def test_filter_categorical(self, loader: BenchmarkLoader):
        """Verify categorical queries (43 expected)."""
        cat = loader.get_by_type("categorical")
        assert len(cat) == 43
        assert all(q["query_type"] == "categorical" for q in cat)

    def test_filter_ood(self, loader: BenchmarkLoader):
        """Verify OOD queries (50 expected)."""
        ood = loader.get_by_type("ood")
        assert len(ood) == 50
        assert all(q["query_type"] == "ood" for q in ood)

    def test_filter_type_case_insensitivity(self, loader: BenchmarkLoader):
        """Verify filtering by query_type is case-insensitive."""
        tax_upper = loader.get_by_type("TAXONOMIC")
        assert len(tax_upper) == 250

    def test_filter_nonexistent_type_returns_empty(self, loader: BenchmarkLoader):
        """Verify filtering for unknown query type returns empty list."""
        unknown = loader.get_by_type("unknown_type")
        assert unknown == []


class TestFilterByDifficultyAndId:
    """Test filtering by difficulty and query retrieval by ID."""

    def test_filter_by_difficulty(self, loader: BenchmarkLoader):
        """Verify difficulty filtering counts (easy: 75, medium: 310, hard: 115)."""
        easy = loader.get_by_difficulty("easy")
        medium = loader.get_by_difficulty("medium")
        hard = loader.get_by_difficulty("hard")

        assert len(easy) == 75
        assert len(medium) == 310
        assert len(hard) == 115
        assert len(easy) + len(medium) + len(hard) == 500

    def test_get_by_id(self, loader: BenchmarkLoader):
        """Verify query retrieval by ID."""
        q = loader.get_by_id("folio_001")
        assert q is not None
        assert q["id"] == "folio_001"
        assert q["source"] == "FOLIO"

        missing = loader.get_by_id("does_not_exist_999")
        assert missing is None


class TestDatasetSplits:
    """Test dataset partitioning and reproducibility."""

    def test_default_splits(self, loader: BenchmarkLoader):
        """Verify default 80/20 train/test split."""
        train, test = loader.get_splits(train_ratio=0.8, seed=42)
        assert len(train) == 400
        assert len(test) == 100
        assert len(train) + len(test) == 500

    def test_splits_are_disjoint(self, loader: BenchmarkLoader):
        """Verify no overlap between train and test sets."""
        train, test = loader.get_splits(train_ratio=0.8, seed=42)
        train_ids = {q["id"] for q in train}
        test_ids = {q["id"] for q in test}
        assert train_ids.isdisjoint(test_ids)

    def test_splits_reproducibility(self, loader: BenchmarkLoader):
        """Verify same seed produces identical splits."""
        train1, test1 = loader.get_splits(train_ratio=0.7, seed=123)
        train2, test2 = loader.get_splits(train_ratio=0.7, seed=123)
        assert [q["id"] for q in train1] == [q["id"] for q in train2]
        assert [q["id"] for q in test1] == [q["id"] for q in test2]

    def test_splits_different_seeds(self, loader: BenchmarkLoader):
        """Verify different seeds produce different permutations."""
        train1, _ = loader.get_splits(train_ratio=0.8, seed=1)
        train2, _ = loader.get_splits(train_ratio=0.8, seed=999)
        assert [q["id"] for q in train1] != [q["id"] for q in train2]

    def test_splits_edge_ratios(self, loader: BenchmarkLoader):
        """Verify 1.0 and 0.0 train ratios."""
        train_all, test_none = loader.get_splits(train_ratio=1.0)
        assert len(train_all) == 500
        assert len(test_none) == 0

        train_none, test_all = loader.get_splits(train_ratio=0.0)
        assert len(train_none) == 0
        assert len(test_all) == 500

    def test_splits_invalid_ratio_raises(self, loader: BenchmarkLoader):
        """Verify ValueError on train_ratio out of [0.0, 1.0]."""
        with pytest.raises(ValueError):
            loader.get_splits(train_ratio=1.5)
        with pytest.raises(ValueError):
            loader.get_splits(train_ratio=-0.1)


class TestSummaryStats:
    """Test summary statistics calculation."""

    def test_summary_stats_content(self, loader: BenchmarkLoader):
        """Verify summary_stats structure and expected distributions."""
        stats = loader.summary_stats()

        assert stats["total_queries"] == 500
        assert stats["benchmark_file"] == "avicenna_benchmark_500.json"

        # Check sources
        assert stats["sources"] == {
            "FOLIO": 200,
            "ProofWriter": 150,
            "Curated_Gold": 100,
            "TruthfulQA_OOD": 50,
        }

        # Check query types
        assert stats["query_types"] == {
            "taxonomic": 250,
            "hypothetical": 157,
            "ood": 50,
            "categorical": 43,
        }

        # Check difficulties
        assert stats["difficulties"] == {
            "medium": 310,
            "hard": 115,
            "easy": 75,
        }

        # Check ground truth
        assert stats["ground_truth_distribution"] == {
            "True": 289,
            "False": 161,
            "OOD": 50,
        }
