"""
Unit Tests for KnowledgeBaseBuilder & DAG Validation
===================================================
"""

import json
import pytest
from pathlib import Path
from avicennaguard.kb.builder import KnowledgeBaseBuilder


class TestKnowledgeBaseBuilder:
    def test_add_taxonomy_edge_and_dag_validation(self):
        builder = KnowledgeBaseBuilder()
        builder.add_taxonomy_edge("golden_eagle", "eagle")
        builder.add_taxonomy_edge("eagle", "raptor")
        builder.add_taxonomy_edge("raptor", "bird")

        is_dag, cycles = builder.validate_acyclicity()
        assert is_dag is True
        assert len(cycles) == 0

        stats = builder.get_statistics()
        assert stats["taxonomy_nodes"] == 4
        assert stats["taxonomy_edges"] == 3
        assert stats["is_dag"] is True

    def test_cycle_detection(self):
        builder = KnowledgeBaseBuilder()
        builder.add_taxonomy_edge("A", "B")
        builder.add_taxonomy_edge("B", "C")
        builder.add_taxonomy_edge("C", "A")  # intentional cycle

        is_dag, cycles = builder.validate_acyclicity()
        assert is_dag is False
        assert len(cycles) > 0

    def test_properties_and_conditionals(self):
        builder = KnowledgeBaseBuilder()
        builder.add_property("bird", "feathers")
        builder.add_property("bird", "wings")
        builder.add_property("mammal", "hair")

        builder.add_conditional("water_freezes", "ice")

        stats = builder.get_statistics()
        assert stats["property_entities"] == 2
        assert stats["total_property_assertions"] == 3
        assert stats["conditional_rules"] == 1

    def test_export_and_reload(self, tmp_path):
        builder = KnowledgeBaseBuilder()
        builder.add_taxonomy_edge("robin", "bird")
        builder.add_property("bird", "lay_eggs")
        builder.add_conditional("rain", "wet_ground")

        out_file = tmp_path / "test_kb.json"
        builder.export_to_json(out_file)

        assert out_file.exists()

        # Reload into new builder
        builder2 = KnowledgeBaseBuilder()
        builder2.load_from_json(out_file)

        stats2 = builder2.get_statistics()
        assert stats2["taxonomy_nodes"] == 2
        assert stats2["total_property_assertions"] == 1
        assert stats2["conditional_rules"] == 1
        assert stats2["is_dag"] is True
