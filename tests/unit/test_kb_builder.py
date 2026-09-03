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

    def test_multi_parent_indexing(self):
        builder = KnowledgeBaseBuilder()
        # Platypus is both a mammal and an oviparous animal (multi-parent)
        builder.add_taxonomy_edge("platypus", "mammal")
        builder.add_taxonomy_edge("platypus", "oviparous_animal")
        builder.add_taxonomy_edge("dog", "mammal")

        parents = builder.get_parents("platypus")
        assert parents == ["mammal", "oviparous_animal"]

        children = builder.get_children("mammal")
        assert "dog" in children
        assert "platypus" in children

        multi_parents = builder.get_multi_parent_nodes()
        assert "platypus" in multi_parents
        assert len(multi_parents["platypus"]) == 2
        assert "dog" not in multi_parents

    def test_ancestor_and_descendant_queries(self):
        builder = KnowledgeBaseBuilder()
        builder.add_taxonomy_edge("poodle", "dog")
        builder.add_taxonomy_edge("dog", "canine")
        builder.add_taxonomy_edge("canine", "mammal")
        builder.add_taxonomy_edge("mammal", "animal")

        ancestors = builder.get_ancestors("poodle")
        assert {"dog", "canine", "mammal", "animal"}.issubset(ancestors)

        descendants = builder.get_descendants("mammal")
        assert {"canine", "dog", "poodle"}.issubset(descendants)

    def test_inherited_properties(self):
        builder = KnowledgeBaseBuilder()
        builder.add_taxonomy_edge("dog", "mammal")
        builder.add_taxonomy_edge("mammal", "animal")

        builder.add_property("animal", "cellular")
        builder.add_property("mammal", "hair")
        builder.add_property("dog", "barks")

        dog_props = builder.get_inherited_properties("dog")
        assert "barks" in dog_props
        assert "hair" in dog_props
        assert "cellular" in dog_props

    def test_self_loop_ignored(self):
        builder = KnowledgeBaseBuilder()
        builder.add_taxonomy_edge("dog", "dog")
        assert builder.g_t.number_of_nodes() == 0
        assert builder.g_t.number_of_edges() == 0

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

    def test_load_extended_kb_and_verify_dag(self):
        kb_path = Path("data/knowledge_bases/knowledge_base_extended.json")
        if not kb_path.exists():
            pytest.skip("knowledge_base_extended.json not found")

        builder = KnowledgeBaseBuilder()
        builder.load_from_json(kb_path)
        stats = builder.get_statistics()

        assert stats["taxonomy_nodes"] == 1500
        assert stats["taxonomy_edges"] == 2156
        assert stats["is_dag"] is True
        assert stats["cycle_count"] == 0
        assert stats["property_entities"] == 418
        assert stats["total_property_assertions"] == 1929
        assert stats["conditional_rules"] == 194
        assert stats["multi_parent_nodes_count"] == 474
