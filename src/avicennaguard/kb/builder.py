"""
AvicennaGuard Knowledge Base Builder & Validator Module
======================================================
Provides programmatic extraction, synthesis, cycle-checking, and export
for AvicennaGuard's multi-relational knowledge base:
  - G_T: Directed Acyclic Graph (DAG) for Taxonomic IS-A chains
  - G_P: Direct and inherited property mappings
  - G_C: Conditional causal / scientific IF-THEN implication rules
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """Programmatic builder and validator for Avicennian multi-relational knowledge bases."""

    def __init__(self):
        self.g_t: nx.DiGraph = nx.DiGraph()
        self.g_p: Dict[str, Set[str]] = {}
        self.g_c: List[Dict[str, str]] = []

    def add_taxonomy_edge(self, child: str, parent: str) -> None:
        """Add a directed IS-A edge: child -> parent."""
        c = child.strip().lower()
        p = parent.strip().lower()
        if c and p and c != p:
            self.g_t.add_edge(c, p)

    def add_property(self, entity: str, prop: str) -> None:
        """Associate a property with an entity."""
        e = entity.strip().lower()
        p = prop.strip().lower()
        if e and p:
            if e not in self.g_p:
                self.g_p[e] = set()
            self.g_p[e].add(p)

    def add_conditional(self, antecedent: str, consequent: str) -> None:
        """Add a conditional implication rule: antecedent => consequent."""
        a = antecedent.strip().lower()
        c = consequent.strip().lower()
        if a and c:
            rule = {"antecedent": a, "consequent": c}
            if rule not in self.g_c:
                self.g_c.append(rule)

    def validate_acyclicity(self) -> Tuple[bool, List[List[str]]]:
        """Verify that G_T is a strictly Acyclic Directed Graph (DAG).
        Returns (is_dag, cycles_found).
        """
        is_dag = nx.is_directed_acyclic_graph(self.g_t)
        cycles = []
        if not is_dag:
            try:
                cycles = list(nx.simple_cycles(self.g_t))
            except Exception as e:
                logger.error(f"Error computing simple cycles: {e}")
        return is_dag, cycles

    def get_statistics(self) -> Dict[str, Any]:
        """Compute structural metrics of the multi-relational KB."""
        is_dag, cycles = self.validate_acyclicity()
        total_properties = sum(len(props) for props in self.g_p.values())
        return {
            "taxonomy_nodes": self.g_t.number_of_nodes(),
            "taxonomy_edges": self.g_t.number_of_edges(),
            "is_dag": is_dag,
            "cycle_count": len(cycles),
            "property_entities": len(self.g_p),
            "total_property_assertions": total_properties,
            "conditional_rules": len(self.g_c),
        }

    def load_from_json(self, filepath: str | Path) -> None:
        """Load an existing knowledge base JSON into the builder."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"KB file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ingest taxonomy
        tax = data.get("taxonomy", {})
        for child, parents in tax.items():
            if isinstance(parents, list):
                for p in parents:
                    self.add_taxonomy_edge(child, p)
            elif isinstance(parents, str):
                self.add_taxonomy_edge(child, parents)

        # Ingest properties
        props = data.get("properties", {})
        for entity, prop_list in props.items():
            if isinstance(prop_list, list):
                for p in prop_list:
                    self.add_property(entity, p)
            elif isinstance(prop_list, str):
                self.add_property(entity, prop_list)

        # Ingest conditionals
        conds = data.get("conditionals", [])
        for rule in conds:
            if isinstance(rule, dict) and "antecedent" in rule and "consequent" in rule:
                self.add_conditional(rule["antecedent"], rule["consequent"])

    def export_to_json(self, filepath: str | Path) -> Dict[str, Any]:
        """Export the verified knowledge base to JSON format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build taxonomy dict
        taxonomy_dict: Dict[str, List[str]] = {}
        for u, v in self.g_t.edges():
            if u not in taxonomy_dict:
                taxonomy_dict[u] = []
            if v not in taxonomy_dict[u]:
                taxonomy_dict[u].append(v)

        # Build properties dict
        properties_dict = {k: sorted(list(v)) for k, v in sorted(self.g_p.items())}

        export_data = {
            "metadata": {
                "system": "AvicennaGuard Knowledge Base",
                "version": "2.0.0",
                "statistics": self.get_statistics(),
            },
            "taxonomy": taxonomy_dict,
            "properties": properties_dict,
            "conditionals": self.g_c,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data
