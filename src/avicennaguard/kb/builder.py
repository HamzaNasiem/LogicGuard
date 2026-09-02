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
        c = child.strip().lower().replace(" ", "_")
        p = parent.strip().lower().replace(" ", "_")
        if c and p and c != p:
            self.g_t.add_edge(c, p)

    def add_property(self, entity: str, prop: str) -> None:
        """Associate a property with an entity."""
        e = entity.strip().lower().replace(" ", "_")
        p = prop.strip().lower().replace(" ", "_")
        if e and p:
            if e not in self.g_p:
                self.g_p[e] = set()
            self.g_p[e].add(p)

    def add_conditional(self, antecedent: str, consequent: str) -> None:
        """Add a conditional implication rule: antecedent => consequent."""
        a = antecedent.strip().lower().replace(" ", "_")
        c = consequent.strip().lower().replace(" ", "_")
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

    def get_parents(self, entity: str) -> List[str]:
        """Get direct taxonomic parents (hypernyms) of an entity."""
        e = entity.strip().lower().replace(" ", "_")
        if e in self.g_t:
            return sorted(list(self.g_t.successors(e)))
        return []

    def get_children(self, entity: str) -> List[str]:
        """Get direct taxonomic children (hyponyms) of an entity."""
        e = entity.strip().lower().replace(" ", "_")
        if e in self.g_t:
            return sorted(list(self.g_t.predecessors(e)))
        return []

    def get_ancestors(self, entity: str) -> Set[str]:
        """Get all taxonomic ancestors (transitive hypernyms) of an entity."""
        e = entity.strip().lower().replace(" ", "_")
        if e in self.g_t:
            return nx.descendants(self.g_t, e)
        return set()

    def get_descendants(self, entity: str) -> Set[str]:
        """Get all taxonomic descendants (transitive hyponyms) of an entity."""
        e = entity.strip().lower().replace(" ", "_")
        if e in self.g_t:
            return nx.ancestors(self.g_t, e)
        return set()

    def get_multi_parent_nodes(self) -> Dict[str, List[str]]:
        """Find all nodes in G_T with multiple direct parents (poly-hierarchy)."""
        multi_parents = {}
        for node in self.g_t.nodes():
            parents = list(self.g_t.successors(node))
            if len(parents) > 1:
                multi_parents[node] = sorted(parents)
        return multi_parents

    def get_inherited_properties(self, entity: str) -> Set[str]:
        """Retrieve all direct and inherited properties for an entity via G_T traversal."""
        e = entity.strip().lower().replace(" ", "_")
        props = set()
        if e in self.g_p:
            props.update(self.g_p[e])
        if e in self.g_t:
            for ancestor in nx.descendants(self.g_t, e):
                if ancestor in self.g_p:
                    props.update(self.g_p[ancestor])
        return props

    def get_statistics(self) -> Dict[str, Any]:
        """Compute structural metrics of the multi-relational KB."""
        is_dag, cycles = self.validate_acyclicity()
        total_properties = sum(len(props) for props in self.g_p.values())
        unique_properties = set()
        for props in self.g_p.values():
            unique_properties.update(props)
        multi_parents = self.get_multi_parent_nodes()

        return {
            "taxonomy_nodes": self.g_t.number_of_nodes(),
            "taxonomy_edges": self.g_t.number_of_edges(),
            "is_dag": is_dag,
            "cycle_count": len(cycles),
            "multi_parent_nodes_count": len(multi_parents),
            "property_entities": len(self.g_p),
            "total_property_assertions": total_properties,
            "unique_properties": len(unique_properties),
            "conditional_rules": len(self.g_c),
        }

    def load_from_json(self, filepath: str | Path) -> None:
        """Load an existing knowledge base JSON into the builder."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"KB file not found at: {path}")

        data = None
        for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    data = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        if data is None:
            raise ValueError(f"Could not decode JSON from {path}")

        # Ingest taxonomy (supports 'taxonomies' or 'taxonomy')
        tax = data.get("taxonomies") or data.get("taxonomy", {})
        if isinstance(tax, dict):
            for child, parents in tax.items():
                if isinstance(parents, list):
                    for p in parents:
                        self.add_taxonomy_edge(child, p)
                elif isinstance(parents, str):
                    self.add_taxonomy_edge(child, parents)

        # Ingest properties
        props = data.get("properties", {})
        if isinstance(props, dict):
            for entity, prop_list in props.items():
                if isinstance(prop_list, (list, set)):
                    for p in prop_list:
                        self.add_property(entity, p)
                elif isinstance(prop_list, str):
                    self.add_property(entity, prop_list)

        # Ingest conditionals (supports dict or list of dicts)
        conds = data.get("conditionals", {})
        if isinstance(conds, dict):
            for condition, consequences in conds.items():
                if isinstance(consequences, list):
                    for c in consequences:
                        self.add_conditional(condition, c)
                elif isinstance(consequences, str):
                    self.add_conditional(condition, consequences)
        elif isinstance(conds, list):
            for rule in conds:
                if isinstance(rule, dict):
                    ant = rule.get("antecedent") or rule.get("condition", "")
                    csq = rule.get("consequent") or rule.get("consequence", "")
                    if ant and csq:
                        self.add_conditional(ant, csq)

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
            "taxonomies": taxonomy_dict,
            "properties": properties_dict,
            "conditionals": self.g_c,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data
