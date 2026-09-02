"""
Knowledge Base Loader and Graph Constructor.

Builds three interdependent directed graphs (NetworkX) from the KB JSON file:
    G_T  — Taxonomy graph:   IS-A hierarchical DAG (BFS reachability)
    G_P  — Property graph:   entity-property associations (inherited via G_T)
    G_C  — Conditional graph: IF-THEN Modus Ponens rules (O(1) adjacency lookup)

KB JSON Format:
{
  "taxonomies":   { "dog": ["canine", "mammal", "animal", "living_thing"], ... },
  "properties":   { "mammal": ["hair", "warm_blood", "backbone", ...], ... },
  "conditionals": { "water_freezes": ["ice", "solid", "becomes_ice"], ... }
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Set

import networkx as nx

logger = logging.getLogger(__name__)


def normalize_term(term: str) -> str:
    """
    Singularize common English plural forms for KB lookup.

    Args:
        term: Raw noun or entity string.

    Returns:
        Normalized singularized lowercase string.
    """
    w = term.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


class KnowledgeBase:
    """
    Three-graph knowledge base for AvicennaGuard Stage 2 validation.

    All reasoning is deterministic graph traversal. No probabilistic
    components exist in this class or any method it calls.
    """

    def __init__(self, kb_path: str | Path) -> None:
        """
        Initialize and load the KnowledgeBase from a JSON file.

        Args:
            kb_path: Path to the knowledge base JSON file.
        """
        self.kb_path = Path(kb_path)
        self._raw: Dict[str, Any] = {}

        # The three directed graphs
        self.G_T: nx.DiGraph = nx.DiGraph()  # Taxonomy (IS-A)
        self.G_P: Dict[str, Set[str]] = {}   # Properties (entity -> set of props)
        self.G_C: nx.DiGraph = nx.DiGraph()  # Conditionals (IF -> THEN)

        self._load()
        self._build_graphs()

    def _load(self) -> None:
        """Load JSON data from disk with multi-encoding fallback."""
        data = None
        for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
            try:
                with open(self.kb_path, "r", encoding=enc) as f:
                    data = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if data is None:
            raise ValueError(f"Could not load or decode KB JSON from {self.kb_path}")
        self._raw = data
        logger.info("KB loaded: %s", self.kb_path.name)

    def _build_graphs(self) -> None:
        """Construct all three graphs from raw KB data."""
        # --- Taxonomy graph (G_T) ---
        tax_data = self._raw.get("taxonomies") or self._raw.get("taxonomy", {})
        if isinstance(tax_data, dict):
            for child, ancestors in tax_data.items():
                child_norm = child.lower().replace(" ", "_")
                anc_list = ancestors if isinstance(ancestors, list) else [ancestors]
                for ancestor in anc_list:
                    ancestor_norm = ancestor.lower().replace(" ", "_")
                    if child_norm and ancestor_norm and child_norm != ancestor_norm:
                        self.G_T.add_edge(child_norm, ancestor_norm)

        # --- Property graph (G_P) ---
        props_data = self._raw.get("properties", {})
        if isinstance(props_data, dict):
            for entity, props in props_data.items():
                entity_norm = entity.lower().replace(" ", "_")
                prop_set: Set[str] = set()
                prop_list = props if isinstance(props, (list, set)) else [props]
                for p in prop_list:
                    p_norm = p.lower().replace(" ", "_")
                    prop_set.add(p_norm)
                    prop_set.add(normalize_term(p_norm))
                    if p_norm.startswith("has_"):
                        base = p_norm[4:]
                        prop_set.add(base)
                        prop_set.add(normalize_term(base))
                self.G_P[entity_norm] = prop_set

        # --- Conditional graph (G_C) ---
        conds = self._raw.get("conditionals", {})
        if isinstance(conds, dict):
            for condition, consequences in conds.items():
                cond_norm = condition.lower().replace(" ", "_")
                conseq_list = consequences if isinstance(consequences, (list, set)) else [consequences]
                for conseq in conseq_list:
                    conseq_norm = conseq.lower().replace(" ", "_")
                    if cond_norm and conseq_norm:
                        self.G_C.add_edge(cond_norm, conseq_norm)
        elif isinstance(conds, list):
            for rule in conds:
                if isinstance(rule, dict):
                    ant = rule.get("antecedent") or rule.get("condition", "")
                    csq = rule.get("consequent") or rule.get("consequence", "")
                    if ant and csq:
                        self.G_C.add_edge(ant.lower().replace(" ", "_"), csq.lower().replace(" ", "_"))

        logger.info(
            "Graphs built — G_T: %d nodes / %d edges | G_P: %d entities | G_C: %d rules",
            self.G_T.number_of_nodes(),
            self.G_T.number_of_edges(),
            len(self.G_P),
            self.G_C.number_of_edges(),
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Summary statistics of the loaded knowledge base graphs.

        Returns:
            Dictionary containing node, edge, entity, and rule counts.
        """
        return {
            "taxonomy_nodes": self.G_T.number_of_nodes(),
            "taxonomy_edges": self.G_T.number_of_edges(),
            "property_entities": len(self.G_P),
            "conditional_rules": self.G_C.number_of_edges(),
            "kb_file": self.kb_path.name,
        }
