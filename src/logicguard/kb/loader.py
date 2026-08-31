"""
Knowledge Base Loader and Graph Constructor.

Builds three interdependent directed graphs (NetworkX) from the KB JSON file:
    G_T  — Taxonomy graph:   115+ nodes, 136+ IS-A edges (BFS traversal)
    G_P  — Property graph:   115+ entity-property associations (inherited via G_T)
    G_C  — Conditional graph: 49+ IF-THEN Modus Ponens rules (O(1) lookup)

KB JSON Format (knowledge_base_extended.json):
{
  "taxonomies":   { "dog": ["canine", "mammal", "animal", "living_thing"], ... },
  "properties":   { "mammal": ["hair", "warm_blood", "backbone", ...], ... },
  "conditionals": { "water_freezes": ["ice", "solid", "becomes_ice"], ... }
}
"""

import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def normalize_term(term: str) -> str:
    """Singularize common English plural forms for KB lookup."""
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
    Three-graph knowledge base for LogicGuard Stage 2 validation.

    All reasoning is deterministic graph traversal. No probabilistic
    components exist in this class or any method it calls.
    """

    def __init__(self, kb_path: str | Path):
        self.kb_path = Path(kb_path)
        self._raw: dict = {}

        # The three directed graphs
        self.G_T: nx.DiGraph = nx.DiGraph()  # Taxonomy (IS-A)
        self.G_P: dict       = {}            # Properties (entity → set of props)
        self.G_C: nx.DiGraph = nx.DiGraph()  # Conditionals (IF → THEN)

        self._load()
        self._build_graphs()

    def _load(self) -> None:
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)
        logger.info("KB loaded: %s", self.kb_path.name)

    def _build_graphs(self) -> None:
        """Construct all three graphs from raw KB data."""
        # --- Taxonomy graph (G_T) ---
        for child, ancestors in self._raw.get("taxonomies", {}).items():
            child_norm = child.lower().replace(" ", "_")
            prev = child_norm
            for ancestor in ancestors:
                ancestor_norm = ancestor.lower().replace(" ", "_")
                self.G_T.add_edge(prev, ancestor_norm)
                prev = ancestor_norm

        # --- Property graph (G_P) ---
        # Stored as dict for O(1) property lookup before BFS inheritance
        for entity, props in self._raw.get("properties", {}).items():
            entity_norm = entity.lower().replace(" ", "_")
            prop_set = set()
            for p in props:
                p_norm = p.lower().replace(" ", "_")
                prop_set.add(p_norm)
                prop_set.add(normalize_term(p_norm))
                if p_norm.startswith("has_"):
                    base = p_norm[4:]
                    prop_set.add(base)
                    prop_set.add(normalize_term(base))
            self.G_P[entity_norm] = prop_set

        # --- Conditional graph (G_C) ---
        for condition, consequences in self._raw.get("conditionals", {}).items():
            cond_norm = condition.lower().replace(" ", "_")
            for conseq in consequences:
                conseq_norm = conseq.lower().replace(" ", "_")
                self.G_C.add_edge(cond_norm, conseq_norm)

        logger.info(
            "Graphs built — G_T: %d nodes / %d edges | G_P: %d entities | G_C: %d rules",
            self.G_T.number_of_nodes(),
            self.G_T.number_of_edges(),
            len(self.G_P),
            self.G_C.number_of_edges(),
        )

    @property
    def stats(self) -> dict:
        return {
            "taxonomy_nodes":      self.G_T.number_of_nodes(),
            "taxonomy_edges":      self.G_T.number_of_edges(),
            "property_entities":   len(self.G_P),
            "conditional_rules":   self.G_C.number_of_edges(),
            "kb_file":             self.kb_path.name,
        }
