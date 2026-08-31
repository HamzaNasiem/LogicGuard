"""
Stage 2: Deterministic BFS Graph Validator.

This is the core of LogicGuard. ALL logical reasoning happens here.
No probabilistic computation enters this module.

Three inference methods (Ibn Sina's Qiyas):
    validate_taxonomic()   — Qiyas al-Haml (IS-A BFS traversal)
    validate_categorical() — Property inheritance with transitive lookup
    validate_hypothetical()— Qiyas al-Istithna (Modus Ponens O(1) check)
"""

import time
import logging
from typing import Optional

import networkx as nx

from logicguard.core.epistemic_states import EpistemicState, QueryType
from logicguard.kb.loader import KnowledgeBase, normalize_term

logger = logging.getLogger(__name__)

# Alias for backward compatibility
_normalize = normalize_term


class BFSValidator:
    """
    Deterministic BFS graph validator implementing Ibn Sina's three Qiyas forms.

    Precision = 100% guarantee:
        A false positive requires BFS to erroneously report no path when
        one exists. BFS graph reachability is provably correct on a correct KB.
        This is a formal guarantee, not an empirical observation.

    Recall gaps:
        When an entity is absent from the KB, the validator returns SHAKK
        and defers to the LLM. These are missed interceptions (FN), not
        false alarms (FP). Honest KB coverage gaps are explicitly reported.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def _resolve(self, term: str, graph: nx.DiGraph | None = None) -> str:
        """Resolve a term to its KB key, trying plural normalization as fallback.

        Checks in order:
        1. Exact match (as-is)
        2. Singular normalized (dogs→dog, corners→corner)
        Returns the first form found in ANY of the KB graphs.
        """
        t = term.lower().replace(" ", "_")
        # Exact match: check all KB graphs
        for g in (self.kb.G_T, self.kb.G_P, self.kb.G_C):
            if t in g:
                return t
        # Singular normalized form
        singular = _normalize(t)
        return singular

    def _prop_matches(self, prop_set: set[str], prop: str) -> bool:
        """Check if a property or its normalized / prefixed forms exist in prop_set."""
        p_norm = _normalize(prop)
        return (
            prop in prop_set
            or p_norm in prop_set
            or f"has_{prop}" in prop_set
            or f"has_{p_norm}" in prop_set
        )

    def validate_taxonomic(
        self, subject: str, predicate: str
    ) -> tuple[Optional[bool], EpistemicState, list[str]]:
        """
        Qiyas al-Haml — Categorical Syllogism (IS-A).

        Checks if subject IS-A predicate via BFS on G_T.
        Time complexity: O(|V_T| + |E_T|)

        Returns:
            (result, epistemic_state, path)
            result: True/False/None (None = SHAKK, entity not in KB)
            path:   BFS path for audit trail (EU AI Act compliance)
        """
        t0 = time.perf_counter()

        s = self._resolve(subject)
        p = self._resolve(predicate)

        # Check KB coverage (SHAKK condition)
        if s not in self.kb.G_T and s not in self.kb.G_P:
            return None, EpistemicState.SHAKK, []

        # Exact match (trivially true: "Is a dog a dog?")
        if s == p:
            return True, EpistemicState.YAQEEN, [s]

        # BFS reachability — O(|V| + |E|)
        try:
            path = nx.shortest_path(self.kb.G_T, source=s, target=p)
            logger.debug("BFS path found: %s", " → ".join(path))
            return True, EpistemicState.YAQEEN, path
        except nx.NetworkXNoPath:
            return False, EpistemicState.YAQEEN, []
        except nx.NodeNotFound:
            return None, EpistemicState.SHAKK, []

    def validate_categorical(
        self, entity: str, prop: str
    ) -> tuple[Optional[bool], EpistemicState]:
        """
        Property inheritance with transitive lookup via G_T.

        has_prop(e, π) ⟺ (e, π) ∈ G_P  ∨  ∃a ∈ anc_G_T(e) : (a, π) ∈ G_P

        This ensures "Do all dogs have hair?" returns YAQEEN even without
        a direct dog→hair edge, via: dog → canine → mammal → hair.
        """
        e = self._resolve(entity)
        p = self._resolve(prop)

        if e not in self.kb.G_P and e not in self.kb.G_T:
            return None, EpistemicState.SHAKK

        # Direct property check
        if e in self.kb.G_P and self._prop_matches(self.kb.G_P[e], p):
            return True, EpistemicState.YAQEEN

        # Inherited property via taxonomy ancestors
        if e in self.kb.G_T:
            ancestors = nx.descendants(self.kb.G_T, e)
            for ancestor in ancestors:
                if ancestor in self.kb.G_P and self._prop_matches(self.kb.G_P[ancestor], p):
                    return True, EpistemicState.YAQEEN

        # Unknown property in KB scope → SHAKK (open-world safe, preserves FP=0)
        if e in self.kb.G_P or e in self.kb.G_T:
            if not self._property_in_kb(p):
                return None, EpistemicState.SHAKK
            return False, EpistemicState.YAQEEN

        return None, EpistemicState.SHAKK

    def _property_in_kb(self, prop: str) -> bool:
        """True if prop appears on any entity in the property graph."""
        p = self._resolve(prop)
        for props in self.kb.G_P.values():
            if self._prop_matches(props, p):
                return True
        return False

    def validate_hypothetical(
        self, condition: str, consequence: str
    ) -> tuple[Optional[bool], EpistemicState]:
        """
        Qiyas al-Istithna — Hypothetical Syllogism (Modus Ponens).

        Checks if condition → consequence edge exists in G_C.
        Time complexity: O(1) adjacency lookup.
        """
        c  = self._resolve(condition,  self.kb.G_C)
        cq = self._resolve(consequence, self.kb.G_C)

        if c not in self.kb.G_C:
            return None, EpistemicState.SHAKK

        if self.kb.G_C.has_edge(c, cq):
            return True, EpistemicState.YAQEEN

        return False, EpistemicState.YAQEEN
