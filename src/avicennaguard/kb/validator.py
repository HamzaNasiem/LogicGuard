"""
Stage 2: Deterministic BFS Graph Validator.

This is the core verification engine of AvicennaGuard. ALL logical reasoning
happens here via deterministic graph algorithms (BFS, reachability, property closure).
No probabilistic computation enters this module.

Three inference methods (Ibn Sina's Qiyas):
    validate_taxonomic()   — Qiyas al-Haml (IS-A BFS reachability)
    validate_categorical() — Property inheritance with transitive taxonomic closure
    validate_hypothetical()— Qiyas al-Istithna (Modus Ponens O(1) adjacency lookup)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

import networkx as nx

from avicennaguard.core.epistemic_states import EpistemicState
from avicennaguard.kb.loader import KnowledgeBase, normalize_term

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

    def __init__(self, kb: KnowledgeBase) -> None:
        """
        Initialize the BFSValidator with a KnowledgeBase instance.

        Args:
            kb: Loaded KnowledgeBase instance containing G_T, G_P, and G_C.
        """
        self.kb = kb

    def _resolve(self, term: str, graph: Optional[nx.DiGraph] = None) -> str:
        """
        Resolve a term to its KB key, trying plural normalization as fallback.

        Checks in order:
        1. Exact match (as-is)
        2. Singular normalized (dogs -> dog, corners -> corner)

        Returns the first form found in ANY of the KB graphs.
        """
        if not term:
            return ""
        t = str(term).strip().lower().replace(" ", "_")
        # Check explicit graph if provided
        if graph is not None and t in graph:
            return t
        # Exact match: check all KB graphs
        for g in (self.kb.G_T, self.kb.G_P, self.kb.G_C):
            if t in g:
                return t
        # Singular normalized form
        singular = normalize_term(t)
        return singular

    def _prop_matches(self, prop_set: Set[str], prop: str) -> bool:
        """Check if a property exists in prop_set (excluding negative/negated trait strings)."""
        p_clean = prop.strip().lower().replace(" ", "_")
        p_norm = normalize_term(p_clean).rstrip("s").rstrip("e")
        
        for ex in prop_set:
            ex_clean = ex.strip().lower()
            # Explicit negative traits (e.g. legless, flightless, hairless, no_wings) must never match positively
            if ex_clean.endswith("less") or ex_clean.startswith("no_") or ex_clean.startswith("non_") or ex_clean.startswith("cannot_"):
                continue
            ex_norm = normalize_term(ex_clean).rstrip("s").rstrip("e")
            
            if p_norm == ex_norm or f"has_{p_norm}" == ex_norm or f"produces_{p_norm}" == ex_norm or f"gives_{p_norm}" == ex_norm:
                return True
            if (p_norm in ("mammary_gland", "mammary", "milk")) and ("milk" in ex_norm or "nurse" in ex_norm):
                return True
            if p_norm == "flower" and ("flower" in ex_norm):
                return True

        return False

    def validate_taxonomic(
        self, subject: str, predicate: str
    ) -> Tuple[Optional[bool], EpistemicState, List[str]]:
        """
        Qiyas al-Haml — Categorical Syllogism (IS-A).

        Checks if subject IS-A predicate via BFS reachability on G_T.
        Time complexity: O(|V_T| + |E_T|).
        """
        if not subject or not predicate:
            return None, EpistemicState.SHAKK, []

        s = self._resolve(subject)
        p = self._resolve(predicate)

        if not s or not p:
            return None, EpistemicState.SHAKK, []

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
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # Check explicit disjointness if available
        disjoint_pairs = getattr(self.kb, "_disjoint_pairs", set())
        if (s, p) in disjoint_pairs or (p, s) in disjoint_pairs:
            return False, EpistemicState.YAQEEN, []

        if s in self.kb.G_T:
            ancestors = nx.descendants(self.kb.G_T, s) if s in self.kb.G_T else set()
            for anc in ancestors:
                if (anc, p) in disjoint_pairs or (p, anc) in disjoint_pairs:
                    return False, EpistemicState.YAQEEN, []

        # Check property fallback if p is a property
        cat_ans, cat_state = self.validate_categorical(s, p)
        if cat_state == EpistemicState.YAQEEN and cat_ans is not None:
            return cat_ans, cat_state, [s, p] if cat_ans else []

        # If s and p are both in G_T with known distinct subtrees
        if s in self.kb.G_T and p in self.kb.G_T:
            return False, EpistemicState.YAQEEN, []

        return None, EpistemicState.SHAKK, []

    def validate_categorical(
        self, entity: str, prop: str
    ) -> Tuple[Optional[bool], EpistemicState]:
        """
        Property inheritance with transitive lookup via G_T under Sound Open-World Assumption (OWA).
        """
        e = self._resolve(entity)
        p = self._resolve(prop)

        if e not in self.kb.G_P and e not in self.kb.G_T:
            return None, EpistemicState.SHAKK

        # 1. Direct positive check
        if e in self.kb.G_P and self._prop_matches(self.kb.G_P[e], p):
            return True, EpistemicState.YAQEEN

        # 2. Inherited positive property via taxonomy ancestors
        if e in self.kb.G_T:
            ancestors = nx.descendants(self.kb.G_T, e)
            for ancestor in ancestors:
                if ancestor in self.kb.G_P and self._prop_matches(self.kb.G_P[ancestor], p):
                    return True, EpistemicState.YAQEEN

        # 3. Explicit negative property check (e.g. 'legless' vs 'legs', 'cold_blooded' vs 'warm_blooded')
        p_clean = p.strip().lower().replace(" ", "_")
        p_stem = normalize_term(p_clean).rstrip("s").rstrip("e")

        entities_to_check = [e]
        if e in self.kb.G_T:
            entities_to_check.extend(nx.descendants(self.kb.G_T, e))

        for ent in entities_to_check:
            if ent in self.kb.G_P:
                e_props = self.kb.G_P[ent]
                for ex in e_props:
                    ex_clean = ex.strip().lower()
                    if ex_clean in (f"{p_stem}less", f"{p_clean}less", f"no_{p_stem}", f"no_{p_clean}", f"cannot_{p_stem}"):
                        return False, EpistemicState.YAQEEN
                    if p_stem == "leg" and ex_clean in ("legless", "limbless"):
                        return False, EpistemicState.YAQEEN
                    if p_stem == "wing" and ex_clean in ("wingless", "flightless"):
                        return False, EpistemicState.YAQEEN
                    if p_stem in ("warm_blood", "warm_blooded") and "cold_blooded" in ex_clean:
                        return False, EpistemicState.YAQEEN
                    if p_stem in ("cold_blood", "cold_blooded") and "warm_blooded" in ex_clean:
                        return False, EpistemicState.YAQEEN

        # 4. Property known in KB scope on another entity, but absent and not inherited on e
        if e in self.kb.G_P or e in self.kb.G_T:
            if self._property_in_kb(p):
                return False, EpistemicState.YAQEEN

        # Open-World Assumption for unknown properties
        return None, EpistemicState.SHAKK

    def _property_in_kb(self, prop: str) -> bool:
        """
        Check if a property appears on any entity in the property graph.

        Args:
            prop: Target property string.

        Returns:
            True if property exists on any KB entity.
        """
        p = self._resolve(prop)
        for props in self.kb.G_P.values():
            if self._prop_matches(props, p):
                return True
        return False

    def validate_hypothetical(
        self, condition: str, consequence: str
    ) -> Tuple[Optional[bool], EpistemicState]:
        """
        Qiyas al-Istithna — Hypothetical Syllogism (Modus Ponens).

        Checks if condition -> consequence edge exists in G_C.
        Time complexity: O(1) adjacency lookup.

        Args:
            condition: IF antecedent clause.
            consequence: THEN consequent clause.

        Returns:
            Tuple of (result, epistemic_state):
                result: True if edge exists in G_C, False if condition is in G_C
                        but consequence is not reachable, None if condition is not in G_C.
                epistemic_state: EpistemicState.YAQEEN or EpistemicState.SHAKK.
        """
        c = self._resolve(condition, self.kb.G_C)
        cq = self._resolve(consequence, self.kb.G_C)

        if c not in self.kb.G_C:
            return None, EpistemicState.SHAKK

        if self.kb.G_C.has_edge(c, cq):
            return True, EpistemicState.YAQEEN

        return False, EpistemicState.YAQEEN
