"""
Dynamic Context-to-Graph Compiler for AvicennaGuard.
===================================================
Parses arbitrary multi-premise natural language contexts into a temporary
local proof DAG K_context = (G_T, G_P, G_C) and unifies it with the global
background ontology G_global via the graph fusion operator (⊕).

Reference:
    Ibn Sina (Avicenna), Kitab al-Burhan (Book of Demonstration), c. 1020 CE.
    Pan et al., Logic-LM (EMNLP 2023), Zhou et al., LINC (ACL 2024).
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx

from avicennaguard.kb.loader import KnowledgeBase, normalize_term

logger = logging.getLogger(__name__)


class ContextGraphBuilder:
    """
    Compiles multi-premise natural language narratives into a dynamic proof graph.
    """

    def __init__(self, global_kb: Optional[KnowledgeBase] = None) -> None:
        """
        Initialize the context graph builder.
        
        Args:
            global_kb: Optional background KnowledgeBase instance.
        """
        self.global_kb = global_kb

    def compile_context(self, premises: List[str]) -> Tuple[nx.DiGraph, Dict[str, Set[str]], nx.DiGraph, Set[Tuple[str, str]]]:
        """
        Compile a list of natural language premises into (G_T, G_P, G_C, disjoint_pairs).
        
        Args:
            premises: List of natural language premise strings.
            
        Returns:
            Tuple of:
                G_T: Taxonomic IS-A directed acyclic graph.
                G_P: Property mapping entity -> set of properties.
                G_C: Conditional implication DAG (Modus Ponens).
                disjoint_pairs: Set of (entity_a, entity_b) disjoint pairs.
        """
        G_T = nx.DiGraph()
        G_P: Dict[str, Set[str]] = {}
        G_C = nx.DiGraph()
        disjoint_pairs: Set[Tuple[str, str]] = set()

        for premise in premises:
            p_text = premise.strip()
            if not p_text:
                continue
                
            self._parse_premise_into_graphs(p_text, G_T, G_P, G_C, disjoint_pairs)

        return G_T, G_P, G_C, disjoint_pairs

    def build_unified_kb(self, premises: List[str]) -> KnowledgeBase:
        """
        Build a unified KnowledgeBase instance fusing local context and global ontology.
        
        Args:
            premises: List of natural language premise strings.
            
        Returns:
            New KnowledgeBase instance containing K_unified = K_context ⊕ G_global.
        """
        c_G_T, c_G_P, c_G_C, c_disjoint = self.compile_context(premises)

        # Create a blank KB structure
        unified_kb = KnowledgeBase.__new__(KnowledgeBase)
        unified_kb.G_T = nx.DiGraph()
        unified_kb.G_P = {}
        unified_kb.G_C = nx.DiGraph()

        # 1. Merge Global Background if available
        if self.global_kb is not None:
            unified_kb.G_T.add_edges_from(self.global_kb.G_T.edges(data=True))
            for k, v in self.global_kb.G_P.items():
                unified_kb.G_P[k] = set(v)
            unified_kb.G_C.add_edges_from(self.global_kb.G_C.edges(data=True))

        # 2. Layer Local Context DAG (Local Context Priority)
        unified_kb.G_T.add_edges_from(c_G_T.edges(data=True))
        for k, v in c_G_P.items():
            if k not in unified_kb.G_P:
                unified_kb.G_P[k] = set()
            unified_kb.G_P[k].update(v)
        unified_kb.G_C.add_edges_from(c_G_C.edges(data=True))

        # Store disjointness metadata
        unified_kb._disjoint_pairs = c_disjoint
        return unified_kb

    def _clean_token(self, token: str) -> str:
        """Clean and normalize entity or property tokens."""
        t = token.strip().lower()
        t = re.sub(r"^(a|an|the|all|every|each|some)\s+", "", t)
        t = re.sub(r"[^\w\s]", "", t)
        t = t.strip().replace(" ", "_")
        return normalize_term(t)

    def _parse_premise_into_graphs(
        self,
        premise: str,
        G_T: nx.DiGraph,
        G_P: Dict[str, Set[str]],
        G_C: nx.DiGraph,
        disjoint: Set[Tuple[str, str]]
    ) -> None:
        """Parse a single premise sentence into corresponding graph structures."""
        text = premise.strip()
        text_clean = text.rstrip(".")

        # Pattern 1: Conditional Implication: "If P, then Q" / "If P then Q"
        m_if = re.search(r"^if\s+(.+?)(?:,\s*then|\s+then)\s+(.+)$", text_clean, re.IGNORECASE)
        if m_if:
            cond = self._clean_token(m_if.group(1))
            conseq = self._clean_token(m_if.group(2))
            if cond and conseq:
                G_C.add_edge(cond, conseq)
                return

        # Pattern 2: Negative / Disjointness: "No X are Y" / "No X is a Y"
        m_no = re.search(r"^no\s+(.+?)\s+(?:are|is(?:\s+a|\s+an)?)\s+(.+)$", text_clean, re.IGNORECASE)
        if m_no:
            sub = self._clean_token(m_no.group(1))
            pred = self._clean_token(m_no.group(2))
            if sub and pred:
                disjoint.add((sub, pred))
                disjoint.add((pred, sub))
                return

        # Pattern 3: Universal Taxonomic: "All X are Y" / "Every X is a Y" / "Each X is a Y"
        m_all = re.search(r"^(?:all|every|each)\s+(.+?)\s+(?:are|is(?:\s+a|\s+an)?)\s+(.+)$", text_clean, re.IGNORECASE)
        if m_all:
            sub = self._clean_token(m_all.group(1))
            pred = self._clean_token(m_all.group(2))
            if sub and pred and sub != pred:
                G_T.add_edge(sub, pred)
                return

        # Pattern 4: Property Assertion: "All X have P" / "Every X has P"
        m_prop = re.search(r"^(?:all|every|each)\s+(.+?)\s+(?:have|has)\s+(.+)$", text_clean, re.IGNORECASE)
        if m_prop:
            sub = self._clean_token(m_prop.group(1))
            prop = self._clean_token(m_prop.group(2))
            if sub and prop:
                if sub not in G_P:
                    G_P[sub] = set()
                G_P[sub].add(prop)
                G_P[sub].add(f"has_{prop}")
                return

        # Pattern 5: Instance / Singular: "[A/An] X is a Y" / "X is an Y" / "X is Y"
        m_is = re.search(r"^(?:a\s+|an\s+)?(.+?)\s+is(?:\s+a|\s+an)?\s+(.+)$", text_clean, re.IGNORECASE)
        if m_is:
            sub = self._clean_token(m_is.group(1))
            pred = self._clean_token(m_is.group(2))
            if sub and pred and sub != pred:
                G_T.add_edge(sub, pred)
                return
