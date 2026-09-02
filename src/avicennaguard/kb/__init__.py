"""
Knowledge Base package for AvicennaGuard.

Provides graph construction, multi-relational loading, DAG cycle validation,
and deterministic Stage 2 BFS reachability verification:
  - KnowledgeBase: Three-graph container (G_T, G_P, G_C)
  - KnowledgeBaseBuilder: Programmatic extraction, synthesis, and DAG validation
  - BFSValidator: Deterministic Stage 2 graph traversal validator
  - normalize_term: Singularization and lexical normalization utility
"""

from avicennaguard.kb.builder import KnowledgeBaseBuilder
from avicennaguard.kb.loader import KnowledgeBase, normalize_term
from avicennaguard.kb.validator import BFSValidator

__all__ = [
    "BFSValidator",
    "KnowledgeBase",
    "KnowledgeBaseBuilder",
    "normalize_term",
]
