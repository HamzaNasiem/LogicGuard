"""
Stage 1 Semantic Parsers package for AvicennaGuard.

Exports neural semantic classifiers, constrained LLM extractors, and
deterministic regex fallback parsers for proposition extraction:
  - DebertaParser: Fast sub-30ms transformer/sklearn classifier + slot extractor
  - LLMParser: Constrained LLM proposition parser (temperature=0.0, JSON mode)
  - RegexParser: Deterministic pattern-matching fallback parser
  - extract_taxonomic, extract_categorical, extract_hypothetical: Typed regex helpers
"""

from avicennaguard.parsers.deberta_parser import DebertaParser
from avicennaguard.parsers.llm_parser import LLMParser
from avicennaguard.parsers.regex_parser import RegexParser
from avicennaguard.parsers.typed_regex import (
    extract_categorical,
    extract_hypothetical,
    extract_taxonomic,
)

__all__ = [
    "DebertaParser",
    "LLMParser",
    "RegexParser",
    "extract_categorical",
    "extract_hypothetical",
    "extract_taxonomic",
]
