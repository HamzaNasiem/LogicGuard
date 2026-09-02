"""
AvicennaGuard: Neuro-Symbolic Middleware for Hallucination Interception in LLMs.

A hybrid two-stage pipeline that enforces deterministic logical constraints
on LLM outputs using the Avicennian (Ibn Sina) syllogistic framework (Qiyas).

Architecture:
    Stage 1 (Probabilistic) — Constrained neural / LLM semantic proposition parser
    Stage 2 (Deterministic) — BFS reachability validator on NetworkX knowledge base

Epistemic States:
    YAQEEN  — Certainty: BFS path confirmed, override LLM
    WAHM    — Illusion: LLM contradicts BFS, intercept hallucination
    SHAKK   — Doubt: entity not in KB, defer to LLM
    ZANN    — Probability: semantic match only, flag with confidence

Authors:
    Hamza Naseem <hamza.naseem2027@gmail.com>
    Moiz Ali <moizk5590@gmail.com>

DOI: https://doi.org/10.5281/zenodo.18745460
"""

from __future__ import annotations

from avicennaguard.core.epistemic_states import (
    EpistemicState,
    QueryType,
    ValidatorResult,
)
from avicennaguard.data.benchmark_loader import BenchmarkLoader
from avicennaguard.eval.benchmark_runner import BenchmarkRunner
from avicennaguard.eval.statistical_analyzer import StatisticalAnalyzer
from avicennaguard.kb.builder import KnowledgeBaseBuilder
from avicennaguard.kb.loader import KnowledgeBase, normalize_term
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.deberta_parser import DebertaParser
from avicennaguard.parsers.llm_parser import LLMParser
from avicennaguard.parsers.regex_parser import RegexParser
from avicennaguard.pipeline.avicennaguard import AvicennaGuard, LogicGuard

__version__ = "2.0.0"
__authors__ = ["Hamza Naseem", "Moiz Ali"]

__all__ = [
    # Metadata
    "__version__",
    "__authors__",
    # Main Pipelines
    "AvicennaGuard",
    "LogicGuard",
    # Core Epistemics
    "EpistemicState",
    "QueryType",
    "ValidatorResult",
    # Knowledge Base
    "KnowledgeBase",
    "KnowledgeBaseBuilder",
    "BFSValidator",
    "normalize_term",
    # Parsers
    "DebertaParser",
    "LLMParser",
    "RegexParser",
    # Evaluation & Data
    "BenchmarkLoader",
    "BenchmarkRunner",
    "StatisticalAnalyzer",
]
