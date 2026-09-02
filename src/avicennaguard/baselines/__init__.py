"""
AvicennaGuard SOTA Baselines Package.
====================================
Standard benchmark baselines for hallucination detection and neuro-symbolic reasoning:

1. SelfCheckGPT (Manakul et al., EMNLP 2023) — Zero-resource stochastic consistency sampling
2. Dense RAG (Lewis et al., NeurIPS 2020)   — Embedding/TF-IDF KB triple retrieval + LLM generation
3. Logic-LM (Pan et al., EMNLP 2023)        — First-Order Logic translation + symbolic solver
"""

from avicennaguard.baselines.dense_rag import (
    DenseEmbeddingRetriever,
    DenseRAGBaseline,
    DenseRAGResult,
    SparseTFIDFRetriever,
    kb_to_facts,
)
from avicennaguard.baselines.logic_lm import (
    LogicLMBaseline,
    LogicLMResult,
    RuleBasedLogicTranslator,
    SymbolicLogicSolver,
)
from avicennaguard.baselines.metrics import (
    compute_classification_metrics,
    compute_group_metrics,
    format_metrics_summary,
    parse_bool_answer,
)
from avicennaguard.baselines.selfcheckgpt import (
    SelfCheckGPTBaseline,
    SelfCheckGPTResult,
)

__all__ = [
    # SelfCheckGPT
    "SelfCheckGPTBaseline",
    "SelfCheckGPTResult",
    # Dense RAG
    "DenseRAGBaseline",
    "DenseRAGResult",
    "DenseEmbeddingRetriever",
    "SparseTFIDFRetriever",
    "kb_to_facts",
    # Logic-LM
    "LogicLMBaseline",
    "LogicLMResult",
    "RuleBasedLogicTranslator",
    "SymbolicLogicSolver",
    # Metrics
    "compute_classification_metrics",
    "compute_group_metrics",
    "format_metrics_summary",
    "parse_bool_answer",
]
