"""
AvicennaGuard: Neuro-Symbolic Middleware for Hallucination Interception in LLMs.

A hybrid two-stage pipeline that enforces deterministic logical constraints
on LLM outputs using the Avicennian (Ibn Sina) syllogistic framework (Qiyas).

Architecture:
    Stage 1 (Probabilistic) — Constrained LLM semantic parser
    Stage 2 (Deterministic) — BFS graph validator on NetworkX knowledge base

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

__version__ = "2.0.0"
__authors__ = ["Hamza Naseem", "Moiz Ali"]
