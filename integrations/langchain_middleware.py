"""
LangChain / LiteLLM middleware wrapper for AvicennaGuard.

Usage:
    from integrations.langchain_middleware import AvicennaGuardMiddleware

    lg = AvicennaGuardMiddleware(kb_path="knowledge_base_1200.json")
    result = lg.guard("Are all fish mammals?", llm_answer="yes")
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

# Allow import from repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.pipeline.avicennaguard import AvicennaGuard


@dataclass
class GuardResult:
    question: str
    epistemic_state: str
    final_answer: str
    intercepted: bool
    covered: bool
    path: list[str]
    raw: dict[str, Any]


class AvicennaGuardMiddleware:
    """Thin wrapper around AvicennaGuard for LLM pipeline integration."""

    def __init__(self, kb_path: str, model: str = "llama3.2:3b"):
        kb = KnowledgeBase(kb_path)
        self._lg = AvicennaGuard(kb, model=model)

    def guard(self, question: str, llm_answer: Optional[str] = None) -> GuardResult:
        r = self._lg.validate(question, llm_answer=llm_answer)
        d = r.to_dict()
        return GuardResult(
            question=question,
            epistemic_state=d["epistemic_state"],
            final_answer=d["final_answer"],
            intercepted=d["intercepted"],
            covered=d["covered"],
            path=d.get("path", []),
            raw=d,
        )


LogicGuardMiddleware = AvicennaGuardMiddleware  # Backward-compatibility alias


def wrap_llm_response(question: str, llm_answer: str, kb_path: str) -> str:
    """One-shot helper: return guarded answer string."""
    return AvicennaGuardMiddleware(kb_path).guard(question, llm_answer).final_answer
