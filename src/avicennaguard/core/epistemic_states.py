"""
Avicennian Epistemic State Classification System.

Ibn Sina (Avicenna, 980-1037 CE) in Kitab al-Shifa' defined four states
of epistemic certainty. AvicennaGuard maps each validator output to one of
these states, replacing binary True/False with a graded certainty framework.

Reference:
    Ibn Sina, Al-Shifa' (Book of Healing), c. 1020 CE.
    Discussed in: Gutas, D. (2014). Avicenna and the Aristotelian Tradition.
    Leiden: Brill, 2nd ed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class EpistemicState(str, Enum):
    """
    Four-state epistemic classification from Ibn Sina's Mantiq (Logic).

    YAQEEN (يقين) — Certainty
        BFS path confirmed in KB. LLM answer overridden with validated answer.
        Precision = 100% guarantee applies here.

    WAHM (وهم) — Illusion
        LLM answer contradicts BFS result. Hallucination intercepted.
        LLM said X, KB proves not-X. User receives KB answer + interception flag.

    SHAKK (شك) — Doubt
        Entity or relation absent from KB scope. LLM answer preserved unchanged.
        This state is the precision guarantee mechanism: we never assert
        certainty we do not have.

    ZANN (ظن) — Probability
        Semantic match found but no formal logical structure available.
        LLM answer returned with confidence flag, no override.
    """

    YAQEEN = "YAQEEN"  # Certainty — BFS confirmed
    WAHM = "WAHM"  # Illusion  — hallucination intercepted
    SHAKK = "SHAKK"  # Doubt     — out of KB scope, defer to LLM
    ZANN = "ZANN"  # Probable  — semantic match, no formal structure


class QueryType(str, Enum):
    """
    Three syllogistic inference forms from Ibn Sina's Qiyas (Syllogism).

    TAXONOMIC     — Qiyas al-Haml (Categorical Syllogism)
                    IS-A hierarchy queries. BFS on taxonomy graph G_T.
                    Example: "Are all dogs mammals?" -> dog -> canine -> mammal

    CATEGORICAL   — Property Inheritance Queries
                    Entity-property associations with transitive inheritance.
                    Example: "Do all fish have hair?" -> fish ⊬ hair -> WAHM

    HYPOTHETICAL  — Qiyas al-Istithna (Hypothetical Syllogism / Modus Ponens)
                    IF-THEN conditional chains. O(1) adjacency lookup on G_C.
                    Example: "If water freezes, does it become ice?"

    NON_LOGICAL   — No formal logical structure detected.
                    Entire query passed to LLM without Stage 2 intervention.
    """

    TAXONOMIC = "taxonomic"
    CATEGORICAL = "categorical"
    HYPOTHETICAL = "hypothetical"
    NON_LOGICAL = "non-logical"


class ValidatorResult:
    """
    Structured output from the Stage 2 BFS graph validator.

    Attributes:
        epistemic_state: One of the four Avicennian epistemic states.
        graph_answer: Boolean result from BFS traversal. None if SHAKK.
        llm_answer: Raw LLM answer before any AvicennaGuard intervention.
        final_answer: Answer returned to user (may differ from llm_answer).
        covered: Whether the KB contains the relevant entities/relations.
        intercepted: True if a hallucination was caught (WAHM state).
        query_type: The syllogistic form detected by Stage 1.
        subject: Primary entity queried.
        predicate: Target class or property in the query.
        path: BFS traversal path (for YAQEEN state, for auditability).
        latency_stage1: Time (ms) for Stage 1 semantic parsing.
        latency_stage2: Time (ms) for Stage 2 BFS validation.
        used_fallback: True if regex fallback parser was used in Stage 1.
    """

    def __init__(
        self,
        epistemic_state: EpistemicState,
        graph_answer: bool | None,
        llm_answer: str,
        final_answer: str,
        covered: bool,
        intercepted: bool,
        query_type: QueryType,
        subject: str = "",
        predicate: str = "",
        path: list[str] | None = None,
        latency_stage1: float = 0.0,
        latency_stage2: float = 0.0,
        used_fallback: bool = False,
    ) -> None:
        """
        Initialize a ValidatorResult instance.

        Args:
            epistemic_state: Graded certainty classification (YAQEEN, WAHM, SHAKK, ZANN).
            graph_answer: Deterministic graph reachability outcome (True/False/None).
            llm_answer: Raw generation text from the underlying LLM.
            final_answer: Resolved output returned to downstream caller.
            covered: Whether query entities/predicates fall within KB scope.
            intercepted: Whether an LLM hallucination contradiction was caught and overridden.
            query_type: Ibn Sina syllogistic query taxonomy classification.
            subject: Primary subject entity.
            predicate: Target predicate / property / consequence.
            path: Graph reachability audit trail steps.
            latency_stage1: Stage 1 parsing latency in milliseconds.
            latency_stage2: Stage 2 graph traversal latency in milliseconds.
            used_fallback: Flag indicating whether fallback parsing was used.
        """
        self.epistemic_state = epistemic_state
        self.graph_answer = graph_answer
        self.llm_answer = llm_answer
        self.final_answer = final_answer
        self.covered = covered
        self.intercepted = intercepted
        self.query_type = query_type
        self.subject = subject
        self.predicate = predicate
        self.path = path or []
        self.latency_stage1 = latency_stage1
        self.latency_stage2 = latency_stage2
        self.used_fallback = used_fallback

    def to_dict(self) -> dict[str, Any]:
        """
        Convert ValidatorResult to a serializable dictionary.

        Returns:
            Dictionary containing state, answers, audit path, and latency breakdown.
        """
        return {
            "epistemic_state": self.epistemic_state.value,
            "graph_answer": self.graph_answer,
            "llm_answer": self.llm_answer,
            "final_answer": self.final_answer,
            "covered": self.covered,
            "intercepted": self.intercepted,
            "query_type": self.query_type.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "path": self.path,
            "latency_ms": {
                "stage1": round(self.latency_stage1, 2),
                "stage2": round(self.latency_stage2, 2),
                "total": round(self.latency_stage1 + self.latency_stage2, 2),
            },
            "used_fallback": self.used_fallback,
        }
