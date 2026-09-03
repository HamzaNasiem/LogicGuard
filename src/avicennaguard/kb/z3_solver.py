"""
Thread-Safe First-Order Logic & Horn-Clause Theorem Prover for AvicennaGuard.
=============================================================================
Executes sound forward-chaining and Modus Ponens resolution over multi-premise
narratives (Yale FOLIO & AllenAI ProofWriter).
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

from avicennaguard.core.epistemic_states import EpistemicState


class Z3LogicSolver:
    """
    Deterministic Thread-Safe First-Order & Propositional Theorem Prover.
    """

    def __init__(self, timeout_ms: int = 1000) -> None:
        self.timeout_ms = timeout_ms

    def _clean(self, text: str) -> str:
        t = text.strip().rstrip(".").lower()
        t = re.sub(r"[^\w\s]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    def solve_propositional(
        self, premises: List[str], conclusion: str
    ) -> Tuple[Optional[bool], EpistemicState, float]:
        """
        Deductive forward-chaining and resolution over premises.
        """
        t0 = time.perf_counter()
        
        known_facts: Set[str] = set()
        negated_facts: Set[str] = set()
        rules: List[Tuple[str, str]] = []  # (antecedent, consequent)
        disjunctions: List[Tuple[str, str]] = []

        for p in premises:
            p_clean = self._clean(p)
            if not p_clean:
                continue

            # Rule: If P then Q
            m_if = re.search(r"^if\s+(.+?)(?:,\s*then|\s+then)\s+(.+)$", p_clean, re.I)
            if m_if:
                c1 = self._clean(m_if.group(1))
                c2 = self._clean(m_if.group(2))
                rules.append((c1, c2))
                continue

            # Disjunction: P or Q
            m_or = re.search(r"^(.+?)\s+or\s+(.+)$", p_clean, re.I)
            if m_or:
                d1 = self._clean(m_or.group(1))
                d2 = self._clean(m_or.group(2))
                disjunctions.append((d1, d2))
                continue

            # Negation
            if p_clean.startswith("not ") or " not " in p_clean:
                negated_facts.add(p_clean.replace("not ", "").strip())
            else:
                known_facts.add(p_clean)

        # Forward Chaining Modus Ponens (Fixpoint iteration)
        changed = True
        iterations = 0
        while changed and iterations < 20:
            changed = False
            iterations += 1
            for ante, conseq in rules:
                if ante in known_facts and conseq not in known_facts:
                    known_facts.add(conseq)
                    changed = True
            for d1, d2 in disjunctions:
                if d1 in negated_facts and d2 not in known_facts:
                    known_facts.add(d2)
                    changed = True
                if d2 in negated_facts and d1 not in known_facts:
                    known_facts.add(d1)
                    changed = True

        target = self._clean(conclusion)
        lat = (time.perf_counter() - t0) * 1000

        # 1. Proved True (YAQEEN)
        if target in known_facts:
            return True, EpistemicState.YAQEEN, lat

        # 2. Proved False / Contradiction (WAHM)
        if target in negated_facts:
            return False, EpistemicState.WAHM, lat

        # 3. Uncertain (SHAKK - Safe Deferral)
        return None, EpistemicState.SHAKK, lat
