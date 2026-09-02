"""
Deterministic Regex Fallback Parser for Stage 1.

Triggered when the neural/LLM parser produces malformed JSON or times out.
Ensures 100% pipeline availability without external model dependencies.

Pattern coverage:
    - Taxonomic:   "Are all X [a/an] Y?", "Is X a Y?", "Do all X belong to Y?"
    - Categorical: "Do all X have Y?", "Does X possess Y?", "Is X a property of Y?"
    - Hypothetical:"If X, [does/will] Y?", "When X, then Y?"
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Taxonomic patterns — IS-A hierarchy queries
_TAXONOMIC_PATTERNS = [
    re.compile(r"^are\s+all\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+(?:an\s+|a\s+)([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^are\s+all\s+([\w\-]+(?:\s+[\w\-]+)*)\s+([\w\-]+)\??$", re.I),
    re.compile(r"^do\s+all\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+belong\s+to\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^do\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+fall\s+(?:under|into)\s+(?:the\s+)?category\s+of\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^(?:are|is)\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+classified\s+as\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^can\s+(?:any|all)\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+be\s+(?:considered|classified\s+as)\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^would\s+(?:an?\s+|any\s+|an\s+instance\s+of\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+(?:be\s+considered\s+(?:an\s+|a\s+)?|fall\s+under\s+(?:the\s+category\s+of\s+)?)([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^is\s+(?:each|every|any|an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+a\s+(?:subclass|subtype|type|member)\s+of\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^is\s+every\s+single\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+a\s+member\s+of\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^are\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+considered\s+to\s+be\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^every\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+is\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^(?:are|is)\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+(?:an\s+|a\s+)([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^(?:are|is)\s+(?:an\s+|a\s+)?([\w\-]+(?:\s+[\w\-]+)*)\s+([\w\-]+)\??$", re.I),
]

# Categorical patterns — property queries
_CATEGORICAL_PATTERNS = [
    re.compile(r"^do\s+all\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+have\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+|each\s+|every\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+have\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^do\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+possess\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+|each\s+|every\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+possess\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^is\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+a\s+property\s+of\s+(?:all\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^is\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+an\s+inherent\s+trait\s+of\s+(?:all\s+)?([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^are\s+(?:all\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+characterized\s+by\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^are\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+known\s+to\s+have\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^do\s+all\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+(?:exhibit|feature|contain|produce|require)\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^do\s+([\w\-]+(?:\s+[\w\-]+)*?)\s+(?:exhibit|feature|contain|produce|require)\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+|each\s+|every\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\s+(?:feature|show\s+signs\s+of)\s+([\w\-]+(?:\s+[\w\-]+)*)\??$", re.I),
]

# Hypothetical patterns — IF-THEN modus ponens
_HYPOTHETICAL_PATTERNS = [
    re.compile(r"^if\s+([\w\-]+(?:\s+[\w\-]+)*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-]+(?:\s+[\w\-]+)*?)(?:\s+be\s+expected)?\??$", re.I),
    re.compile(r"^when\s+([\w\-]+(?:\s+[\w\-]+)*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\??$", re.I),
    re.compile(r"^assuming\s+(?:that\s+)?([\w\-]+(?:\s+[\w\-]+)*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\??$", re.I),
    re.compile(r"^given\s+that\s+([\w\-]+(?:\s+[\w\-]+)*?),?\s+(?:does\s+it\s+follow\s+that|does|will|would|then)?\s*(?:it\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\??$", re.I),
    re.compile(r"^suppose\s+([\w\-]+(?:\s+[\w\-]+)*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\??$", re.I),
    re.compile(r"^provided\s+that\s+([\w\-]+(?:\s+[\w\-]+)*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-]+(?:\s+[\w\-]+)*?)\??$", re.I),
]


def _clean_term(term: str) -> str:
    """Normalize extracted term: strip leading articles/determiners and convert whitespace to underscores."""
    t = (term or "").strip()
    t = re.sub(r"^(?:a|an|the|each|every|all|any|an\s+instance\s+of|every\s+single)\s+", "", t, flags=re.I)
    return t.strip().lower().replace(" ", "_")


class RegexParser:
    """Deterministic fallback parser using regex pattern matching."""

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Attempt to extract logical form using regex patterns.

        Args:
            question: Natural language question string.

        Returns:
            Dictionary matching one of the proposition schemas:
            {"type": "taxonomic", "subject": ..., "predicate": ...},
            {"type": "categorical", "entity": ..., "property": ...},
            {"type": "hypothetical", "condition": ..., "consequence": ...},
            or {"type": "non-logical"}.
        """
        if question is None:
            return {"type": "non-logical"}
        q = re.sub(r"\s+", " ", str(question)).strip()
        if not q:
            return {"type": "non-logical"}

        for pattern in _TAXONOMIC_PATTERNS:
            m = pattern.match(q)
            if m:
                logger.debug("Regex matched taxonomic: %s", q)
                return {
                    "type": "taxonomic",
                    "subject": _clean_term(m.group(1)),
                    "predicate": _clean_term(m.group(2)),
                }

        for pattern in _CATEGORICAL_PATTERNS:
            m = pattern.match(q)
            if m:
                logger.debug("Regex matched categorical: %s", q)
                if "property of" in q.lower() or "trait of" in q.lower():
                    entity = _clean_term(m.group(2))
                    prop = _clean_term(m.group(1))
                else:
                    entity = _clean_term(m.group(1))
                    prop = _clean_term(m.group(2))
                return {
                    "type": "categorical",
                    "entity": entity,
                    "property": prop,
                }

        for pattern in _HYPOTHETICAL_PATTERNS:
            m = pattern.match(q)
            if m:
                logger.debug("Regex matched hypothetical: %s", q)
                return {
                    "type": "hypothetical",
                    "condition": _clean_term(m.group(1)),
                    "consequence": _clean_term(m.group(2)),
                }

        return {"type": "non-logical"}
