"""
Deterministic Regex Fallback Parser for Stage 1.

Triggered when the constrained LLM produces malformed JSON.
Ensures 100% pipeline availability without any LLM dependency.

Pattern coverage:
    - Taxonomic:   "Are all X [a/an] Y?", "Is X a Y?", "Do all X belong to Y?"
    - Categorical: "Do all X have Y?", "Does X possess Y?"
    - Hypothetical:"If X, [does/will] Y?", "If X then Y?"
"""

import re
import logging

logger = logging.getLogger(__name__)

# Taxonomic patterns — IS-A hierarchy queries
_TAXONOMIC_PATTERNS = [
    re.compile(r"^are\s+all\s+(\w[\w\s]*?)\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^do\s+all\s+(\w[\w\s]*?)\s+belong\s+to\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^(?:are|is)\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\s+classified\s+as\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^every\s+(\w[\w\s]*?)\s+is\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^is\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\??$", re.I),
]

# Categorical patterns — property queries
_CATEGORICAL_PATTERNS = [
    re.compile(r"^do\s+all\s+(\w[\w\s]*?)\s+have\s+(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+)?(\w[\w\s]*?)\s+have\s+(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^do\s+(\w[\w\s]*?)\s+possess\s+(\w[\w\s]*?)\??$", re.I),
]

# Hypothetical patterns — IF-THEN modus ponens
_HYPOTHETICAL_PATTERNS = [
    re.compile(r"^if\s+(\w[\w\s]*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?(\w[\w\s]*?)\??$", re.I),
    re.compile(r"^when\s+(\w[\w\s]*?),?\s+(?:does|will|would)?\s*(?:it\s+)?(\w[\w\s]*?)\??$", re.I),
]


def _clean_term(term: str) -> str:
    """Normalize extracted term: strip leading articles and convert whitespace to underscores."""
    t = term.strip()
    t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.I)
    return t.strip().lower().replace(" ", "_")


class RegexParser:
    """Deterministic fallback parser using pattern matching."""

    def parse(self, question: str) -> dict:
        """
        Attempt to extract logical form using regex patterns.
        Returns {"type": "non-logical"} if no pattern matches.
        """
        q = question.strip()

        for pattern in _TAXONOMIC_PATTERNS:
            m = pattern.match(q)
            if m:
                logger.debug("Regex matched taxonomic: %s", q)
                return {
                    "type": "taxonomic",
                    "subject":   _clean_term(m.group(1)),
                    "predicate": _clean_term(m.group(2)),
                }

        for pattern in _CATEGORICAL_PATTERNS:
            m = pattern.match(q)
            if m:
                logger.debug("Regex matched categorical: %s", q)
                return {
                    "type":     "categorical",
                    "entity":   _clean_term(m.group(1)),
                    "property": _clean_term(m.group(2)),
                }

        for pattern in _HYPOTHETICAL_PATTERNS:
            m = pattern.match(q)
            if m:
                logger.debug("Regex matched hypothetical: %s", q)
                return {
                    "type":        "hypothetical",
                    "condition":   _clean_term(m.group(1)),
                    "consequence": _clean_term(m.group(2)),
                }

        return {"type": "non-logical"}
