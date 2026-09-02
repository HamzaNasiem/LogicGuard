"""
Stage 1: Constrained LLM Semantic Parser.

The LLM in Stage 1 is caged — it can ONLY output one of four JSON schemas.
It NEVER answers the user's question. All logical content is distrusted.
The LLM identifies only the logical form of the query.

Constraints applied:
    1. Restrictive system prompt prohibiting answering
    2. Temperature = 0.0 (deterministic extraction)
    3. Forced JSON output via format='json'
    4. Hard limit: 80 output tokens

If the LLM produces malformed JSON, the regex fallback in regex_parser.py
ensures 100% pipeline availability.

This design resolves the Semantic Parsing Bottleneck:
    "Would every creature classified as a dog fall under the mammal category?"
    ↕ (logically identical, structurally incompatible)
    "Are dogs mammals?"

Both produce: {"type": "taxonomic", "subject": "dog", "predicate": "mammal"}
"""

import json
import logging
import time
from typing import Optional

from avicennaguard.core.epistemic_states import QueryType
from avicennaguard.parsers.regex_parser import RegexParser

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are a logical form extractor. Your ONLY job is to identify
the logical structure of a question. You MUST NOT answer the question.

Output ONLY valid JSON matching one of these four schemas:

Schema 1 — Taxonomic (Is X a subtype of Y?):
{"type": "taxonomic", "subject": "X", "predicate": "Y"}

Schema 2 — Categorical (Does class X have property Y?):
{"type": "categorical", "entity": "X", "property": "Y"}

Schema 3 — Hypothetical (If X then Y?):
{"type": "hypothetical", "condition": "X", "consequence": "Y"}

Schema 4 — No logical structure:
{"type": "non-logical"}

Output ONLY the JSON object. No explanation. No answer."""


class LLMParser:
    """
    Constrained LLM semantic parser for Stage 1.

    Uses locally hosted Ollama models. Model-agnostic: any model that
    supports JSON-forced output can serve as the Stage 1 parser.
    """

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self._regex_fallback = RegexParser()
        self._parse_stats = {"success": 0, "regex_fallback": 0, "failure": 0}

    def parse(self, question: str) -> tuple[dict, bool]:
        """
        Parse natural language question into structured logical form.

        Returns:
            (parsed_dict, used_fallback)
            parsed_dict:    One of the four JSON schemas above.
            used_fallback:  True if regex fallback was triggered.

        Never raises an exception — returns {"type": "non-logical"} on
        complete failure to ensure 100% pipeline availability.
        """
        t0 = time.perf_counter()
        result, used_fallback = self._call_llm(question)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        if used_fallback:
            self._parse_stats["regex_fallback"] += 1
        else:
            self._parse_stats["success"] += 1

        result["_parse_latency_ms"] = round(elapsed, 2)
        return result, used_fallback

    def _call_llm(self, question: str) -> tuple[dict, bool]:
        try:
            import ollama

            resp = ollama.chat(
                model=self.model,
                format="json",
                options={"temperature": 0.0, "num_predict": 80, "seed": 42},
                messages=[
                    {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                    {"role": "user",   "content": question},
                ],
            )
            raw = resp["message"]["content"].strip()
            parsed = json.loads(raw)

            # Validate schema
            if "type" not in parsed:
                raise ValueError("Missing 'type' field in LLM output")

            # If LLM says non-logical, try regex to verify — regex is more reliable
            # for simple structural queries like "Are all X Y?"
            if parsed.get("type") == "non-logical":
                regex_result = self._regex_fallback.parse(question)
                if regex_result.get("type") != "non-logical":
                    logger.debug("Regex override: LLM said non-logical but regex found %s", regex_result["type"])
                    return regex_result, True

            return parsed, False

        except Exception as e:
            logger.warning("LLM parse failed (%s). Trying regex fallback.", e)
            self._parse_stats["failure"] += 1
            return self._regex_fallback.parse(question), True

    @property
    def parse_stats(self) -> dict:
        """Returns Stage 1 reliability metrics for paper reporting."""
        total = sum(self._parse_stats.values())
        if total == 0:
            return self._parse_stats
        return {
            **self._parse_stats,
            "total": total,
            "success_rate": round(self._parse_stats["success"] / total * 100, 1),
            "fallback_rate": round(self._parse_stats["regex_fallback"] / total * 100, 1),
        }
