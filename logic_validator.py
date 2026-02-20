# logic_validator.py

"""
logic_validator.py — LogicGuard Main Validator (v2, with SemanticParser)
=========================================================================
Pipeline:
  Question → SemanticParser (LLaMA or Regex) → Parsed JSON
                                                     ↓
                                          NetworkX Graph Lookup
                                                     ↓
                                      Epistemic State (YAQEEN/WAHM/etc.)

The key architectural insight for IEEE:
  - PARSING  : probabilistic (LLM assists with messy natural language)
  - REASONING: 100% deterministic (NetworkX BFS — never probabilistic)
"""

import json
import logging
from typing import Dict, Optional
from knowledge_graph import KnowledgeGraph
from semantic_parser import SemanticParser

logger = logging.getLogger(__name__)


class LogicValidator:
    """
    Unified validation pipeline.
    Replaces regex-based template matching with LLM-assisted semantic parsing
    while keeping all logical inference fully deterministic.
    """

    def __init__(self,
                 kb_path: str = None,          # Legacy param (ignored — we use NetworkX)
                 use_ollama: bool = True,       # Set False for offline/fast mode
                 model: str = "llama3.2:3b"):  # Ollama model name
        self.kg     = KnowledgeGraph()
        self.parser = SemanticParser(
            model=model,
            use_ollama=use_ollama
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def validate(self, question: str, llm_answer: str = '') -> Dict:
        """
        Full pipeline: parse → graph lookup → epistemic classification.

        Returns dict with keys:
          is_logical_question : bool
          template_used       : 'taxonomic'|'categorical'|'hypothetical'|None
          logically_valid     : True|False|None
          certainty           : 0-100
          proof               : human-readable proof string
          epistemic_state     : 'YAQEEN'|'WAHM'|'SHAKK'|'ZANN'|'UNKNOWN'
          method              : description of reasoning method
          parsed_structure    : the JSON from SemanticParser
          parser_mode         : 'ollama' or 'regex_fallback'
        """
        # Step 1 — Semantic Parsing (LLM or Regex)
        parsed = self.parser.parse(question)

        base = {
            'parsed_structure': parsed,
            'parser_mode':      self.parser.mode,
        }

        q_type = parsed.get("type", "non-logical")

        # Step 2 — Route to graph lookup based on parsed type
        if q_type == "taxonomic":
            return {**base, **self._verify_taxonomic(parsed)}

        elif q_type == "categorical":
            return {**base, **self._verify_categorical(parsed)}

        elif q_type == "hypothetical":
            return {**base, **self._verify_hypothetical(parsed)}

        else:
            # Non-logical — skip to semantic/LLM evaluation
            return {
                **base,
                'is_logical_question': False,
                'template_used':       None,
                'logically_valid':     None,
                'certainty':           0,
                'proof':               'No formal logical structure detected',
                'epistemic_state':     'UNKNOWN',
                'method':              'Non-Logical',
                'fallback_to_semantic': True,
            }

    def classify(self, question: str) -> Optional[str]:
        """Quick classification only. Returns type or None."""
        parsed = self.parser.parse(question)
        t = parsed.get("type")
        return t if t != "non-logical" else None

    def graph_stats(self) -> dict:
        return self.kg.stats()

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH LOOKUP METHODS (100% deterministic — no LLM involvement)
    # ─────────────────────────────────────────────────────────────────────────

    def _verify_taxonomic(self, parsed: Dict) -> Dict:
        """BFS subset check on taxonomy graph."""
        subject   = parsed.get("subject",   "").lower().strip()
        predicate = parsed.get("predicate", "").lower().strip()

        if not subject or not predicate:
            return self._parse_failed("taxonomic", "Missing subject/predicate")

        is_subset, path = self.kg.is_subset(subject, predicate)

        if is_subset:
            return {
                'is_logical_question': True,
                'template_used':       'taxonomic',
                'logically_valid':     True,
                'certainty':           100,
                'proof':               f'BFS proof: {" → ".join(path)}',
                'epistemic_state':     'YAQEEN',
                'method':              'Qiyas al-Haml (Taxonomic BFS)',
                'fallback_to_semantic': False,
            }

        return {
            'is_logical_question': True,
            'template_used':       'taxonomic',
            'logically_valid':     False,
            'certainty':           0,
            'proof':               f'No BFS path: {subject} ⊄ {predicate}',
            'epistemic_state':     'WAHM',
            'method':              'Qiyas al-Haml (Failed)',
            'fallback_to_semantic': False,
        }

    def _verify_categorical(self, parsed: Dict) -> Dict:
        """Property inheritance check via taxonomy + property graph."""
        entity = parsed.get("entity",   "").lower().strip()
        prop   = parsed.get("property", "").lower().strip()

        if not entity or not prop:
            return self._parse_failed("categorical", "Missing entity/property")

        has_prop, proof = self.kg.has_property(entity, prop)

        if has_prop:
            return {
                'is_logical_question': True,
                'template_used':       'categorical',
                'logically_valid':     True,
                'certainty':           100,
                'proof':               f'Property graph: {proof}',
                'epistemic_state':     'YAQEEN',
                'method':              'Categorical Reasoning (Graph)',
                'fallback_to_semantic': False,
            }

        return {
            'is_logical_question': True,
            'template_used':       'categorical',
            'logically_valid':     False,
            'certainty':           0,
            'proof':               f'Not in property graph: {entity} ⊬ {prop}',
            'epistemic_state':     'WAHM',
            'method':              'Categorical Reasoning (Failed)',
            'fallback_to_semantic': False,
        }

    def _verify_hypothetical(self, parsed: Dict) -> Dict:
        """Modus Ponens lookup in conditional graph."""
        condition   = parsed.get("condition",   "").lower().strip()
        consequence = parsed.get("consequence", "").lower().strip()

        if not condition or not consequence:
            return self._parse_failed("hypothetical", "Missing condition/consequence")

        valid, proof = self.kg.check_conditional(condition, consequence)

        if valid:
            return {
                'is_logical_question': True,
                'template_used':       'hypothetical',
                'logically_valid':     True,
                'certainty':           100,
                'proof':               proof,
                'epistemic_state':     'YAQEEN',
                'method':              'Qiyas al-Istithna (Modus Ponens)',
                'fallback_to_semantic': False,
            }

        return {
            'is_logical_question': True,
            'template_used':       'hypothetical',
            'logically_valid':     False,
            'certainty':           0,
            'proof':               f'Conditional unverified: {condition} → {consequence}',
            'epistemic_state':     'WAHM',
            'method':              'Qiyas al-Istithna (Failed)',
            'fallback_to_semantic': False,
        }

    def _parse_failed(self, template_type: str, reason: str) -> Dict:
        return {
            'is_logical_question': True,
            'template_used':       template_type,
            'logically_valid':     None,
            'certainty':           0,
            'proof':               f'Parse failed: {reason}',
            'epistemic_state':     'UNKNOWN',
            'method':              'Parse Failed',
            'fallback_to_semantic': True,
        }