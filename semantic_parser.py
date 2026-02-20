# semantic_parser.py

"""
semantic_parser.py — LogicGuard Translation Layer
===================================================
Author: LogicGuard Research Team
Purpose: Uses LLaMA 3.2 (via Ollama) purely as a semantic parser.
         The LLM is NEVER asked to answer questions — only to extract
         formal logical structure from natural language.

Architecture:
  Messy NL Question
       ↓
  SemanticParser (LLaMA 3.2 as parser)
       ↓
  Structured JSON: {type, subject, predicate, ...}
       ↓
  NetworkX Graph Lookup (deterministic)
       ↓
  Epistemic State: YAQEEN / WAHM / SHAKK / ZANN

Why this matters for IEEE:
  This solves the "Semantic Parsing Bottleneck" — our previous regex-only
  approach required syntactically perfect questions. This layer enables
  LogicGuard to handle natural, messy human language while keeping
  all logical REASONING fully deterministic (no LLM guessing).

Output JSON Schemas (strictly one of four):
  {"type": "taxonomic",   "subject": "X",    "predicate": "Y"}
  {"type": "categorical", "entity": "X",     "property": "Y"}
  {"type": "hypothetical","condition": "X",  "consequence": "Y"}
  {"type": "non-logical"}
"""

import json
import re
import time
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  — Extremely restrictive. The LLM is caged to be only a parser.
# ─────────────────────────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """You are a formal logic extraction engine. You are NOT an assistant. You do NOT answer questions. You do NOT explain anything. You have ONE job: analyze the user's question and extract its formal logical structure into a JSON object.

STRICT OUTPUT RULES:
- Output ONLY a raw JSON object. No markdown. No explanation. No preamble.
- Choose EXACTLY ONE of the four schemas below.
- All values must be lowercase single words or short phrases.
- Never include the answer to the question. Never add extra keys.

SCHEMA 1 — TAXONOMIC (Is A a type/kind/subset of B?):
{"type": "taxonomic", "subject": "<singular lowercase noun>", "predicate": "<singular lowercase noun>"}
Use when: question asks if one category IS a kind of another.
Examples:
  "Are all dogs mammals?" → {"type": "taxonomic", "subject": "dog", "predicate": "mammal"}
  "I wonder, would every single eagle count as a bird?" → {"type": "taxonomic", "subject": "eagle", "predicate": "bird"}
  "Is a square always a rectangle?" → {"type": "taxonomic", "subject": "square", "predicate": "rectangle"}

SCHEMA 2 — CATEGORICAL (Do members of class A possess property B?):
{"type": "categorical", "entity": "<singular lowercase noun>", "property": "<lowercase property word>"}
Use when: question asks if a class of things has, needs, gives, or does something.
Examples:
  "Do all mammals have hair?" → {"type": "categorical", "entity": "mammal", "property": "hair"}
  "Would every bird necessarily lay eggs?" → {"type": "categorical", "entity": "bird", "property": "lay_eggs"}
  "Do living things eventually die?" → {"type": "categorical", "entity": "living_thing", "property": "die"}

SCHEMA 3 — HYPOTHETICAL (If condition A is true, does consequence B follow?):
{"type": "hypothetical", "condition": "<short lowercase phrase>", "consequence": "<short lowercase word or phrase>"}
Use when: question has an IF-THEN structure or causal relationship.
Examples:
  "If it is raining, is the ground wet?" → {"type": "hypothetical", "condition": "raining", "consequence": "ground_wet"}
  "When water freezes, does it become ice?" → {"type": "hypothetical", "condition": "water freezes", "consequence": "ice"}
  "Assuming there is fire, would heat be present?" → {"type": "hypothetical", "condition": "fire", "consequence": "heat"}

SCHEMA 4 — NON-LOGICAL (No formal logical structure):
{"type": "non-logical"}
Use when: question asks for facts, opinions, definitions, stories, or anything that is NOT a subset/property/conditional claim.
Examples:
  "Where is Paris?" → {"type": "non-logical"}
  "Who invented the telephone?" → {"type": "non-logical"}
  "What is the capital of France?" → {"type": "non-logical"}

Now extract the logical structure from the user's question. Output ONLY the JSON object:"""


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK REGEX PARSER  (used if Ollama is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_word(word: str) -> str:
    """Singularize common plurals for KB lookup."""
    IRREGULAR = {
        'mice':'mouse','geese':'goose','teeth':'tooth','feet':'foot',
        'men':'man','women':'woman','children':'child','people':'person',
        'buses':'bus','viruses':'virus','fungi':'fungus','cacti':'cactus',
        'fish':'fish','sheep':'sheep','deer':'deer','moose':'moose',
    }
    w = word.strip().lower()
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w.endswith('ies') and len(w) > 3: return w[:-3]+'y'
    if w.endswith(('shes','ches','xes','sses')): return w[:-2]
    if w.endswith('ves') and len(w) > 3: return w[:-3]+'f'
    if w.endswith('s') and not w.endswith(('ss','us')): return w[:-1]
    return w

def _regex_fallback_parse(question: str) -> Dict:
    """
    Regex-based fallback parser. Used when Ollama is offline.
    Identical logic to the old logic_templates.py — always available,
    no network required.
    """
    q = question.lower().replace('?','').strip()
    VERBS = ['have','need','give','lay','produce','die','grow',
             'possess','contain','carry']

    # ── Hypothetical ──────────────────────────────────────────────
    if q.startswith('if ') or (' if ' in q and q.index('if') < 5):
        after = q.split('if',1)[1].strip()
        if 'then' in after:
            parts = after.split('then',1)
        elif ',' in after:
            parts = after.split(',',1)
        else:
            return {"type": "non-logical"}
        if len(parts) < 2:
            return {"type": "non-logical"}
        cond = parts[0].strip()
        cons = parts[1].strip()
        for f in ["it is ","it's ","there is ","there are ","there ",
                  "you are ","we are "]:
            cond = cond.replace(f,'')
        for p in ["a ","an ","the "]:
            if cond.startswith(p): cond = cond[len(p):]
        for pat in ["is there ","is the ","does it ","are they ",
                    "are you ","will it ","is it ","is ","does ","are "]:
            if cons.startswith(pat): cons = cons[len(pat):]; break
        return {"type":"hypothetical","condition":cond.strip(),
                "consequence":cons.strip()}

    # ── Taxonomic ─────────────────────────────────────────────────
    if 'are all' in q:
        after = q.split('are all',1)[1].strip()
        for f in [' a ',' an ',' the ']: after = after.replace(f,' ')
        parts = after.split()
        if len(parts) >= 2:
            return {"type":"taxonomic",
                    "subject":   _normalize_word(parts[0]),
                    "predicate": _normalize_word(parts[1])}

    # ── Categorical ───────────────────────────────────────────────
    if 'do all' in q and any(v in q for v in VERBS):
        after = q.split('do all',1)[1]
        splitter, split_pos = None, len(after)+1
        for v in VERBS:
            idx = after.find(v)
            if idx != -1 and idx < split_pos:
                split_pos, splitter = idx, v
        if splitter:
            parts = after.split(splitter,1)
            entity = _normalize_word(parts[0].strip())
            prop   = parts[1].strip() if len(parts)>1 else ''
            if not prop: prop = splitter
            for f in ['a ','an ','the ']:
                if prop.startswith(f): prop = prop[len(f):]
            return {"type":"categorical",
                    "entity":   entity,
                    "property": prop.strip().replace(' ','_')}

    return {"type": "non-logical"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEMANTIC PARSER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SemanticParser:
    """
    Two-mode semantic parser:
      - PRIMARY  : LLaMA 3.2 via Ollama (handles messy natural language)
      - FALLBACK : Regex-based parser  (offline / fast path)

    The parser NEVER answers questions. It only extracts structure.
    All logical reasoning stays 100% deterministic in the NetworkX layer.
    """

    # Strict JSON schema validator
    REQUIRED_KEYS = {
        "taxonomic":   {"subject", "predicate"},
        "categorical": {"entity", "property"},
        "hypothetical":{"condition", "consequence"},
        "non-logical": set(),
    }

    def __init__(self,
                 model: str = "llama3.2:3b",
                 use_ollama: bool = True,
                 timeout_seconds: int = 15,
                 max_retries: int = 2):
        """
        Args:
            model          : Ollama model name (must match `ollama list`)
            use_ollama     : Set False to force regex fallback (testing / offline)
            timeout_seconds: Per-call timeout to avoid blocking
            max_retries    : Retry attempts if LLM returns malformed JSON
        """
        self.model    = model
        self.timeout  = timeout_seconds
        self.retries  = max_retries
        self._ollama_available = False

        if use_ollama:
            self._ollama_available = self._check_ollama()

        if self._ollama_available:
            logger.info(f"[SemanticParser] PRIMARY mode: Ollama ({model})")
        else:
            logger.info("[SemanticParser] FALLBACK mode: Regex parser (Ollama offline)")

    def _check_ollama(self) -> bool:
        """Silently probe Ollama. Returns True if reachable."""
        try:
            import ollama
            ollama.list()   # fast ping
            return True
        except Exception as e:
            logger.warning(f"[SemanticParser] Ollama unavailable: {e}")
            return False

    def _call_ollama(self, question: str) -> Optional[Dict]:
        """
        Call LLaMA 3.2 with the restrictive system prompt.
        Forces JSON output via Ollama's format='json' parameter.
        """
        import ollama

        for attempt in range(1, self.retries + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    format="json",          # Forces JSON-only output
                    options={
                        "temperature": 0.0, # Deterministic — no creativity
                        "num_predict": 80,  # Short output only (just JSON)
                        "stop": ["\n\n"],   # Stop after first complete JSON
                    },
                    messages=[
                        {
                            "role": "system",
                            "content": PARSER_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                )

                raw_text = response["message"]["content"].strip()
                parsed   = self._extract_and_validate_json(raw_text)

                if parsed:
                    logger.debug(f"[SemanticParser] LLM parsed: {parsed}")
                    return parsed

                logger.warning(f"[SemanticParser] Attempt {attempt}: "
                               f"Invalid JSON from LLM: {raw_text[:80]}")

            except Exception as e:
                logger.warning(f"[SemanticParser] Attempt {attempt} error: {e}")
                if attempt < self.retries:
                    time.sleep(0.5)

        return None  # All retries failed — caller will use fallback

    def _extract_and_validate_json(self, raw: str) -> Optional[Dict]:
        """
        Robustly extract and validate JSON from LLM output.
        Handles: extra whitespace, markdown fences, trailing text.
        """
        # Strip markdown code fences if present
        raw = re.sub(r'```(?:json)?', '', raw).strip()

        # Try direct parse first
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object within the text
            match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        # Validate schema
        return self._validate_schema(data)

    def _validate_schema(self, data: Dict) -> Optional[Dict]:
        """
        Strict schema validation. Rejects any JSON that doesn't match
        one of the four expected structures exactly.
        """
        if not isinstance(data, dict):
            return None

        q_type = data.get("type", "").lower().strip()

        if q_type not in self.REQUIRED_KEYS:
            return None

        required = self.REQUIRED_KEYS[q_type]
        if not required.issubset(data.keys()):
            return None

        # Normalize all string values to lowercase
        clean = {"type": q_type}
        for k in required:
            val = str(data[k]).lower().strip()
            # Normalize spaces to underscores in multi-word values
            if k in {"property", "consequence"}:
                val = val.replace(' ', '_')
            clean[k] = val

        return clean

    def parse(self, question: str) -> Dict:
        """
        Main entry point. Returns a validated JSON dict.

        Flow:
          1. Try Ollama (LLaMA 3.2) → handles messy NL
          2. If Ollama fails → Regex fallback → handles standard syntax
          3. If both fail → returns non-logical

        Returns one of:
          {"type": "taxonomic",    "subject": X,   "predicate": Y}
          {"type": "categorical",  "entity": X,    "property": Y}
          {"type": "hypothetical", "condition": X, "consequence": Y}
          {"type": "non-logical"}
        """
        result = None

        # Primary: Ollama LLM parser
        if self._ollama_available:
            result = self._call_ollama(question)
            if result and result["type"] != "non-logical":
                return result
            # If LLM says non-logical, double-check with regex
            regex_result = _regex_fallback_parse(question)
            if regex_result["type"] != "non-logical":
                # Regex found structure that LLM missed — trust regex
                logger.info("[SemanticParser] Regex override: LLM said non-logical "
                            "but regex found structure")
                return regex_result
            return result or {"type": "non-logical"}

        # Fallback: Regex parser (offline / fast)
        result = _regex_fallback_parse(question)
        return result

    @property
    def mode(self) -> str:
        return "ollama" if self._ollama_available else "regex_fallback"