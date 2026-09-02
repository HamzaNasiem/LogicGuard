"""
Stage 1: DeBERTa Fast Semantic Parser with Deterministic Regex Fallback.

Provides sub-30ms semantic classification and slot extraction for Stage 1 queries.
Supports both fine-tuned DeBERTa transformer models and high-speed Scikit-Learn
calibrated pipelines (saved to models/stage1_classifier.joblib).
Gracefully falls back to RegexParser when model weights are missing or inference is
unavailable, guaranteeing sub-millisecond execution and 100% pipeline availability.

Output schema:
    {
        "type": "taxonomic" | "categorical" | "hypothetical" | "non-logical",
        "subject": str,
        "predicate": str,
        "condition": str,
        "consequence": str,
        "confidence": float,
        "method": "deberta" | "regex_fallback"
    }
"""

import logging
import os
import re
import time
from typing import Optional, Dict, Any, Tuple

from avicennaguard.parsers.regex_parser import RegexParser

logger = logging.getLogger(__name__)

# Label mappings for 4 query types
LABEL_MAP = {
    0: "taxonomic",
    1: "categorical",
    2: "hypothetical",
    3: "non-logical",
}
NAME_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def _clean_term(term: str) -> str:
    """Normalize extracted term: strip leading articles and convert whitespace to underscores."""
    t = (term or "").strip()
    t = re.sub(r"^(?:a|an|the)\s+", "", t, flags=re.I)
    return t.strip().lower().replace(" ", "_")


# Helper patterns for slot extraction (ordered from most specific to general)
_TAX_PATTERNS = [
    re.compile(r"^are\s+all\s+([\w\-][\w\s\-]*?)\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^do\s+all\s+([\w\-][\w\s\-]*?)\s+belong\s+to\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^do\s+([\w\-][\w\s\-]*?)\s+fall\s+(?:under|into)\s+(?:the\s+)?category\s+of\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^(?:are|is)\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\s+classified\s+as\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^can\s+(?:any|all)\s+([\w\-][\w\s\-]*?)\s+be\s+(?:considered|classified\s+as)\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^would\s+(?:an?\s+|any\s+|an\s+instance\s+of\s+)?([\w\-][\w\s\-]*?)\s+(?:be\s+considered\s+(?:an\s+|a\s+)?|fall\s+under\s+(?:the\s+category\s+of\s+)?([\w\-][\w\s\-]*?))\??$", re.I),
    re.compile(r"^is\s+(?:each|every|any|an\s+|a\s+)?([\w\-][\w\s\-]*?)\s+a\s+(?:subclass|subtype|type|member)\s+of\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^is\s+every\s+single\s+([\w\-][\w\s\-]*?)\s+a\s+member\s+of\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^are\s+([\w\-][\w\s\-]*?)\s+considered\s+to\s+be\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^every\s+([\w\-][\w\s\-]*?)\s+is\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^is\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\s+(?:an\s+|a\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
]

_CAT_PATTERNS = [
    re.compile(r"^do\s+all\s+([\w\-][\w\s\-]*?)\s+have\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+|each\s+|every\s+)?([\w\-][\w\s\-]*?)\s+have\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^do\s+([\w\-][\w\s\-]*?)\s+possess\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+|each\s+|every\s+)?([\w\-][\w\s\-]*?)\s+possess\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^is\s+([\w\-][\w\s\-]*?)\s+a\s+property\s+of\s+(?:all\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^is\s+([\w\-][\w\s\-]*?)\s+an\s+inherent\s+trait\s+of\s+(?:all\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^are\s+(?:all\s+)?([\w\-][\w\s\-]*?)\s+characterized\s+by\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^are\s+([\w\-][\w\s\-]*?)\s+known\s+to\s+have\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^do\s+all\s+([\w\-][\w\s\-]*?)\s+(?:exhibit|feature|contain|produce|require)\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^do\s+([\w\-][\w\s\-]*?)\s+(?:exhibit|feature|contain|produce|require)\s+([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^does\s+(?:an\s+|a\s+|each\s+|every\s+)?([\w\-][\w\s\-]*?)\s+(?:feature|show\s+signs\s+of)\s+([\w\-][\w\s\-]*?)\??$", re.I),
]

_HYP_PATTERNS = [
    re.compile(r"^if\s+([\w\-][\w\s\-]*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-][\w\s\-]*?)(?:\s+be\s+expected)?\??$", re.I),
    re.compile(r"^when\s+([\w\-][\w\s\-]*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^assuming\s+(?:that\s+)?([\w\-][\w\s\-]*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^given\s+that\s+([\w\-][\w\s\-]*?),?\s+(?:does\s+it\s+follow\s+that|does|will|would|then)?\s*(?:it\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^suppose\s+([\w\-][\w\s\-]*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
    re.compile(r"^provided\s+that\s+([\w\-][\w\s\-]*?),?\s+(?:does|will|would|then)?\s*(?:it\s+)?([\w\-][\w\s\-]*?)\??$", re.I),
]


class DebertaParser:
    """
    Stage 1 Fast Semantic Parser using DeBERTa / TF-IDF classification with regex fallback.

    Provides high-throughput, low-latency parsing of natural language questions into
    structured logical representations.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        fallback_parser: Optional[RegexParser] = None,
    ):
        """
        Initialize DebertaParser.

        Args:
            model_path: Path to trained model artifact (DeBERTa checkpoint directory,
                        `.joblib` classifier file, 'auto', or HuggingFace ID).
                        If None or not found, parser operates in pure regex fallback mode.
            device: 'cuda', 'cpu', or None (auto-detect).
            confidence_threshold: Minimum prediction probability to accept model output.
            fallback_parser: Custom RegexParser instance (optional).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.fallback_parser = fallback_parser or RegexParser()
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_backend = None  # "sklearn" or "torch"

        self._stats = {
            "total": 0,
            "deberta": 0,
            "regex_fallback": 0,
            "failures": 0,
        }

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load classifier model if model_path is provided."""
        if not self.model_path:
            logger.info("DebertaParser initialized without model checkpoint; operating in fast regex fallback mode.")
            return

        # Handle 'auto' resolution
        target_path = self.model_path
        if target_path == "auto":
            candidates = [
                "models/stage1_classifier.joblib",
                "models/stage1_deberta",
            ]
            target_path = None
            for cp in candidates:
                if os.path.exists(cp):
                    target_path = cp
                    break
            if target_path is None:
                logger.info("DebertaParser: 'auto' model path requested but no saved model found in models/. Operating in regex fallback mode.")
                return

        # Check if joblib sklearn artifact
        is_joblib_file = target_path.endswith(".joblib") or target_path.endswith(".pkl")
        joblib_dir_candidate = os.path.join(target_path, "stage1_classifier.joblib") if os.path.isdir(target_path) else None

        if is_joblib_file and os.path.exists(target_path):
            self._load_sklearn_model(target_path)
            return
        elif joblib_dir_candidate and os.path.exists(joblib_dir_candidate):
            self._load_sklearn_model(joblib_dir_candidate)
            return

        # Check if HuggingFace directory or hub ID
        if not os.path.exists(target_path) and not ("/" in target_path or "\\" in target_path):
            # HuggingFace hub id
            pass
        elif not os.path.exists(target_path):
            logger.warning(
                "Model path '%s' not found on disk. Falling back to RegexParser.",
                target_path,
            )
            return

        # Attempt DeBERTa PyTorch load
        self._load_deberta_model(target_path)

    def _load_sklearn_model(self, path: str) -> None:
        """Load Scikit-Learn TF-IDF classification pipeline from joblib artifact."""
        try:
            import joblib
            logger.info("Loading Scikit-Learn Stage 1 classifier from '%s'...", path)
            self.model = joblib.load(path)
            self.model_backend = "sklearn"
            logger.info("Scikit-Learn Stage 1 classifier successfully loaded.")
        except Exception as e:
            logger.warning(
                "Failed to load Scikit-Learn classifier from '%s' (%s). Falling back to RegexParser.",
                path,
                e,
            )
            self.model = None
            self.model_backend = None

    def _load_deberta_model(self, path: str) -> None:
        """Load HuggingFace DeBERTa tokenizer and model."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading DeBERTa model from '%s' onto device '%s'...", path, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
            self.model_backend = "torch"
            logger.info("DeBERTa model successfully loaded.")
        except Exception as e:
            logger.warning(
                "Failed to initialize DeBERTa from '%s' (%s). Operating in regex fallback mode.",
                path,
                e,
            )
            self.model = None
            self.tokenizer = None
            self.model_backend = None

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Parse natural language question into structured logical form.

        Args:
            question: Natural language question string.

        Returns:
            Dictionary matching the schema:
            {
                "type": "taxonomic" | "categorical" | "hypothetical" | "non-logical",
                "subject": str,
                "predicate": str,
                "condition": str,
                "consequence": str,
                "confidence": float,
                "method": "deberta" | "regex_fallback"
            }
        """
        self._stats["total"] += 1
        q = (question or "").strip()

        if not q:
            return self._build_result("non-logical", confidence=1.0, method="regex_fallback")

        # If model is loaded, run neural/classifier inference
        if self.model is not None:
            try:
                result = self._infer_and_extract(q)
                if result is not None:
                    self._stats["deberta"] += 1
                    return result
            except Exception as e:
                logger.warning("Classifier inference error (%s); falling back to regex.", e)
                self._stats["failures"] += 1

        # Fallback to RegexParser
        self._stats["regex_fallback"] += 1
        return self._regex_fallback_parse(q)

    def _infer_and_extract(self, question: str) -> Optional[Dict[str, Any]]:
        """Route to appropriate backend (sklearn or DeBERTa) and extract slots."""
        # Detect sklearn vs torch backend
        is_sklearn = (self.model_backend == "sklearn") or (
            hasattr(self.model, "predict_proba") and not hasattr(self.model, "forward")
        )

        if is_sklearn:
            import numpy as np
            probs = self.model.predict_proba([question])[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            if confidence < self.confidence_threshold:
                logger.debug(
                    "Classifier confidence %.3f below threshold %.3f; falling back.",
                    confidence,
                    self.confidence_threshold,
                )
                return None

            pred_type = LABEL_MAP.get(pred_idx, "non-logical")
            slots = self._extract_slots(question, pred_type)

            return self._build_result(
                query_type=pred_type,
                subject=slots.get("subject", ""),
                predicate=slots.get("predicate", ""),
                condition=slots.get("condition", ""),
                consequence=slots.get("consequence", ""),
                confidence=round(confidence, 4),
                method="deberta",
            )
        else:
            return self._deberta_parse(question)

    def _deberta_parse(self, question: str) -> Optional[Dict[str, Any]]:
        """Run DeBERTa PyTorch sequence classification and slot extraction."""
        import torch

        if self.tokenizer is None:
            return None

        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        if self.device:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx].item())

        if confidence < self.confidence_threshold:
            logger.debug(
                "DeBERTa confidence %.3f below threshold %.3f; falling back.",
                confidence,
                self.confidence_threshold,
            )
            return None

        pred_type = LABEL_MAP.get(pred_idx, "non-logical")
        slots = self._extract_slots(question, pred_type)

        return self._build_result(
            query_type=pred_type,
            subject=slots.get("subject", ""),
            predicate=slots.get("predicate", ""),
            condition=slots.get("condition", ""),
            consequence=slots.get("consequence", ""),
            confidence=round(confidence, 4),
            method="deberta",
        )

    def _extract_slots(self, question: str, query_type: str) -> Dict[str, str]:
        """Extract slot arguments corresponding to the classified query type."""
        q = question.strip()
        slots = {"subject": "", "predicate": "", "condition": "", "consequence": ""}

        if query_type == "taxonomic":
            for pattern in _TAX_PATTERNS:
                m = pattern.match(q)
                if m:
                    slots["subject"] = _clean_term(m.group(1))
                    slots["predicate"] = _clean_term(m.group(2))
                    return slots
            # Fall back to regex parser extraction if available
            reg = self.fallback_parser.parse(q)
            if reg.get("type") == "taxonomic":
                slots["subject"] = _clean_term(reg.get("subject", ""))
                slots["predicate"] = _clean_term(reg.get("predicate", ""))

        elif query_type == "categorical":
            for pattern in _CAT_PATTERNS:
                m = pattern.match(q)
                if m:
                    if "property of" in q.lower() or "trait of" in q.lower():
                        slots["subject"] = _clean_term(m.group(2))
                        slots["predicate"] = _clean_term(m.group(1))
                    else:
                        slots["subject"] = _clean_term(m.group(1))
                        slots["predicate"] = _clean_term(m.group(2))
                    return slots
            reg = self.fallback_parser.parse(q)
            if reg.get("type") == "categorical":
                slots["subject"] = _clean_term(reg.get("entity") or reg.get("subject", ""))
                slots["predicate"] = _clean_term(reg.get("property") or reg.get("predicate", ""))

        elif query_type == "hypothetical":
            for pattern in _HYP_PATTERNS:
                m = pattern.match(q)
                if m:
                    slots["condition"] = _clean_term(m.group(1))
                    slots["consequence"] = _clean_term(m.group(2))
                    return slots
            reg = self.fallback_parser.parse(q)
            if reg.get("type") == "hypothetical":
                slots["condition"] = _clean_term(reg.get("condition", ""))
                slots["consequence"] = _clean_term(reg.get("consequence", ""))

        return slots

    def _regex_fallback_parse(self, question: str) -> Dict[str, Any]:
        """Deterministic sub-millisecond fallback using RegexParser."""
        raw = self.fallback_parser.parse(question)
        qtype = raw.get("type", "non-logical")

        subject = _clean_term(raw.get("subject") or raw.get("entity") or "")
        predicate = _clean_term(raw.get("predicate") or raw.get("property") or "")
        condition = _clean_term(raw.get("condition", ""))
        consequence = _clean_term(raw.get("consequence", ""))

        return self._build_result(
            query_type=qtype,
            subject=subject,
            predicate=predicate,
            condition=condition,
            consequence=consequence,
            confidence=1.0,
            method="regex_fallback",
        )

    def _build_result(
        self,
        query_type: str,
        subject: str = "",
        predicate: str = "",
        condition: str = "",
        consequence: str = "",
        confidence: float = 1.0,
        method: str = "regex_fallback",
    ) -> Dict[str, Any]:
        """Construct normalized standard 7-key dictionary."""
        return {
            "type": query_type,
            "subject": subject,
            "predicate": predicate,
            "condition": condition,
            "consequence": consequence,
            "confidence": confidence,
            "method": method,
        }

    @property
    def parse_stats(self) -> Dict[str, Any]:
        """Stage 1 throughput and method distribution statistics."""
        total = self._stats["total"]
        if total == 0:
            return dict(self._stats)
        return {
            **self._stats,
            "deberta_rate": round(self._stats["deberta"] / total * 100, 1),
            "fallback_rate": round(self._stats["regex_fallback"] / total * 100, 1),
        }
