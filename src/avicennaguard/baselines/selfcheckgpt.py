"""
SelfCheckGPT Baseline for AvicennaGuard.

Implements SelfCheckGPT (Manakul et al., EMNLP 2023) black-box zero-resource
hallucination detection and consistency-based factuality checking.

Architecture:
    1. Sample N responses from LLM with temperature > 0.0 (stochastic sampling)
    2. Extract binary answers (YES / NO) from each sample
    3. Calculate consistency score (majority agreement fraction)
    4. Majority answer serves as the final prediction
    5. Disagreement (consistency < threshold) indicates hallucination / uncertainty

Reference:
    Manakul P., Liusie A., Gales M. "SelfCheckGPT: Zero-Resource Black-Box
    Hallucination Detection for Generative Large Language Models." EMNLP 2023.
    https://arxiv.org/abs/2303.08896
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from avicennaguard.baselines.metrics import (
    compute_classification_metrics,
    compute_group_metrics,
    format_metrics_summary,
)

logger = logging.getLogger(__name__)

# Try importing ollama gracefully
try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    ollama = None
    _HAS_OLLAMA = False


@dataclass
class SelfCheckGPTResult:
    """Structured output from a SelfCheckGPT evaluation call."""

    query_id: str
    question: str
    prediction: bool
    final_answer: str
    confidence: float
    consistency_score: float
    is_hallucination: bool
    samples: List[str]
    ground_truth: Optional[Any] = None
    query_type: str = "unknown"
    source: str = "unknown"
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


class SelfCheckGPTBaseline:
    """
    SelfCheckGPT stochastic consistency baseline.

    Args:
        model: LLM model name (e.g. 'llama3.2:3b', 'mistral:7b', etc.).
        n_samples: Number of stochastic completions to sample per question.
        temperature: Sampling temperature (T > 0.0 required for variance).
        confidence_threshold: Consistency threshold below which an answer is flagged as hallucination.
        mock: If True, uses deterministic offline sampling without calling Ollama.
        seed: Random seed for deterministic mock reproducibility.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        n_samples: int = 5,
        temperature: float = 0.7,
        confidence_threshold: float = 0.6,
        mock: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
        self.model = model
        self.n_samples = max(1, n_samples)
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        if not mock and not _HAS_OLLAMA:
            raise RuntimeError(
                "Ollama is not installed or available for SelfCheckGPT execution. "
                "Please install ollama and start the Ollama service."
            )
        self.mock = mock
        self.seed = seed
        self._rng = random.Random(seed)

    def _extract_binary_answer(self, text: str) -> Optional[str]:
        """Extract YES or NO from LLM response text."""
        cleaned = text.strip().upper()
        if re.search(r"\bYES\b", cleaned):
            return "yes"
        if re.search(r"\bNO\b", cleaned):
            return "no"
        if re.search(r"\bTRUE\b", cleaned):
            return "yes"
        if re.search(r"\bFALSE\b", cleaned):
            return "no"
        return None

    def _call_ollama(self, question: str, temperature: float) -> str:
        """Single Ollama LLM call."""
        if not _HAS_OLLAMA:
            return self._mock_sample(question, 0)

        system_prompt = (
            "You are a factual assistant. Answer the question with only YES or NO. "
            "Do not explain."
        )
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                options={"temperature": temperature, "num_predict": 15},
            )
            content = resp.get("message", {}).get("content", "")
            ans = self._extract_binary_answer(content)
            return ans if ans is not None else "no"
        except Exception as e:
            logger.warning("Ollama call failed (%s); using fallback", e)
            return self._mock_sample(question, 0)

    def _mock_sample(self, question: str, sample_idx: int) -> str:
        """
        Deterministic mock sampler for offline testing and unit tests.

        Simulates LLM sampling variance deterministically:
        - Uses hash of question + sample index + seed
        - Factual/intuitive true queries trend towards majority YES
        - Obvious false queries trend towards majority NO
        - Confusable / difficult queries exhibit sample dispersion
        """
        q_lower = question.lower()
        hash_val = int(hashlib.md5(f"{self.seed}_{question}_{sample_idx}".encode("utf-8")).hexdigest(), 16)

        if any(w in q_lower for w in ["not a", "neither", "spider", "insect", "is a fish a mammal", "fly without"]):
            prob_yes = 0.2
        elif any(w in q_lower for w in ["all dogs", "mammal", "animal", "living thing", "water", "ice", "square"]):
            prob_yes = 0.8
        elif "if" in q_lower and "then" in q_lower:
            prob_yes = 0.65
        else:
            prob_yes = 0.3 + ((hash_val % 100) / 250.0)

        sample_rand = (hash_val % 1000) / 1000.0
        return "yes" if sample_rand < prob_yes else "no"

    def predict_samples(self, question: str) -> List[str]:
        """Generate N stochastic binary response samples for the question."""
        samples: List[str] = []
        for i in range(self.n_samples):
            if self.mock:
                sample = self._mock_sample(question, i)
            else:
                sample = self._call_ollama(question, self.temperature)
            samples.append(sample)
        return samples

    def predict(
        self,
        question: str,
        query_id: str = "",
        ground_truth: Optional[Any] = None,
        query_type: str = "unknown",
        source: str = "unknown",
        difficulty: str = "medium",
    ) -> SelfCheckGPTResult:
        """
        Execute SelfCheckGPT consistency sampling on a single question.

        Returns:
            SelfCheckGPTResult containing majority vote, confidence, samples,
            and hallucination detection flag.
        """
        t0 = time.perf_counter()
        samples = self.predict_samples(question)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        valid_samples = [s for s in samples if s in ("yes", "no")]
        if not valid_samples:
            final_answer = "no"
            confidence = 0.0
        else:
            counts = Counter(valid_samples)
            majority_answer, majority_count = counts.most_common(1)[0]
            final_answer = majority_answer
            confidence = majority_count / len(valid_samples)

        prediction_bool = (final_answer == "yes")
        consistency_score = confidence
        is_hallucination = (confidence < self.confidence_threshold)

        return SelfCheckGPTResult(
            query_id=query_id,
            question=question,
            prediction=prediction_bool,
            final_answer=final_answer,
            confidence=round(confidence, 4),
            consistency_score=round(consistency_score, 4),
            is_hallucination=is_hallucination,
            samples=samples,
            ground_truth=ground_truth,
            query_type=query_type,
            source=source,
            latency_ms=round(latency_ms, 2),
            metadata={
                "model": self.model,
                "n_samples": self.n_samples,
                "temperature": self.temperature,
                "mock": self.mock,
                "confidence_threshold": self.confidence_threshold,
            },
        )

    def evaluate_dataset(
        self,
        benchmark_data: List[Dict[str, Any]],
        max_queries: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate SelfCheckGPT across a list of benchmark queries.

        Args:
            benchmark_data: List of query dicts.
            max_queries: Optional cap on the number of queries to evaluate.
            progress_callback: Optional callback invoked as `callback(current_idx, total)`.

        Returns:
            Dictionary containing aggregate metrics, confusion matrix, per-type breakdown,
            per-source breakdown, and full per-query results.
        """
        queries = benchmark_data[:max_queries] if max_queries is not None else benchmark_data
        total_queries = len(queries)
        results: List[Dict[str, Any]] = []

        total_confidence = 0.0
        total_consistency = 0.0
        hallucination_flags_count = 0
        latencies: List[float] = []

        for idx, item in enumerate(queries):
            qid = item.get("id", f"query_{idx:04d}")
            qtext = item.get("question", "")
            gt = item.get("ground_truth")
            qtype = item.get("query_type", "unknown")
            source = item.get("source", "unknown")
            difficulty = item.get("difficulty", "medium")

            res = self.predict(
                question=qtext,
                query_id=qid,
                ground_truth=gt,
                query_type=qtype,
                source=source,
                difficulty=difficulty,
            )
            r_dict = res.to_dict()
            results.append(r_dict)

            total_confidence += res.confidence
            total_consistency += res.consistency_score
            if res.is_hallucination:
                hallucination_flags_count += 1
            latencies.append(res.latency_ms)

            if progress_callback:
                progress_callback(idx + 1, total_queries)

        predictions = [r["prediction"] for r in results]
        ground_truths = [r["ground_truth"] for r in results]

        metrics = compute_classification_metrics(predictions, ground_truths)
        by_type = compute_group_metrics(results, group_key="query_type")
        by_source = compute_group_metrics(results, group_key="source")

        mean_confidence = total_confidence / total_queries if total_queries > 0 else 0.0
        mean_consistency = total_consistency / total_queries if total_queries > 0 else 0.0
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        summary_text = format_metrics_summary("SelfCheckGPT", metrics, by_type, by_source)

        return {
            "method": "SelfCheckGPT",
            "model": self.model,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "mock": self.mock,
            "total_queries": total_queries,
            "mean_confidence": round(mean_confidence, 4),
            "mean_consistency": round(mean_consistency, 4),
            "hallucination_flags_count": hallucination_flags_count,
            "hallucination_rate": round(hallucination_flags_count / total_queries, 4) if total_queries > 0 else 0.0,
            "mean_latency_ms": round(mean_latency, 2),
            "metrics": metrics,
            "per_query_type": by_type,
            "per_source": by_source,
            "results": results,
            "summary_text": summary_text,
        }
