"""
Dense RAG (Retrieval-Augmented Generation) Baseline for AvicennaGuard.

Implements an embedding-based and TF-IDF hybrid RAG baseline (Lewis et al., NeurIPS 2020)
grounded against the 1,500-node AvicennaGuard Knowledge Base.

Architecture:
    1. Knowledge Corpus: Converts the 1,500+ KB graph relations (taxonomies, properties,
       conditionals) into natural-language declarative fact sentences.
    2. Dense Retriever: Encodes KB facts into dense embeddings using SentenceTransformers
       (or fast TF-IDF similarity in mock/offline mode).
    3. Context Augmentation: Retrieves top-K semantically relevant KB facts and injects
       them into the LLM context prompt.
    4. Generation / Reasoning: LLM answers YES/NO given the retrieved context (T=0.0).

Reference:
    Lewis P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
    NeurIPS 2020. https://arxiv.org/abs/2005.11401
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from avicennaguard.baselines.metrics import (
    compute_classification_metrics,
    compute_group_metrics,
    format_metrics_summary,
)

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    _HAS_SKLEARN = False

try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    ollama = None
    _HAS_OLLAMA = False


DEFAULT_KB_PATHS = [
    Path("data/knowledge_bases/knowledge_base_extended.json"),
    Path("data/knowledge_bases/knowledge_base.json"),
    Path(__file__).resolve().parents[3] / "data" / "knowledge_bases" / "knowledge_base_extended.json",
    Path(__file__).resolve().parents[3] / "data" / "knowledge_bases" / "knowledge_base.json",
    Path.cwd() / "data" / "knowledge_bases" / "knowledge_base_extended.json",
]


def kb_to_facts(kb_source: Union[str, Path, dict]) -> List[str]:
    """
    Convert Knowledge Base graph into natural-language declarative fact sentences.

    Extracts:
      - Taxonomy IS-A edges -> "A {child} is a {ancestor}." / "All {child}s are {ancestor}s."
      - Entity properties -> "A {entity} has {property}."
      - Conditionals -> "If {condition}, then {consequence}."

    Args:
        kb_source: Path to KB JSON file or pre-loaded dictionary.

    Returns:
        List of unique declarative fact strings.
    """
    if isinstance(kb_source, (str, Path)):
        p = Path(kb_source)
        if not p.exists():
            raise FileNotFoundError(f"KB file not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
    elif isinstance(kb_source, dict):
        kb_data = kb_source
    else:
        raise ValueError(f"Unsupported KB source type: {type(kb_source)}")

    facts: List[str] = []
    seen = set()

    def add_fact(fact: str) -> None:
        clean = fact.strip()
        if clean and clean not in seen:
            seen.add(clean)
            facts.append(clean)

    # 1. Taxonomies (IS-A)
    taxonomies = kb_data.get("taxonomies", {})
    for entity, ancestors in taxonomies.items():
        e_clean = entity.replace("_", " ")
        for anc in ancestors:
            anc_clean = anc.replace("_", " ")
            add_fact(f"A {e_clean} is a {anc_clean}.")
            add_fact(f"All {e_clean}s are {anc_clean}s.")

    # 2. Properties (HAS)
    properties = kb_data.get("properties", {})
    for entity, props in properties.items():
        e_clean = entity.replace("_", " ")
        for prop in props:
            p_clean = prop.replace("_", " ")
            if p_clean.startswith("has "):
                p_clean = p_clean[4:]
            add_fact(f"A {e_clean} has {p_clean}.")
            add_fact(f"All {e_clean}s have {p_clean}.")

    # 3. Conditionals (IF-THEN)
    conditionals = kb_data.get("conditionals", {})
    for cond, consequences in conditionals.items():
        c_clean = cond.replace("_", " ")
        for conseq in consequences:
            cq_clean = conseq.replace("_", " ")
            add_fact(f"If {c_clean}, then {cq_clean}.")

    return facts


class SparseTFIDFRetriever:
    """TF-IDF / keyword similarity retriever for fast and deterministic offline retrieval."""

    def __init__(self, facts: List[str]) -> None:
        """
        Initialize SparseTFIDFRetriever.

        Args:
            facts: List of declarative fact strings.
        """
        self.facts = facts
        self.stop_words = {
            "a", "an", "the", "is", "are", "all", "do", "does", "have",
            "has", "it", "if", "then", "true", "that", "be", "of", "in",
            "for", "to", "and", "or", "not", "no", "yes", "what", "which",
        }
        if _HAS_SKLEARN and len(facts) > 0:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.fact_vectors = self.vectorizer.fit_transform(facts)
        else:
            self.vectorizer = None
            self.fact_vectors = None

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-K most similar facts with relevance scores.

        Args:
            query: Query text string.
            top_k: Number of nearest facts to retrieve.

        Returns:
            List of (fact_string, relevance_score) tuples.
        """
        if not self.facts:
            return []

        if self.vectorizer is not None and self.fact_vectors is not None:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.fact_vectors).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(sims[idx])
                if score > 0.0:
                    results.append((self.facts[idx], round(score, 4)))
            if results:
                return results

        # Pure Python fallback token-overlap scoring
        q_tokens = set(self._tokenize(query))
        scored: List[Tuple[str, float]] = []
        for fact in self.facts:
            f_tokens = set(self._tokenize(fact))
            if not f_tokens:
                continue
            intersection = q_tokens.intersection(f_tokens)
            if intersection:
                overlap_score = len(intersection) / (len(q_tokens) + 1e-5)
                scored.append((fact, round(overlap_score, 4)))

        scored.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
        if scored:
            return scored[:top_k]
        return [(self.facts[i], 0.1) for i in range(min(top_k, len(self.facts)))]


class DenseEmbeddingRetriever:
    """Embedding-based dense retriever using SentenceTransformers."""

    def __init__(self, facts: List[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize DenseEmbeddingRetriever.

        Args:
            facts: List of declarative knowledge base facts.
            model_name: SentenceTransformer encoder model name.
        """
        self.facts = facts
        self.model_name = model_name
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is not installed")

        self.encoder = SentenceTransformer(model_name)
        embeddings = self.encoder.encode(
            facts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.fact_embeddings = np.array(embeddings, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-K semantic nearest facts using cosine similarity.

        Args:
            query: Natural language query string.
            top_k: Number of nearest facts to return.

        Returns:
            List of (fact_string, similarity_score) tuples.
        """
        if not self.facts:
            return []
        q_emb = self.encoder.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        q_emb = np.array(q_emb, dtype=np.float32)
        # Cosine similarity via dot product of normalized vectors
        scores = np.dot(self.fact_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.facts[idx], round(float(scores[idx]), 4)) for idx in top_indices]


@dataclass
class DenseRAGResult:
    """Structured output from a Dense RAG query evaluation."""

    query_id: str
    question: str
    prediction: bool
    final_answer: str
    retrieved_facts: List[str]
    similarity_scores: List[float]
    context_used: str
    ground_truth: Optional[Any] = None
    query_type: str = "unknown"
    source: str = "unknown"
    latency_retrieval_ms: float = 0.0
    latency_generation_ms: float = 0.0
    latency_total_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


class DenseRAGBaseline:
    """
    Dense RAG baseline integrating KB triple retrieval and augmented LLM generation.

    Args:
        kb_path: Path to knowledge base JSON file (auto-resolved if None).
        model: LLM generation model (e.g. 'llama3.2:3b').
        embedding_model: SentenceTransformer model name for dense encoding.
        top_k: Number of relevant KB facts to retrieve per query.
        use_dense: If True and sentence-transformers is available, uses dense embeddings;
                   otherwise falls back to TF-IDF retriever.
        mock: If True, operates in deterministic offline mock mode.
    """

    def __init__(
        self,
        kb_path: Optional[Union[str, Path]] = None,
        model: str = "llama3.2:3b",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        use_dense: bool = True,
        mock: bool = False,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.top_k = max(1, top_k)
        self.mock = mock or (not _HAS_OLLAMA)
        self.use_dense = use_dense and _HAS_SENTENCE_TRANSFORMERS and (not mock)

        self.kb_path = self._resolve_kb_path(kb_path)
        self.facts = kb_to_facts(self.kb_path)
        self._init_retriever()

    @staticmethod
    def _resolve_kb_path(kb_path: Optional[Union[str, Path]]) -> Path:
        if kb_path is not None:
            p = Path(kb_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"Provided KB path does not exist: {p}")

        for candidate in DEFAULT_KB_PATHS:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError("Could not locate AvicennaGuard knowledge base file in default paths.")

    def _init_retriever(self) -> None:
        """Initialize dense or sparse retriever."""
        if self.use_dense and _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.retriever = DenseEmbeddingRetriever(self.facts, model_name=self.embedding_model)
                self.retriever_type = "dense_embedding"
                return
            except Exception as e:
                logger.warning("Dense retriever init failed (%s); falling back to TF-IDF", e)

        self.retriever = SparseTFIDFRetriever(self.facts)
        self.retriever_type = "sparse_tfidf"

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Retrieve top-K KB facts for query."""
        k = top_k if top_k is not None else self.top_k
        return self.retriever.retrieve(query, top_k=k)

    def build_augmented_prompt(self, question: str, retrieved_facts: List[str]) -> str:
        """Construct the context-augmented LLM prompt."""
        if not retrieved_facts:
            context_block = "No relevant knowledge found."
        else:
            context_block = "\n".join(f"- {f}" for f in retrieved_facts)

        return (
            f"Context information:\n"
            f"{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Based strictly on the provided context and logical reasoning, answer with ONLY 'YES' or 'NO'. "
            f"Do not provide explanations."
        )

    def _call_ollama_with_context(self, prompt: str) -> str:
        """Query LLM with augmented context."""
        if not _HAS_OLLAMA:
            return "no"

        system_msg = "You are a factual assistant. Answer strictly with YES or NO based on the context."
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0, "num_predict": 10},
            )
            content = resp.get("message", {}).get("content", "").strip().upper()
            if re.search(r"\bYES\b", content) or "TRUE" in content:
                return "yes"
            if re.search(r"\bNO\b", content) or "FALSE" in content:
                return "no"
            return "no"
        except Exception as e:
            logger.warning("Ollama call failed in RAG baseline: %s", e)
            return "no"

    def _mock_reasoning(self, question: str, retrieved_facts: List[str]) -> str:
        """Deterministic mock reasoning over retrieved facts for offline testing."""
        q_lower = question.lower()
        context_text = " ".join(retrieved_facts).lower()

        # Check explicit negative patterns
        if "spider" in q_lower and "insect" in q_lower:
            return "no"
        if "fish" in q_lower and "mammal" in q_lower:
            return "no"
        if "whale" in q_lower and "fish" in q_lower:
            return "no"
        if "not" in q_lower or "neither" in q_lower:
            return "no"

        # Check explicit positive support
        tokens = [t for t in re.findall(r"[a-z]+", q_lower) if len(t) > 3 and t not in ("true", "that", "does", "have")]
        if len(tokens) >= 2:
            t1, t2 = tokens[0], tokens[1]
            if (t1 in context_text and t2 in context_text) or any(t1 in f.lower() and t2 in f.lower() for f in retrieved_facts):
                return "yes"

        # Check general high context overlap
        overlap = sum(1 for t in tokens if t in context_text)
        return "yes" if overlap >= max(1, len(tokens) // 2) else "no"

    def predict(
        self,
        question: str,
        query_id: str = "",
        ground_truth: Optional[Any] = None,
        query_type: str = "unknown",
        source: str = "unknown",
        top_k: Optional[int] = None,
    ) -> DenseRAGResult:
        """
        Execute full RAG retrieval + augmented reasoning for a single query.
        """
        k = top_k if top_k is not None else self.top_k

        # 1. Retrieval
        t_ret_0 = time.perf_counter()
        retrieved = self.retrieve(question, top_k=k)
        ret_latency_ms = (time.perf_counter() - t_ret_0) * 1000.0

        facts = [r[0] for r in retrieved]
        scores = [r[1] for r in retrieved]
        context_str = "\n".join(facts)

        # 2. Generation / Reasoning
        t_gen_0 = time.perf_counter()
        if self.mock:
            answer = self._mock_reasoning(question, facts)
        else:
            prompt = self.build_augmented_prompt(question, facts)
            answer = self._call_ollama_with_context(prompt)
        gen_latency_ms = (time.perf_counter() - t_gen_0) * 1000.0

        pred_bool = (answer == "yes")
        total_latency_ms = ret_latency_ms + gen_latency_ms

        return DenseRAGResult(
            query_id=query_id,
            question=question,
            prediction=pred_bool,
            final_answer=answer,
            retrieved_facts=facts,
            similarity_scores=scores,
            context_used=context_str,
            ground_truth=ground_truth,
            query_type=query_type,
            source=source,
            latency_retrieval_ms=round(ret_latency_ms, 2),
            latency_generation_ms=round(gen_latency_ms, 2),
            latency_total_ms=round(total_latency_ms, 2),
            metadata={
                "model": self.model,
                "retriever_type": self.retriever_type,
                "top_k": k,
                "mock": self.mock,
                "num_kb_facts": len(self.facts),
            },
        )

    def evaluate_dataset(
        self,
        benchmark_data: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        max_queries: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Dense RAG baseline across benchmark dataset.
        """
        queries = benchmark_data[:max_queries] if max_queries is not None else benchmark_data
        total_queries = len(queries)
        results: List[Dict[str, Any]] = []

        ret_latencies: List[float] = []
        gen_latencies: List[float] = []
        tot_latencies: List[float] = []

        for idx, item in enumerate(queries):
            qid = item.get("id", f"query_{idx:04d}")
            qtext = item.get("question", "")
            gt = item.get("ground_truth")
            qtype = item.get("query_type", "unknown")
            source = item.get("source", "unknown")

            res = self.predict(
                question=qtext,
                query_id=qid,
                ground_truth=gt,
                query_type=qtype,
                source=source,
                top_k=top_k,
            )
            results.append(res.to_dict())
            ret_latencies.append(res.latency_retrieval_ms)
            gen_latencies.append(res.latency_generation_ms)
            tot_latencies.append(res.latency_total_ms)

            if progress_callback:
                progress_callback(idx + 1, total_queries)

        predictions = [r["prediction"] for r in results]
        ground_truths = [r["ground_truth"] for r in results]

        metrics = compute_classification_metrics(predictions, ground_truths)
        by_type = compute_group_metrics(results, group_key="query_type")
        by_source = compute_group_metrics(results, group_key="source")

        mean_ret_lat = sum(ret_latencies) / len(ret_latencies) if ret_latencies else 0.0
        mean_gen_lat = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0.0
        mean_tot_lat = sum(tot_latencies) / len(tot_latencies) if tot_latencies else 0.0

        summary_text = format_metrics_summary("Dense RAG", metrics, by_type, by_source)

        return {
            "method": "Dense RAG",
            "model": self.model,
            "retriever_type": self.retriever_type,
            "top_k": top_k or self.top_k,
            "mock": self.mock,
            "total_queries": total_queries,
            "num_kb_facts": len(self.facts),
            "mean_latency_retrieval_ms": round(mean_ret_lat, 2),
            "mean_latency_generation_ms": round(mean_gen_lat, 2),
            "mean_latency_total_ms": round(mean_tot_lat, 2),
            "metrics": metrics,
            "per_query_type": by_type,
            "per_source": by_source,
            "results": results,
            "summary_text": summary_text,
        }
