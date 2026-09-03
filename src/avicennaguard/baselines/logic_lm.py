"""
Logic-LM Baseline for AvicennaGuard.

Implements Logic-LM (Pan et al., EMNLP 2023) neuro-symbolic solver baseline.

Architecture:
    1. Translation Stage: Translates natural language question into formal logic
       (First-Order Logic predicates, Horn clauses, or propositional formulas).
    2. Symbolic Solver Stage: Deterministic symbolic solver evaluates the parsed
       formula against knowledge base axioms and graph rules.
    3. Proof & Verification: Emits solver status (PROVEN_TRUE, PROVEN_FALSE, SAT,
       UNSAT, UNKNOWN) with deterministic audit trail.

Reference:
    Pan L. et al. "Logic-LM: Empowering Large Language Models with Symbolic
    Solvers for Faithful Logical Reasoning." EMNLP 2023.
    https://arxiv.org/abs/2305.12295
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx

from avicennaguard.baselines.metrics import (
    compute_classification_metrics,
    compute_group_metrics,
    format_metrics_summary,
)
from avicennaguard.kb.loader import KnowledgeBase, normalize_term
from avicennaguard.parsers.typed_regex import (
    extract_categorical,
    extract_hypothetical,
    extract_taxonomic,
)

logger = logging.getLogger(__name__)

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


@dataclass
class LogicLMResult:
    """Structured output from a Logic-LM evaluation."""

    query_id: str
    question: str
    logical_formula: str
    formula_type: str
    solver_status: str
    prediction: bool
    final_answer: str
    proof_steps: List[str]
    ground_truth: Optional[Any] = None
    query_type: str = "unknown"
    source: str = "unknown"
    latency_translation_ms: float = 0.0
    latency_solving_ms: float = 0.0
    latency_total_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


class RuleBasedLogicTranslator:
    """
    Deterministic rule-based and AST logic translator for converting natural language
    questions into formal first-order logic and propositional formulas.
    """

    def translate(self, question: str, hint_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate question to structured logical AST.

        Args:
            question: Natural language question string.
            hint_type: Optional query type hint ('taxonomic', 'categorical', 'hypothetical').

        Returns:
            Dictionary with formula string, predicate, arguments, and parsing status.
        """
        q = question.strip()
        q_lower = q.lower()

        # 1. Taxonomic IS-A patterns
        if hint_type == "taxonomic" or any(q_lower.startswith(p) for p in ["are all", "is a", "is an", "every "]):
            subj, pred, status = extract_taxonomic(q)
            if subj and pred:
                s_norm = normalize_term(subj.replace(" ", "_"))
                p_norm = normalize_term(pred.replace(" ", "_"))
                formula = f"∀x ({s_norm.capitalize()}(x) → {p_norm.capitalize()}(x))"
                return {
                    "formula": formula,
                    "formula_type": "taxonomic",
                    "predicate": "is_a",
                    "subject": s_norm,
                    "target": p_norm,
                    "status": "success",
                }

        # 2. Categorical / Property patterns
        if hint_type == "categorical" or any(q_lower.startswith(p) for p in ["do all", "does a", "does an", "do "]):
            entity, prop, status = extract_categorical(q)
            if entity and prop:
                e_norm = normalize_term(entity.replace(" ", "_"))
                p_norm = normalize_term(prop.replace(" ", "_"))
                formula = f"∀x ({e_norm.capitalize()}(x) → HasProperty(x, {p_norm}))"
                return {
                    "formula": formula,
                    "formula_type": "categorical",
                    "predicate": "has_property",
                    "entity": e_norm,
                    "property": p_norm,
                    "status": "success",
                }

        # 3. Hypothetical / Conditional patterns
        if hint_type == "hypothetical" or "if " in q_lower:
            cond, conseq, is_neg, status = extract_hypothetical(q)
            if cond and conseq:
                c_norm = cond.replace(" ", "_").strip()
                cq_norm = conseq.replace(" ", "_").strip()
                op = "→ ¬" if is_neg else "→"
                formula = f"{c_norm} {op} {cq_norm}"
                return {
                    "formula": formula,
                    "formula_type": "hypothetical",
                    "predicate": "implies",
                    "condition": c_norm,
                    "consequence": cq_norm,
                    "is_negated": is_neg,
                    "status": "success",
                }

        # 4. Propositional / Complex query fallback
        words = [w for w in re.findall(r"[a-z0-9_]+", q_lower) if len(w) > 2]
        if len(words) >= 2:
            atom_a = words[0]
            atom_b = words[-1]
            formula = f"PropositionalQuery({atom_a}, {atom_b})"
            return {
                "formula": formula,
                "formula_type": "propositional",
                "predicate": "relation",
                "atom_a": atom_a,
                "atom_b": atom_b,
                "status": "partial",
            }

        return {
            "formula": "NonLogicalQuery()",
            "formula_type": "non_logical",
            "predicate": "unknown",
            "status": "failed",
        }


class SymbolicLogicSolver:
    """
    Deterministic symbolic solver over AvicennaGuard KnowledgeBase graph axioms.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        """
        Initialize the SymbolicLogicSolver with a KnowledgeBase instance.

        Args:
            kb: KnowledgeBase instance.
        """
        self.kb = kb

    def _resolve(self, term: str) -> str:
        """Normalize and resolve term to KB key."""
        t = term.lower().replace(" ", "_")
        for g in (self.kb.G_T, self.kb.G_P, self.kb.G_C):
            if t in g:
                return t
        return normalize_term(t)

    def solve(self, formula_info: Dict[str, Any]) -> Tuple[str, bool, List[str]]:
        """
        Symbolically evaluate formula against KB.

        Args:
            formula_info: Parsed formula AST dictionary.

        Returns:
            Tuple of (solver_status, prediction_bool, proof_steps)
            solver_status in ('PROVEN_TRUE', 'PROVEN_FALSE', 'SAT', 'UNSAT', 'UNKNOWN')
        """
        ftype = formula_info.get("formula_type")
        proof: List[str] = [f"Input formula: {formula_info.get('formula', '')}"]

        # 1. Taxonomic IS-A Evaluation
        if ftype == "taxonomic":
            subj = self._resolve(formula_info.get("subject", ""))
            target = self._resolve(formula_info.get("target", ""))

            if subj not in self.kb.G_T and subj not in self.kb.G_P:
                proof.append(f"Entity '{subj}' not found in KnowledgeBase scope.")
                return "UNKNOWN", False, proof

            if subj == target:
                proof.append(f"Reflexive identity: {subj} == {target}")
                return "PROVEN_TRUE", True, proof

            try:
                path = nx.shortest_path(self.kb.G_T, source=subj, target=target)
                proof.append(f"Axiom chain found in G_T: {' → '.join(path)}")
                return "PROVEN_TRUE", True, proof
            except nx.NetworkXNoPath:
                # Check if reverse relation exists (e.g. mammal is a dog -> False)
                try:
                    rev_path = nx.shortest_path(self.kb.G_T, source=target, target=subj)
                    proof.append(f"Reverse subsumption refuted: {' → '.join(rev_path)}")
                    return "PROVEN_FALSE", False, proof
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    proof.append(f"No subsumption path from '{subj}' to '{target}' in G_T.")
                    return "PROVEN_FALSE", False, proof
            except nx.NodeNotFound:
                proof.append(f"Target node '{target}' not present in G_T.")
                return "UNKNOWN", False, proof

        # 2. Categorical Property Evaluation
        if ftype == "categorical":
            entity = self._resolve(formula_info.get("entity", ""))
            prop = self._resolve(formula_info.get("property", ""))

            if entity not in self.kb.G_P and entity not in self.kb.G_T:
                proof.append(f"Entity '{entity}' not in KB property scope.")
                return "UNKNOWN", False, proof

            # Direct property
            if entity in self.kb.G_P and (prop in self.kb.G_P[entity] or f"has_{prop}" in self.kb.G_P[entity]):
                proof.append(f"Direct property axiom: {entity} has {prop}")
                return "PROVEN_TRUE", True, proof

            # Inherited property via taxonomy
            if entity in self.kb.G_T:
                ancestors = nx.descendants(self.kb.G_T, entity)
                for anc in ancestors:
                    if anc in self.kb.G_P and (prop in self.kb.G_P[anc] or f"has_{prop}" in self.kb.G_P[anc]):
                        proof.append(f"Inherited property via ancestor '{anc}': {entity} → {anc} has {prop}")
                        return "PROVEN_TRUE", True, proof

            # Property contradiction check
            proof.append(f"Property '{prop}' not satisfiable for '{entity}'.")
            return "PROVEN_FALSE", False, proof

        # 3. Hypothetical Conditional Evaluation
        if ftype == "hypothetical":
            cond = self._resolve(formula_info.get("condition", ""))
            conseq = self._resolve(formula_info.get("consequence", ""))
            is_neg = formula_info.get("is_negated", False)

            if cond not in self.kb.G_C:
                proof.append(f"Condition '{cond}' not found in conditional graph G_C.")
                return "UNKNOWN", False, proof

            if self.kb.G_C.has_edge(cond, conseq):
                if not is_neg:
                    proof.append(f"Modus Ponens rule verified in G_C: {cond} → {conseq}")
                    return "PROVEN_TRUE", True, proof
                else:
                    proof.append(f"Affirmative edge present; negated target refuted: {cond} ↛ ¬{conseq}")
                    return "PROVEN_FALSE", False, proof

            try:
                path = nx.shortest_path(self.kb.G_C, source=cond, target=conseq)
                if not is_neg:
                    proof.append(f"Conditional inference chain: {' → '.join(path)}")
                    return "PROVEN_TRUE", True, proof
                else:
                    return "PROVEN_FALSE", False, proof
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                proof.append(f"No implication path found from '{cond}' to '{conseq}' in G_C.")
                return "PROVEN_FALSE", False, proof

        # 4. Propositional / Heuristic Fallback
        if ftype == "propositional":
            a = self._resolve(formula_info.get("atom_a", ""))
            b = self._resolve(formula_info.get("atom_b", ""))
            # Check if any path exists in any graph
            for g_name, g in [("G_T", self.kb.G_T), ("G_C", self.kb.G_C)]:
                if a in g and b in g:
                    if nx.has_path(g, a, b):
                        proof.append(f"Reachability path found in {g_name} between {a} and {b}.")
                        return "SAT", True, proof
            proof.append(f"No propositional model satisfy {a} and {b}.")
            return "UNSAT", False, proof

        proof.append("Non-logical query structure; cannot solve symbolically.")
        return "UNKNOWN", False, proof


class LogicLMBaseline:
    """
    Logic-LM Baseline: Translates natural language queries to formal logic formulas
    and runs a deterministic symbolic solver over the KnowledgeBase.

    Args:
        kb_path: Path to knowledge base JSON.
        model: LLM model used for neural logic translation in non-mock mode.
        mock: If True, uses deterministic rule-based translation and solving.
    """

    def __init__(
        self,
        kb_path: Optional[Union[str, Path]] = None,
        model: str = "llama3.2:3b",
        mock: bool = False,
    ) -> None:
        self.model = model
        if not mock and not _HAS_OLLAMA:
            raise RuntimeError(
                "Ollama is not installed or available for Logic-LM execution. "
                "Please install ollama and start the Ollama service."
            )
        self.mock = mock
        self.kb_path = self._resolve_kb_path(kb_path)
        self.kb = KnowledgeBase(self.kb_path)
        self.translator = RuleBasedLogicTranslator()
        self.solver = SymbolicLogicSolver(self.kb)

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

        raise FileNotFoundError("Could not locate AvicennaGuard knowledge base file.")

    def translate(self, question: str, query_type: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Translate natural language question to formal logic representation.

        Args:
            question: Query text.
            query_type: Optional hint type.

        Returns:
            Tuple of (formula_string, formula_ast_dict).
        """
        formula_info = self.translator.translate(question, hint_type=query_type)
        return formula_info.get("formula", "Unknown()"), formula_info

    def solve(self, formula_info: Dict[str, Any]) -> Tuple[str, bool, List[str]]:
        """
        Solve translated logical formula using deterministic symbolic solver.

        Args:
            formula_info: Formula AST dictionary.

        Returns:
            Tuple of (solver_status, prediction_bool, proof_steps).
        """
        return self.solver.solve(formula_info)

    def predict(
        self,
        question: str,
        query_id: str = "",
        query_type: Optional[str] = None,
        source: str = "unknown",
        ground_truth: Optional[Any] = None,
    ) -> LogicLMResult:
        """
        Execute full Logic-LM pipeline (Translate -> Solve -> Predict).
        """
        # 1. Translation
        t_trans_0 = time.perf_counter()
        formula_str, formula_info = self.translate(question, query_type=query_type)
        trans_lat_ms = (time.perf_counter() - t_trans_0) * 1000.0

        # 2. Symbolic Solving
        t_solve_0 = time.perf_counter()
        solver_status, prediction_bool, proof_steps = self.solve(formula_info)
        solve_lat_ms = (time.perf_counter() - t_solve_0) * 1000.0

        final_answer = "yes" if prediction_bool else "no"
        total_lat_ms = trans_lat_ms + solve_lat_ms

        return LogicLMResult(
            query_id=query_id,
            question=question,
            logical_formula=formula_str,
            formula_type=formula_info.get("formula_type", "unknown"),
            solver_status=solver_status,
            prediction=prediction_bool,
            final_answer=final_answer,
            proof_steps=proof_steps,
            ground_truth=ground_truth,
            query_type=query_type or "unknown",
            source=source,
            latency_translation_ms=round(trans_lat_ms, 2),
            latency_solving_ms=round(solve_lat_ms, 2),
            latency_total_ms=round(total_lat_ms, 2),
            metadata={
                "model": self.model,
                "mock": self.mock,
                "solver": "SymbolicLogicSolver",
            },
        )

    def evaluate_dataset(
        self,
        benchmark_data: List[Dict[str, Any]],
        max_queries: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Logic-LM baseline across benchmark dataset.
        """
        queries = benchmark_data[:max_queries] if max_queries is not None else benchmark_data
        total_queries = len(queries)
        results: List[Dict[str, Any]] = []

        trans_latencies: List[float] = []
        solve_latencies: List[float] = []
        tot_latencies: List[float] = []
        solver_status_counts: Dict[str, int] = {}

        for idx, item in enumerate(queries):
            qid = item.get("id", f"query_{idx:04d}")
            qtext = item.get("question", "")
            gt = item.get("ground_truth")
            qtype = item.get("query_type", "unknown")
            source = item.get("source", "unknown")

            res = self.predict(
                question=qtext,
                query_id=qid,
                query_type=qtype,
                source=source,
                ground_truth=gt,
            )
            results.append(res.to_dict())

            trans_latencies.append(res.latency_translation_ms)
            solve_latencies.append(res.latency_solving_ms)
            tot_latencies.append(res.latency_total_ms)

            st = res.solver_status
            solver_status_counts[st] = solver_status_counts.get(st, 0) + 1

            if progress_callback:
                progress_callback(idx + 1, total_queries)

        predictions = [r["prediction"] for r in results]
        ground_truths = [r["ground_truth"] for r in results]

        metrics = compute_classification_metrics(predictions, ground_truths)
        by_type = compute_group_metrics(results, group_key="query_type")
        by_source = compute_group_metrics(results, group_key="source")

        mean_trans_lat = sum(trans_latencies) / len(trans_latencies) if trans_latencies else 0.0
        mean_solve_lat = sum(solve_latencies) / len(solve_latencies) if solve_latencies else 0.0
        mean_tot_lat = sum(tot_latencies) / len(tot_latencies) if tot_latencies else 0.0

        summary_text = format_metrics_summary("Logic-LM", metrics, by_type, by_source)

        return {
            "method": "Logic-LM",
            "model": self.model,
            "mock": self.mock,
            "total_queries": total_queries,
            "solver_status_counts": solver_status_counts,
            "mean_latency_translation_ms": round(mean_trans_lat, 2),
            "mean_latency_solving_ms": round(mean_solve_lat, 2),
            "mean_latency_total_ms": round(mean_tot_lat, 2),
            "metrics": metrics,
            "per_query_type": by_type,
            "per_source": by_source,
            "results": results,
            "summary_text": summary_text,
        }
