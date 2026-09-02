# 🛡️ AvicennaGuard v2.0
### *Deterministic Hallucination Interception in Large Language Models Using Avicennian Syllogistic Frameworks*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Architecture-Neuro--Symbolic-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Knowledge%20Base-1%2C500%20Nodes-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Benchmark-500%20Queries-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Precision-100%25%20(FP%3D0)-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tests-145%2F145%20Passed-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.18745460">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18745460.svg" alt="DOI"/>
  </a>
  &nbsp;
  <a href="https://github.com/HamzaNasiem/LogicGuard/tree/avicennaguard">
    <img src="https://img.shields.io/badge/GitHub-AvicennaGuard%20v2.0-181717?style=flat&logo=github"/>
  </a>
</p>

<p align="center">
  <i>"Probabilistic AI guesses. AvicennaGuard formally proves."</i>
</p>

---

## 🌟 What Is AvicennaGuard?

Large Language Models (LLMs) operate via probabilistic next-token prediction:
$$\mathcal{P}(w_t \mid w_1, w_2, \dots, w_{t-1})$$
Consequently, even state-of-the-art LLMs (GPT-4o, Claude 3.5, LLaMA-3.2, Mistral) confidently generate structurally impossible claims:
- *"Not all squares are rectangles"* (Euclidean geometric impossibility)
- *"Fish have hair"* (Taxonomic category violation)
- *"Spiders are insects"* (Arachnid vs Insect cross-branch contradiction)

**AvicennaGuard** is a model-agnostic, pre-delivery neuro-symbolic middleware that computationally formalizes the 1,000-year-old epistemic logic of **Ibn Sina (Avicenna, 980–1037 CE)** (*Kitab al-Shifa*):
1. **Stage 1 ($Ta\d{s}awwur$ / Concept Formation):** Sub-millisecond Neural Semantic Parser that extracts structured proposition slots without letting the LLM generate unverified text.
2. **Stage 2 ($Ta\d{s}d\bar{i}q$ / Assent & Proof):** Deterministic Breadth-First Search (BFS) graph traversal over a multi-relational Knowledge Base ($G_T, G_P, G_C$) with **mathematically guaranteed Zero False Positives ($\text{FPR} = 0$)**.

---

## 🚀 Key Experimental Results (500-Query Benchmark)

Evaluated on the standardized **500-query non-circular benchmark** (`avicenna_benchmark_500.json`) comprising **200 FOLIO** (Yale NLP), **150 ProofWriter** (AllenAI), **100 Curated Gold Syllogisms**, and **50 TruthfulQA Out-Of-Domain** questions:

### 1. Comparison with State-of-the-Art Baselines

| Method / Baseline | Accuracy | Precision | Recall | F1 Score | False Positives | Guard Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw LLM (LLaMA3.2-3B)** | 85.0% | 100.0% | 75.0% | 85.7% | 0 | --- |
| **SelfCheckGPT ($N=5$)** | 85.5% | 96.5% | 67.5% | 79.4% | 2 | ~4,400 ms |
| **RAG-Sparse (BM25)** | 84.0% | 100.0% | 61.5% | 76.1% | 0 | ~6,700 ms |
| **RAG-Dense (MiniLM / mpnet)** | 80.0% - 82.0% | 100.0% | 51.8% - 56.6% | 68.2% - 72.3% | 0 | ~5,600 - 6,100 ms |
| **Logic-LM (Symbolic Solver)** | 44.0% | 80.0% | 12.9% | 22.2% | 9 | ~1,200 ms |
| **AvicennaGuard (Ours)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0 (Guaranteed)** | **$< 0.1$ ms** |

*Key finding:* AvicennaGuard executes in **$<0.1\text{ms}$** ($>40,000\times$ faster than sampling/retrieval baselines) while maintaining **Zero False Alarms**.

---

## 🏛️ Epistemic Architecture & Ibn Sina's Logic

AvicennaGuard replaces naive binary True/False classification with Ibn Sina's 4-state graded epistemology:

| Epistemic State | Arabic Term | Operational Meaning | Action Taken |
| :--- | :--- | :--- | :--- |
| **YAQEEN 🟢** | *Yaqīn* (Certainty) | Reachability path verified in $G_T / G_P / G_C$ | Confirm LLM answer & emit audit trail |
| **WAHM 🔴** | *Wahm* (Illusion) | Deterministic graph refutes LLM claim | **INTERCEPT & Correct** before delivery |
| **SHAKK 🟠** | *Shakk* (Doubt) | Query entity out-of-domain (OOD) | **Safe Deferral** (Never intervene; preserves FPR=0) |
| **ZANN 🟡** | *Za\d{n}n* (Conjecture) | Regex heuristic match only | Pass-through with epistemic flag |

```
                              USER QUERY
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │        STAGE 1: NEURAL SEMANTIC PARSER           │
        │   - DebertaParser (<1.0 ms inference latency)    │
        │   - Trained Multi-ngram Model (99.60% Val Acc)   │
        │   - Strict Regex Fallback (<0.1 ms)              │
        └─────────────────────────┬────────────────────────┘
                                  │ Proposition JSON
                                  ▼
        ┌──────────────────────────────────────────────────┐
        │       STAGE 2: DETERMINISTIC BFS GRAPH ENGINE    │
        │   - KnowledgeBase (1,500 Nodes, 418 Properties)  │
        │   - Directed BFS Traversal (0.04 - 0.09 ms)      │
        │   - 4-State Epistemic Output Engine              │
        └─────────────────────────┬────────────────────────┘
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      ▼                           ▼                           ▼
 [ YAQEEN ]                  [ WAHM ]                  [ SHAKK / ZANN ]
(Certainty)                 (Illusion)                  (Doubt / OOD)
Confirm LLM             INTERCEPT & Correct           Safe Pass-Through
(FP = 0)                (Hallucination Caught)        (Zero False Alarms)
```

---

## 📂 Repository Structure

```
LogicGuard/
├── src/avicennaguard/
│   ├── api/                     # FastAPI middleware microservice
│   ├── baselines/               # SOTA Baselines (SelfCheckGPT, Dense RAG, Logic-LM)
│   ├── core/                    # Epistemic states (YAQEEN, WAHM, SHAKK, ZANN)
│   ├── data/                    # BenchmarkLoader module
│   ├── eval/                    # BenchmarkRunner, StatisticalAnalyzer, LatexGenerator
│   ├── kb/                      # Three-Graph Loader (G_T, G_P, G_C) & BFS Validator
│   ├── parsers/                 # DebertaParser (<1ms) & RegexParser
│   └── pipeline/                # AvicennaGuard master pipeline
├── data/
│   ├── benchmarks/              # avicenna_benchmark_500.json (FOLIO, ProofWriter, etc.)
│   ├── knowledge_bases/         # knowledge_base_extended.json (1,500 nodes, 418 props)
│   └── training/                # 5,000 synthetic pairs (stage1_train, stage1_val)
├── docs/paper/tables/           # Automated IEEE LaTeX tables (Tables I–V)
├── models/                      # stage1_classifier.joblib (99.60% accuracy)
├── scripts/                     # Benchmark runners, training, and table generator scripts
└── tests/                       # 145 Unit & Integration Tests (100% Green)
```

---

## 🛠️ Quickstart & Installation

```bash
# Clone the repository and switch to avicennaguard branch
git clone https://github.com/HamzaNasiem/LogicGuard.git
cd LogicGuard
git checkout avicennaguard

# Install dependencies
pip install -e .
```

### Run Full Test Suite (145 Tests)
```bash
python -m pytest tests/ -v
# Output: 145 passed in ~13s (100% Green)
```

### Run Multi-Model 500-Query Benchmark
```bash
python scripts/run_benchmark_eval.py --models all --benchmark data/benchmarks/avicenna_benchmark_500.json
```

### Run SOTA Baselines
```bash
python scripts/run_baseline_selfcheck.py
python scripts/run_baseline_rag.py
python scripts/run_baseline_logic_lm.py
```

### Generate Publication-Ready LaTeX Tables
```bash
python scripts/generate_paper_artifacts.py
# Outputs LaTeX Tables I–V in docs/paper/tables/
```

---

## 📜 Citation

```bibtex
@misc{naseem2026avicennaguard,
  author    = {Naseem, Hamza and Ali, Moiz},
  title     = {AvicennaGuard: A Neuro-Symbolic Middleware for Deterministic Hallucination Interception in Large Language Models Using Avicennian Syllogistic Frameworks},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18745460},
  url       = {https://doi.org/10.5281/zenodo.18745460}
}
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for details.

## Contributing

Open an issue before submitting major changes. Pull requests welcome for:
- KB extensions (new taxonomies, properties, conditionals)
- New query types or evaluation domains
- Stage 1 parser improvements

---

<p align="center">
  Built on classical logic and modern AI.<br>
  <i>Ibn Sina (980–1037 CE) formalized deductive logic. We made it intercept LLM hallucinations.</i><br><br>
  <a href="https://doi.org/10.5281/zenodo.18745460">📄 Read the Paper</a> &nbsp;·&nbsp;
  <a href="https://github.com/HamzaNasiem/LogicGuard">💻 View Code</a>
</p>