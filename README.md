# 🛡️ LogicGuard
### *Deterministic Hallucination Interception in Large Language Models Using Aristotelian-Avicennian Syllogistic Frameworks*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-Multi--Model-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Engine-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/IEEE-Under%20Review-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Precision-100%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.18745460">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18745460.svg" alt="DOI"/>
  </a>
  &nbsp;
  <a href="https://github.com/HamzaNasiem/LogicGuard">
    <img src="https://img.shields.io/badge/GitHub-LogicGuard-181717?style=flat&logo=github"/>
  </a>
</p>

<p align="center">
  <i>"Probabilistic AI guesses. LogicGuard formally proves."</i>
</p>

---

## What Is This?

Large Language Models are powerful semantic engines — but they are fundamentally unreliable deductive reasoners. Operating on token probabilities, they confidently produce logically impossible outputs:

- *"Not all squares are rectangles"* — logically impossible by Euclidean definition
- *"Fish have hair"* — structurally false by taxonomic classification
- *"Spiders are insects"* — a cross-branch error that any KB can definitively refute

In 2023, Google's Bard AI hallucinated one claim during a live demonstration and erased **$100 billion** in market capitalization the same day.

**LogicGuard** is a hybrid neuro-symbolic middleware that sits between LLMs and users. It computationally implements the 1,000-year-old syllogistic framework (*Qiyas / Mantiq*) of Ibn Sina (Avicenna), building a deterministic interceptor that catches and corrects structural hallucinations before they reach the user.

The core architectural insight: **parsing is probabilistic; reasoning must be deterministic.**

---

## Key Results

Evaluated on a **100-query original syllogism set** (circular-eval removed; 175 total with 75 KB-generated excluded) — spanning biological taxonomy, geometric relations, and physical conditionals — using three open-weight LLMs running locally via Ollama:

> **Note:** Research eval defaults to `--parser regex` (~0.05ms Stage 1). Production API uses LLM parser (~2–7s Stage 1). Use `--parser both` in step2 to compare.

### Accuracy Improvement

| Model | Baseline | +LogicGuard | Δ |
|-------|----------|-------------|---|
| LLaMA2-7B | 60.0% | **94.3%** | +34.3 pp |
| Mistral-7B | 94.9% | **97.7%** | +2.8 pp |
| LLaMA3.2-3B | 84.6% | **96.6%** | +12.0 pp |

### Precision / Recall / F1

| Model | Precision | Recall | F1 | Spec. | FP |
|-------|-----------|--------|----|-------|----|
| LLaMA2-7B +LG | **100%** | 90.9% | 95.2% | **100%** | **0** |
| Mistral-7B +LG | **100%** | 96.4% | 98.1% | **100%** | **0** |
| LLaMA3.2-3B +LG | **100%** | 94.5% | 97.2% | **100%** | **0** |

**Precision = 100% and FP = 0 across 525 total evaluations (175 queries × 3 models).**

### Hallucination Interception

| Model | LLM Errors | Intercepted | Rate |
|-------|-----------|-------------|------|
| LLaMA2-7B | 70 | 62 | **88.6%** |
| Mistral-7B | 9 | 9 | **100.0%** |
| LLaMA3.2-3B | 27 | 25 | **92.6%** |

### Out-of-Domain Generalization

Applied to the full **TruthfulQA benchmark** (790 general-knowledge questions) without any LLM calls:

- **99.5% non-interference rate** — LogicGuard correctly deferred to the LLM on 786/790 questions
- Only 4 questions (0.5%) matched KB patterns — proving zero KB-test co-derivation

---

## Architecture

LogicGuard is a **two-stage neuro-symbolic pipeline**:

```
User Question (natural language)
        │
        ▼
┌──────────────────────────────────┐
│  Stage 1: Neural Semantic Parser │  ← LLM constrained to JSON-only output
│                                  │    Temperature=0. Never answers.
│  {"type": "taxonomic",           │    Falls back to regex if Ollama offline.
│   "subject": "dog",              │
│   "predicate": "mammal"}         │
└────────────────┬─────────────────┘
                 │  Structured JSON proposition
                 ▼  (no logical content trusted)
┌──────────────────────────────────┐
│  Stage 2: BFS Graph Validator    │  ← 100% deterministic. No probability.
│                                  │    NetworkX directed semantic graph.
│  dog → canine → mammal → ✓      │    115 nodes, 136 IS-A edges.
│  graph answer: TRUE              │
└────────────────┬─────────────────┘
                 │
                 ▼
        YAQEEN / WAHM / ZANN / SHAKK
```

### The Key Contribution

The LLM in Stage 1 is **caged**: it can only output one of four JSON schemas. It never answers the question. All actual logical reasoning happens in Stage 2, which is pure graph traversal with mathematical guarantees.

If the query falls outside the KB (Shakk state), LogicGuard **does not intervene** — it defers to the LLM. This deliberate pass-through is why Precision stays at 100%: the system never overclaims certainty on queries it cannot formally adjudicate.

---

## Three Forms of Ibn Sina's Qiyas (Syllogism)

### 1. Qiyas al-Haml — Taxonomic (IS-A)
```
"Are all dogs mammals?"
→ JSON:  {"type": "taxonomic", "subject": "dog", "predicate": "mammal"}
→ BFS:   dog → canine → mammal ✓ (path found)
→ State: YAQEEN (Certainty — override LLM if wrong)

"Are all animals dogs?"
→ BFS:   No path from animal → dog ✗
→ State: WAHM (Illusion — intercept LLM hallucination)
```

### 2. Qiyas al-Istithna — Hypothetical (Modus Ponens)
```
"If water freezes, does it become ice?"
→ JSON:  {"type": "hypothetical", "condition": "water freezes", "consequence": "ice"}
→ Check: water_freezes → ice ∈ G_C ✓
→ State: YAQEEN
```

### 3. Categorical — Property Inheritance
```
"Do all birds have feathers?"
→ JSON:  {"type": "categorical", "entity": "bird", "property": "feathers"}
→ Check: bird → feathers ∈ G_P ✓ (direct or inherited)
→ State: YAQEEN

"Do all fish have hair?"
→ Check: fish ⊬ hair ✗
→ State: WAHM
```

---

## Epistemic State Classification

LogicGuard replaces binary True/False with Ibn Sina's four-state epistemic framework:

| State | Meaning | When | Action |
|-------|---------|------|--------|
| **Yaqeen** 🟢 | Certainty | BFS path confirmed in KB | Override LLM with validated answer |
| **Zann** 🟡 | Probability | Semantic match; no formal structure | Return LLM answer with confidence flag |
| **Shakk** 🟠 | Doubt | Entity absent from KB scope | Defer to LLM — no intervention |
| **Wahm** 🔴 | Illusion | LLM answer contradicts BFS result | Intercept and flag structural hallucination |

The **Shakk** state is the precision guarantee: when LogicGuard is uncertain, it says so and defers. It never invents certainty it does not have.

---

## Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally

```bash
# Install models (choose what you want to evaluate)
ollama pull llama2          # 7B — higher error rate baseline
ollama pull mistral         # 7B — strong mid-range baseline
ollama pull llama3.2:3b     # 3B — compact edge model

# Clone
git clone https://github.com/HamzaNasiem/LogicGuard.git
cd LogicGuard

# Install dependencies
pip install networkx pandas numpy matplotlib ollama scikit-learn
```

---

## Usage

### Full Pipeline (Steps 1–5)
```bash
# Requires ProofWriter dataset (download separately — not in repo)
python run_all.py --proofwriter_dir path/to/proofwriter-dataset-V2020.12.3

# Steps only (if KB and queries already built)
python run_all.py --steps 2,3
python run_all.py --steps 4,5   # TruthfulQA + paper tables
```

### Step-by-Step

```bash
# Step 1: Build Knowledge Base from ProofWriter
python step1_proofwriter_extractor.py --proofwriter_dir ./proofwriter-dataset-V2020.12.3

# Step 2: Run multi-model evaluation (requires Ollama + models)
python step2_multi_model_runner.py

# Step 3: Compute metrics (Precision/Recall/F1, confusion matrices)
python step3_metrics.py

# Step 4: TruthfulQA generalization test (no LLM needed — fast)
python step4_truthfulqa_validation.py --csv truthfulqa.csv --kb knowledge_base_extended.json

# Step 5: Generate all IEEE paper tables and text
python step5_generate_paper_tables.py
```

### Use as a Library

```python
from step2_multi_model_runner import LogicGuardValidator
import json

with open("knowledge_base_extended.json") as f:
    kb = json.load(f)

validator = LogicGuardValidator(kb)
result = validator.validate("Are all squares rectangles?", "taxonomic")

print(result["epistemic_state"])   # YAQEEN
print(result["graph_answer"])      # True
print(result["covered"])           # True
```

---

## Repository Structure

```
LogicGuard/
│
├── step1_proofwriter_extractor.py  # ProofWriter → KB builder
├── step2_multi_model_runner.py     # Multi-model evaluation engine
├── step3_metrics.py                # P/R/F1, confusion matrices, reports
├── step4_truthfulqa_validation.py  # Out-of-domain generalization test
├── step5_generate_paper_tables.py  # IEEE paper tables
├── run_all.py                      # Master pipeline runner (Steps 1–5)
├── knowledge_base.json             # Base KB (hand-curated)
└── knowledge_base_extended.json    # KB after ProofWriter extension
```

---

## Knowledge Base

Three interconnected directed graphs built on top of ProofWriter triples:

```python
# Taxonomy (IS-A hierarchy) — 115 nodes, 136 edges
dog → canine → mammal → animal → living_thing
square → rectangle → quadrilateral → polygon → shape
spider → arachnid → invertebrate → animal → living_thing

# Property (with transitive inheritance) — 115 associations
mammal   → {hair, warm_blood, backbone, gives_milk, ...}
bird     → {feathers, wings, beak, lay_eggs, ...}
reptile  → {scales, cold_blood, ...}
insect   → {six_legs, exoskeleton, ...}
arachnid → {eight_legs, ...}

# Conditional (Modus Ponens rules) — 49 rules
raining          → {ground_wet, wet}
water_freezes    → {ice, solid, becomes_ice}
fire_present     → {heat, smoke, oxygen_consumed}
metal_heated     → {expands}
```

---

## Why Precision = 100% Is Not a Suspicious Claim

For a probabilistic system, 100% precision on any non-trivial dataset would rightly invite scrutiny. LogicGuard's Stage 2 is not probabilistic.

A false positive requires the BFS algorithm to erroneously determine that an IS-A path does not exist when it does. **This is computationally impossible given a correct KB.** BFS either finds a path or it doesn't, and its answer is verified by the graph structure itself.

What *is* empirical — and where real uncertainty resides — is the Recall figure (90.9–96.4%), which reflects genuine KB coverage gaps. These are reported honestly.

The formal scope: **Precision = 100% within KB-covered queries.** The system explicitly returns Shakk and defers on queries outside this scope. The 99.5% non-interference rate on TruthfulQA demonstrates this scope is conservatively applied.

---

## Reproducibility

All experiments run locally on commodity hardware (CPU-only), no GPU required:

```bash
# Fixed seed for full reproducibility
# All Ollama calls: temperature=0.0, seed=42

python run_all.py --proofwriter_dir proofwriter-dataset-V2020.12.3
# Runtime: ~44 minutes (all 3 models, 175 queries × 2 configs each)
```

---

## Citation

If you use LogicGuard in your research, please cite both the paper and the code:

**Paper (Zenodo preprint):**
```bibtex
@misc{naseem2026logicguard,
  author    = {Naseem, Hamza and Ali, Moiz},
  title     = {LogicGuard: A Neuro-Symbolic Middleware for Deterministic
               Hallucination Interception in Large Language Models
               Using Aristotelian-Avicennian Syllogistic Frameworks},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18745460},
  url       = {https://doi.org/10.5281/zenodo.18745460}
}
```

---

## Future Work

- **ConceptNet / Wikidata integration** — Replace the manually curated KB with 8M+ semantic relationships via public APIs
- **Legal and medical domains** — Statutes as conditionals, symptom-disease mappings
- **Fine-tuned Stage 1 parser** — BERT-based sequence classifier to eliminate LLM dependency in parsing
- **Multi-hop conditionals** — Explicit chaining semantics for nested IF-THEN inference
- **Real-time API** — FastAPI wrapper for enterprise hallucination guardrail deployment

---

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