# 🛡️ LogicGuard
### *Deterministic Validation of LLM Hallucinations Using Aristotelian-Avicennian Syllogistic Frameworks*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ollama-LLaMA%203.2-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NetworkX-Graph%20Engine-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/IEEE-Under%20Review-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Logic%20Accuracy-100%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <i>"Probabilistic AI guesses. LogicGuard formally proves."</i>
</p>

---

## 📖 What Is This?

Large Language Models (LLMs) are extraordinary semantic engines — but they are fundamentally broken logical reasoners. Operating on token probabilities, they confidently hallucinate logically impossible claims:

- *"Not all squares are rectangles"* — logically impossible, yet LLMs say it
- *"Fish have hair"* — structurally false, yet LLMs rationalize it
- In 2023, Google Bard hallucinated one fact during a live demo and erased **$100 billion** in market value in a single day

**LogicGuard** is a neuro-symbolic middleware that solves the structural reasoning problem. It computationally models the 1,000-year-old Aristotelian-Avicennian syllogistic logic (**Qiyas / Mantiq**) of Ibn Sina (Avicenna), building a deterministic interceptor that sits between users and LLMs.

When an LLM violates strict deductive logic, **LogicGuard overrides it before the answer reaches the user**.

---

## 🚀 Key Results

Evaluated on a **hybrid dataset of 890 questions** — combining the full TruthfulQA benchmark with 100 custom logical syllogisms — using a local **Llama 3.2 (3B)** model:

| Metric | LLM Baseline (Llama 3.2) | **LogicGuard** |
|--------|--------------------------|----------------|
| Logic Q Accuracy | ~60% | **100.0% (Yaqeen)** |
| Non-Logic Q Accuracy | 31.3% | 31.3% (Zann/Shakk) |
| Logical Hallucinations Caught | 0% | **100% intercepted** |

> The non-logical accuracy is identical — LogicGuard does **not** degrade conversational performance. It only activates where it can be formally certain.

---

## 🧠 Architecture

LogicGuard is a **three-layer neuro-symbolic pipeline**:

```
User Question (messy natural language)
        │
        ▼
┌───────────────────────────────┐
│   Layer 1: Semantic Parser    │  ← LLaMA 3.2 as a constrained JSON-only
│   (semantic_parser.py)        │    extractor. Temperature=0. Never answers.
│                               │    Falls back to regex if Ollama offline.
└───────────────┬───────────────┘
                │  Structured JSON:
                │  {"type": "taxonomic", "subject": "dog", "predicate": "mammal"}
                ▼
┌───────────────────────────────┐
│   Layer 2: NetworkX Graph     │  ← 100% deterministic. BFS traversal on
│   (knowledge_graph.py)        │    directed semantic graph. No probability.
│                               │    115 nodes, 136 IS-A edges.
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Layer 3: Epistemic State    │  ← Ibn Sina's 4-state classification
│   (logic_validator.py)        │
└───────────────────────────────┘
        │
        ▼
   YAQEEN / WAHM / ZANN / SHAKK
```

### The Key Architectural Insight (IEEE Contribution)

> **PARSING is probabilistic. REASONING is 100% deterministic.**

The LLM (Layer 1) is caged — it can only output one of four JSON schemas. It never answers the question. All actual logical reasoning happens in Layer 2, which is pure graph traversal with mathematical guarantees.

---

## 🔬 Three Forms of Ibn Sina's Qiyas (Syllogism)

### 1. Qiyas al-Haml — Taxonomic Logic
*"Is A a kind of B?"* → BFS traversal on IS-A taxonomy graph

```
"Are all dogs mammals?" 
→ Parser: {"type": "taxonomic", "subject": "dog", "predicate": "mammal"}
→ Graph:  dog → canine → mammal ✓
→ State:  YAQEEN (100% Certainty)

"Are all animals dogs?"
→ Graph:  No path from animal → dog ✗
→ State:  WAHM (Illusion — LLM hallucination intercepted)
```

### 2. Qiyas al-Istithna — Hypothetical Logic (Modus Ponens)
*"If A, then B?"* → Conditional edge lookup

```
"If water freezes, does it become ice?"
→ Parser: {"type": "hypothetical", "condition": "water freezes", "consequence": "ice"}
→ Graph:  water_freezes → ice ✓
→ State:  YAQEEN
```

### 3. Categorical — Property Inheritance
*"Do all X have property Y?"* → Property graph with transitive inheritance

```
"Do all birds lay eggs?"
→ Parser: {"type": "categorical", "entity": "bird", "property": "lay_eggs"}
→ Graph:  bird → lay_eggs ✓ (via property graph)
→ State:  YAQEEN

"Do all fish have hair?"
→ Graph:  fish ⊬ hair ✗
→ State:  WAHM
```

---

## 🟢 Epistemic State Classification

LogicGuard replaces binary Pass/Fail with Ibn Sina's classical epistemic framework:

| State | Symbol | Meaning | When |
|-------|--------|---------|------|
| **Yaqeen** | 🟢 | Certainty (100%) | Formally proven via graph traversal |
| **Zann** | 🟡 | Probability (>50%) | Not logical, but LLM semantically correct |
| **Shakk** | 🟠 | Doubt (<50%) | Ambiguous factual answer |
| **Wahm** | 🔴 | Illusion (0%) | LLM contradicted a proven logical rule |

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally with `llama3.2:3b` pulled

```bash
# Install Ollama model
ollama pull llama3.2:3b

# Clone and install
git clone https://github.com/HamzaNasiem/LogicGuard.git
cd LogicGuard
pip install networkx pandas numpy matplotlib ollama
```

---

## 🎮 Usage

### Quick Test (No Ollama needed — verifies your setup)
```bash
python quick_test.py
```
Expected output: `24/24 PASSED — Logic templates working perfectly!`

### Interactive Chatbot Demo
```bash
python chatbot.py
```
```
🤖 LogicGuard Chatbot Active

🧑 You: Are all penguins birds?
🤖 LLM says: Yes, penguins are birds...
🔬 LogicGuard Analysis:
   • Type: TAXONOMIC
   • Epistemic State: YAQEEN
   • Proof: BFS proof: penguin → bird
```

### Run the Full 890-Question Experiment
```bash
python run.py
```
> ⚠️ Requires `truthfulqa.csv` in the same directory. Takes 15–30 minutes depending on hardware.

### Use as a Library
```python
from logic_validator import LogicValidator

# With Ollama (handles messy natural language)
validator = LogicValidator(use_ollama=True, model="llama3.2:3b")

# Offline / fast mode (regex fallback)
validator = LogicValidator(use_ollama=False)

result = validator.validate("Are all squares rectangles?")
print(result['epistemic_state'])  # YAQEEN
print(result['proof'])            # BFS proof: square → rectangle
```

---

## 📁 Repository Structure

```
LogicGuard/
│
├── knowledge_graph.py     # NetworkX directed semantic graph
│                          # 115 taxonomy nodes, 136 IS-A edges
│                          # BFS transitive inference engine
│
├── semantic_parser.py     # Translation Layer (LLaMA 3.2 as JSON-only parser)
│                          # Handles messy natural language → structured JSON
│                          # temperature=0, format='json', max_tokens=80
│
├── logic_validator.py     # Main pipeline: parse → graph → epistemic state
│
├── logic_templates.py     # Regex fallback parser (offline mode)
│
├── chatbot.py             # Interactive demo
├── quick_test.py          # 24-question offline test suite
├── run.py                 # Full 890-question IEEE experiment
└── truthfulqa.csv         # TruthfulQA benchmark dataset
```

---

## 🧪 The Knowledge Graph

The backbone of LogicGuard is a **directed semantic graph** with three sub-graphs:

```python
# Taxonomy graph (IS-A hierarchy)
dog → canine → mammal → animal → living_thing
square → rectangle → quadrilateral → polygon → shape

# Property graph (with inheritance)
mammal → {hair, fur, gives_milk, warm_blooded, backbone, ...}
bird   → {feathers, wings, lay_eggs, beak, ...}

# Conditional graph (Modus Ponens)
raining     → {ground_wet, wet, sky_cloudy}
water_freezes → {ice, solid, becomes_ice}
fire        → {heat, smoke, light, dangerous}
```

**Graph Statistics:**
- Taxonomy: 115 nodes, 136 edges
- Properties: 97 nodes, 115 edges  
- Conditionals: 51 nodes, 49 edges

---

## 🔑 The Semantic Parser (Translation Layer)

The biggest academic contribution of this work is the **strict separation of parsing from reasoning**:

```python
# LLM is caged with this system prompt (simplified):
"""
You are a formal logic extraction engine.
You do NOT answer questions.
Output ONLY a raw JSON object — one of four schemas:

{"type": "taxonomic",    "subject": X,    "predicate": Y}
{"type": "categorical",  "entity": X,     "property": Y}  
{"type": "hypothetical", "condition": X,  "consequence": Y}
{"type": "non-logical"}
"""

# Called with temperature=0, format='json', max_tokens=80
# If Ollama unavailable → automatic regex fallback
```

This enables LogicGuard to handle **linguistically messy** questions that break traditional regex:

| Messy Question | Parsed As |
|----------------|-----------|
| *"I wonder, do all those creatures we call dogs fall under the mammal category?"* | `{"type": "taxonomic", "subject": "dog", "predicate": "mammal"}` |
| *"Would every bird necessarily lay eggs?"* | `{"type": "categorical", "entity": "bird", "property": "lay_eggs"}` |
| *"Assuming there is fire, would heat be present?"* | `{"type": "hypothetical", "condition": "fire", "consequence": "heat"}` |

---

## 📄 Citation

If you use LogicGuard in your academic research, please cite:

```bibtex
@article{naseem2026logicguard,
  title     = {LogicGuard: Deterministic Validation of Large Language Model 
               Hallucinations Using Aristotelian-Avicennian Syllogistic Frameworks},
  author    = {Naseem, Hamza},
  journal   = {IEEE (Under Review)},
  year      = {2026}
}
```

---

## 🔭 Future Work

- **Knowledge Graph Integration** — Connect to ConceptNet/Wikidata APIs to replace the manually curated graph with millions of semantic relationships
- **Domain Expansion** — Legal reasoning (statutes as conditionals), medical diagnosis (symptom-disease mappings), mathematical proof verification
- **Fine-tuned Parser** — Replace the general-purpose LLaMA parser with a task-specific model trained purely on logic extraction
- **Real-time API** — FastAPI wrapper for enterprise integration as a drop-in hallucination guardrail

---

## 🤝 Contributing

Contributions are welcome. If you're interested in formal logic, neuro-symbolic AI, or knowledge graphs:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/conceptnet-integration`)
3. Submit a pull request with a clear description

Open an issue first for major changes.

---

<p align="center">
  Built with 🔬 classical logic and modern AI.<br>
  <i>Ibn Sina (980–1037 CE) formalized deductive logic. We made it run on Python.</i>
</p>