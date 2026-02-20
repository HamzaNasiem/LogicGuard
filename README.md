# 🛡️ LogicGuard: Deterministic Interceptor for LLM Hallucinations

> *"Probabilistic AI guesses. LogicGuard formally proves."*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper: IEEE](https://img.shields.io/badge/Paper-IEEE_Under_Review-red.svg)](#)

## 📖 Overview
Large Language Models (LLMs) are phenomenal semantic engines, but they are fundamentally flawed logical reasoners. They operate on token probabilities, frequently leading to structural hallucinations—confident but logically impossible statements. 

**LogicGuard** is a neuro-symbolic middleware framework that solves this. By computationally modeling the 1,000-year-old Aristotelian-Avicennian syllogistic logic (Mantiq), LogicGuard intercepts LLM outputs, validating them against a deterministic Knowledge Base before they reach the user. 

If an LLM violates strict deductive logic, LogicGuard overrides it, converting the "Black Box" into a verifiable, mathematically sound pipeline.

## 🚀 Key Results (Massive 900+ Question Evaluation)
We tested a local `Llama-3.2 (3B)` model on a hybrid dataset combining the **TruthfulQA** benchmark (800+ factual queries) with 100 strict logical syllogisms. 

| Evaluation Metric | LLM Baseline (Llama 3.2) | LogicGuard (Our System) |
| :--- | :---: | :---: |
| **Logic Q Accuracy** | ~60.0% | **100.0%** (Yaqeen) |
| **Non-Logic Q Accuracy** | 31.3% | 31.3% (Zann/Shakk) |
| **Hallucinations Caught**| 0% | **100% of logical errors intercepted** |

*(See the `ieee_results_chart.png` in this repository for the visual plot).*

## 🧠 Epistemic State Classification
Instead of a simple Pass/Fail, LogicGuard grades AI output using classical Avicennian epistemic states:
- 🟢 **Yaqeen (Certainty - 100%):** Formally proven true via strict set-theory and deductive rules.
- 🟡 **Zann (Probability - >50%):** Logic doesn't apply, but semantic factual matching is high.
- 🟠 **Shakk (Doubt - <50%):** Ambiguous factual statement.
- 🔴 **Wahm (Illusion - 0%):** The LLM contradicted a proven logical rule. LogicGuard flags this as a hallucination.

## ⚙️ Architecture Workflow
1. **User Prompt:** "Are all squares rectangles?"
2. **LLM Generation:** LLM probabilistically generates an answer (e.g., "No, a square is a specific shape...")
3. **LogicGuard Parsing:** Extracts entities (`square`, `rectangle`) and the logical operator.
4. **Graph Traversal:** Searches the JSON Knowledge Base for transitive/subset relationships.
5. **Interceptor Action:** LogicGuard detects the LLM hallucination, overrides the output, and returns **WAHM (Illusion)**. 

## 🛠️ Installation & Usage

### 1. Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com/) installed locally with the `llama3.2:3b` model pulled (`ollama run llama3.2:3b`).

### 2. Clone the Repository
```bash
git clone [https://github.com/HamzaNasiem/LogicGuard.git](https://github.com/HamzaNasiem/LogicGuard.git)
cd LogicGuard
pip install pandas numpy matplotlib
```

### 3. Run the Massive Evaluation
To reproduce the exact results submitted for the IEEE paper (processing the 900+ hybrid dataset):

```bash
python run_massive_updated_900.py
```
Note: This will take 15-30 minutes depending on your local hardware. Upon completion, it will generate a massive_experiment_results.csv and an auto-generated Matplotlib char


## 📄 Citation

If you use this framework or dataset in your academic research, please cite our upcoming IEEE paper:

```bibtex
@article{naseem2026logicguard,
  title={LogicGuard: Deterministic Validation of Large Language Model Hallucinations Using Aristotelian-Avicennian Syllogistic Frameworks},
  author={Naseem, Hamza},
  journal={IEEE (Under Review)},
  year={2026}
}
```

## 🤝 Future Work & Contributions

Currently, **LogicGuard** acts as a successful **Proof of Concept** utilizing exact template matching (The Semantic Parsing Bottleneck). 

### 🚀 Phase 2 Focus
*   **NLP Translation Layer:** Building a dedicated layer to condense messy, natural language into formal symbolic structures.
*   **Knowledge Graphs:** Integrating massive datasets like **ConceptNet** or **Wikidata**.

> [!TIP]
> **Contributions are highly encouraged!** If you're interested in formal logic or NLP, feel free to open an **issue** or submit a **pull request**.
