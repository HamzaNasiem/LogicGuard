# AvicennaGuard Scripts & Utilities

This directory contains standalone execution scripts, benchmark evaluation runners, training workflows, stress-testing suites, and publication artifact generators for the AvicennaGuard research framework.

## Overview of Scripts

| Script | Purpose |
| :--- | :--- |
| [`run_benchmark_eval.py`](file:///d:/research/LogicGuard/scripts/run_benchmark_eval.py) | Executes the 500-query multi-model benchmark comparison across 5 LLMs. |
| [`run_baseline_rag.py`](file:///d:/research/LogicGuard/scripts/run_baseline_rag.py) | Standalone evaluation runner for Dense RAG baseline (Lewis et al., NeurIPS 2020). |
| [`run_baseline_selfcheck.py`](file:///d:/research/LogicGuard/scripts/run_baseline_selfcheck.py) | Standalone evaluation runner for SelfCheckGPT consistency baseline (Manakul et al., EMNLP 2023). |
| [`run_baseline_logic_lm.py`](file:///d:/research/LogicGuard/scripts/run_baseline_logic_lm.py) | Standalone runner for Logic-LM neuro-symbolic solver baseline (Pan et al., EMNLP 2023). |
| [`reproduce_and_audit_all_baselines.py`](file:///d:/research/LogicGuard/scripts/reproduce_and_audit_all_baselines.py) | Comprehensive reproduction, empirical cross-validation, and baseline audit script. |
| [`audit_stage1_adversarial.py`](file:///d:/research/LogicGuard/scripts/audit_stage1_adversarial.py) | Adversarial stress-test and latency benchmark for the Stage 1 Semantic Parser. |
| [`run_kb_stress_test.py`](file:///d:/research/LogicGuard/scripts/run_kb_stress_test.py) | Formal DAG verification, cycle-detection, and stress audit on the multi-relational KB. |
| [`prepare_deberta_data.py`](file:///d:/research/LogicGuard/scripts/prepare_deberta_data.py) | Synthetic dataset generator for Stage 1 parser training across 4 query classes. |
| [`train_stage1_classifier.py`](file:///d:/research/LogicGuard/scripts/train_stage1_classifier.py) | Trains and validates the fast Stage 1 intent classifier (`models/stage1_classifier.joblib`). |
| [`expand_kb_wordnet.py`](file:///d:/research/LogicGuard/scripts/expand_kb_wordnet.py) | Expands the taxonomic ontology using WordNet synsets. |
| [`generate_paper_artifacts.py`](file:///d:/research/LogicGuard/scripts/generate_paper_artifacts.py) | Generates publication LaTeX tables, McNemar significance tests, and markdown summaries. |
| [`generate_paper_figures.py`](file:///d:/research/LogicGuard/scripts/generate_paper_figures.py) | Generates publication-grade 300 DPI figures for the IEEE paper into `docs/figures/`. |

## Quick Usage

```bash
# Run multi-model benchmark evaluation
python scripts/run_benchmark_eval.py --model llama3.2:3b --limit 50

# Run baseline audit and reproduction
python scripts/reproduce_and_audit_all_baselines.py

# Run KB formal verification and stress test
python scripts/run_kb_stress_test.py

# Generate all publication LaTeX tables and figures
python scripts/generate_paper_artifacts.py
python scripts/generate_paper_figures.py
```
