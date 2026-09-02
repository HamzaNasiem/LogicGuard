# Baseline Comparison Experiments

This folder contains implementations and runners for comparison systems.
These are required for IEEE journal submission (Phase 2 of the roadmap).

## Planned Baselines

### baseline_logic_lm.py
Replication of Logic-LM (Pan et al., EMNLP 2023) on the same 175-query dataset.
Logic-LM translates NLP to symbolic representations for Z3 solver.
Reference: https://github.com/teacherpeterpan/Logic-LLM

### baseline_selfcheck.py
SelfCheckGPT (Manakul et al., EMNLP 2023) on same queries.
Measures consistency across multiple LLM samples (stochastic approach).
Key comparison: SelfCheckGPT cannot guarantee FP=0 (probabilistic decision).

### baseline_raw_llm.py
Already implemented in step2_multi_model_runner.py as the baseline condition.
This is the "no intervention" condition for all three models.

## Comparison Table Target (for IEEE paper Section VI)

| System          | Approach          | FP=0 Guarantee | Epistemic Grading | Latency |
|-----------------|-------------------|----------------|-------------------|---------|
| Raw LLM         | Probabilistic     | No             | No                | Baseline|
| SelfCheckGPT    | Sampling variance | No             | No                | 5-10x   |
| Logic-LM        | Symbolic solver   | Partial        | No                | TBD     |
| **AvicennaGuard**  | **BFS + KB**      | **Yes**        | **Yes (4-state)** | **TBD** |
