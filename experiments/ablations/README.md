# Ablation Studies

Required for IEEE journal submission. Each ablation tests one component
in isolation to justify every design choice in the paper.

## Study 1 — KB Component Ablation (ablation_kb_components.py)
Tests each graph independently:
    - G_T only (taxonomy, no properties, no conditionals)
    - G_P only (properties, no taxonomy, no conditionals)
    - G_C only (conditionals only)
    - G_T + G_P (no conditionals)
    - G_T + G_P + G_C (full system — current)

Expected outcome: Full system outperforms any subset. Justifies three-graph design.

## Study 2 — Stage 1 Temperature Ablation (ablation_stage1_temperature.py)
Tests Stage 1 parser at different temperatures:
    - T=0.0 (current — deterministic extraction)
    - T=0.3
    - T=0.7
    - T=1.0

Expected outcome: T=0.0 maximizes parse consistency. Justifies design choice.

## Study 3 — KB Size Ablation (ablation_kb_size.py)
Tests system at different KB sizes:
    - 50 nodes (minimal)
    - 115 nodes (current)
    - 300 nodes (phase 2)
    - 1000+ nodes (ConceptNet extended)

Expected outcome: Recall improves with KB size; Precision stays 100%.
Demonstrates scalability of architecture.

## Study 4 — Parser Fallback Ablation (ablation_parser_fallback.py)
Tests pipeline with and without regex fallback:
    - LLM parser only (no regex fallback)
    - Regex parser only (no LLM)
    - LLM + regex fallback (current)

Expected outcome: Hybrid approach maximizes parse success rate.
