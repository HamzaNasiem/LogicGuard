# Figures for IEEE Paper

## Current Figures (in paper)

- Fig1_pipeline_architecture.png  — Two-stage pipeline flowchart
- Fig3_accuracy_comparison.png    — Accuracy before/after AvicennaGuard per model
- Fig4_hallucination_rates.png    — Interception rates per model

## Planned Figures (Phase 2)

- Fig_ablation_kb_components.png  — Ablation study: each graph component
- Fig_kb_size_vs_recall.png       — KB scale vs recall curve
- Fig_latency_breakdown.png       — Stage 1 vs Stage 2 latency per query type
- Fig_baseline_comparison.png     — AvicennaGuard vs Logic-LM vs SelfCheckGPT

## Generation

All figures generated programmatically by:
    experiments/steps/step5_generate_paper_tables.py

To regenerate:
    python experiments/steps/step5_generate_paper_tables.py \
        --results data/results/phase2/all_model_results.json \
        --output  docs/figures/
