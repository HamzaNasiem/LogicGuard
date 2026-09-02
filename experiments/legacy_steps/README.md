# Legacy Step Pipeline (Phase 1 Archive)

This directory preserves the historical standalone step scripts (`step1` through `step5` and associated variants) developed during Phase 1 of the AvicennaGuard / LogicGuard research project.

## Archived Steps

- `step1_proofwriter_extractor.py`: ProofWriter and deductive logic extractor.
- `step2_multi_model_runner.py`: Initial multi-model benchmark evaluation runner.
- `step2b_openrouter_eval.py`: OpenRouter cloud API evaluation script.
- `step3_metrics.py`: Metrics calculation script.
- `step3b_statistical.py`: Statistical significance and confidence interval calculations.
- `step4_truthfulqa_validation.py`: TruthfulQA out-of-domain safe deferral validation.
- `step5_generate_paper_tables.py`: Legacy LaTeX table generator.
- `step_case_study.py`: Qualitative case study generator.
- `step_conceptnet_expansion.py`: ConceptNet ontology expansion script.
- `step_folio_extended.py`: FOLIO reasoning evaluation script.
- `step_generate_comparison_table.py`: SOTA comparison table generator.
- `step_generate_dataset.py`: Legacy synthetic dataset builder.
- `step_rag_baseline.py` & `step_rag_dense_baseline.py`: Initial RAG baseline implementations.
- `step_selfcheckgpt_baseline.py`: Initial SelfCheckGPT baseline implementation.

## Note on Architecture Evolution

For all current research, production use, and extended experimentation, use the modular package architecture:
- Core library: [`src/avicennaguard/`](file:///d:/research/LogicGuard/src/avicennaguard)
- Unit & integration tests: [`tests/`](file:///d:/research/LogicGuard/tests)
- Modern CLI runners: [`scripts/`](file:///d:/research/LogicGuard/scripts) and [`run_all.py`](file:///d:/research/LogicGuard/run_all.py)
