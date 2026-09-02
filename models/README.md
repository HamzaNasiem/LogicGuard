# AvicennaGuard Models Directory

This directory stores serialized model weights, classifier checkpoints, and tokenizer assets for the AvicennaGuard neuro-symbolic framework.

## Model Inventory

- **`stage1_classifier.joblib`**: Serialized Scikit-Learn TF-IDF + Logistic Regression pipeline for Stage 1 query intent classification (`taxonomic`, `categorical`, `hypothetical`, `non_logical`). Achieves sub-millisecond classification latency (0.012 ms/query).

## Training & Updating

To re-train or update the Stage 1 classifier:
```bash
python scripts/prepare_deberta_data.py
python scripts/train_stage1_classifier.py
```
