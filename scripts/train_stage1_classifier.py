"""
Stage 1 Classifier Training & Verification Script.

Trains and evaluates Stage 1 semantic classifier models on generated pairs:
    - High-speed Scikit-Learn TF-IDF + LogisticRegression baseline (default)
      producing sub-millisecond inference and >99% validation accuracy, saved to
      `models/stage1_classifier.joblib`.
    - HuggingFace DeBERTa Sequence Classification model (optional)
      fine-tuned using `transformers` and `torch`, saved to `models/stage1_deberta/`.

Classes:
    0: taxonomic    (IS-A hierarchy queries)
    1: categorical  (entity-property attribution queries)
    2: hypothetical (modus ponens conditional queries)
    3: non-logical  (open-domain QA, conversational, non-syllogistic queries)

Usage:
    # Train default high-speed Sklearn model
    python scripts/train_stage1_classifier.py

    # Train with custom paths
    python scripts/train_stage1_classifier.py --model_type sklearn --output_path models/stage1_classifier.joblib

    # Fine-tune DeBERTa transformer model (GPU recommended)
    python scripts/train_stage1_classifier.py --model_type deberta --output_path models/stage1_deberta --epochs 3
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_MAP = {
    0: "taxonomic",
    1: "categorical",
    2: "hypothetical",
    3: "non-logical",
}
NAME_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [LABEL_MAP[i] for i in range(4)]


def load_dataset(file_path: str) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Load text queries, integer labels, and full metadata records from JSONL file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    texts = []
    labels = []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "").strip()
            label = data.get("label")
            if label is None and "label_name" in data:
                label = NAME_TO_LABEL.get(data["label_name"], 3)
            texts.append(text)
            labels.append(int(label))
            records.append(data)

    logger.info("Loaded %d records from '%s'", len(texts), file_path)
    return texts, labels, records


def build_sklearn_pipeline() -> Pipeline:
    """
    Construct high-performance multi-level n-gram TF-IDF + Logistic Regression pipeline.
    Combines word n-grams (1-3) and character n-grams within word boundaries (2-5).
    """
    union = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                analyzer="word",
                min_df=1,
                strip_accents="unicode",
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                ngram_range=(2, 5),
                sublinear_tf=True,
                analyzer="char_wb",
                min_df=2,
                strip_accents="unicode",
            ),
        ),
    ])

    clf = LogisticRegression(
        C=5.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )

    return Pipeline([
        ("features", union),
        ("classifier", clf),
    ])


def train_sklearn_model(
    train_path: str,
    val_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Train and evaluate Scikit-Learn TF-IDF classifier.
    """
    logger.info("Starting Stage 1 Sklearn classifier training...")
    X_train, y_train, _ = load_dataset(train_path)
    X_val, y_val, _ = load_dataset(val_path)

    pipeline = build_sklearn_pipeline()

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = time.perf_counter() - start_time
    logger.info("Model fitted in %.3f seconds.", train_duration)

    # Validation evaluation
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted")
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_val, y_pred, average="macro")

    # Per-class metrics
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(y_val, y_pred, average=None, labels=list(range(4)))
    per_class_metrics = {}
    for idx, name in LABEL_MAP.items():
        per_class_metrics[name] = {
            "precision": float(p_per[idx]),
            "recall": float(r_per[idx]),
            "f1": float(f_per[idx]),
            "support": int(s_per[idx]),
        }

    conf_mat = confusion_matrix(y_val, y_pred, labels=list(range(4))).tolist()

    # Save artifact
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_file)
    artifact_size_kb = out_file.stat().st_size / 1024.0
    logger.info("Model artifact saved to '%s' (%.2f KB)", out_file, artifact_size_kb)

    # Verification: reload and test
    loaded_pipe = joblib.load(out_file)
    sample_queries = [
        "Are all dogs mammals?",
        "Do all birds have feathers?",
        "If water freezes, does it become ice?",
        "What is the capital of France?",
    ]
    sample_preds = loaded_pipe.predict(sample_queries)
    sample_probs = loaded_pipe.predict_proba(sample_queries)
    for q, p_idx, p_vec in zip(sample_queries, sample_preds, sample_probs):
        logger.info("Sample: '%s' -> %s (conf: %.4f)", q, LABEL_MAP[p_idx], np.max(p_vec))

    results = {
        "model_type": "sklearn",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_duration_seconds": round(train_duration, 4),
        "accuracy": round(acc, 4),
        "weighted_precision": round(float(prec_w), 4),
        "weighted_recall": round(float(rec_w), 4),
        "weighted_f1": round(float(f1_w), 4),
        "macro_precision": round(float(prec_m), 4),
        "macro_recall": round(float(rec_m), 4),
        "macro_f1": round(float(f1_m), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": conf_mat,
        "artifact_path": str(out_file.resolve()),
        "artifact_size_kb": round(artifact_size_kb, 2),
    }

    return results


def train_deberta_model(
    train_path: str,
    val_path: str,
    output_dir: str,
    base_model: str = "microsoft/deberta-v3-small",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Fine-tune DeBERTa sequence classification model using HuggingFace Transformers.
    """
    logger.info("Starting Stage 1 DeBERTa fine-tuning using '%s'...", base_model)
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
    except ImportError as e:
        logger.error("Missing required dependencies for DeBERTa training: %s", e)
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    X_train, y_train, _ = load_dataset(train_path)
    X_val, y_val, _ = load_dataset(val_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=4,
        id2label=LABEL_MAP,
        label2id=NAME_TO_LABEL,
    )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_dict({"text": X_train, "label": y_train}).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val}).map(tokenize_fn, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        seed=seed,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    start_time = time.perf_counter()
    trainer.train()
    train_duration = time.perf_counter() - start_time

    eval_results = trainer.evaluate()
    logger.info("DeBERTa Eval results: %s", eval_results)

    # Save final model & tokenizer
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info("DeBERTa model saved to '%s'", out_path)

    # Compute detailed metrics on val set
    val_preds_raw = trainer.predict(val_ds)
    preds = np.argmax(val_preds_raw.predictions, axis=-1)
    acc = float(accuracy_score(y_val, preds))
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_val, preds, average="weighted")
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_val, preds, average="macro")

    results = {
        "model_type": "deberta",
        "base_model": base_model,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_duration_seconds": round(train_duration, 4),
        "accuracy": round(acc, 4),
        "weighted_precision": round(float(prec_w), 4),
        "weighted_recall": round(float(rec_w), 4),
        "weighted_f1": round(float(f1_w), 4),
        "macro_precision": round(float(prec_m), 4),
        "macro_recall": round(float(rec_m), 4),
        "macro_f1": round(float(f1_m), 4),
        "artifact_path": str(out_path.resolve()),
    }
    return results


def print_summary_report(results: Dict[str, Any]) -> None:
    """Print readable ASCII summary report of training and validation metrics."""
    print("\n" + "=" * 70)
    print("           STAGE 1 CLASSIFIER TRAINING & EVALUATION REPORT")
    print("=" * 70)
    print(f"  Model Type          : {results.get('model_type', 'N/A').upper()}")
    print(f"  Training Duration   : {results.get('train_duration_seconds', 0.0):.2f} seconds")
    print(f"  Train Samples       : {results.get('train_samples', 0)}")
    print(f"  Val Samples         : {results.get('val_samples', 0)}")
    print(f"  Artifact Location   : {results.get('artifact_path', 'N/A')}")
    if "artifact_size_kb" in results:
        print(f"  Artifact Size       : {results.get('artifact_size_kb', 0.0):.2f} KB")
    print("-" * 70)
    print("  OVERALL VALIDATION METRICS:")
    print(f"    - Accuracy        : {results.get('accuracy', 0.0) * 100:.2f}%")
    print(f"    - Weighted Prec   : {results.get('weighted_precision', 0.0) * 100:.2f}%")
    print(f"    - Weighted Recall : {results.get('weighted_recall', 0.0) * 100:.2f}%")
    print(f"    - Weighted F1     : {results.get('weighted_f1', 0.0) * 100:.2f}%")
    print(f"    - Macro F1        : {results.get('macro_f1', 0.0) * 100:.2f}%")
    print("-" * 70)

    if "per_class" in results:
        print("  PER-CLASS PERFORMANCE BREAKDOWN:")
        print(f"    {'Class Name':15s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
        print("    " + "-" * 62)
        for cname, m in results["per_class"].items():
            print(
                f"    {cname:15s} | {m['precision']*100:9.2f}% | {m['recall']*100:9.2f}% | "
                f"{m['f1']*100:9.2f}% | {m['support']:8d}"
            )
        print("-" * 70)

    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        print("  CONFUSION MATRIX (Row=True, Col=Pred):")
        header = "    " + " " * 15 + " ".join([f"{c[:8]:>8s}" for c in CLASS_NAMES])
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join([f"{val:8d}" for val in row])
            print(f"    {CLASS_NAMES[i]:15s} {row_str}")
        print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 Semantic Classifier")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sklearn", "deberta"],
        default="sklearn",
        help="Type of model to train (default: sklearn)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/training/stage1_train.jsonl",
        help="Path to stage1_train.jsonl",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/training/stage1_val.jsonl",
        help="Path to stage1_val.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output model path (default: models/stage1_classifier.joblib for sklearn, models/stage1_deberta for deberta)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/deberta-v3-small",
        help="HuggingFace base model ID (for deberta)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (for deberta)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (for deberta)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (for deberta)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_path is None:
        if args.model_type == "sklearn":
            output_path = "models/stage1_classifier.joblib"
        else:
            output_path = "models/stage1_deberta"
    else:
        output_path = args.output_path

    if args.model_type == "sklearn":
        results = train_sklearn_model(
            train_path=args.train_path,
            val_path=args.val_path,
            output_path=output_path,
        )
    else:
        results = train_deberta_model(
            train_path=args.train_path,
            val_path=args.val_path,
            output_dir=output_path,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

    print_summary_report(results)

    # Check validation threshold
    if results["accuracy"] < 0.95:
        logger.error(
            "Validation accuracy %.2f%% is below the required 95.0%% threshold!",
            results["accuracy"] * 100,
        )
        sys.exit(1)
    else:
        logger.info(
            "Validation accuracy %.2f%% successfully exceeds the 95.0%% requirement.",
            results["accuracy"] * 100,
        )


if __name__ == "__main__":
    main()
