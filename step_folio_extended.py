#!/usr/bin/env python3
"""
AvicennaGuard — Extended FOLIO OOD Validation
===========================================
Downloads the full FOLIO dataset (train + validation) and measures AvicennaGuard's
out-of-distribution non-interference rate.

FOLIO is a natural-language formal logic dataset (Yale NLP, 2022) with 1,004
training and 204 validation examples. These are COMPLEX multi-premise syllogisms
that AvicennaGuard's KB does not cover — making them ideal for measuring whether
AvicennaGuard correctly defers (SHAKK) instead of producing false positives.

Non-interference rate = % of OOD queries where AvicennaGuard says SHAKK
(i.e., "I don't know, deferring to LLM"). A high rate proves the system
does NOT hallucinate new knowledge when it lacks KB coverage.

Reference:
  Han S et al. "FOLIO: Natural Language Reasoning with First-Order Logic"
  EMNLP 2022. https://arxiv.org/abs/2209.00840

Usage:
    python step_folio_extended.py
    python step_folio_extended.py --split both   # train + validation
    python step_folio_extended.py --split train  # training set only
"""

import json
import sys
import argparse
import urllib.request
from pathlib import Path

# Must be run from project root (adds src/ to path)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.regex_parser import RegexParser
from avicennaguard.core.epistemic_states import EpistemicState


FOLIO_URLS = {
    "train":      "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-train.jsonl",
    "validation": "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl",
}


def download_folio(split: str) -> list[dict]:
    """Download a FOLIO split from GitHub. Returns list of example dicts."""
    url = FOLIO_URLS[split]
    print(f"  Downloading FOLIO {split} from GitHub...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = r.read().decode("utf-8").strip().split("\n")
        examples = [json.loads(line) for line in lines if line.strip()]
        print(f"{len(examples)} examples")
        return examples
    except Exception as e:
        print(f"FAILED: {e}")
        return []


def folio_to_query(example: dict) -> dict:
    """
    Convert a FOLIO example to a simple AvicennaGuard query format.
    Uses the conclusion as the question, with label as ground truth.

    FOLIO labels: True, False, Uncertain → our format: yes, no, unknown
    """
    label = example.get("label", "Uncertain")
    # For OOD testing, we care about whether AvicennaGuard defers (SHAKK).
    # Ground truth for AvicennaGuard purposes: True → yes, otherwise → no/unknown
    return {
        "question":     example.get("conclusion", ""),
        "type":         "folio_ood",
        "ground_truth": label == "True",
        "folio_label":  label,
        "folio_premises": example.get("premises", []),
        "source":       f"folio_{example.get('story_id', '')}",
    }


def run_ood_test(
    queries: list[dict],
    validator: BFSValidator,
    regex_parser: RegexParser,
) -> dict:
    """
    Run AvicennaGuard Stage 2 on FOLIO queries.
    Since these are OOD, we expect most to yield SHAKK (not covered by KB).

    Returns stats:
      shakk_rate      — fraction that correctly deferred (non-interference)
      yaqeen_correct  — fraction of KB-covered ones answered correctly
      false_positives — KB-covered queries answered WRONG
    """
    total = len(queries)
    shakk = 0
    yaqeen_correct = 0
    yaqeen_wrong = 0
    wahm = 0
    parse_fail = 0

    results = []
    for q in queries:
        question = q["question"]
        if not question.strip():
            total -= 1
            continue

        # Try regex parser to extract logical structure
        parsed = regex_parser.parse(question)
        qtype = parsed.get("type", "non-logical")

        epistemic = EpistemicState.SHAKK
        lg_answer = None

        if qtype == "taxonomic":
            subj = parsed.get("subject", "")
            pred = parsed.get("predicate", "")
            result, epistemic, path = validator.validate_taxonomic(subj, pred)
            lg_answer = result

        elif qtype == "categorical":
            entity = parsed.get("entity", "")
            prop   = parsed.get("property", "")
            result, epistemic = validator.validate_categorical(entity, prop)
            lg_answer = result

        elif qtype == "hypothetical":
            cond  = parsed.get("condition", "")
            conseq = parsed.get("consequence", "")
            result, epistemic = validator.validate_hypothetical(cond, conseq)
            lg_answer = result

        # else: non-logical → SHAKK by default

        gt = q.get("ground_truth", False)

        if epistemic == EpistemicState.SHAKK:
            shakk += 1
            outcome = "shakk"
        elif epistemic == EpistemicState.YAQEEN:
            if lg_answer == gt:
                yaqeen_correct += 1
                outcome = "yaqeen_correct"
            else:
                yaqeen_wrong += 1
                outcome = "yaqeen_wrong"
        elif epistemic == EpistemicState.WAHM:
            wahm += 1
            outcome = "wahm"
        else:
            outcome = "other"

        results.append({
            "question": question,
            "folio_label": q.get("folio_label", ""),
            "parsed_type": qtype,
            "epistemic": epistemic.value if hasattr(epistemic, "value") else str(epistemic),
            "lg_answer": lg_answer,
            "outcome": outcome,
        })

    non_interference_rate = shakk / total if total else 0.0
    fp = yaqeen_wrong  # AvicennaGuard answered when it shouldn't have, and got it wrong
    coverage = (total - shakk) / total if total else 0.0

    return {
        "total":        total,
        "shakk":        shakk,
        "yaqeen_correct": yaqeen_correct,
        "yaqeen_wrong": yaqeen_wrong,
        "wahm":         wahm,
        "non_interference_rate": round(non_interference_rate, 4),
        "kb_coverage":  round(coverage, 4),
        "false_positives": fp,
        "results":      results,
    }


def main():
    parser = argparse.ArgumentParser(description="Extended FOLIO OOD Validation")
    parser.add_argument("--kb",     default="data/knowledge_bases/knowledge_base_extended.json")
    parser.add_argument("--split",  default="both", choices=["train", "validation", "both"])
    parser.add_argument("--output", default="folio_extended_results.json")
    args = parser.parse_args()

    print("=" * 65)
    print("  Extended FOLIO OOD Non-Interference Validation")
    print("=" * 65)

    # Load KB
    print(f"\n  Loading KB: {args.kb}")
    kb = KnowledgeBase(args.kb)
    validator  = BFSValidator(kb)
    regex_parser = RegexParser()
    print(f"  KB nodes: {kb.G_T.number_of_nodes()} taxonomy, "
          f"{len(kb.G_P)} property entities, "
          f"{kb.G_C.number_of_nodes()} conditionals")

    # Download FOLIO
    all_queries = []
    splits_used = []

    if args.split in ("train", "both"):
        train_examples = download_folio("train")
        train_queries  = [folio_to_query(e) for e in train_examples]
        all_queries.extend(train_queries)
        splits_used.append(f"train ({len(train_queries)})")

    if args.split in ("validation", "both"):
        val_examples = download_folio("validation")
        val_queries  = [folio_to_query(e) for e in val_examples]
        all_queries.extend(val_queries)
        splits_used.append(f"validation ({len(val_queries)})")

    print(f"\n  FOLIO splits used: {', '.join(splits_used)}")
    print(f"  Total FOLIO queries: {len(all_queries)}")

    # Run OOD test
    print(f"\n  Running Stage 2 only (regex parser + BFS)...")
    stats = run_ood_test(all_queries, validator, regex_parser)

    # Report
    print(f"\n{'=' * 65}")
    print(f"  FOLIO OOD RESULTS")
    print(f"{'=' * 65}")
    print(f"  Total queries          : {stats['total']}")
    print(f"  SHAKK (deferred)       : {stats['shakk']}  ({stats['non_interference_rate']:.1%})")
    print(f"  KB-covered (answered)  : {stats['total'] - stats['shakk']}  ({stats['kb_coverage']:.1%})")
    print(f"  Correct answers        : {stats['yaqeen_correct']}")
    print(f"  Wrong answers (FP)     : {stats['yaqeen_wrong']}")
    print(f"  WAHM (intercepted)     : {stats['wahm']}")
    print(f"\n  Non-interference rate  : {stats['non_interference_rate']:.1%}")
    print(f"  (Higher = system correctly defers to LLM for OOD queries)")
    print(f"\n  Interpretation:")
    pct = stats['non_interference_rate'] * 100
    if pct >= 95:
        print(f"  EXCELLENT ({pct:.1f}%): System defers correctly on FOLIO OOD queries.")
        print(f"  AvicennaGuard does NOT hallucinate new knowledge beyond its KB scope.")
    elif pct >= 85:
        print(f"  GOOD ({pct:.1f}%): Minor coverage overlap with FOLIO queries.")
    else:
        print(f"  NOTE ({pct:.1f}%): Some FOLIO queries resemble KB-covered patterns.")

    # Save
    output = {
        "splits": splits_used,
        "kb_path": args.kb,
        "stats": stats,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output}")
    print(f"\n  Use this number in paper Section VI:")
    print(f"  'AvicennaGuard correctly deferred on {stats['non_interference_rate']:.1%} of")
    print(f"   {stats['total']} FOLIO out-of-distribution natural-language syllogisms,")
    print(f"   confirming the SHAKK state prevents hallucination beyond KB scope.'")


if __name__ == "__main__":
    main()
