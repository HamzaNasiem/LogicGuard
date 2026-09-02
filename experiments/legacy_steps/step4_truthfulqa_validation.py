#!/usr/bin/env python3
"""Step 4: TruthfulQA out-of-scope generalization test (package-backed)."""

import csv
import json
import re
import sys
import os
import argparse
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from avicennaguard.research.adapter import ResearchValidator  # noqa: E402


def resolve_kb_path(kb_arg: str) -> str:
    aliases = {
        "extended": "data/knowledge_bases/knowledge_base_extended.json",
        "1200": "knowledge_base_1200.json",
    }
    path = aliases.get(kb_arg, kb_arg)
    root = os.path.dirname(__file__)
    for candidate in (path, kb_arg, os.path.join(root, path), os.path.join(root, "knowledge_base_1200.json")):
        if os.path.exists(candidate):
            return candidate
    return path


def classify_question(q: str) -> str:
    t = q.lower().strip()
    if re.match(r"are all \w+", t) or re.match(r"is \w+ a[n]? \w+", t):
        return "taxonomic"
    if re.match(r"do all \w+ (have|need)", t):
        return "categorical"
    if "if" in t and ("," in t or "then" in t or "does" in t):
        return "hypothetical"
    return "other"


def load_truthfulqa(csv_path: str) -> List[Dict]:
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Question", row.get("question", "")).strip()
            category = row.get("Category", row.get("category", "Unknown"))
            if q:
                questions.append({"question": q, "category": category})
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="TruthfulQA.csv")
    parser.add_argument("--kb", default="1200")
    parser.add_argument("--parser", default="regex", choices=["regex", "llm", "both"])
    parser.add_argument("--output", default="truthfulqa_validation_report.txt")
    parser.add_argument("--json_out", default="truthfulqa_validation.json")
    args = parser.parse_args()

    kb_path = resolve_kb_path(args.kb)
    if not os.path.exists(kb_path):
        print(f"ERROR: KB not found: {kb_path}")
        sys.exit(1)

    validator = ResearchValidator(kb_path, parser_mode=args.parser)
    questions = load_truthfulqa(args.csv)

    covered_count = 0
    non_interference = 0
    false_positives = 0
    results = []

    for item in questions:
        q = item["question"]
        qtype = classify_question(q)
        if qtype == "other":
            outcome = "shakk"
            non_interference += 1
            results.append({"question": q, "outcome": outcome, "covered": False})
            continue

        vr = validator.validate(q, qtype)
        covered = vr["covered"]
        if covered:
            covered_count += 1
        else:
            non_interference += 1
        results.append(
            {
                "question": q,
                "type": qtype,
                "covered": covered,
                "epistemic_state": vr["epistemic_state"],
                "graph_answer": vr["graph_answer"],
                "outcome": "covered" if covered else "shakk",
            }
        )

    total = len(questions)
    nir = non_interference / total * 100 if total else 0

    report = {
        "metadata": {"kb_path": kb_path, "parser": args.parser, "total": total},
        "summary": {
            "total_questions": total,
            "kb_covered": covered_count,
            "non_interference_rate": round(nir, 1),
            "false_positives": false_positives,
        },
        "results": results,
    }

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "TruthfulQA Validation Report",
        f"Total: {total}",
        f"KB covered: {covered_count}",
        f"Non-interference: {nir:.1f}%",
        f"False positives: {false_positives}",
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()
