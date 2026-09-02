#!/usr/bin/env python3
"""
Simplified Logic-LM style baseline on AvicennaGuard query set.

Uses the same BFS validator as AvicennaGuard (symbolic check) without
LLM override logic — approximates neuro-symbolic solver comparison.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from avicennaguard.research.adapter import ResearchValidator  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="extended_queries.json")
    parser.add_argument("--kb", default="1200")
    parser.add_argument("--output", default="logic_lm_baseline_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    kb_path = root / "knowledge_base_1200.json"
    if args.kb == "extended":
        kb_path = root / "data/knowledge_bases/knowledge_base_extended.json"

    with open(root / args.queries, encoding="utf-8") as f:
        queries = json.load(f)["queries"]
    queries = [q for q in queries if q.get("source", "original") == "original"]

    validator = ResearchValidator(kb_path, parser_mode="regex")
    correct = 0
    results = []

    for q in queries:
        vr = validator.validate(q["question"], q["type"])
        pred = vr["graph_answer"] if vr["covered"] else None
        ok = pred == q["ground_truth"] if pred is not None else False
        if ok:
            correct += 1
        results.append(
            {
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "prediction": pred,
                "covered": vr["covered"],
                "correct": ok,
            }
        )

    acc = correct / len(queries) if queries else 0
    out = {"accuracy": acc, "n": len(queries), "results": results}
    out_path = root / args.output
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Logic-LM baseline accuracy: {acc:.1%} ({correct}/{len(queries)})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
