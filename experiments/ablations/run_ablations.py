#!/usr/bin/env python3
"""LogicGuard ablation studies: KB components and parser modes."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from logicguard.kb.loader import KnowledgeBase  # noqa: E402
from logicguard.research.adapter import ResearchValidator  # noqa: E402


def make_taxonomy_only_kb(kb_path: Path) -> Path:
    with open(kb_path, encoding="utf-8") as f:
        data = json.load(f)
    slim = {"taxonomies": data.get("taxonomies", {}), "properties": {}, "conditionals": {}}
    out = kb_path.parent / "knowledge_base_taxonomy_only.json"
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return out


def run_ablation(name: str, kb_path: Path, queries: list, parser_mode: str) -> dict:
    validator = ResearchValidator(kb_path, parser_mode=parser_mode)
    correct = 0
    covered = 0
    for q in queries:
        vr = validator.validate(q["question"], q["type"])
        if vr["covered"]:
            covered += 1
        if vr["covered"] and vr["graph_answer"] == q["ground_truth"]:
            correct += 1
    n = len(queries)
    return {
        "name": name,
        "parser": parser_mode,
        "accuracy": round(correct / n * 100, 1) if n else 0,
        "coverage": round(covered / n * 100, 1) if n else 0,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="extended_queries.json")
    parser.add_argument("--kb", default="knowledge_base_1200.json")
    parser.add_argument("--output", default="ablation_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    kb_path = root / args.kb
    with open(root / args.queries, encoding="utf-8") as f:
        queries = [q for q in json.load(f)["queries"] if q.get("source", "original") == "original"]

    tax_only = make_taxonomy_only_kb(kb_path)
    results = [
        run_ablation("full_kb_regex", kb_path, queries, "regex"),
        run_ablation("taxonomy_only_regex", tax_only, queries, "regex"),
    ]

    out = {"ablations": results}
    out_path = root / args.output
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['name']}: acc={r['accuracy']}% cov={r['coverage']}%")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
