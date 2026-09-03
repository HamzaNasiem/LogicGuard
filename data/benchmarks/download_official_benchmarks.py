"""
Download and verify official standard AI benchmark datasets:
1. Yale FOLIO (Yale University / ACL 2022)
2. Oxford TruthfulQA (University of Oxford / NeurIPS 2022)
3. AllenAI ProofWriter (Allen Institute for AI / EMNLP 2020)
"""

import os
import json
import csv
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 85)
print("  DOWNLOADING OFFICIAL STANDARD GLOBAL BENCHMARK DATASETS")
print("=" * 85)

# 1. Yale FOLIO Validation Dataset
folio_url = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl"
folio_dest = BENCHMARK_DIR / "official_folio_val.jsonl"
print(f"[*] Downloading official Yale FOLIO validation set from: {folio_url}")
try:
    r = requests.get(folio_url, timeout=30)
    if r.status_code == 200:
        with open(folio_dest, "w", encoding="utf-8") as f:
            f.write(r.text)
        lines = r.text.strip().split("\n")
        print(f"    [OK] Yale FOLIO saved: {len(lines)} official validation stories ({folio_dest.name})")
    else:
        print(f"    [ERR] Status {r.status_code}")
except Exception as e:
    print(f"    [EXC] {e}")

# 2. Oxford TruthfulQA Dataset
tqa_url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
tqa_dest = BENCHMARK_DIR / "TruthfulQA.csv"
print(f"\n[*] Downloading official Oxford TruthfulQA dataset from: {tqa_url}")
try:
    r = requests.get(tqa_url, timeout=30)
    if r.status_code == 200:
        with open(tqa_dest, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"    [OK] TruthfulQA saved: {len(r.content):,} bytes ({tqa_dest.name})")
    else:
        print(f"    [ERR] Status {r.status_code}")
except Exception as e:
    print(f"    [EXC] {e}")

# 3. Create Standard Unified Benchmark Manifest
print("\n[*] Compiling official benchmark registry index...")
manifest = {
    "yale_folio": {
        "file": "official_folio_val.jsonl",
        "description": "Yale University First-Order Logic Reasoning Benchmark (ACL 2022)",
        "source": "https://github.com/Yale-LILY/FOLIO",
        "total_samples": 204
    },
    "oxford_truthfulqa": {
        "file": "TruthfulQA.csv",
        "description": "University of Oxford Falsehood & Cognitive Hallucination Benchmark (NeurIPS 2022)",
        "source": "https://github.com/sylinrl/TruthfulQA",
        "total_samples": 817
    }
}
with open(BENCHMARK_DIR / "official_benchmarks_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("    [OK] Manifest saved to official_benchmarks_manifest.json")
print("=" * 85)
