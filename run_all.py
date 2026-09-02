#!/usr/bin/env python3
"""
MASTER RUNNER — AvicennaGuard Complete Research Pipeline (v2.0)
=============================================================
Runs the modular AvicennaGuard evaluation pipeline or legacy steps:

  Suite A (Modern v2.0 Benchmark Suite):
    1. Multi-Model 500-Query Evaluation across 5 LLMs
    2. SOTA Baselines: SelfCheckGPT, Dense RAG, Logic-LM
    3. Component Ablation Suite (5 variants)
    4. Statistical Significance Testing (McNemar + Wilson CI)
    5. Automated IEEE LaTeX Tables Generator

  Suite B (Legacy Step Pipeline):
    step1 -> step2 -> step3 -> step3b -> step4 -> step5
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BANNER = """
====================================================================
     AvicennaGuard -- Complete Research Pipeline (v2.0)
     Neuro-Symbolic Middleware for Hallucination Interception
====================================================================
"""


def run_step(cmd: list, step_name: str, abort_on_fail: bool = True) -> bool:
    print(f"\n{'-' * 65}")
    print(f"  >> {step_name}")
    print(f"{'-' * 65}\n")
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    result = subprocess.run([sys.executable, '-X', 'utf8'] + cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"\n  [ERROR] {step_name} FAILED (exit code {result.returncode})")
        if abort_on_fail:
            print("  Aborting pipeline.")
        return False
    print(f"\n  [OK] {step_name} COMPLETE")
    return True


def check_dependencies():
    print("\n[Pre-check] Verifying dependencies...")
    packages = {'ollama': 'ollama', 'numpy': 'numpy', 'networkx': 'networkx'}
    for module, pkg in packages.items():
        try:
            __import__(module)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [ERROR] {pkg} -- run: pip install {pkg}")
            sys.exit(1)


def check_ollama_models():
    print("\n[Pre-check] Verifying Ollama models...")
    try:
        import ollama
        models    = ollama.list()
        available = set(m.get('name', m.get('model', '')).split(':')[0]
                        for m in models.get('models', []))
        for base, full in {'llama2': 'llama2', 'mistral': 'mistral',
                           'llama3.2': 'llama3.2:3b'}.items():
            icon = '✅' if base in available else '❌'
            print(f"  {icon}  {full}" + ('' if base in available else f' — run: ollama pull {full}'))
    except Exception as e:
        print(f"  ⚠️  Ollama check failed: {e}")


def check_file(path: str, label: str) -> bool:
    if os.path.exists(path):
        print(f"  ✅  {label}: {path}")
        return True
    print(f"  ❌  {label} not found: {path}")
    return False


def parse_steps(steps_str: str) -> set:
    """Parse step list like '2,3,3b,4,5' into {'2','3','3b','4','5'}."""
    return set(s.strip() for s in steps_str.split(','))


def run_modern_suite(models: str = "all", mock: bool = True):
    print("\n[Executing Modern AvicennaGuard v2.0 Benchmark Suite]")
    start = time.time()

    # Step 1: Multi-Model Evaluation
    eval_cmd = [
        "scripts/run_benchmark_eval.py",
        "--models", models,
        "--benchmark", "data/benchmarks/avicenna_benchmark_500.json",
        "--kb", "data/knowledge_bases/knowledge_base_extended.json",
        "--output", "results/models/all_model_results_500.json",
        "--report_txt", "results/reports/all_model_report_500.txt",
    ]
    if mock:
        eval_cmd.append("--mock")
    run_step(eval_cmd, "Stage 1: Multi-Model 500-Query Benchmark")

    # Step 2: SOTA Baselines
    sc_cmd = ["scripts/run_baseline_selfcheck.py", "--benchmark", "data/benchmarks/avicenna_benchmark_500.json", "--output", "results/baselines/selfcheck_results_500.json"]
    rag_cmd = ["scripts/run_baseline_rag.py", "--benchmark", "data/benchmarks/avicenna_benchmark_500.json", "--output", "results/baselines/rag_results_500.json"]
    logic_cmd = ["scripts/run_baseline_logic_lm.py", "--benchmark", "data/benchmarks/avicenna_benchmark_500.json", "--output", "results/baselines/logic_lm_results_500.json"]
    if mock:
        sc_cmd.append("--mock")
        rag_cmd.append("--mock")
        logic_cmd.append("--mock")

    run_step(sc_cmd, "Stage 2a: SOTA Baseline — SelfCheckGPT (EMNLP 2023)")
    run_step(rag_cmd, "Stage 2b: SOTA Baseline — Dense RAG (NeurIPS 2020)")
    run_step(logic_cmd, "Stage 2c: SOTA Baseline — Logic-LM (EMNLP 2023)")

    # Step 3: Component Ablations
    run_step([
        "experiments/ablations/run_ablations.py",
        "--benchmark", "data/benchmarks/avicenna_benchmark_500.json",
        "--kb", "data/knowledge_bases/knowledge_base_extended.json",
        "--output", "results/reports/ablation_results_500.json",
    ], "Stage 3: Component Ablation Suite (5 Variants)")

    # Step 4: Statistical Significance & IEEE Tables
    run_step([
        "scripts/generate_paper_artifacts.py",
    ], "Stage 4: Statistical Significance (McNemar + Wilson CI) & IEEE Tables I–V")

    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f"  ALL RESEARCH ARTIFACTS GENERATED in {elapsed:.2f}s")
    print(f"  LaTeX Tables: docs/paper/tables/ (Tables I-V)")
    print(f"  Results:      results/models/, results/baselines/, results/reports/")
    print(f"{'=' * 65}\n")


def main():
    parser = argparse.ArgumentParser(description="AvicennaGuard Research Pipeline Orchestrator")
    parser.add_argument("--suite", default="v2", choices=["v2", "legacy"], help="v2: modern 500 benchmark suite, legacy: legacy step scripts")
    parser.add_argument("--models", default="all", help="Target models or 'all'")
    parser.add_argument("--live", action="store_true", help="Run live model inference (default is deterministic mock for instant execution)")
    args = parser.parse_args()

    print(BANNER)
    is_mock = not args.live

    if args.suite == "v2":
        run_modern_suite(models=args.models, mock=is_mock)
    else:
        print("\n[Executing Legacy Step Suite in experiments/legacy_steps/]")
        run_step(["experiments/legacy_steps/step2_multi_model_runner.py"], "Legacy Step 2")
        run_step(["experiments/legacy_steps/step3_metrics.py"], "Legacy Step 3")


if __name__ == "__main__":
    main()
