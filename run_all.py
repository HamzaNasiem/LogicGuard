#!/usr/bin/env python3
"""
MASTER RUNNER — LogicGuard Complete Pipeline
=============================================
Runs all 5 steps in sequence:
  Step 1: ProofWriter extraction + KB extension
  Step 2: Multi-model evaluation (Llama2, Mistral, Llama3.2 + LogicGuard)
  Step 3: Metrics, confusion matrices, P/R/F1
  Step 4: TruthfulQA out-of-scope generalization test
  Step 5: IEEE paper tables + copy-paste text generator

Usage:
    python run_all.py --proofwriter_dir proofwriter-dataset-V2020.12.3

Optional — skip steps already done:
    python run_all.py --start_from 4    # skip to step4 (TruthfulQA)
    python run_all.py --start_from 5    # skip to step5 (paper tables)
    python run_all.py --steps 4,5       # run only steps 4 and 5
"""

import os
import sys
import argparse
import subprocess
import json
import time

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          LogicGuard — Complete Pipeline (Steps 1–5)             ║
║          Baseline → Evaluation → Metrics → TruthfulQA → Paper   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_step(cmd: list, step_name: str, abort_on_fail: bool = True) -> bool:
    """Run a step, return True if successful."""
    print(f"\n{'━' * 65}")
    print(f"  ▶  {step_name}")
    print(f"{'━' * 65}\n")
    result = subprocess.run([sys.executable] + cmd, check=False)
    if result.returncode != 0:
        print(f"\n  ❌  {step_name} FAILED (exit code {result.returncode})")
        if abort_on_fail:
            print("  Aborting pipeline.")
        return False
    print(f"\n  ✅  {step_name} COMPLETE")
    return True


def check_dependencies():
    """Check required packages are installed."""
    print("\n[Pre-check] Verifying dependencies...")
    missing = []
    packages = {
        'ollama':   'ollama',
        'numpy':    'numpy',
    }
    # scikit-learn optional (step3)
    try:
        from sklearn.metrics import f1_score  # noqa
        print(f"  ✅  scikit-learn")
    except ImportError:
        print(f"  ⚠️  scikit-learn not found (step3 may warn) — pip install scikit-learn")

    for module, pkg in packages.items():
        try:
            __import__(module)
            print(f"  ✅  {pkg}")
        except ImportError:
            print(f"  ❌  {pkg} NOT installed — run: pip install {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n  Install missing packages: pip install {' '.join(missing)}")
        sys.exit(1)


def check_ollama_models():
    """Verify Ollama models are available."""
    print("\n[Pre-check] Verifying Ollama models...")
    try:
        import ollama
        models = ollama.list()
        available = set()
        for m in models.get('models', []):
            n = m.get('name', m.get('model', ''))
            available.add(n.split(':')[0])

        required = {'llama2': 'llama2', 'mistral': 'mistral', 'llama3.2': 'llama3.2:3b'}
        all_ok = True
        for base, full in required.items():
            if base in available:
                print(f"  ✅  {full}")
            else:
                print(f"  ❌  {full} — run: ollama pull {full}")
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"  ⚠️  Ollama check failed: {e}")
        return False


def check_file(path: str, label: str) -> bool:
    """Check if a required file exists."""
    if os.path.exists(path):
        print(f"  ✅  {label}: {path}")
        return True
    else:
        print(f"  ❌  {label} not found: {path}")
        return False


def main():
    parser = argparse.ArgumentParser(description='LogicGuard Complete Pipeline')
    parser.add_argument('--proofwriter_dir', type=str,
                        default='proofwriter-dataset-V2020.12.3',
                        help='Path to ProofWriter dataset directory')
    parser.add_argument('--queries', type=str,
                        default='extended_queries.json',
                        help='Path to extended queries JSON')
    parser.add_argument('--kb', type=str,
                        default='knowledge_base_extended.json',
                        help='Path to KB JSON')
    parser.add_argument('--truthfulqa', type=str,
                        default='truthfulqa.csv',
                        help='Path to TruthfulQA CSV file')
    parser.add_argument('--start_from', type=int, default=1,
                        help='Start from step N (1-5)')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated list of steps to run (e.g. "4,5")')
    parser.add_argument('--skip_model_check', action='store_true',
                        help='Skip Ollama model availability check')
    args = parser.parse_args()

    print(BANNER)

    # Determine which steps to run
    if args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
    else:
        steps_to_run = set(range(args.start_from, 6))

    print(f"  Steps to run: {sorted(steps_to_run)}")
    print(f"  Working dir : {os.getcwd()}")

    # ── Pre-checks ─────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  PRE-FLIGHT CHECKS")
    print(f"{'─' * 65}")

    if (1 in steps_to_run or 2 in steps_to_run):
        check_dependencies()
        if not args.skip_model_check and 2 in steps_to_run:
            if not check_ollama_models():
                print("\n  ⚠️  Some models missing — continuing anyway (step2 will skip unavailable models)")

    # ── File checks ────────────────────────────────────────────────
    print("\n[Pre-check] Required files:")
    all_files_ok = True

    if 1 in steps_to_run:
        if not check_file(args.proofwriter_dir, 'ProofWriter dataset'):
            print(f"    ℹ️  Step 1 will use KB-generated queries only (still valid)")

    if 2 in steps_to_run and 1 not in steps_to_run:
        if not check_file(args.queries, 'Extended queries'):
            all_files_ok = False
        if not check_file(args.kb, 'Knowledge base'):
            all_files_ok = False

    if 3 in steps_to_run and not os.path.exists('all_model_results.json'):
        print(f"  ❌  all_model_results.json missing — run step2 first")
        all_files_ok = False

    if 4 in steps_to_run:
        check_file(args.truthfulqa, 'TruthfulQA CSV')  # warning only

    if not all_files_ok:
        print("\n  ⛔ Missing required files. Exiting.")
        sys.exit(1)

    # ── Run steps ──────────────────────────────────────────────────
    start_time = time.time()

    if 1 in steps_to_run:
        ok = run_step(
            ['step1_proofwriter_extractor.py',
             '--proofwriter_dir', args.proofwriter_dir,
             '--kb_path',         'knowledge_base.json',
             '--output_kb',       args.kb,
             '--output_queries',  args.queries],
            'STEP 1 — ProofWriter Extraction + KB Extension'
        )
        if not ok:
            sys.exit(1)

    if 2 in steps_to_run:
        ok = run_step(
            ['step2_multi_model_runner.py',
             '--queries', args.queries,
             '--kb',      args.kb,
             '--output',  'all_model_results.json'],
            'STEP 2 — Multi-Model Evaluation (LLM × 3 + LogicGuard)'
        )
        if not ok:
            sys.exit(1)

    if 3 in steps_to_run:
        ok = run_step(
            ['step3_metrics.py',
             '--results', 'all_model_results.json'],
            'STEP 3 — Metrics, Confusion Matrices, P/R/F1'
        )
        if not ok:
            sys.exit(1)

    if 4 in steps_to_run:
        tqa_args = ['step4_truthfulqa_validation.py',
                    '--csv',    args.truthfulqa,
                    '--kb',     args.kb,
                    '--output', 'truthfulqa_validation_report.txt',
                    '--json_out', 'truthfulqa_validation.json']
        ok = run_step(
            tqa_args,
            'STEP 4 — TruthfulQA Out-of-Scope Generalization Test',
            abort_on_fail=False  # Non-fatal: TruthfulQA may not be present
        )
        if not ok:
            print("  ⚠️  Step 4 skipped/failed — step5 will show placeholder for Table VI")

    if 5 in steps_to_run:
        ok = run_step(
            ['step5_generate_paper_tables.py'],
            'STEP 5 — IEEE Paper Tables Generator'
        )
        if not ok:
            sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═' * 65}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"{'═' * 65}")

    print("\n  Output files:")
    output_files = [
        ('knowledge_base_extended.json',       'Extended KB'),
        ('extended_queries.json',              'Query set (175 queries)'),
        ('all_model_results.json',             'All model results'),
        ('metrics_report.txt',                 'Metrics report (human-readable)'),
        ('metrics_report.json',                'Metrics report (machine-readable)'),
        ('truthfulqa_validation_report.txt',   'TruthfulQA generalization report'),
        ('truthfulqa_validation.json',         'TruthfulQA validation data'),
        ('paper_tables_final.txt',             '★ IEEE PAPER CONTENT (copy-paste ready)'),
    ]
    for fname, label in output_files:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"  ✅  {label:<42} {fname} ({size:,}B)")
        else:
            print(f"  —   {label:<42} {fname} (not generated)")

    print()
    print("  ★ Open paper_tables_final.txt for IEEE-ready content:")
    print("    - Tables I through VI")
    print("    - Ready-to-paste Results section paragraphs")
    print("    - Reviewer response paragraphs (R1–R5)")
    print()
    print(f"{'═' * 65}\n")


if __name__ == '__main__':
    main()