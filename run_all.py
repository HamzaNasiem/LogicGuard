#!/usr/bin/env python3
"""
MASTER RUNNER — AvicennaGuard Complete Pipeline  (v2 — IEEE journal grade)
========================================================================
Runs all steps in sequence:

  Step 1  : ProofWriter extraction + KB extension
  Step 2  : Multi-model evaluation  (--filter_source original = no circular eval)
  Step 3  : Metrics, confusion matrices, P/R/F1
  Step 3b : Statistical significance (McNemar's test, p-values, CI)  ← NEW
  Step 4  : TruthfulQA out-of-scope generalization test
  Step 5  : IEEE paper tables + copy-paste text generator

Usage:
    # Full pipeline (IEEE journal submission)
    python run_all.py --proofwriter_dir proofwriter-dataset-V2020.12.3

    # Skip to stats + paper tables (data already collected)
    python run_all.py --steps 3b,4,5

    # Only regenerate paper tables
    python run_all.py --steps 5

    # Single model only
    python run_all.py --steps 2,3,3b,5 --models llama32_3b
"""

import os
import sys
import argparse
import subprocess
import time

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     AvicennaGuard — Complete Pipeline  (v2 — IEEE journal grade)    ║
║     Step 1 → 2 → 3 → 3b → 4 → 5                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_step(cmd: list, step_name: str, abort_on_fail: bool = True) -> bool:
    print(f"\n{'━' * 65}")
    print(f"  ▶  {step_name}")
    print(f"{'━' * 65}\n")
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    result = subprocess.run([sys.executable, '-X', 'utf8'] + cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"\n  ❌  {step_name} FAILED (exit code {result.returncode})")
        if abort_on_fail:
            print("  Aborting pipeline.")
        return False
    print(f"\n  ✅  {step_name} COMPLETE")
    return True


def check_dependencies():
    print("\n[Pre-check] Verifying dependencies...")
    packages = {'ollama': 'ollama', 'numpy': 'numpy', 'networkx': 'networkx'}
    for module, pkg in packages.items():
        try:
            __import__(module)
            print(f"  ✅  {pkg}")
        except ImportError:
            print(f"  ❌  {pkg} — run: pip install {pkg}")
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


def main():
    parser = argparse.ArgumentParser(description='AvicennaGuard Complete Pipeline v2')
    parser.add_argument('--proofwriter_dir',  default='proofwriter-dataset-V2020.12.3')
    parser.add_argument('--queries',          default='extended_queries.json')
    parser.add_argument('--kb',               default='knowledge_base_extended.json')
    parser.add_argument('--truthfulqa',       default='truthfulqa.csv')
    parser.add_argument('--models',           default='all',
                        help='Comma-separated model keys or "all"')
    parser.add_argument('--filter_source',    default='original',
                        choices=['original', 'all'],
                        help='original = no circular eval (recommended for IEEE)')
    parser.add_argument('--steps',            default='1,2,3,3b,4,5',
                        help='Steps to run (e.g. "2,3,3b,5")')
    parser.add_argument('--skip_model_check', action='store_true')
    args = parser.parse_args()

    print(BANNER)
    steps = parse_steps(args.steps)
    print(f"  Steps to run    : {sorted(steps)}")
    print(f"  Source filter   : {args.filter_source.upper()}")
    print(f"  Models          : {args.models}")
    print(f"  Working dir     : {os.getcwd()}")

    if args.filter_source == 'original':
        print(f"\n  ✓ Circular evaluation eliminated — ORIGINAL queries only")
    else:
        print(f"\n  ⚠️  ALL queries including kb_generated — NOT for IEEE submission")

    # ── Pre-checks ───────────────────────────────────────────────────
    print(f"\n{'─' * 65}  PRE-FLIGHT CHECKS  {'─' * 5}")
    if '2' in steps:
        check_dependencies()
        if not args.skip_model_check:
            check_ollama_models()

    start_time = time.time()

    # ── Step 1 ───────────────────────────────────────────────────────
    if '1' in steps:
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

    # ── Step 2 ───────────────────────────────────────────────────────
    if '2' in steps:
        ok = run_step(
            ['step2_multi_model_runner.py',
             '--queries',       args.queries,
             '--kb',            args.kb,
             '--output',        'all_model_results.json',
             '--models',        args.models,
             '--filter_source', args.filter_source],
            'STEP 2 — Multi-Model Evaluation + AvicennaGuard (no circular eval)'
        )
        if not ok:
            sys.exit(1)

    # ── Step 3 ───────────────────────────────────────────────────────
    if '3' in steps:
        ok = run_step(
            ['step3_metrics.py', '--results', 'all_model_results.json'],
            'STEP 3 — Metrics, Confusion Matrices, P/R/F1'
        )
        if not ok:
            sys.exit(1)

    # ── Step 3b (NEW) ────────────────────────────────────────────────
    if '3b' in steps:
        ok = run_step(
            ['step3b_statistical.py',
             '--results',     'all_model_results.json',
             '--output_json', 'statistical_significance.json',
             '--output_txt',  'statistical_report.txt'],
            'STEP 3b — Statistical Significance (McNemar + CI + Latency)'
        )
        if not ok:
            print("  ⚠️  Step 3b failed — Tables VII/VIII/IX will show placeholders in step5")

    # ── Step 4 ───────────────────────────────────────────────────────
    if '4' in steps:
        ok = run_step(
            ['step4_truthfulqa_validation.py',
             '--csv',      args.truthfulqa,
             '--kb',       args.kb,
             '--output',   'truthfulqa_validation_report.txt',
             '--json_out', 'truthfulqa_validation.json'],
            'STEP 4 — TruthfulQA Out-of-Scope Generalization Test',
            abort_on_fail=False
        )
        if not ok:
            print("  ⚠️  Step 4 skipped — Table VI will show placeholder")

    # ── Step 5 ───────────────────────────────────────────────────────
    if '5' in steps:
        ok = run_step(
            ['step5_generate_paper_tables.py'],
            'STEP 5 — IEEE Paper Tables Generator (9 tables)'
        )
        if not ok:
            sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═' * 65}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"{'═' * 65}")

    output_files = [
        ('knowledge_base_extended.json',     'Extended KB'),
        ('all_model_results.json',           'All model results'),
        ('metrics_report.txt',               'Metrics report'),
        ('statistical_significance.json',    'Statistical significance (McNemar)'),
        ('statistical_report.txt',           'Statistical report (Tables VII-IX)'),
        ('truthfulqa_validation.json',       'TruthfulQA validation'),
        ('paper_tables_final.txt',           '★ IEEE PAPER CONTENT (copy-paste ready)'),
    ]
    print("\n  Output files:")
    for fname, label in output_files:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"  ✅  {label:<44} {fname} ({size:,}B)")
        else:
            print(f"  —   {label:<44} {fname}")

    print()
    print("  ★ paper_tables_final.txt contains:")
    print("    TABLE I-VI   — Original tables (from conference submission)")
    print("    TABLE VII    — Stage 1 Parser Robustness  [NEW]")
    print("    TABLE VIII   — Statistical Significance   [NEW — p-values]")
    print("    TABLE IX     — Latency Analysis            [NEW]")
    print("    Reviewer responses R1-R5 (updated for circular eval fix)")
    print(f"\n{'═' * 65}\n")


if __name__ == '__main__':
    main()
