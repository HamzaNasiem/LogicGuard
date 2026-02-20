# quick_test.py
"""
LogicGuard - Quick Test (No Ollama / No Internet needed)
Run this first to verify all templates work correctly.
Usage: python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from logic_validator import LogicValidator

print("=" * 65)
print("  LOGICGUARD — QUICK TEST (No LLM Required)")
print("=" * 65)

v = LogicValidator(use_ollama=False)

tests = [
    # ── Taxonomic ──────────────────────────────────────── expected
    ("Are all dogs mammals?",                   'YAQEEN'),
    ("Are all cats mammals?",                   'YAQEEN'),
    ("Are all whales mammals?",                 'YAQEEN'),
    ("Are all birds animals?",                  'YAQEEN'),
    ("Are all squares rectangles?",             'YAQEEN'),
    ("Are all triangles polygons?",             'YAQEEN'),
    ("Are all cars vehicles?",                  'YAQEEN'),
    ("Are all apples fruits?",                  'YAQEEN'),
    ("Are all sharks fish?",                    'YAQEEN'),
    ("Are all eagles birds?",                   'YAQEEN'),

    # ── Hypothetical ───────────────────────────────────
    ("If it is raining, is the ground wet?",    'YAQEEN'),
    ("If there is fire, is there heat?",        'YAQEEN'),
    ("If water freezes, does it become ice?",   'YAQEEN'),
    ("If water boils, is it hot?",              'YAQEEN'),
    ("If the sun is shining, is it daytime?",   'YAQEEN'),

    # ── Categorical ────────────────────────────────────
    ("Do all mammals have hair?",               'YAQEEN'),
    ("Do all birds have feathers?",             'YAQEEN'),
    ("Do all squares have four sides?",         'YAQEEN'),
    ("Do all fish have gills?",                 'YAQEEN'),
    ("Do all humans have a heart?",             'YAQEEN'),
    ("Do all living things need water?",        'YAQEEN'),

    # ── Non-logical (should NOT match any template) ───
    ("What is the capital of France?",          'UNKNOWN'),
    ("Who wrote Romeo and Juliet?",             'UNKNOWN'),
    ("How many continents are there?",          'UNKNOWN'),
]

passed = 0
failed = 0

print()
for question, expected_state in tests:
    result = v.validate(question)
    state  = result['epistemic_state']
    ok     = (state == expected_state)

    icon = "✅" if ok else "❌"
    passed += ok
    failed += (not ok)

    label = result.get('template_used', 'none') or 'none'
    print(f"  {icon} [{state:8}] [{label:12}] {question}")
    if not ok:
        print(f"       Expected : {expected_state}")
        print(f"       Proof    : {result['proof']}")

print()
print("=" * 65)
print(f"  PASSED : {passed} / {len(tests)}")
print(f"  FAILED : {failed} / {len(tests)}")
if failed == 0:
    print("  🎉 ALL TESTS PASSED — Logic templates working perfectly!")
    print("  ✅ Safe to run full experiment with Ollama.")
else:
    print(f"  ⚠️  {failed} test(s) failed — fix before running full experiment.")
print("=" * 65)