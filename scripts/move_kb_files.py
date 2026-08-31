"""
One-time migration script: move KB files from project root to data/knowledge_bases/

Run once:
    python scripts/move_kb_files.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
KB_DIR = ROOT / "data" / "knowledge_bases"
KB_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_MOVE = [
    "knowledge_base.json",
    "knowledge_base_extended.json",
]

for fname in FILES_TO_MOVE:
    src = ROOT / fname
    dst = KB_DIR / fname
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {fname} → data/knowledge_bases/{fname}")
    elif dst.exists():
        print(f"Already exists: data/knowledge_bases/{fname}")
    else:
        print(f"NOT FOUND at root: {fname}")

print("\nDone. Update configs/default.yaml KB paths if needed.")
print("Keep root KB files for backwards compatibility with legacy step*.py scripts.")
