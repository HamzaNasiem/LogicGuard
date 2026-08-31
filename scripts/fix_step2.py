"""One-off fix for corrupted step2_multi_model_runner.py"""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "step2_multi_model_runner.py"
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

# Keep header through use_logicguard line (0-indexed: 0..142)
header = lines[:143]

# Find second complete evaluate_model signature (with delay line)
start_tail = None
for i, line in enumerate(lines):
    if line.startswith("    delay:           float = 0.3,") and i > 200:
        start_tail = i
        break

if start_tail is None:
    raise SystemExit("Could not find tail start")

# Include delay line and rest; skip duplicate def block header
tail = lines[start_tail:]
fixed = header + tail
path.write_text("".join(fixed), encoding="utf-8")
print(f"Fixed: {len(lines)} -> {len(fixed)} lines")
