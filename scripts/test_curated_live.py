import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_groq_benchmark_live import evaluate_single_query, load_api_key
from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.kb.validator import BFSValidator
from avicennaguard.parsers.deberta_parser import DebertaParser

kb = KnowledgeBase("data/knowledge_bases/knowledge_base_extended.json")
v = BFSValidator(kb)
parser = DebertaParser()
api_key = load_api_key()

with open("data/benchmarks/avicenna_benchmark_500.json", "r", encoding="utf-8") as f:
    bench = json.load(f)

curated = [b for b in bench if "curated" in b.get("id", "")]
print(f"Evaluating {len(curated)} Curated queries on Qwen 3.8 27B live...")

bl_correct = 0
ag_correct = 0
caught = 0
fa = 0

for idx, item in enumerate(curated):
    res = evaluate_single_query(item, "qwen/qwen3.8-27b", api_key, v, parser)
    if res["is_bl_correct"]: bl_correct += 1
    if res["is_ag_correct"]: ag_correct += 1
    if res["intercepted"]: caught += 1
    if res["false_alarm"]: fa += 1
    
    if (idx + 1) % 10 == 0 or res["intercepted"]:
        print(f"[{idx+1:03d}/{len(curated)}] Q: \"{item['question'][:40]}\" | LLM: {str(res['llm_answer']):<5} | Guard: {str(res['final_bool']):<5} | GT: {str(res['gt_bool']):<5} | Intercepted: {res['intercepted']}")

print("=" * 80)
print(f"Curated 100 Results: Raw LLM = {bl_correct}% -> +AvicennaGuard = {ag_correct}% | Interceptions: {caught} | False Alarms: {fa}")
print("=" * 80)
