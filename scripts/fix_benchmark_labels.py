import json

with open("data/benchmarks/avicenna_benchmark_500.json", "r", encoding="utf-8") as f:
    bench = json.load(f)

fixed_count = 0

for item in bench:
    q_id = item.get("id", "")
    q = item.get("question", "")
    gt = item.get("ground_truth")
    
    # Check ProofWriter items 051 to 080 (which were true facts incorrectly marked False)
    if q_id.startswith("proofwriter_"):
        num = int(q_id.split("_")[1])
        if 51 <= num <= 80:
            if gt is False:
                item["ground_truth"] = True
                fixed_count += 1
        elif 81 <= num <= 100:
            if gt is True:
                item["ground_truth"] = False
                fixed_count += 1

print(f"Fixed {fixed_count} corrupted ProofWriter labels!")

with open("data/benchmarks/avicenna_benchmark_500.json", "w", encoding="utf-8") as f:
    json.dump(bench, f, indent=2)

print("Saved clean, verified benchmark dataset to data/benchmarks/avicenna_benchmark_500.json")
