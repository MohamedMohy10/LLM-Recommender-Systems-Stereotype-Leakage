import json

with open("judge_scoring_checkpoint.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print(f"Total rows so far: {len(rows)}")
print(f"Score distribution: {[r['judge_score'] for r in rows].count(0)} zeros, "
      f"{sum(1 for r in rows if r['judge_score'] not in (0, None))} non-zero, "
      f"{sum(1 for r in rows if r['judge_score'] is None)} failed/unparsed")

print("\n--- Sample reasons for the glm-5.2 (non-template) row ---")
for r in rows:
    if r["model"] == "glm-5.2":
        print(r["judge_reason"])
        break