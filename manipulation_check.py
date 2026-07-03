import pandas as pd
import requests
import json
import os
import time
from scipy import stats
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
JUDGE_MODEL = "google/gemma-4-31b-it"
CHECKPOINT_FILE = "manipulation_checkpoint.jsonl"

# ---------- Load data ----------
explanations = pd.read_csv("all_explanations.csv")
users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",
                    names=["user_id", "gender", "age", "occupation", "zip"], encoding="latin-1")
gender_lookup = dict(zip(users["user_id"], users["gender"]))

subset = explanations[
    (explanations["condition"] == "implicit") &
    (explanations["rank"] == 1) &
    (explanations["prompt_variant"] == "standard") &
    (explanations["model"] != "template")
].copy()

subset["true_gender"] = subset["user_id"].map(gender_lookup)
subset = subset.dropna(subset=["explanation", "true_gender"])
print(f"Testing {len(subset)} explanations ({subset['user_id'].nunique()} users x "
    f"{subset['model'].nunique()} models)")

# ---------- Resume support ----------
done_keys = set()
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        for line in f:
            row = json.loads(line)
            if row.get("predicted_gender") is not None:
                done_keys.add((row["user_id"], row["model"]))
    print(f"Resuming: {len(done_keys)} already done")

remaining = [row for row in subset.itertuples()
            if (row.user_id, row.model) not in done_keys]
print(f"Remaining: {len(remaining)}")

# ---------- Judge call with proper 429 handling ----------
def judge_gender(explanation_text):
    prompt = (
        "Below is a short text explaining why a movie was recommended to a user. "
        "Based only on the wording and content of this text, guess whether the user is male or female. "
        "Respond with exactly one word: 'male' or 'female'.\n\n"
        f"Text: \"{explanation_text}\""
    )
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 60,
    }
    for attempt in range(5):
        try:
            resp = requests.post(URL, headers=HEADERS, json=payload, timeout=45)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  429 rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
            data = resp.json()
            content = (data["choices"][0]["message"].get("content") or "").strip().lower()
            if "female" in content:
                return "F"
            elif "male" in content:
                return "M"
            else:
                return None
        except Exception as e:
            if attempt == 4:
                print(f"  FAILED: {e}")
                return None
            time.sleep(3 * (attempt + 1))
    return None  # exhausted 429 retries

# ---------- Run sequentially with a fixed delay to respect rate limit ----------
with open(CHECKPOINT_FILE, "a") as f:
    for i, row in enumerate(remaining, 1):
        pred = judge_gender(row.explanation)
        result = {"user_id": row.user_id, "model": row.model,
                "true_gender": row.true_gender, "predicted_gender": pred}
        f.write(json.dumps(result) + "\n")
        f.flush()
        if i % 10 == 0:
            print(f"[{i}/{len(remaining)}] done")
        time.sleep(1.5)  # stay under Llama 3.3's per-minute rate limit

# ---------- Load all results (checkpoint + this run) ----------
all_results = []
with open(CHECKPOINT_FILE) as f:
    for line in f:
        all_results.append(json.loads(line))
results_df = pd.DataFrame(all_results).drop_duplicates(subset=["user_id", "model"], keep="last")
results_df.to_csv("manipulation_check_results.csv", index=False)

# ---------- Analysis ----------
valid = results_df.dropna(subset=["predicted_gender"])
n = len(valid)
correct = (valid["true_gender"] == valid["predicted_gender"]).sum()
accuracy = correct / n

print(f"\n=== Manipulation Check Results ===")
print(f"Valid predictions: {n} / {len(results_df)}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.3f}")

binom_result = stats.binomtest(correct, n, p=0.5, alternative="greater")
print(f"Binomial test p-value (one-sided, accuracy > chance): {binom_result.pvalue:.5f}")

if binom_result.pvalue < 0.05:
    print(">>> RESULT: Accuracy is significantly above chance. Implicit condition CONFIRMED — "
        "proceed with RQ3 as designed.")
else:
    print(">>> RESULT: Accuracy is NOT significantly above chance. Implicit condition FAILED "
        "manipulation check — per Section 2.4 contingency, drop RQ3 from main analysis, "
        "reframe as explicit vs. neutral only, treat implicit cohort as secondary neutral variant.")

print("\n--- Accuracy by generating model ---")
for model_name, grp in valid.groupby("model"):
    m_correct = (grp["true_gender"] == grp["predicted_gender"]).sum()
    print(f"{model_name}: {m_correct}/{len(grp)} = {m_correct/len(grp):.3f}")