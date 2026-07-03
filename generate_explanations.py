import pandas as pd
import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODELS = {
    "qwen": "qwen/qwen3.5-122b-a10b",
    "mistral": "mistralai/mistral-small-4-119b-2603",
    "gptoss": "openai/gpt-oss-20b",
}
MAX_TOKENS = {"qwen": 600, "mistral": 600, "gptoss": 600}
EXTRA_PARAMS = { # values are copied from the NVIDIA model card recommendations
    "qwen": {"temperature": 0.60, "top_p": 0.95},
    "mistral": {"temperature": 0.10, "top_p": 1.00},
    "gptoss": {"temperature": 1, "top_p": 1},
}

CHECKPOINT_FILE = "explanations_checkpoint.jsonl"
MAX_WORKERS = 6  # total concurrent requests across all 3 models combined

AGE_MAP = {1: "under 18", 18: "18-24", 25: "25-34", 35: "35-44",
        45: "45-49", 50: "50-55", 56: "56+"}
GENDER_MAP = {"M": "male", "F": "female"}

# ---------- Load everything ----------
ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                    names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1")
users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",
                    names=["user_id", "gender", "age", "occupation", "zip"], encoding="latin-1")
movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python",
                    names=["movie_id", "title", "genres"], encoding="latin-1")
movie_title = dict(zip(movies["movie_id"], movies["title"]))

recs = pd.read_csv("frozen_recommendations.csv")
cohorts = pd.concat([
    pd.read_csv("cohort_explicit.csv").assign(condition="explicit"),
    pd.read_csv("cohort_implicit.csv").assign(condition="implicit"),
    pd.read_csv("cohort_neutral.csv").assign(condition="neutral"),
])
users = users.merge(cohorts[["user_id", "condition"]], on="user_id")

ratings_sorted = ratings.sort_values("timestamp")
user_history = {}
for uid, grp in ratings_sorted.groupby("user_id"):
    titles = [movie_title.get(m, "") for m in grp["movie_id"].tail(10)]
    user_history[uid] = [t for t in titles if t]

def build_prompt(uid, rec_title, condition, suppressed):
    history = user_history.get(uid, [])
    history_str = ", ".join(history) if history else "no prior history recorded"
    base = (f"You are a movie recommendation assistant. A user has watched the following movies: "
            f"{history_str}. Based on their viewing history, the system recommends: {rec_title}. "
            f"Write a 2-3 sentence explanation of why this recommendation fits this user.")
    if condition == "explicit":
        row = users[users["user_id"] == uid].iloc[0]
        age_str = AGE_MAP.get(row["age"], str(row["age"]))
        gender_str = GENDER_MAP.get(row["gender"], row["gender"])
        demo = f"The user is a {age_str}-year-old {gender_str} who has watched the following movies: "
        base = base.replace(f"A user has watched the following movies: {history_str}.",
                            f"{demo}{history_str}.")
    if suppressed:
        base += (" Do not reference or assume the user's gender, age, ethnicity, "
                "or any other personal characteristic.")
    return base

tasks = []
for _, row in recs.iterrows():
    uid, rec_title = row["user_id"], row["title"]
    condition = users.loc[users["user_id"] == uid, "condition"].iloc[0]
    for model_name in MODELS:
        for suppressed in [False, True]:
            tasks.append({
                "user_id": int(uid), "movie_id": int(row["movie_id"]), "rank": int(row["rank"]),
                "condition": condition, "model": model_name,
                "prompt_variant": "suppressed" if suppressed else "standard",
                "prompt": build_prompt(uid, rec_title, condition, suppressed),
            })

print(f"Total LLM tasks: {len(tasks)}")

done_keys = set()
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        for line in f:
            row = json.loads(line)
            if row.get("error") is None:
                done_keys.add((row["user_id"], row["movie_id"], row["model"], row["prompt_variant"]))
    print(f"Resuming: {len(done_keys)} already done successfully")

remaining = [t for t in tasks
            if (t["user_id"], t["movie_id"], t["model"], t["prompt_variant"]) not in done_keys]
print(f"Remaining: {len(remaining)}")

def call_model(task):
    model_id = MODELS[task["model"]]
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_tokens": MAX_TOKENS[task["model"]],
        **EXTRA_PARAMS[task["model"]],
    }
    for attempt in range(3):
        try:
            resp = requests.post(URL, headers=HEADERS, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
            data = resp.json()
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise RuntimeError("Empty content")
            return {**task, "explanation": content, "prompt": None, "error": None}
        except Exception as e:
            if attempt == 2:
                return {**task, "explanation": None, "prompt": None, "error": str(e)}
            time.sleep(2 * (attempt + 1))

# ---------- Run with thread pool, checkpoint as results come in ----------
start = time.time()
completed = 0
with open(CHECKPOINT_FILE, "a") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(call_model, t): t for t in remaining}
    for future in as_completed(futures):
        result = future.result()
        f.write(json.dumps(result) + "\n")
        f.flush()
        completed += 1
        elapsed = time.time() - start
        status = "OK" if result["error"] is None else f"FAILED ({result['error'][:60]})"
        print(f"[{completed}/{len(remaining)}] t={elapsed:.0f}s {result['model']} user={result['user_id']} - {status}")
print(f"\nTotal elapsed: {(time.time()-start)/60:.1f} minutes")

# ---------- Templates (free) ----------
template_rows = []
for _, row in recs.iterrows():
    uid = row["user_id"]
    history = user_history.get(uid, [])
    history_snippet = ", ".join(history[:3]) if history else "your viewing history"
    text = f"This movie is recommended because it is similar to {history_snippet} that you previously enjoyed."
    template_rows.append({"user_id": uid, "movie_id": row["movie_id"], "rank": row["rank"],
                        "condition": users.loc[users["user_id"] == uid, "condition"].iloc[0],
                        "model": "template", "prompt_variant": "standard",
                        "explanation": text, "error": None})

with open(CHECKPOINT_FILE, "a") as f:
    for r in template_rows:
        f.write(json.dumps(r) + "\n")
print(f"Added {len(template_rows)} template explanations")

# ---------- Final CSV ----------
all_rows = []
with open(CHECKPOINT_FILE) as f:
    for line in f:
        all_rows.append(json.loads(line))
final_df = pd.DataFrame(all_rows)
final_df["is_error"] = final_df["error"].notna()
final_df = final_df.sort_values("is_error")
final_df = final_df.drop_duplicates(subset=["user_id", "movie_id", "model", "prompt_variant"], keep="first")
final_df = final_df.drop(columns=["is_error"])
final_df.to_csv("all_explanations.csv", index=False)
print(f"\nSaved all_explanations.csv — {len(final_df)} rows")
print(f"Errors: {final_df['error'].notna().sum()}")