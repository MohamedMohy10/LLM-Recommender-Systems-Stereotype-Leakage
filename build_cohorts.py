import pandas as pd
import numpy as np

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                    names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1")
users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",
                    names=["user_id", "gender", "age", "occupation", "zip"], encoding="latin-1")

# --- Step 1: movie-level gender skew z-score (Section 2.4 method) ---
merged = ratings.merge(users[["user_id", "gender"]], on="user_id")
p = (merged["gender"] == "F").mean()

movie_stats = merged.groupby("movie_id").agg(
    n_ratings=("rating", "count"),
    n_female=("gender", lambda x: (x == "F").sum())
).reset_index()
movie_stats["f_m"] = movie_stats["n_female"] / movie_stats["n_ratings"]
movie_stats = movie_stats[movie_stats["n_ratings"] >= 20]

movie_stats["z"] = (movie_stats["f_m"] - p) / np.sqrt(p * (1 - p) / movie_stats["n_ratings"])
movie_z = dict(zip(movie_stats["movie_id"], movie_stats["z"]))

# --- Step 2: per-user average |z| ---
merged["z"] = merged["movie_id"].map(movie_z)
user_skew = merged.dropna(subset=["z"]).groupby("user_id")["z"].mean().rename("avg_z")
user_skew_abs = user_skew.abs().rename("avg_abs_z")

users = users.merge(user_skew.reset_index(), on="user_id", how="left")
users = users.merge(user_skew_abs.reset_index(), on="user_id", how="left")
users = users.dropna(subset=["avg_z"])

pop_mean = users["avg_abs_z"].mean()
pop_std = users["avg_abs_z"].std()
threshold = pop_mean + 2 * pop_std
print(f"Population mean |z|: {pop_mean:.3f}, std: {pop_std:.3f}, implicit threshold: {threshold:.3f}")

# --- Step 3: Implicit cohort (50 users, balanced skew direction) ---
implicit_pool = users[users["avg_abs_z"] > threshold].copy()
print(f"Users above implicit threshold: {len(implicit_pool)}")

n_each = 25  # 25 female-skewed + 25 male-skewed = 50
female_skewed = implicit_pool[implicit_pool["avg_z"] > 0].sort_values("avg_abs_z", ascending=False)
male_skewed = implicit_pool[implicit_pool["avg_z"] < 0].sort_values("avg_abs_z", ascending=False)

implicit_cohort = pd.concat([female_skewed.head(n_each), male_skewed.head(n_each)])
print(f"Implicit cohort size: {len(implicit_cohort)} "
      f"({len(female_skewed.head(n_each))} female-skewed, {len(male_skewed.head(n_each))} male-skewed)")

used_ids = set(implicit_cohort["user_id"])

# --- Step 4: Explicit cohort (50 users, 25F/25M, excluding implicit) ---
remaining = users[~users["user_id"].isin(used_ids)]
explicit_f = remaining[remaining["gender"] == "F"].sample(n=25, random_state=RANDOM_SEED)
explicit_m = remaining[remaining["gender"] == "M"].sample(n=25, random_state=RANDOM_SEED)
explicit_cohort = pd.concat([explicit_f, explicit_m])
used_ids |= set(explicit_cohort["user_id"])
print(f"Explicit cohort size: {len(explicit_cohort)} (25F / 25M)")

# --- Step 5: Neutral cohort (25 users, low skew, excluding used) ---
remaining2 = users[~users["user_id"].isin(used_ids)]
neutral_pool = remaining2[remaining2["avg_abs_z"] < pop_mean].copy()
neutral_cohort = neutral_pool.sample(n=25, random_state=RANDOM_SEED)
print(f"Neutral cohort size: {len(neutral_cohort)}")

# --- Save locally ---
implicit_cohort.assign(condition="implicit").to_csv("cohort_implicit.csv", index=False)
explicit_cohort.assign(condition="explicit").to_csv("cohort_explicit.csv", index=False)
neutral_cohort.assign(condition="neutral").to_csv("cohort_neutral.csv", index=False)

print("\nSaved cohort_implicit.csv, cohort_explicit.csv, cohort_neutral.csv")
print(f"\nTOTAL USERS: {len(implicit_cohort) + len(explicit_cohort) + len(neutral_cohort)}")