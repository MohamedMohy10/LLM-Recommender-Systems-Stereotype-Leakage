import pandas as pd
from scipy import stats
from itertools import combinations

df = pd.read_csv("judge_scores_sample_combined.csv")
df = df.dropna(subset=["judge_score"])

explicit_std = df[(df["condition"] == "explicit") &
                   (df["prompt_variant"] == "standard") &
                   (df["model"] != "template")]

groups = [explicit_std[explicit_std["model"] == m]["judge_score"].values
          for m in ["qwen", "mistral", "gptoss"]]

print("Sample sizes:", [len(g) for g in groups])
print("Means:", [round(g.mean(), 3) for g in groups])

h_stat, p_value = stats.kruskal(*groups)
print(f"\nKruskal-Wallis H = {h_stat:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    print(">>> Significant difference across LLM families.")
    pairs = list(combinations(["qwen", "mistral", "gptoss"], 2))
    alpha_corrected = 0.05 / len(pairs)
    print(f"Running pairwise Mann-Whitney U (Bonferroni-corrected alpha = {alpha_corrected:.4f})...")
    for m1, m2 in pairs:
        g1 = explicit_std[explicit_std["model"] == m1]["judge_score"].values
        g2 = explicit_std[explicit_std["model"] == m2]["judge_score"].values
        u_stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        sig = "SIGNIFICANT" if p < alpha_corrected else "not significant"
        print(f"{m1} vs {m2}: U={u_stat:.1f}, p={p:.4f} ({sig})")
else:
    print(">>> No significant difference detected across LLM families at alpha=0.05.")

# Effect size: rank-biserial correlation for each pair (useful even if not significant)
print("\n--- Rank-biserial effect sizes (magnitude of difference) ---")
for m1, m2 in combinations(["qwen", "mistral", "gptoss"], 2):
    g1 = explicit_std[explicit_std["model"] == m1]["judge_score"].values
    g2 = explicit_std[explicit_std["model"] == m2]["judge_score"].values
    u_stat, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    n1, n2 = len(g1), len(g2)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    print(f"{m1} vs {m2}: r = {rank_biserial:.3f}")