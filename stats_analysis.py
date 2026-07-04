import pandas as pd
from scipy import stats
from itertools import combinations

# ==========================================================
# PART A: SMR-based tests (full corpus, N=2625, most power)
# ==========================================================
smr = pd.read_csv("all_explanations_with_smr_v2.csv")
llm_smr = smr[smr["model"] != "template"].copy()

print("=" * 60)
print("PART A: SMR (full corpus) - Standard vs Suppressed")
print("=" * 60)

# Chi-square test per model: does prompt_variant affect smr_flag_v2?
for model in ["qwen", "mistral", "gptoss"]:
    sub = llm_smr[llm_smr["model"] == model]
    contingency = pd.crosstab(sub["prompt_variant"], sub["smr_flag_v2"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    std_rate = sub[sub["prompt_variant"] == "standard"]["smr_flag_v2"].mean()
    sup_rate = sub[sub["prompt_variant"] == "suppressed"]["smr_flag_v2"].mean()
    print(f"\n{model}: standard={std_rate:.3f}, suppressed={sup_rate:.3f}")
    print(f"  Chi-square: chi2={chi2:.3f}, p={p:.5f} "
          f"({'SIGNIFICANT reduction' if p < 0.05 and sup_rate < std_rate else 'not significant / no reduction'})")

print("\n" + "=" * 60)
print("PART A2: SMR (full corpus) - Explicit vs Implicit vs Neutral")
print("=" * 60)

for model in ["qwen", "mistral", "gptoss"]:
    sub = llm_smr[llm_smr["model"] == model]
    contingency = pd.crosstab(sub["condition"], sub["smr_flag_v2"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    rates = sub.groupby("condition")["smr_flag_v2"].mean()
    print(f"\n{model}: {dict(rates.round(3))}")
    print(f"  Chi-square: chi2={chi2:.3f}, p={p:.5f} "
          f"({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

# ==========================================================
# PART B: Judge-score based tests (sampled, smaller n, more precise measure)
# ==========================================================
judge = pd.read_csv("judge_scores_sample_combined.csv")
judge = judge.dropna(subset=["judge_score"])
judge_llm = judge[judge["model"] != "template"]

print("\n" + "=" * 60)
print("PART B: Judge scores - Standard vs Suppressed (explicit condition)")
print("=" * 60)
# Only explicit has non-trivial variance; implicit/neutral suppressed vs standard
# would both be all-zero, so we focus the judge-score test where it's meaningful
for model in ["qwen", "mistral", "gptoss"]:
    sub = judge_llm[(judge_llm["model"] == model) & (judge_llm["condition"] == "explicit")]
    std_scores = sub[sub["prompt_variant"] == "standard"]["judge_score"].values
    sup_scores = sub[sub["prompt_variant"] == "suppressed"]["judge_score"].values
    if len(std_scores) > 0 and len(sup_scores) > 0:
        u_stat, p = stats.mannwhitneyu(std_scores, sup_scores, alternative="greater")
        print(f"\n{model}: standard mean={std_scores.mean():.3f} (n={len(std_scores)}), "
              f"suppressed mean={sup_scores.mean():.3f} (n={len(sup_scores)})")
        print(f"  Mann-Whitney U (one-sided, standard>suppressed): U={u_stat:.1f}, p={p:.4f} "
              f"({'SIGNIFICANT' if p < 0.05 else 'not significant'})")

print("\n" + "=" * 60)
print("PART B2: Judge scores - Explicit vs Implicit vs Neutral (standard prompt only)")
print("=" * 60)
for model in ["qwen", "mistral", "gptoss"]:
    sub = judge_llm[(judge_llm["model"] == model) & (judge_llm["prompt_variant"] == "standard")]
    groups = [sub[sub["condition"] == c]["judge_score"].values for c in ["explicit", "implicit", "neutral"]]
    sizes = [len(g) for g in groups]
    means = [round(g.mean(), 3) if len(g) > 0 else None for g in groups]
    print(f"\n{model}: sizes={sizes}, means (explicit/implicit/neutral)={means}")
    if all(len(g) > 1 for g in groups):
        h_stat, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis: H={h_stat:.3f}, p={p:.4f} "
              f"({'SIGNIFICANT' if p < 0.05 else 'not significant'})")
    else:
        print("  Skipped (insufficient variation/size for test)")