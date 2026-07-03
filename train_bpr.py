import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------- Load data ----------
ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                    names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1")
movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python",
                    names=["movie_id", "title", "genres"], encoding="latin-1")

# Only positive implicit feedback: rating >= 3 (as specified in Section 4)
ratings = ratings[ratings["rating"] >= 3].copy()

# Re-index users/movies to contiguous ids for the embedding tables
user_ids = ratings["user_id"].unique()
movie_ids = ratings["movie_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}
idx2movie = {i: m for m, i in movie2idx.items()}

ratings["u_idx"] = ratings["user_id"].map(user2idx)
ratings["m_idx"] = ratings["movie_id"].map(movie2idx)

n_users, n_movies = len(user_ids), len(movie_ids)
print(f"Users: {n_users}, Movies: {n_movies}, Interactions: {len(ratings)}")

# ---------- Temporal leave-one-out split (Section 4) ----------
ratings = ratings.sort_values(["user_id", "timestamp"])
train_rows, val_rows, test_rows = [], [], []
for uid, grp in ratings.groupby("user_id"):
    grp = grp.sort_values("timestamp")
    if len(grp) < 3:
        train_rows.append(grp)  # too few interactions, keep all in train
        continue
    test_rows.append(grp.iloc[[-1]])
    val_rows.append(grp.iloc[[-2]])
    train_rows.append(grp.iloc[:-2])

train_df = pd.concat(train_rows)
print(f"Train interactions: {len(train_df)}")

# Build per-user positive item sets (for negative sampling + exclusion)
user_pos_items = train_df.groupby("u_idx")["m_idx"].apply(set).to_dict()

# ---------- BPR-MF model ----------
class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i, j):
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        j_e = self.item_emb(j)
        pos_score = (u_e * i_e).sum(dim=1)
        neg_score = (u_e * j_e).sum(dim=1)
        return pos_score, neg_score

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BPRMF(n_users, n_movies, dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

train_pairs = list(zip(train_df["u_idx"].values, train_df["m_idx"].values))

def sample_batch(batch_size=1024):
    us, pos, neg = [], [], []
    for _ in range(batch_size):
        u, i = train_pairs[np.random.randint(len(train_pairs))]
        j = np.random.randint(n_movies)
        while j in user_pos_items.get(u, set()):
            j = np.random.randint(n_movies)
        us.append(u); pos.append(i); neg.append(j)
    return (torch.tensor(us, device=device), torch.tensor(pos, device=device),
            torch.tensor(neg, device=device))

# ---------- Train ----------
n_epochs = 20
steps_per_epoch = len(train_pairs) // 1024

for epoch in range(n_epochs):
    total_loss = 0
    for _ in range(steps_per_epoch):
        u, i, j = sample_batch()
        pos_score, neg_score = model(u, i, j)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs} - loss: {total_loss/steps_per_epoch:.4f}")

# ---------- Generate frozen top-3 lists for our 125 cohort users ONLY ----------
cohort_dfs = [pd.read_csv(f"cohort_{c}.csv") for c in ["explicit", "implicit", "neutral"]]
cohort_users = pd.concat(cohort_dfs)["user_id"].unique()
print(f"\nGenerating recommendations for {len(cohort_users)} cohort users")

model.eval()
all_movie_idx = torch.arange(n_movies, device=device)
results = []

with torch.no_grad():
    for uid in cohort_users:
        if uid not in user2idx:
            print(f"WARNING: user {uid} not in training data, skipping")
            continue
        u_idx = user2idx[uid]
        u_vec = model.user_emb(torch.tensor([u_idx], device=device))
        scores = (u_vec @ model.item_emb.weight.T).squeeze()
        seen = user_pos_items.get(u_idx, set())
        scores_np = scores.cpu().numpy()
        scores_np[list(seen)] = -1e9  # exclude already-watched
        top3_idx = np.argsort(-scores_np)[:3]
        top3_movie_ids = [idx2movie[i] for i in top3_idx]
        top3_titles = movies.set_index("movie_id").loc[top3_movie_ids, "title"].tolist()
        for rank, (mid, title) in enumerate(zip(top3_movie_ids, top3_titles), start=1):
            results.append({"user_id": uid, "rank": rank, "movie_id": mid, "title": title})

rec_df = pd.DataFrame(results)
rec_df.to_csv("frozen_recommendations.csv", index=False)
print(f"\nSaved frozen_recommendations.csv — {len(rec_df)} rows "
    f"({len(cohort_users)} users x 3 recs)")
print(rec_df.head(9))