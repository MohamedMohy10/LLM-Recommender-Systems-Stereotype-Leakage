import pandas as pd

# Ratings: UserID::MovieID::Rating::Timestamp
ratings = pd.read_csv(
    "ml-1m/ratings.dat", sep="::", engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
    encoding="latin-1"
)

# Users: UserID::Gender::Age::Occupation::Zip-code
users = pd.read_csv(
    "ml-1m/users.dat", sep="::", engine="python",
    names=["user_id", "gender", "age", "occupation", "zip"],
    encoding="latin-1"
)

# Movies: MovieID::Title::Genres
movies = pd.read_csv(
    "ml-1m/movies.dat", sep="::", engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

print("=== Ratings ===")
print(f"Total ratings: {len(ratings)}")
print(ratings.head())

print("\n=== Users ===")
print(f"Total users: {len(users)}")
print("Gender distribution:")
print(users["gender"].value_counts())
print("\nAge bucket distribution (MovieLens uses coded buckets, not raw age):")
print(users["age"].value_counts().sort_index())

print("\n=== Movies ===")
print(f"Total movies: {len(movies)}")
print(movies.head())

print("\n=== Ratings per user (sanity check for cohort feasibility) ===")
ratings_per_user = ratings.groupby("user_id").size()
print(ratings_per_user.describe())