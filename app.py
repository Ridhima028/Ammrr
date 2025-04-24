import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    ratings_url = "https://raw.githubusercontent.com/Ridhima028/Ammrr/main/ratings.dat"
    movies_url = "https://raw.githubusercontent.com/Ridhima028/Ammrr/main/movies.dat"

    ratings = pd.read_csv(ratings_url, sep="::", engine="python",
                          names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv(movies_url, sep="::", engine="python",
                         names=["movieId", "title", "genres"])
    return ratings, movies

ratings, movies = load_data()

# Pivot the ratings data
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute similarity matrix
similarity = cosine_similarity(user_movie_matrix)

# Recommend movies
def get_recommendations(user_id, ratings, movies, top_n=10):
    user_index = user_id - 1  # userId starts from 1

    # User similarity scores
    sim_scores = similarity[user_index]

    # Weighted ratings
    weighted_scores = sim_scores @ user_movie_matrix.values
    user_seen = user_movie_matrix.iloc[user_index].values > 0
    weighted_scores[user_seen] = 0  # Don't recommend seen movies

    # Get top N recommendations
    top_movie_indices = weighted_scores.argsort()[::-1][:top_n]
    movie_ids = user_movie_matrix.columns[top_movie_indices]
    recommended = movies[movies['movieId'].isin(movie_ids)]

    return recommended

# Streamlit UI
st.title("🎬 Movie Recommender (Pandas + scikit-learn)")
user_ids = ratings['userId'].unique()
selected_user = st.selectbox("Select User ID", sorted(user_ids))

if st.button("Get Recommendations"):
    recommendations = get_recommendations(int(selected_user), ratings, movies)
    st.subheader("🎯 Top 10 Recommended Movies")
    st.table(recommendations[['movieId', 'title', 'genres']].reset_index(drop=True))
