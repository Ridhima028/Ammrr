import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
   ratings_url = "https://raw.githubusercontent.com/Ridhima028/Ammrr/main/ratings.dat"
   movies_url = "https://raw.githubusercontent.com/Ridhima028/Ammrr/main/movies.dat"
   ratings = pd.read_csv("ratings.dat", sep="::", engine="python", 
                      names=["userId", "movieId", "rating", "timestamp"])

   movies = pd.read_csv("movies.dat", sep="::", engine="python", 
                     names=["movieId", "title", "genres"])

   return ratings, movies

ratings, movies = load_data()

# Train model
@st.cache_resource
def train_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model, trainset

model, trainset = train_model(ratings)

# Recommend top N movies for a user
def get_recommendations(user_id, model, trainset, movies, n=10):
    all_movie_ids = movies['movieId'].unique()
    seen = set(j for (j, _) in trainset.ur[trainset.to_inner_uid(user_id)]) if trainset.knows_user(user_id) else set()
    unseen = [i for i in all_movie_ids if trainset.knows_item(i) and trainset.to_inner_iid(i) not in seen]

    predictions = [model.predict(user_id, i) for i in unseen]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    top_movie_ids = [int(pred.iid) for pred in top_n]
    recommended = movies[movies['movieId'].isin(top_movie_ids)]
    return recommended

# Streamlit UI
st.title("🎬 Movie Recommender System")
st.write("Made with MovieLens 1M and Surprise")

user_ids = ratings['userId'].unique()
selected_user = st.selectbox("Select User ID", user_ids)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(int(selected_user), model, trainset, movies, n=10)
    st.subheader("🎯 Top 10 Recommended Movies")
    st.table(recommendations[['movieId', 'title', 'genres']].reset_index(drop=True))

