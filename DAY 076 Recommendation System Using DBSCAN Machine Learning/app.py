import streamlit as st
import pandas as pd
import random

# Load the movie data
df_movies = pd.read_csv("clustered_movies.csv")

# Function to recommend movies
def recommend_movie(movie_name: str, n_recommendations=5):
    # Convert the input movie name to lowercase for case-insensitive matching
    movie_name = movie_name.lower()

    # Create a new column 'name' with lowercase movie names for comparison
    df_movies['name'] = df_movies['name'].str.lower()

    # Find the movie that matches the input name
    movie = df_movies[df_movies['name'].str.contains(movie_name, na=False)]

    if not movie.empty:
        # Get the cluster label of the input movie
        cluster = movie['dbscan_clusters'].values[0]

        # Get all movies in the same cluster
        cluster_movies = df_movies[df_movies['dbscan_clusters'] == cluster]

        # If there are more than the requested number of movies, randomly select them
        if len(cluster_movies) >= n_recommendations:
            recommended_movies = random.sample(list(cluster_movies['name']), n_recommendations)
        else:
            # If fewer, return all the movies in the cluster
            recommended_movies = list(cluster_movies['name'])

        return recommended_movies
    else:
        return ["Movie not found in the database."]


#================================================Streamlit UI Creation==================================================
# Streamlit App UI
st.title("Movie Recommendation System Using DBSCAN Clustering")
st.write("---------------------------------------------------------")
st.subheader("Support Databases:")
st.write("1: Netflix TV Shows and Movies")
st.write("1: HBO Max TV Shows and Movies")
st.write("1: Amazon Prime TV Shows and Movies")
st.write('---------------------------------------------------------')
# Searchable dropdown for movie name
# Get the list of movie names for the dropdown
movie_names = df_movies['name'].values
movie_name = st.selectbox("Search for a movie you like:", options=movie_names)

# Button to trigger the recommendation
if st.button("Recommend Movies"):
    st.write("### We recommend you these movies:")
    recommendations = recommend_movie(movie_name)
    st.dataframe(recommendations)

