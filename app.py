import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("movie_small.pkl", "rb") as file:
        return pickle.load(file)

data = load_model()
titles = data["titles"]
vectors = data["vectors"]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f1020, #251337);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #6c35de, #e83e8c);
    text-align: center;
    margin-bottom: 25px;
}
.movie-card {
    background: #29243b;
    border: 1px solid #514368;
    border-radius: 14px;
    padding: 20px;
    margin: 8px 0;
}
.stButton > button {
    background: #e83e8c;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Discover your next favourite movie!</p>
</div>
""", unsafe_allow_html=True)

st.subheader("🍿 Find your next movie")

selected_movie = st.selectbox(
    "Choose a movie",
    titles
)

number = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("✨ Recommend Movies"):
    idx = titles.index(selected_movie)

    scores = cosine_similarity(
        vectors[idx], vectors
    ).flatten()

    scores[idx] = -1
    best_indices = scores.argsort()[::-1][:number]

    st.success(f"Movies similar to {selected_movie}")

    cols = st.columns(2)

    for j, movie_idx in enumerate(best_indices):
        with cols[j % 2]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <h3>🎥 {titles[movie_idx]}</h3>
                    <p>Recommended based on cast and director.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.caption("CineMatch | Content-Based Movie Recommendation System")
