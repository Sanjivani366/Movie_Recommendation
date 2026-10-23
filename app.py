import streamlit as st
import pickle
import html
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b0b16, #21112f);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #6c35de, #e83e8c);
    padding: 35px 20px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.hero h1 {
    color: white !important;
    font-size: 44px;
    margin-bottom: 8px;
}

.hero p {
    color: white !important;
    font-size: 18px;
}

.movie-card {
    background: linear-gradient(145deg, #29243b, #171525);
    border: 1px solid #514368;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 20px;
    min-height: 220px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 12px;
    display: block;
    background: #20152f;
}

.movie-title {
    color: #ffffff !important;
    font-size: 19px;
    font-weight: 700;
    margin-top: 14px;
    margin-bottom: 8px;
}

.movie-description {
    color: #d0c7e5 !important;
    font-size: 13px;
}

div.stButton > button {
    background: linear-gradient(90deg, #e83e8c, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    border: 1px solid white;
    color: white;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("movie_small.pkl", "rb") as file:
        return pickle.load(file)

try:
    data = load_model()
    titles = data["titles"]
    vectors = data["vectors"]
except Exception as error:
    st.error("Model load झाला नाही. movie_small.pkl file तपास.")
    st.exception(error)
    st.stop()


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Discover your next favourite movie!</p>
    <p>✨ Find movies based on similar cast and directors</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SEARCH SECTION ----------------
st.subheader("🍿 Find Your Next Movie")
st.write("तुला आवडणारा movie निवड आणि त्याच्यासारखे movies शोध.")

selected_movie = st.selectbox(
    "🎥 Select a movie",
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
        vectors[idx],
        vectors
    ).flatten()

    # Selected movie स्वतः recommendation मध्ये येऊ नये
    scores[idx] = -1

    best_indices = scores.argsort()[::-1][:number]

    st.markdown("---")
    st.subheader(f"🎯 Movies similar to {selected_movie}")
    st.write("तुझ्यासाठी निवडलेले movies 🍿")

    cols = st.columns(3)

    for j, movie_idx in enumerate(best_indices):
        movie_title = titles[movie_idx]
        safe_title = html.escape(str(movie_title))
        image_title = html.escape(str(movie_title), quote=True)

        # Placeholder image: movie title shown on the poster
        poster_url = (
            "https://placehold.co/400x600/20152f/ffffff"
            "?font=roboto&text=" +
            html.escape(str(movie_title).replace(" ", "+"), quote=True)
        )

        with cols[j % 3]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <img
                        class="poster"
                        src="{poster_url}"
                        alt="{image_title}"
                    >
                    <div class="movie-title">{safe_title}</div>
                    <div class="movie-description">
                        🎭 Recommended based on cast and director
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center style='color:#bdb1d6;'>"
    "🎬 CineMatch | Content-Based Movie Recommendation System"
    "<br>Made with Python & Streamlit ❤️"
    "</center>",
    unsafe_allow_html=True
)

