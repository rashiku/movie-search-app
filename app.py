import streamlit as st
import pandas as pd
import numpy as np
import ast
import faiss
import pickle
import requests
import json
import os
import re

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from rapidfuzz import fuzz
from rapidfuzz import process

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------

load_dotenv()

TMDB_API_KEY = (
    st.secrets.get("TMDB_API_KEY")
    or os.getenv("TMDB_API_KEY")
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Movie Search",
    layout="wide"
)

st.title(
    "🎬 AI Semantic Movie Search"
)

# ---------------------------------------------------
# CONSTANTS
# ---------------------------------------------------

SEMANTIC_WEIGHT = 0.55
BM25_WEIGHT = 0.30
FUZZY_WEIGHT = 0.15

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
POSTER_CACHE_FILE = "poster_cache.json"
REQUEST_TIMEOUT = 10

# ---------------------------------------------------
# REQUEST SESSION
# ---------------------------------------------------

session = requests.Session()

# ---------------------------------------------------
# CACHE KEY HELPER
# ---------------------------------------------------

def create_cache_key(title, release_date):
    year = ""
    if pd.notna(release_date):
        year = str(release_date)[:4]
    return title.strip().lower() + "_" + (year if year else "unknown")

# ---------------------------------------------------
# SAFE LITERAL EVAL
# ---------------------------------------------------

def safe_literal_eval(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []

# ---------------------------------------------------
# LOAD / SAVE CACHE
# ---------------------------------------------------

def load_poster_cache():
    if os.path.exists(POSTER_CACHE_FILE):
        with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_poster_cache(cache):
    with open(POSTER_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

poster_cache = load_poster_cache()

# ---------------------------------------------------
# TMDB POSTER FETCH
# ---------------------------------------------------

def fetch_tmdb_poster(title, release_date=""):
    cache_key = create_cache_key(title, release_date)
    if cache_key in poster_cache:
        return poster_cache[cache_key]
    year = ""
    if pd.notna(release_date):
        year = str(release_date)[:4]
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title, "year": year}
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        poster_url = None
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                poster_url = POSTER_BASE_URL + poster_path
        poster_cache[cache_key] = poster_url
        save_poster_cache(poster_cache)
        return poster_url
    except Exception as e:
        print(f"Error fetching {title}: {e}")
    poster_cache[cache_key] = None
    save_poster_cache(poster_cache)
    return None

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_parquet("processed_movies.parquet")
    list_columns = ["genres_list", "keywords_list", "cast_list", "director_list", "production_companies_list"]

    for col in list_columns:
        df[col] = df[col].apply(safe_literal_eval)
        df[col] = df[col].apply(lambda x: tuple(str(item) for item in x if not isinstance(item, list)) if isinstance(x, list) else tuple())

    # Weighted search text
    df["search_text"] = (
        (df["title"].fillna("") + " ") * 3 +
        (df["overview"].fillna("") + " ") * 1 +
        df["genres_list"].apply(lambda x: (" ".join(x) + " ") * 3) +
        df["keywords_list"].apply(lambda x: (" ".join(x) + " ") * 4) +
        df["cast_list"].apply(lambda x: (" ".join(x) + " ") * 2) +
        df["director_list"].apply(lambda x: (" ".join(x) + " ") * 3) +
        df["production_companies_list"].apply(lambda x: (" ".join(x) + " ") * 2)
    )
    return df

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    with open("model_info.pkl", "rb") as f:
        info = pickle.load(f)
    return SentenceTransformer(info["model_name"], device="cpu")

# ---------------------------------------------------
# LOAD FAISS
# ---------------------------------------------------

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("movie_index.faiss")

# ---------------------------------------------------
# BUILD BM25
# ---------------------------------------------------

@st.cache_resource
def build_bm25(df):
    tokenized_corpus = [re.findall(r"\w+", text.lower()) for text in df["search_text"]]
    return BM25Okapi(tokenized_corpus)

# ---------------------------------------------------
# INITIALIZE
# ---------------------------------------------------

df = load_data()
model = load_model()
faiss_index = load_faiss_index()
bm25 = build_bm25(df)
movie_titles = [str(title) for title in df["title"].fillna("")]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Search Settings")

top_k = st.sidebar.slider("Top-K Results", min_value=5, max_value=50, value=15, step=5)
rating_range = st.sidebar.slider("⭐ IMDb Rating Range", min_value=0.0, max_value=10.0, value=(5.0, 10.0), step=0.1)

all_genres = sorted({str(genre) for genres in df["genres_list"] if isinstance(genres, (list, tuple)) for genre in genres if not isinstance(genre, list)})
selected_genre = st.selectbox("🎭 Filter by genre", ["All"] + all_genres)

search_mode = st.radio("Mode", ["Semantic Search", "Movie Recommendations"])

# ---------------------------------------------------
# FUZZY TITLE MATCHING
# ---------------------------------------------------

def find_best_title_match(user_input, titles, threshold=70):
    result = process.extractOne(user_input, titles, scorer=fuzz.ratio)
    if result is None:
        return None
    matched_title, score, _ = result
    if score >= threshold:
        return matched_title
    return None

# ---------------------------------------------------
# HYBRID SEARCH
# ---------------------------------------------------

def hybrid_search(query, top_k=15, genre_filter="All", rating_range=(0.0, 10.0)):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    candidate_count = max(top_k * 50, 500)

    semantic_scores, semantic_indices = faiss_index.search(query_embedding, candidate_count)
    semantic_scores = semantic_scores[0]
    semantic_indices = semantic_indices[0]

    candidate_df = df.iloc[semantic_indices].copy()

    tokenized_query = re.findall(r"\w+", query.lower())
    all_bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores = np.array(all_bm25_scores[semantic_indices], dtype=np.float32)

    fuzzy_scores = np.array([fuzz.partial_ratio(query.lower(), title.lower()) for title in candidate_df["title"]], dtype=np.float32) / 100.0

    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

    final_scores = SEMANTIC_WEIGHT * semantic_scores + BM25_WEIGHT * bm25_scores + FUZZY_WEIGHT * fuzzy_scores
    candidate_df["score"] = final_scores

    # ---------------------------------------------------
    # GENRE FILTER
    # ---------------------------------------------------
    if genre_filter != "All":
        candidate_df = candidate_df[candidate_df["genres_list"].apply(lambda x: isinstance(x, (list, tuple)) and genre_filter in x)]

    # ---------------------------------------------------
    # RATING FILTER
    # ---------------------------------------------------
    candidate_df = candidate_df[candidate_df["vote_average"].fillna(0).between(rating_range[0], rating_range[1])]

    candidate_df = candidate_df.sort_values(by="score", ascending=False)
    return candidate_df.head(top_k)

# ---------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------

def recommend_movies(movie_title, top_k=15, rating_range=(0.0, 10.0)):
    matched_title = find_best_title_match(movie_title, movie_titles)
    if matched_title is None:
        st.error("Movie not found.")
        return pd.DataFrame()
    if matched_title != movie_title:
        st.info(f"Showing results for: {matched_title}")
    movie_title = matched_title

    matching_movies = df[df["title"].str.casefold() == movie_title.casefold()]
    if len(matching_movies) == 0:
        return pd.DataFrame()

    idx = matching_movies.index[0]
    query_embedding = np.expand_dims(faiss_index.reconstruct(int(idx)).astype(np.float32), axis=0)
    scores, indices = faiss_index.search(query_embedding, top_k + 1)
    results = df.iloc[indices[0]].copy()
    results["score"] = scores[0]
    results = results[results["title"] != movie_title]

    results = results[results["vote_average"].fillna(0).between(rating_range[0], rating_range[1])]
    return results.head(top_k)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

results = None

if search_mode == "Semantic Search":
    query = st.text_input("Describe the movie you want:")
    if query:
        with st.spinner("Searching movies..."):
            results = hybrid_search(query=query, top_k=top_k, genre_filter=selected_genre, rating_range=rating_range)
else:
    recommendation_input_mode = st.radio("Recommendation Input Method", ["Dropdown Selection", "Type Movie Name"])
    selected_movie = None
    if recommendation_input_mode == "Dropdown Selection":
        selected_movie = st.selectbox("Choose a movie", sorted(movie_titles))
    else:
        selected_movie = st.text_input("Type movie name:")

    if selected_movie:
        with st.spinner("Finding similar movies..."):
            results = recommend_movies(selected_movie, top_k=top_k, rating_range=rating_range)

# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------

if results is not None and len(results) > 0:
    st.subheader(f"Top {len(results)} Results")
    for i in range(0, len(results), 5):
        row_results = results.iloc[i:i+5]
        cols = st.columns(5)
        for col, (_, row) in zip(cols, row_results.iterrows()):
            with col:
                poster_url = fetch_tmdb_poster(row["title"], row["release_date"])
                if poster_url:
                    st.image(poster_url, width=220)
                st.subheader(row["title"])
                st.caption(f"⭐ {row['vote_average']}")
                st.caption(f"📅 {row['release_date']}")
                genres = row["genres_list"]
                if isinstance(genres, (list, tuple)):
                    st.caption(" | ".join(genres))
                st.write(str(row["overview"])[:220] + "...")
                st.caption(f"Score: {row['score']:.3f}")