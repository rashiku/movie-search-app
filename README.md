# 🎬 AI Semantic Movie Search App

An AI-powered movie recommendation and semantic movie search application built using Streamlit, Sentence Transformers, FAISS, BM25, and TMDB APIs.

The app allows users to:

- Search movies using natural language descriptions
- Get semantically similar movie recommendations
- Combine semantic search + keyword search + fuzzy matching
- View movie posters automatically fetched from TMDB
- Cache posters locally for faster performance

---

# 🚀 Features

## ✅ AI Semantic Search

Search movies using descriptions like:

- "mind bending sci fi movies"
- "dark psychological thriller"
- "emotional space adventure"

The app uses Sentence Transformers embeddings with FAISS similarity search.

---

## ✅ Hybrid Recommendation System

The recommendation score combines:

| Method | Weight |
|---|---|
| Semantic Similarity | 55% |
| BM25 Keyword Search | 30% |
| Fuzzy Matching | 15% |

This improves both relevance and search robustness.

---

## ✅ Movie Poster Fetching

Movie posters are automatically fetched from TMDB API.

Features:

- Local poster caching
- Faster repeated loading
- Reduced API calls

---

## ✅ Multiple Search Modes

The app supports:

- Search by movie description
- Search by movie title
- Recommendation from selected movie
- Recommendation from typed movie name

---

## ✅ Fast Retrieval Using FAISS

FAISS is used for efficient vector similarity search over movie embeddings.

---

## ✅ Optimized Storage Using Parquet

The project uses Parquet instead of CSV for:

- Smaller file sizes
- Faster loading
- GitHub compatibility
- Better memory efficiency

---

# 🛠️ Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- BM25
- RapidFuzz
- Pandas
- NumPy
- PyArrow
- TMDB API

---

# 📂 Project Structure

```bash
movie-search-app/
│
├── app.py
├── generate_embeddings.py
├── update_posters.py
├── requirements.txt
├── README.md
│
├── processed_movies.parquet
├── movie_index.faiss
├── embeddings.npy
├── model_info.pkl
│
├── poster_cache.json
│
└── .streamlit/
    └── secrets.toml
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd movie-search-app
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 TMDB API Setup

Create a TMDB API key from:

https://www.themoviedb.org/settings/api

---

## Create `.streamlit/secrets.toml`

```toml
TMDB_API_KEY = "your_tmdb_api_key"
```

---

# 🧠 Generate Embeddings

Run:

```bash
python generate_embeddings.py
```

This script:

- Cleans movie metadata
- Generates embeddings
- Builds FAISS index
- Saves optimized parquet dataset

Generated files:

- `processed_movies.parquet`
- `movie_index.faiss`
- `embeddings.npy`
- `model_info.pkl`

---

# ▶️ Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

# 🖼️ Update Posters Automatically

Run:

```bash
python update_posters.py
```

This script automatically:

- Fetches posters for all movies
- Stores them in local cache
- Avoids manual dropdown selection

---

# 📸 Example Searches

Try searching:

- "space survival movie"
- "psychological thriller with twist ending"
- "emotional animated movie"
- "crime mafia family drama"

---

# 📈 Future Improvements

Possible future upgrades:

- Genre filtering improvements
- Actor/director search
- Advanced ranking models
- Collaborative filtering
- User watchlists
- LLM-powered movie explanations
- Cloud deployment

---

# 🌐 Deployment

You can deploy easily on:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Railway

---

# 🙌 Acknowledgements

- TMDB API
- Streamlit
- Sentence Transformers
- FAISS by Meta
- RapidFuzz

---

# 📜 License

This project is for educational and learning purposes.