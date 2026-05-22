import pandas as pd
import numpy as np
import ast
import faiss
import pickle
import requests
import os

from dotenv import load_dotenv

from sentence_transformers import (
    SentenceTransformer
)

# ---------------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------------

load_dotenv()

TMDB_API_KEY = os.getenv(
    "TMDB_API_KEY"
)

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

TOP_MOVIES = 15000

BATCH_SIZE = 8

TMDB_IMAGE_BASE = (
    "https://image.tmdb.org/t/p/w500"
)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def safe_literal_eval(text):

    try:

        return ast.literal_eval(text)

    except Exception:

        return []

def parse_names(data):

    parsed = safe_literal_eval(data)

    return [

        item["name"]

        for item in parsed

        if isinstance(item, dict)
        and "name" in item
    ]

def parse_cast(data):

    parsed = safe_literal_eval(data)

    return [

        actor["name"]

        for actor in parsed[:5]

        if isinstance(actor, dict)
        and "name" in actor
    ]

def parse_director(data):

    parsed = safe_literal_eval(data)

    directors = []

    for item in parsed:

        if (
            isinstance(item, dict)
            and item.get("job") == "Director"
        ):

            directors.append(
                item["name"]
            )

    return directors

# ---------------------------------------------------
# OPTIONAL TMDB POSTER FETCH
# ---------------------------------------------------

def fetch_tmdb_poster(
    title,
    release_date=""
):

    try:

        year = ""

        if pd.notna(release_date):

            year = str(
                release_date
            )[:4]

        url = (
            "https://api.themoviedb.org/"
            "3/search/movie"
        )

        params = {

            "api_key":
            TMDB_API_KEY,

            "query":
            title,

            "year":
            year
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        results = data.get(
            "results",
            []
        )

        if len(results) == 0:

            return None

        poster_path = results[
            0
        ].get("poster_path")

        if poster_path:

            return (
                TMDB_IMAGE_BASE +
                poster_path
            )

    except Exception:

        pass

    return None

# ---------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------

print("Loading datasets...")

movies_df = pd.read_csv(
    "movies_metadata.csv",
    low_memory=False
)

keywords_df = pd.read_csv(
    "keywords.csv"
)

credits_df = pd.read_csv(
    "credits.csv"
)

# ---------------------------------------------------
# FIX IDS
# ---------------------------------------------------

movies_df = movies_df[
    movies_df["id"]
    .astype(str)
    .str.isnumeric()
]

movies_df["id"] = movies_df[
    "id"
].astype(int)

keywords_df["id"] = keywords_df[
    "id"
].astype(int)

credits_df["id"] = credits_df[
    "id"
].astype(int)

# ---------------------------------------------------
# MERGE DATASETS
# ---------------------------------------------------

print("Merging datasets...")

df = movies_df.merge(
    keywords_df,
    on="id",
    how="left"
)

df = df.merge(
    credits_df,
    on="id",
    how="left"
)

# ---------------------------------------------------
# KEEP REQUIRED COLUMNS
# ---------------------------------------------------

df = df[
    [
        "title",
        "overview",
        "genres",
        "poster_path",
        "vote_average",
        "release_date",
        "production_companies",
        "keywords",
        "cast",
        "crew"
    ]
]

# ---------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------

print("Cleaning data...")

fill_columns = [

    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew",
    "production_companies"
]

for col in fill_columns:

    df[col] = df[col].fillna(
        "[]"
    )

# ---------------------------------------------------
# PARSE LIST FEATURES
# ---------------------------------------------------

df["genres_list"] = df[
    "genres"
].apply(parse_names)

df["keywords_list"] = df[
    "keywords"
].apply(parse_names)

df["cast_list"] = df[
    "cast"
].apply(parse_cast)

df["director_list"] = df[
    "crew"
].apply(parse_director)

df["production_companies_list"] = df[
    "production_companies"
].apply(parse_names)

# ---------------------------------------------------
# REMOVE EMPTY OVERVIEW
# ---------------------------------------------------

df = df[
    df["overview"]
    .astype(str)
    .str.strip() != ""
]

# ---------------------------------------------------
# LIMIT MOVIES
# ---------------------------------------------------

df = df.head(
    TOP_MOVIES
)

# ---------------------------------------------------
# RESET INDEX
# ---------------------------------------------------

df = df.reset_index(
    drop=True
)

# ---------------------------------------------------
# CREATE SEARCH TEXT
# ---------------------------------------------------

print("Creating search text...")

df["search_text"] = (

    df["title"].fillna("") + " " +

    df["overview"].fillna("") + " " +

    df["genres_list"].apply(
        lambda x: " ".join(x)
    ) + " " +

    df["keywords_list"].apply(
        lambda x: " ".join(x)
    ) + " " +

    df["cast_list"].apply(
        lambda x: " ".join(x)
    ) + " " +

    df["director_list"].apply(
        lambda x: " ".join(x)
    ) + " " +

    df["production_companies_list"]
    .apply(
        lambda x: " ".join(x)
    )
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

print("Loading model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

# ---------------------------------------------------
# GENERATE EMBEDDINGS
# ---------------------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    df["search_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype(
    np.float16
)

# ---------------------------------------------------
# CREATE FAISS INDEX
# ---------------------------------------------------

print("Creating FAISS index...")

faiss_embeddings = embeddings.astype(
    np.float32
)

index = faiss.IndexFlatIP(
    faiss_embeddings.shape[1]
)

index.add(
    faiss_embeddings
)

# ---------------------------------------------------
# REMOVE LARGE TEMP COLUMN
# ---------------------------------------------------

df.drop(
    columns=["search_text"],
    inplace=True
)

# ---------------------------------------------------
# CONVERT LIST COLUMNS
# ---------------------------------------------------

list_columns = [

    "genres_list",
    "keywords_list",
    "cast_list",
    "director_list",
    "production_companies_list"
]

for col in list_columns:

    df[col] = df[col].apply(
        lambda x: str(x)
    )

# ---------------------------------------------------
# SAVE FAISS INDEX
# ---------------------------------------------------

print("Saving FAISS index...")

faiss.write_index(
    index,
    "movie_index.faiss"
)

# ---------------------------------------------------
# SAVE PARQUET FILE
# ---------------------------------------------------

print("Saving parquet file...")

df.to_parquet(
    "processed_movies.parquet",
    index=False
)

# ---------------------------------------------------
# SAVE EMBEDDINGS
# ---------------------------------------------------

print("Saving embeddings...")

np.save(
    "embeddings.npy",
    embeddings
)

# ---------------------------------------------------
# SAVE MODEL INFO
# ---------------------------------------------------

print("Saving model info...")

with open(
    "model_info.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "model_name": MODEL_NAME
        },
        f
    )

# ---------------------------------------------------
# DONE
# ---------------------------------------------------

print("\nDone.")
print(
    "Saved files:"
)

print(
    "- processed_movies.parquet"
)

print(
    "- movie_index.faiss"
)

print(
    "- embeddings.npy"
)

print(
    "- model_info.pkl"
)