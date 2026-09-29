"""
rag_utils.py
RAG corpus building (text descriptions of historical 30-day windows,
embedded with TF-IDF and indexed with FAISS) and retrieval.
"""

import json
import os

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_PATH = "index.faiss"
VECTORIZER_PATH = "vectorizer.joblib"
DOCUMENTS_PATH = "documents.json"
METADATA_PATH = "metadata.json"


def describe_window(window: pd.DataFrame) -> str:
    """Produce a short, deterministic text description of a value window,
    used both to build the RAG corpus and to describe the query window at
    inference time - the two must use the same description function so
    they land in the same embedding space."""
    values = window["value"]
    change_pct = ((values.iloc[-1] - values.iloc[0]) / (abs(values.iloc[0]) + 1e-9)) * 100
    direction = "up" if change_pct > 1 else ("down" if change_pct < -1 else "flat")
    return (
        f"Window from {window['date'].iloc[0].date()} to {window['date'].iloc[-1].date()}. "
        f"Trend {direction} ({change_pct:+.1f}%). "
        f"Average value {values.mean():.2f}, min {values.min():.2f}, max {values.max():.2f}."
    )


def build_and_save_index(df: pd.DataFrame, output_dir: str, window: int = 30) -> int:
    """Build a TF-IDF + FAISS index over rolling `window`-sized historical
    slices of the series, and save it to `output_dir`. Returns the number
    of windows indexed."""
    docs, meta = [], []
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window:i]
        docs.append(describe_window(w))
        meta.append({
            "start_date": str(w["date"].iloc[0].date()),
            "end_date": str(w["date"].iloc[-1].date()),
        })

    if not docs:
        raise ValueError(f"Not enough rows ({len(df)}) to build even one {window}-day window.")

    vectorizer = TfidfVectorizer(max_features=256)
    matrix = vectorizer.fit_transform(docs).toarray().astype("float32")

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, INDEX_PATH))
    joblib.dump(vectorizer, os.path.join(output_dir, VECTORIZER_PATH))
    with open(os.path.join(output_dir, DOCUMENTS_PATH), "w") as f:
        json.dump(docs, f)
    with open(os.path.join(output_dir, METADATA_PATH), "w") as f:
        json.dump(meta, f)

    return len(docs)


def load_index(vector_store_dir: str):
    """Returns (index, vectorizer, docs, meta), or raises if any file is
    missing/corrupt - callers should catch and treat RAG as unavailable."""
    index = faiss.read_index(os.path.join(vector_store_dir, INDEX_PATH))
    vectorizer = joblib.load(os.path.join(vector_store_dir, VECTORIZER_PATH))
    with open(os.path.join(vector_store_dir, DOCUMENTS_PATH)) as f:
        docs = json.load(f)
    with open(os.path.join(vector_store_dir, METADATA_PATH)) as f:
        meta = json.load(f)
    return index, vectorizer, docs, meta


def retrieve_top_k(query_text: str, index, vectorizer, docs, meta, k: int = 3):
    """Returns the k most similar historical windows to `query_text`, each
    as {"text", "start_date", "end_date", "distance"} - lower distance
    means more similar (FAISS L2 distance on the TF-IDF vectors)."""
    query_vec = vectorizer.transform([query_text]).toarray().astype("float32")
    k = min(k, len(docs))
    distances, indices = index.search(query_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": docs[idx],
            "start_date": meta[idx]["start_date"],
            "end_date": meta[idx]["end_date"],
            "distance": float(dist),
        })
    return results
