import os
import pickle
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_STORE_PATH = "./vector_store.pkl"
_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _load_store() -> Dict:
    if os.path.exists(_STORE_PATH):
        with open(_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return {"ids": [], "embeddings": None, "metadatas": []}


def _save_store(store: Dict) -> None:
    with open(_STORE_PATH, "wb") as f:
        pickle.dump(store, f)


# ---------------------------------------------------------------------------
# Core interface
# ---------------------------------------------------------------------------

def already_exists(id: str) -> bool:
    """Return True if the article ID is already in the store."""
    return id in _load_store()["ids"]


def add_article(id: str, text: str, metadata: Dict) -> None:
    """Embed text and persist id + embedding + metadata."""
    store = _load_store()
    if id in store["ids"]:
        return
    embedding = _get_model().encode([text])[0]
    store["ids"].append(id)
    store["metadatas"].append(metadata)
    if store["embeddings"] is None:
        store["embeddings"] = embedding.reshape(1, -1)
    else:
        store["embeddings"] = np.vstack([store["embeddings"], embedding.reshape(1, -1)])
    _save_store(store)


def search(query: str, n_results: int = 5) -> List[Dict]:
    """Return top-N articles by cosine similarity, each with a 'score' field."""
    store = _load_store()
    if not store["ids"] or store["embeddings"] is None:
        return []
    query_vec = _get_model().encode([query])[0]
    embeddings = store["embeddings"]
    denom = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    denom = np.where(denom == 0, 1e-10, denom)
    similarities = np.dot(embeddings, query_vec) / denom
    n = min(n_results, len(store["ids"]))
    top_indices = np.argsort(similarities)[::-1][:n]
    results = []
    for idx in top_indices:
        meta = store["metadatas"][idx].copy()
        meta["score"] = round(float(similarities[idx]), 3)
        results.append(meta)
    return results


def get_all(filters: Optional[Dict] = None) -> List[Dict]:
    """Return all stored articles, optionally filtered by exact metadata values."""
    metadatas = _load_store()["metadatas"]
    if not filters:
        return metadatas
    return [m for m in metadatas if all(m.get(k) == v for k, v in filters.items())]


# ---------------------------------------------------------------------------
# Compatibility wrappers — match the interface called by app.py
# ---------------------------------------------------------------------------

def store(article: Dict, sentiment: Dict) -> bool:
    """Persist article + sentiment. Returns False if already stored."""
    if already_exists(article["id"]):
        return False
    metadata = {
        "title":      article["title"],
        "source":     article["source"],
        "link":       article["link"],
        "published":  article["published"],
        "fetched_at": article["fetched_at"],
        "sentiment":  sentiment["sentiment"],
        "confidence": sentiment["confidence"],
        "reason":     sentiment["reason"],
        "date":       date.today().isoformat(),
    }
    add_article(article["id"], article["title"], metadata)
    return True


def get_today() -> List[Dict]:
    """Return all articles stored today."""
    return get_all(filters={"date": date.today().isoformat()})


def trend(days: int = 14) -> List[Dict]:
    """Daily sentiment counts for the last `days` days."""
    daily: Dict[str, Dict] = defaultdict(lambda: {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0})
    for m in get_all():
        d = m.get("date", "")
        s = m.get("sentiment", "NEUTRAL")
        if d:
            daily[d][s] += 1
    rows = []
    for d, counts in sorted(daily.items())[-days:]:
        total = sum(counts.values()) or 1
        rows.append({
            "date":         d,
            "bullish":      counts["BULLISH"],
            "bearish":      counts["BEARISH"],
            "neutral":      counts["NEUTRAL"],
            "bullish_pct":  round(counts["BULLISH"] / total * 100, 1),
            "bearish_pct":  round(counts["BEARISH"] / total * 100, 1),
            "neutral_pct":  round(counts["NEUTRAL"] / total * 100, 1),
        })
    return rows
