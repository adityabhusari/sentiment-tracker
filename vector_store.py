from collections import defaultdict
from datetime import date
from typing import Dict, List

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_DB_PATH = "./chroma_db"
_COLLECTION = "market_sentiment"

_ef = DefaultEmbeddingFunction()


def _col():
    client = chromadb.PersistentClient(path=_DB_PATH)
    return client.get_or_create_collection(name=_COLLECTION, embedding_function=_ef)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def store(article: Dict, sentiment: Dict) -> bool:
    """Persist article + sentiment. Returns False if it already existed."""
    col = _col()
    if col.get(ids=[article["id"]])["ids"]:
        return False

    col.add(
        ids=[article["id"]],
        documents=[article["title"]],
        metadatas=[{
            "title": article["title"],
            "source": article["source"],
            "link": article["link"],
            "published": article["published"],
            "fetched_at": article["fetched_at"],
            "sentiment": sentiment["sentiment"],
            "confidence": sentiment["confidence"],
            "reason": sentiment["reason"],
            "date": date.today().isoformat(),
        }],
    )
    return True


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_today() -> List[Dict]:
    """Return all articles stored today."""
    col = _col()
    results = col.get(where={"date": date.today().isoformat()}, include=["metadatas"])
    return results["metadatas"] or []


def search(query: str, n: int = 10) -> List[Dict]:
    """Semantic search across all stored articles."""
    col = _col()
    total = col.count()
    if total == 0:
        return []
    results = col.query(
        query_texts=[query],
        n_results=min(n, total),
        include=["metadatas", "distances"],
    )
    rows = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        rows.append({**meta, "score": round(max(0.0, 1 - dist), 3)})
    return rows


def trend(days: int = 14) -> List[Dict]:
    """Daily sentiment counts for the last `days` days."""
    col = _col()
    all_meta = col.get(include=["metadatas"])["metadatas"]

    daily: Dict[str, Dict] = defaultdict(lambda: {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0})
    for m in all_meta:
        d = m.get("date", "")
        s = m.get("sentiment", "NEUTRAL")
        if d:
            daily[d][s] += 1

    rows = []
    for d, counts in sorted(daily.items())[-days:]:
        total = sum(counts.values()) or 1
        rows.append({
            "date": d,
            "bullish": counts["BULLISH"],
            "bearish": counts["BEARISH"],
            "neutral": counts["NEUTRAL"],
            "bullish_pct": round(counts["BULLISH"] / total * 100, 1),
            "bearish_pct": round(counts["BEARISH"] / total * 100, 1),
            "neutral_pct": round(counts["NEUTRAL"] / total * 100, 1),
        })
    return rows
