import hashlib
from datetime import datetime
from typing import List, Dict

import feedparser

RSS_FEEDS = {
    "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "ET Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}


def fetch_headlines() -> List[Dict]:
    """Fetch latest headlines from all configured RSS feeds."""
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                link = entry.get("link", "")
                summary = entry.get("summary", "")
                published = entry.get("published", datetime.now().isoformat())

                articles.append({
                    "id": hashlib.md5(link.encode()).hexdigest(),
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "source": source,
                    "published": published,
                    "fetched_at": datetime.now().isoformat(),
                })
        except Exception as e:
            print(f"[fetcher] Error fetching {source}: {e}")

    return articles
