import json
import os
from typing import Dict

from groq import Groq

_client: Groq | None = None


def init_client(api_key: str | None = None) -> None:
    global _client
    _client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))


def _get_client() -> Groq:
    if _client is None:
        init_client()
    return _client


SYSTEM_PROMPT = """\
You are a financial sentiment analyzer specializing in Indian equity markets (NSE/BSE).
Given a news headline, classify its market sentiment as exactly one of:
  BULLISH  — likely to push prices or indices upward
  BEARISH  — likely to push prices or indices downward
  NEUTRAL  — informational; no clear directional impact

Reply with valid JSON only, no extra text:
{"sentiment": "BULLISH|BEARISH|NEUTRAL", "confidence": <0.0–1.0>, "reason": "<≤15 words>"}
"""


def analyze(headline: str) -> Dict:
    """Return sentiment dict with keys: sentiment, confidence, reason."""
    try:
        response = _get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Headline: {headline}"},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "sentiment": result.get("sentiment", "NEUTRAL").upper(),
            "confidence": float(result.get("confidence", 0.5)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"sentiment": "NEUTRAL", "confidence": 0.0, "reason": f"Error: {e}"}
