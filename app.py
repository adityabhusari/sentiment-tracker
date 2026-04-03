import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fetcher
import sentiment as sa
import vector_store as vs

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Market Sentiment Tracker",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Indian Market Sentiment Tracker")
st.caption("Powered by Moneycontrol · ET Markets · Groq Llama 3.3 · ChromaDB")

# ---------------------------------------------------------------------------
# Sidebar — API key + refresh
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at console.groq.com",
    )
    if api_key:
        sa.init_client(api_key)

    st.divider()
    st.subheader("Fetch & Analyze")
    run_btn = st.button("Fetch Latest Headlines", type="primary", use_container_width=True)
    st.caption("Pulls RSS feeds, sends each headline to Llama 3.3, and stores results.")

# ---------------------------------------------------------------------------
# Pipeline — run when button clicked
# ---------------------------------------------------------------------------
if run_btn:
    if not api_key:
        st.sidebar.error("Enter your Groq API key first.")
    else:
        with st.spinner("Fetching headlines…"):
            articles = fetcher.fetch_headlines()

        if not articles:
            st.warning("No articles fetched. Check your internet connection.")
        else:
            progress = st.progress(0, text="Analyzing sentiment…")
            new_count = 0
            for i, article in enumerate(articles):
                result = sa.analyze(article["title"])
                stored = vs.store(article, result)
                if stored:
                    new_count += 1
                progress.progress((i + 1) / len(articles), text=f"Analyzed {i+1}/{len(articles)}")
                time.sleep(0.05)          # avoid Groq rate-limit bursts
            progress.empty()
            st.sidebar.success(f"Done — {new_count} new articles stored ({len(articles)} fetched).")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔥 Today's Pulse", "🔍 Search History", "📊 Trend Chart"])

# ── Tab 1: Today's Pulse ──────────────────────────────────────────────────
with tab1:
    articles_today = vs.get_today()

    if not articles_today:
        st.info("No articles for today yet. Click **Fetch Latest Headlines** in the sidebar.")
    else:
        bullish = [a for a in articles_today if a["sentiment"] == "BULLISH"]
        bearish = [a for a in articles_today if a["sentiment"] == "BEARISH"]
        neutral = [a for a in articles_today if a["sentiment"] == "NEUTRAL"]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", len(articles_today))
        c2.metric("Bullish 🟢", len(bullish))
        c3.metric("Bearish 🔴", len(bearish))
        c4.metric("Neutral ⚪", len(neutral))

        # Donut chart
        fig = go.Figure(go.Pie(
            labels=["Bullish", "Bearish", "Neutral"],
            values=[len(bullish), len(bearish), len(neutral)],
            hole=0.55,
            marker_colors=["#22c55e", "#ef4444", "#94a3b8"],
            textinfo="label+percent",
        ))
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Filter
        filter_choice = st.radio(
            "Show", ["All", "Bullish", "Bearish", "Neutral"], horizontal=True
        )
        filtered = articles_today if filter_choice == "All" else [
            a for a in articles_today if a["sentiment"] == filter_choice.upper()
        ]

        BADGE = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}

        for art in sorted(filtered, key=lambda x: x.get("fetched_at", ""), reverse=True):
            badge = BADGE.get(art["sentiment"], "⚪")
            conf = int(art.get("confidence", 0) * 100)
            with st.expander(f"{badge} {art['title']}", expanded=False):
                cols = st.columns([2, 1, 1])
                cols[0].markdown(f"**Source:** {art['source']}")
                cols[1].markdown(f"**Confidence:** {conf}%")
                cols[2].markdown(f"**Sentiment:** {art['sentiment']}")
                st.markdown(f"**Reason:** {art.get('reason', '—')}")
                st.markdown(f"[Read full article]({art['link']})")

# ── Tab 2: Semantic Search ────────────────────────────────────────────────
with tab2:
    st.subheader("Search Stored News Semantically")
    query = st.text_input("Search query", placeholder="e.g. RBI interest rate decision")
    n_results = st.slider("Max results", 3, 20, 8)

    if query:
        with st.spinner("Searching…"):
            results = vs.search(query, n=n_results)

        if not results:
            st.info("No results found. Fetch some articles first.")
        else:
            BADGE = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
            for r in results:
                badge = BADGE.get(r["sentiment"], "⚪")
                score_pct = int(r["score"] * 100)
                with st.expander(f"{badge} {r['title']} — relevance {score_pct}%"):
                    cols = st.columns([2, 1, 1, 1])
                    cols[0].markdown(f"**Source:** {r['source']}")
                    cols[1].markdown(f"**Date:** {r.get('date', '—')}")
                    cols[2].markdown(f"**Sentiment:** {r['sentiment']}")
                    cols[3].markdown(f"**Confidence:** {int(r.get('confidence', 0)*100)}%")
                    st.markdown(f"**Reason:** {r.get('reason', '—')}")
                    st.markdown(f"[Read full article]({r['link']})")

# ── Tab 3: Trend Chart ────────────────────────────────────────────────────
with tab3:
    st.subheader("Sentiment Trend Over Time")
    days = st.slider("Days of history", 3, 30, 14)
    trend_data = vs.trend(days=days)

    if not trend_data:
        st.info("Not enough data yet. Fetch articles on multiple days to see trends.")
    else:
        df = pd.DataFrame(trend_data)

        # Stacked bar — absolute counts
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df["date"], y=df["bullish"],
            name="Bullish", marker_color="#22c55e",
        ))
        fig_bar.add_trace(go.Bar(
            x=df["date"], y=df["bearish"],
            name="Bearish", marker_color="#ef4444",
        ))
        fig_bar.add_trace(go.Bar(
            x=df["date"], y=df["neutral"],
            name="Neutral", marker_color="#94a3b8",
        ))
        fig_bar.update_layout(
            barmode="stack",
            title="Daily Article Count by Sentiment",
            xaxis_title="Date",
            yaxis_title="Articles",
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=60),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Line chart — percentage
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df["date"], y=df["bullish_pct"],
            name="Bullish %", line=dict(color="#22c55e", width=2),
            mode="lines+markers",
        ))
        fig_line.add_trace(go.Scatter(
            x=df["date"], y=df["bearish_pct"],
            name="Bearish %", line=dict(color="#ef4444", width=2),
            mode="lines+markers",
        ))
        fig_line.add_trace(go.Scatter(
            x=df["date"], y=df["neutral_pct"],
            name="Neutral %", line=dict(color="#94a3b8", width=2),
            mode="lines+markers",
        ))
        fig_line.update_layout(
            title="Daily Sentiment Share (%)",
            xaxis_title="Date",
            yaxis_title="Share (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=60),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.dataframe(
            df[["date", "bullish", "bearish", "neutral", "bullish_pct", "bearish_pct", "neutral_pct"]],
            use_container_width=True,
            hide_index=True,
        )
