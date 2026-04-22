"""
therapy_ai/ui/pages/progress.py

Progress dashboard: mood trends, session history, and fine-tune export.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from therapy_ai.memory.session_store import SessionStore
from therapy_ai.memory.exporter import FinetuneExporter


def render() -> None:
    store: SessionStore = st.session_state.store

    st.markdown("## 📊 Your Progress")

    # ── Mood chart ────────────────────────────────────────────────────────────
    mood_data = store.mood_series()
    if mood_data:
        df = pd.DataFrame(mood_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["mood_before"],
            name="Before session", mode="lines+markers",
            line=dict(color="#6c9bcf", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["mood_after"],
            name="After session", mode="lines+markers",
            line=dict(color="#5ab08d", width=2),
        ))
        fig.update_layout(
            title="Mood Over Time",
            yaxis=dict(title="Mood (1–10)", range=[0, 11]),
            xaxis=dict(title="Date"),
            legend=dict(orientation="h"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Complete a few sessions to see your mood trend here.")

    st.divider()

    # ── Session history ───────────────────────────────────────────────────────
    st.markdown("### Session History")
    sessions = store.list_sessions(limit=30)

    if not sessions:
        st.info("No sessions yet.")
        return

    for s in sessions:
        label = f"{s.created_at.strftime('%Y-%m-%d %H:%M')} — Mood: {s.mood_before}→{s.mood_after or '?'}"
        if s.rating:
            label += f"  ({'★' * s.rating}{'☆' * (5 - s.rating)})"
        with st.expander(label):
            if s.themes:
                st.markdown(f"**Themes:** {', '.join(s.themes)}")
            if s.note:
                st.markdown(f"**Note:** {s.note}")
            if st.button(f"🗑️ Delete session", key=f"del_{s.id}"):
                store.delete_session(s.id)
                st.success("Session deleted.")
                st.rerun()

    st.divider()

    # ── Fine-tune export ──────────────────────────────────────────────────────
    st.markdown("### Export for Fine-Tuning")
    st.caption(
        "When you have 50+ sessions, you can export your data to train a "
        "personalised LoRA adapter."
    )
    col1, col2 = st.columns(2)
    with col1:
        min_rating = st.selectbox(
            "Minimum session rating to include",
            options=[None, 3, 4, 5],
            format_func=lambda x: "All sessions" if x is None else f"{x}+ stars",
        )
    with col2:
        val_split = st.slider("Validation split", 0.05, 0.30, 0.15, 0.05)

    if st.button("Export JSONL", type="primary"):
        exporter = FinetuneExporter(store)
        counts = exporter.export(
            val_split=val_split,
            min_rating=min_rating,
        )
        st.success(
            f"Exported {counts['train']} train / {counts['val']} val records "
            f"→ `data/finetune/`"
        )
