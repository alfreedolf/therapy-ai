"""
therapy_ai/ui/pages/chat.py

Main therapy chat page.
Handles session lifecycle, streaming generation, safety routing,
mood check-in, and session close flow.
"""

from __future__ import annotations

import time
import streamlit as st

from therapy_ai.core.safety import SafetyLevel
from therapy_ai.domain import Message
from therapy_ai.memory.session_store import SessionStore


def render() -> None:
    store:   SessionStore = st.session_state.store
    engine                = st.session_state.engine
    safety                = st.session_state.safety
    builder               = st.session_state.context_builder

    st.markdown("## 💬 Therapy Session")

    # ── Start / continue session ──────────────────────────────────────────────
    if st.session_state.current_session is None:
        _start_session_ui(store)
        return

    session_id = st.session_state.current_session

    # ── Render chat history ───────────────────────────────────────────────────
    messages = store.load_messages(session_id)
    for msg in messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    # ── Input bar ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("How are you feeling today?")

    if user_input:
        # Safety check BEFORE passing to LLM
        safety_result = safety.check_input(user_input)

        if safety_result.level == SafetyLevel.BLOCKED:
            st.error("⛔ " + safety_result.message)
            return

        if safety_result.level == SafetyLevel.INJECTION:
            st.warning("⚠️ " + safety_result.message)
            return

        if safety_result.level == SafetyLevel.CRISIS:
            _render_crisis_banner(safety_result.message)
            # Still store the message but flag it
            store.add_message(session_id, "user", user_input, is_flagged=True)
            with st.chat_message("user"):
                st.markdown(user_input)
            # Don't continue to LLM — show resources only
            return

        # Normal path: persist user message
        store.add_message(session_id, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build context — pass user_input for semantic memory retrieval
        context = builder.build(session_id, latest_user_text=user_input)
        context.append(Message(role="user", content=user_input))

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response: list[str] = []
            gen = engine.stream(context)

            try:
                while True:
                    token = next(gen)
                    full_response.append(token)
                    placeholder.markdown("".join(full_response) + "▌")
            except StopIteration as e:
                result = e.value

            response_text = "".join(full_response)
            # Safety check on output
            response_text = safety.check_output(response_text)
            placeholder.markdown(response_text)

        store.add_message(
            session_id,
            "assistant",
            response_text,
            tokens=result.completion_tokens if result else None,
        )
        st.rerun()

    # ── Session controls ──────────────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("End Session ✓", use_container_width=True, type="primary"):
            _end_session_ui(store, session_id)


# ── Sub-renders ───────────────────────────────────────────────────────────────

def _start_session_ui(store: SessionStore) -> None:
    st.markdown("### How are you feeling right now?")
    mood = st.slider("Mood (1 = very low, 10 = great)", 1, 10, 5)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Session 🌿", use_container_width=True, type="primary"):
            sid = store.create_session(mood_before=mood)
            st.session_state.current_session = sid
            # Warm-up: load model in background
            st.session_state.manager.load()
            st.rerun()

    with col2:
        _render_privacy_reminder()


def _end_session_ui(store: SessionStore, session_id: str) -> None:
    st.markdown("### How do you feel now?")
    mood_after = st.slider("Mood after session", 1, 10, 5, key="mood_after")
    rating     = st.select_slider(
        "Rate this session", options=[1, 2, 3, 4, 5], value=3, key="session_rating"
    )
    note = st.text_area("Optional: any thoughts to remember?", key="session_note")

    if st.button("Save & Close", type="primary"):
        store.close_session(
            session_id,
            mood_after=mood_after,
            rating=rating,
            note=note or None,
        )
        st.session_state.current_session = None
        st.success("Session saved. Take care of yourself 🌱")
        time.sleep(1.5)
        st.rerun()


def _render_crisis_banner(message: str) -> None:
    st.error("🚨 **I'm concerned about what you've shared.**")
    for line in message.split("\n"):
        if line.strip():
            st.markdown(line)
    st.info(
        "Please consider reaching out to a real person right now. "
        "You deserve support beyond what I can offer."
    )


def _render_privacy_reminder() -> None:
    st.info(
        "🔒 This session is encrypted locally.\n"
        "Nothing leaves your device."
    )
