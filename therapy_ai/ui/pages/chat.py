"""
therapy_ai/ui/pages/chat.py

Main therapy chat page.
Handles session lifecycle, streaming generation, safety routing,
and the evidence-based session-closing ritual.

Closing-mode logic
──────────────────
Turn count is tracked in st.session_state["turn_count"].
Once it reaches the threshold defined in config (session_closing.auto_close_after_turns),
a [CLOSING_MODE] marker is injected as a system message so the model follows the
SESSION CLOSING protocol in system_prompt.txt.
The user can also trigger closing manually at any time via the sidebar button.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from therapy_ai.core.safety import SafetyLevel
from therapy_ai.domain import Message
from therapy_ai.memory.session_store import SessionStore
from therapy_ai.core.model_manager import ModelManager
from therapy_ai.core.inference import InferenceEngine
from therapy_ai.ui.closing import (
    ClosingConfig,
    inject_closing_marker,
    should_auto_close,
    should_show_wrapup_button,
)

CONFIG_PATH = Path("config/default_config.yaml")


# ── Cached engine (loads model once, survives Streamlit reruns) ───────────────

@st.cache_resource
def get_cached_engine() -> InferenceEngine:
    manager = ModelManager.from_config(CONFIG_PATH)
    return InferenceEngine(manager)


# ── Session-state bootstrap ───────────────────────────────────────────────────

def _ensure_state() -> None:
    defaults = {
        "engine":          get_cached_engine(),
        "current_session": None,
        "turn_count":      0,       # counts completed user turns in this session
        "closing_mode":    False,   # True once the closing ritual is active
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    _ensure_state()
    closing_cfg = ClosingConfig.from_yaml(CONFIG_PATH)

    store:   SessionStore  = st.session_state.store
    engine:  InferenceEngine = st.session_state.engine
    safety                   = st.session_state.safety
    builder                  = st.session_state.context_builder

    st.markdown("## 💬 Therapy Session")

    # ── No active session → start screen ─────────────────────────────────────
    if st.session_state.current_session is None:
        _render_start_ui(store, closing_cfg)
        return

    session_id   = st.session_state.current_session
    turn_count   = st.session_state.turn_count
    closing_mode = st.session_state.closing_mode

    # Auto-trigger closing mode when threshold is reached
    if should_auto_close(turn_count, closing_cfg) and not closing_mode:
        st.session_state.closing_mode = True
        closing_mode = True

    # ── Closing-mode banner ───────────────────────────────────────────────────
    if closing_mode:
        st.info(
            "🌿 **We're moving towards the close of this session.** "
            "Take your time — there's no rush.",
            icon=None,
        )

    # ── Sidebar: wrap-up controls ─────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        if should_show_wrapup_button(turn_count, closing_cfg, closing_mode):
            if st.button("🌿 Wrap up session", use_container_width=True):
                st.session_state.closing_mode = True
                st.rerun()

        if closing_mode:
            st.caption("✦ Closing ritual in progress")

        # Always available: hard end
        if st.button("End Session ✓", use_container_width=True, type="primary"):
            _render_end_ui(store, session_id)
            return

    # ── Chat history ──────────────────────────────────────────────────────────
    messages = store.load_messages(session_id)
    for msg in messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    # ── Input ─────────────────────────────────────────────────────────────────
    placeholder_text = (
        "Take your time to respond..." if closing_mode
        else "How are you feeling today?"
    )
    user_input = st.chat_input(placeholder_text)

    if user_input:
        # Safety check before LLM
        safety_result = safety.check_input(user_input)

        if safety_result.level == SafetyLevel.BLOCKED:
            st.error("⛔ " + safety_result.message)
            return

        if safety_result.level == SafetyLevel.INJECTION:
            st.warning("⚠️ " + safety_result.message)
            return

        if safety_result.level == SafetyLevel.CRISIS:
            _render_crisis_banner(safety_result.message)
            store.add_message(session_id, "user", user_input, is_flagged=True)
            with st.chat_message("user"):
                st.markdown(user_input)
            return

        # Persist user message
        store.add_message(session_id, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build context, injecting [CLOSING_MODE] when active
        context = builder.build(session_id, latest_user_text=user_input)
        if closing_mode:
            context = inject_closing_marker(context)
        context.append(Message(role="user", content=user_input))

        # Stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response: list[str] = []

            gen = engine.stream(context)
            try:
                while True:
                    token = next(gen)
                    full_response.append(token)
                    response_placeholder.markdown("".join(full_response) + "▌")
            except StopIteration as e:
                result = e.value

            response_text = "".join(full_response)
            response_text = safety.check_output(response_text)
            response_placeholder.markdown(response_text)

        store.add_message(
            session_id, "assistant", response_text,
            tokens=result.completion_tokens if result else None,
        )

        # Increment turn count after a completed exchange
        st.session_state.turn_count += 1
        st.rerun()

    # ── Soft end-session area (shown below chat when closing) ─────────────────
    if closing_mode:
        st.divider()
        st.caption("Ready to close when you are.")
        if st.button("Save & close session 🌱", use_container_width=True, type="primary"):
            _render_end_ui(store, session_id)


# ── Sub-renders ───────────────────────────────────────────────────────────────

def _render_start_ui(store: SessionStore, closing_cfg: ClosingConfig) -> None:
    st.markdown("### How are you feeling right now?")
    mood = st.slider("Mood (1 = very low, 10 = great)", 1, 10, 5)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Session 🌿", use_container_width=True, type="primary"):
            sid = store.create_session(mood_before=mood)
            st.session_state.current_session = sid
            st.session_state.turn_count = 0
            st.session_state.closing_mode = False
            st.rerun()
    with col2:
        st.info("🔒 This session is encrypted locally. Nothing leaves your device.")

    # Show configured session length as a friendly hint
    threshold = closing_cfg.auto_close_after_turns
    if threshold > 0:
        st.caption(
            f"Sessions gently guide towards a close after around {threshold} exchanges."
        )


def _render_end_ui(store: SessionStore, session_id: str) -> None:
    """Mood check-out + save."""
    st.markdown("### How do you feel now?")
    mood_after = st.slider("Mood after session", 1, 10, 5, key="mood_after")
    rating = st.select_slider(
        "Rate this session", options=[1, 2, 3, 4, 5], value=3, key="session_rating"
    )
    note = st.text_area("Optional: any thoughts to remember?", key="session_note")

    if st.button("Confirm & close", type="primary"):
        store.close_session(
            session_id,
            mood_after=mood_after,
            rating=rating,
            note=note or None,
        )
        # Reset all session-level state
        st.session_state.current_session = None
        st.session_state.turn_count = 0
        st.session_state.closing_mode = False
        st.success("Session saved. Take care of yourself 🌱")
        st.rerun()


def _render_crisis_banner(message: str) -> None:
    st.error("🚨 **I'm concerned about what you've shared.**")
    for line in message.split("\n"):
        if line.strip():
            st.markdown(line)
    st.info("Please consider reaching out to a real person right now.")


if __name__ == "__main__":
    render()
