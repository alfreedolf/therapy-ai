"""
therapy_ai/ui/pages/chat.py

Main therapy chat page.
Handles session lifecycle, streaming generation, safety routing, vault-locked
sessions, and the evidence-based session-closing ritual.

Closing-mode logic
──────────────────
Turn count is tracked in st.session_state["turn_count"].
Once it reaches the threshold defined in config (session_closing.auto_close_after_turns),
a [CLOSING_MODE] marker is injected as a system message so the model follows the
SESSION CLOSING protocol in system_prompt.txt. The user can also trigger closing
manually, by a wind-down keyphrase, or by a timed-session limit.

Vault logic
───────────
A vault session is locked with its own passphrase (separate from the app
passphrase). Its content is encrypted with a vault key; the passphrase is held
in st.session_state["vault_passphrase"] for the active session only.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from therapy_ai.core.inference import InferenceEngine
from therapy_ai.core.model_manager import ModelManager
from therapy_ai.core.safety import SafetyLevel
from therapy_ai.domain import Message
from therapy_ai.memory.session_store import SessionStore
from therapy_ai.ui.closing import (
    ClosingConfig,
    detect_acknowledgement,
    detect_closing_intent,
    detect_save_intent,
    inject_closing_marker,
    should_auto_close,
    should_show_wrapup_button,
    should_time_close,
    time_remaining_seconds,
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
        "engine":            get_cached_engine(),
        "current_session":   None,
        "turn_count":        0,       # completed user turns in this session
        "closing_mode":      False,   # True once the closing ritual is active
        "closing_turns":     0,       # assistant turns run in closing mode
        "checkout":          False,   # True once the user asked to end
        "started_at":        None,    # epoch seconds the session began (time-box)
        "duration_min":      None,    # optional user-chosen session length (min)
        "save_by_default":   True,    # whether the session is kept on close
        "save_override":     None,    # "save"/"discard" if the user said so
        "vault_passphrase":  None,    # in-memory vault secret for this session
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    _ensure_state()
    closing_cfg = ClosingConfig.from_yaml(CONFIG_PATH)

    store:   SessionStore    = st.session_state.store
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
    vault_pass   = st.session_state.get("vault_passphrase")

    # ── Vault unlock gate ─────────────────────────────────────────────────────
    # If this session is a vault and we don't hold a valid passphrase in memory,
    # prompt to unlock before anything is loaded or sent.
    if store.is_vault(session_id):
        if not vault_pass or not store.verify_vault_passphrase(session_id, vault_pass):
            _render_vault_unlock_ui(store, session_id)
            return

    # ── Checkout form takes over the main column when requested ───────────────
    if st.session_state.checkout:
        _render_end_ui(store, session_id)
        return

    # Auto-trigger closing mode when the turn threshold is reached
    if should_auto_close(turn_count, closing_cfg) and not closing_mode:
        st.session_state.closing_mode = True
        closing_mode = True

    # Time-box trigger: enter closing when the session is near its chosen limit
    if (
        not closing_mode
        and should_time_close(
            st.session_state.started_at,
            st.session_state.duration_min,
            closing_cfg,
        )
    ):
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
        _render_time_remaining(st.session_state.started_at, st.session_state.duration_min)

        if should_show_wrapup_button(turn_count, closing_cfg, closing_mode):
            if st.button("🌿 Wrap up session", use_container_width=True):
                st.session_state.closing_mode = True
                st.rerun()

        if closing_mode:
            st.caption("✦ Closing ritual in progress")

        # Always available: hard end. Sets a flag; the form renders in the
        # main column on the next run (not inside the sidebar).
        if st.button("End Session ✓", use_container_width=True, type="primary"):
            st.session_state.checkout = True
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    messages = store.load_messages(session_id, vault_passphrase=vault_pass)
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
            store.add_message(
                session_id, "user", user_input, is_flagged=True,
                vault_passphrase=vault_pass,
            )
            with st.chat_message("user"):
                st.markdown(user_input)
            return

        # Explicit save/discard intent ("save this" / "don't save this")
        # overrides the start-screen toggle for this session.
        intent = detect_save_intent(user_input, closing_cfg)
        if intent is not None:
            st.session_state.save_override = intent

        # Acknowledgement (2a): a short "ok / ok talk soon" ONLY counts as an
        # end-signal when we're already in closing mode — otherwise a bare "ok"
        # mid-conversation would end the session prematurely. Persist the
        # message first so it isn't lost, then go to the check-out form.
        if closing_mode and detect_acknowledgement(user_input, closing_cfg):
            store.add_message(session_id, "user", user_input, vault_passphrase=vault_pass)
            st.session_state.checkout = True
            st.rerun()

        # Keyphrase trigger: if the user signals they want to leave
        # ("I have to go", "thanks, talk soon"), enter closing mode so the
        # model's reply to THIS message is already the start of the ritual.
        if not closing_mode and detect_closing_intent(user_input, closing_cfg):
            st.session_state.closing_mode = True
            closing_mode = True

        # Persist user message
        store.add_message(session_id, "user", user_input, vault_passphrase=vault_pass)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build context. Inject [CLOSING_MODE] only for the FIRST closing turn —
        # once the model has delivered one closing message, stop re-injecting so
        # it doesn't repeat the full goodbye every subsequent turn.
        context = builder.build(session_id, latest_user_text=user_input)
        inject_closing = closing_mode and st.session_state.closing_turns == 0
        if inject_closing:
            context = inject_closing_marker(context)
        context.append(Message(role="user", content=user_input))

        # Stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response: list[str] = []

            result = None  # guard: generator may yield nothing before StopIteration
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
            tokens=result.completion_tokens if result is not None else None,
            vault_passphrase=vault_pass,
        )

        # Increment turn count after a completed exchange
        st.session_state.turn_count += 1
        # Track closing turns so the ritual is delivered once, not repeated.
        if closing_mode:
            st.session_state.closing_turns += 1
        st.rerun()

    # ── Soft end-session area (shown below chat when closing) ─────────────────
    if closing_mode:
        st.divider()
        if st.session_state.closing_turns >= 1:
            # The ritual has been delivered — steer gently to the close.
            st.caption("We've reached a natural place to close. Whenever you're ready.")
        else:
            st.caption("Ready to close when you are.")
        if st.button("Save & close session 🌱", use_container_width=True, type="primary"):
            st.session_state.checkout = True
            st.rerun()


# ── Sub-renders ───────────────────────────────────────────────────────────────

def _render_start_ui(store: SessionStore, closing_cfg: ClosingConfig) -> None:
    import time

    st.markdown("### How are you feeling right now?")
    mood = st.slider("Mood (1 = very low, 10 = great)", 1, 10, 5)

    # Optional timed session. "No limit" leaves duration unset.
    duration_label = st.selectbox(
        "Session length",
        ["No limit", "3 minutes", "15 minutes", "30 minutes", "50 minutes"],
        index=0,
        help="With a limit, the session gently begins closing as the time nears its end.",
    )
    duration_min = None if duration_label == "No limit" else int(duration_label.split()[0])

    # Whether this session is kept after it ends. Default on — the app's value
    # (mood trends, memory, fine-tuning) depends on saved sessions. The user can
    # still override at the end by saying e.g. "don't save this".
    save_by_default = st.checkbox(
        "Save this session",
        value=True,
        help="If off, the session is discarded when it ends (with a confirmation).",
    )

    # ── Vault lock (optional) ─────────────────────────────────────────────────
    lock_vault = st.checkbox(
        "🔒 Lock this session (vault)",
        help=(
            "Encrypt this session with its own passphrase, separate from your "
            "app passphrase. No one — not even someone who knows your app "
            "passphrase — can read it without the vault passphrase."
        ),
    )
    vault_passphrase: str | None = None
    if lock_vault:
        vault_passphrase = st.text_input(
            "Vault passphrase",
            type="password",
            help="You will need this exact passphrase to reopen the session.",
        )
        st.warning(
            "⚠️ There is no recovery. If you forget this passphrase, the "
            "session is permanently unreadable."
        )

    col1, col2 = st.columns(2)
    with col1:
        start_disabled = lock_vault and not vault_passphrase
        if st.button(
            "Start Session 🌿",
            use_container_width=True,
            type="primary",
            disabled=start_disabled,
        ):
            sid = store.create_session(
                mood_before=mood,
                vault_passphrase=vault_passphrase if lock_vault else None,
            )
            st.session_state.current_session = sid
            st.session_state.turn_count = 0
            st.session_state.closing_mode = False
            st.session_state.closing_turns = 0
            st.session_state.checkout = False
            st.session_state.started_at = time.time()
            st.session_state.duration_min = duration_min
            st.session_state.save_by_default = save_by_default
            st.session_state.save_override = None
            # Hold the vault passphrase in memory for THIS session only.
            st.session_state.vault_passphrase = vault_passphrase if lock_vault else None
            st.rerun()
    with col2:
        st.info("🔒 This session is encrypted locally. Nothing leaves your device.")

    # Show configured session length as a friendly hint
    threshold = closing_cfg.auto_close_after_turns
    if threshold > 0:
        st.caption(
            f"Sessions gently guide towards a close after around {threshold} exchanges."
        )


def _reset_session_state() -> None:
    """Clear all per-session state after a session ends."""
    st.session_state.current_session = None
    st.session_state.turn_count = 0
    st.session_state.closing_mode = False
    st.session_state.closing_turns = 0
    st.session_state.checkout = False
    st.session_state.started_at = None
    st.session_state.duration_min = None
    st.session_state.save_override = None
    st.session_state.vault_passphrase = None


def _render_end_ui(store: SessionStore, session_id: str) -> None:
    """
    Session check-out. Honours the save decision: an explicit in-chat override
    ("save"/"discard") wins over the start-screen toggle. Discarding is
    irreversible, so it is gated behind a confirmation.
    """
    # Resolve the effective save decision.
    override = st.session_state.get("save_override")
    will_save = (
        override != "discard"
        if override is not None
        else st.session_state.get("save_by_default", True)
    )

    # ── Discard path ──────────────────────────────────────────────────────────
    if not will_save:
        st.markdown("### Discard this session?")
        st.warning(
            "This session will be **permanently deleted** — messages and all. "
            "This cannot be undone."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Delete permanently", type="primary", use_container_width=True):
                store.delete_session(session_id)
                _reset_session_state()
                st.success("Session discarded. Nothing was kept.")
                st.rerun()
        with col_b:
            if st.button("← Keep it after all", use_container_width=True):
                # Flip to saving: clear the discard override and fall through to
                # the save form on the next run.
                st.session_state.save_override = "save"
                st.session_state.save_by_default = True
                st.rerun()
        return

    # ── Save path ──────────────────────────────────────────────────────────────
    st.markdown("### How do you feel now?")
    mood_after = st.slider("Mood after session", 1, 10, 5, key="mood_after")
    rating = st.select_slider(
        "Rate this session", options=[1, 2, 3, 4, 5], value=3, key="session_rating"
    )
    note = st.text_area("Optional: any thoughts to remember?", key="session_note")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Confirm & close", type="primary", use_container_width=True):
            store.close_session(
                session_id,
                mood_after=mood_after,
                rating=rating,
                note=note or None,
            )
            _reset_session_state()
            st.success("Session saved. Take care of yourself 🌱")
            st.rerun()
    with col_b:
        # Let the user back out of the checkout form and keep talking.
        if st.button("← Not yet", use_container_width=True):
            st.session_state.checkout = False
            st.rerun()


def _render_time_remaining(started_at, duration_min) -> None:
    """Sidebar caption showing time left in a timed session."""
    remaining = time_remaining_seconds(started_at, duration_min)
    if remaining is None:
        return
    mins, secs = divmod(max(0, int(remaining)), 60)
    st.caption(f"⏳ ~{mins}:{secs:02d} remaining")


def _render_vault_unlock_ui(store: SessionStore, session_id: str) -> None:
    """Prompt for a vault session's passphrase before its content is shown."""
    st.markdown("### 🔒 This session is locked")
    st.caption(
        "Enter the vault passphrase you set when you created this session. "
        "There is no recovery if it is lost."
    )
    attempt = st.text_input("Vault passphrase", type="password", key="vault_unlock_input")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Unlock", type="primary", use_container_width=True):
            if store.verify_vault_passphrase(session_id, attempt):
                st.session_state.vault_passphrase = attempt
                st.rerun()
            else:
                st.error("Incorrect vault passphrase.")
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_session = None
            st.session_state.vault_passphrase = None
            st.rerun()


def _render_crisis_banner(message: str) -> None:
    st.error("🚨 **I'm concerned about what you've shared.**")
    for line in message.split("\n"):
        if line.strip():
            st.markdown(line)
    st.info("Please consider reaching out to a real person right now.")


if __name__ == "__main__":
    render()
