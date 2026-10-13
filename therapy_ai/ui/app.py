"""
therapy_ai/ui/app.py

Streamlit entry point. Bootstraps the app, enforces passphrase gate,
and routes to chat / progress / settings pages.

Run with:  streamlit run therapy_ai/ui/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Therapy AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

from therapy_ai.core.backend import BackendFactory
from therapy_ai.core.model_manager import ModelManager, ModelConfig
from therapy_ai.core.inference import InferenceEngine
from therapy_ai.core.safety import SafetyGuard
from therapy_ai.memory.session_store import SessionStore
from therapy_ai.memory.context_builder import ContextBuilder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("config/default_config.yaml")


# ── Session-state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "authenticated":  False,
        "store":          None,
        "manager":        None,
        "engine":         None,
        "safety":         None,
        "context_builder":None,
        "current_session":None,
        "lifecycle_token":None,
        "last_activity":  None,   # epoch seconds; for inactivity auto-lock
        "page":           "chat",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Passphrase gate ───────────────────────────────────────────────────────────

def _render_passphrase_gate() -> None:
    st.markdown("## 🔐 Therapy AI — Local & Private")
    st.markdown(
        "All your data is encrypted locally on this device. "
        "Your passphrase is **never stored** — it derives your encryption key in memory."
    )
    st.divider()
    with st.form("passphrase_form"):
        phrase = st.text_input(
            "Enter your passphrase",
            type="password",
            placeholder="Choose something memorable...",
        )
        submitted = st.form_submit_button("Unlock", use_container_width=True)

    if submitted:
        if len(phrase) < 8:
            st.error("Passphrase must be at least 8 characters.")
            return
        try:
            _bootstrap(phrase)
        except Exception as e:  # noqa: BLE001 — surface auth/state errors to the user
            from therapy_ai.memory.session_store import (
                AuthenticationError,
                WorkspaceStateError,
            )
            if isinstance(e, AuthenticationError):
                st.error("Incorrect passphrase.")
            elif isinstance(e, WorkspaceStateError):
                st.error(
                    "This workspace needs attention before it can be opened "
                    f"({e}). It was not unlocked and nothing was changed."
                )
            else:
                st.error(f"Could not open workspace: {e}")
            return
        st.rerun()


def _bootstrap(passphrase: str) -> None:
    """
    Verified workspace open. Initialise an empty workspace or unlock an existing
    verified one. Legacy/ambiguous states raise WorkspaceStateError and are NOT
    silently unlocked — the user is told to use the one-time retirement/recovery
    path instead.
    """
    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    locale = cfg.get("ui", {}).get("locale", "EN")

    store = SessionStore()
    store.open_verified(passphrase)   # raises on wrong pass / legacy / corrupt

    manager = ModelManager.from_config(CONFIG_PATH)
    engine  = InferenceEngine(manager)
    safety  = SafetyGuard(locale=locale)
    builder = ContextBuilder(store)

    st.session_state.update({
        "authenticated":   True,
        "store":           store,
        "manager":         manager,
        "engine":          engine,
        "safety":          safety,
        "context_builder": builder,
        "lifecycle_token": store.lifecycle_token,
    })


# ── Sidebar navigation ────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🌿 Therapy AI")
        backend_info = BackendFactory.detect()
        st.caption(f"Running on: **{backend_info.device_name}**")
        st.caption(f"Memory: {backend_info.available_memory_gb:.1f} GB")
        st.divider()

        pages = {
            "💬 Session": "chat",
            "📊 Progress": "progress",
            "⚙️ Settings": "settings",
        }
        for label, key in pages.items():
            if st.button(label, use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.divider()
        st.caption("All data is local & encrypted.")
        if st.button("🔒 Lock App", use_container_width=True):
            _lock_and_clear()
            st.rerun()


def _lock_and_clear() -> None:
    """
    Service-level revocation: lock the store (drops key + revokes lifecycle
    token), invalidate the in-memory semantic index, and clear ALL session
    state so no decrypted content, cache, or in-flight result survives.
    """
    store = st.session_state.get("store")
    builder = st.session_state.get("context_builder")
    try:
        if builder is not None and getattr(builder, "_retriever", None) is not None:
            builder._retriever.invalidate_index()
    except Exception:
        pass
    try:
        if store is not None:
            store.lock()
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def _enforce_inactivity_lock() -> None:
    """
    Configurable inactivity auto-lock. Reads ui.lock_timeout_minutes from config
    (0/absent disables). Locks and clears state if idle beyond the threshold.
    """
    import time
    import yaml

    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
    except Exception:
        return
    timeout_min = cfg.get("ui", {}).get("lock_timeout_minutes", 0) or 0
    if timeout_min <= 0:
        return
    now = time.time()
    last = st.session_state.get("last_activity")
    if last is not None and (now - last) > timeout_min * 60:
        _lock_and_clear()
        st.warning("Workspace locked due to inactivity. Please unlock again.")
        st.stop()
    st.session_state.last_activity = now


# ── Main router ───────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    if not st.session_state.authenticated:
        _render_passphrase_gate()
        return

    _enforce_inactivity_lock()
    _render_sidebar()

    page = st.session_state.get("page", "chat")
    if page == "chat":
        from pages.chat import render as render_chat
        render_chat()
    elif page == "progress":
        from pages.progress import render as render_progress
        render_progress()
    elif page == "settings":
        from pages.settings import render as render_settings
        render_settings()


if __name__ == "__main__":
    main()
