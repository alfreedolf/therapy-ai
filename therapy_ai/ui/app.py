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
        "passphrase":     None,
        "store":          None,
        "manager":        None,
        "engine":         None,
        "safety":         None,
        "context_builder":None,
        "current_session":None,
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
        _bootstrap(phrase)
        st.rerun()


def _bootstrap(passphrase: str) -> None:
    """Initialise all services after passphrase entry."""
    store   = SessionStore(passphrase=passphrase)
    manager = ModelManager.from_config(CONFIG_PATH)
    engine  = InferenceEngine(manager)
    safety  = SafetyGuard(locale="DE")
    builder = ContextBuilder(store)

    st.session_state.update({
        "authenticated":   True,
        "passphrase":      passphrase,
        "store":           store,
        "manager":         manager,
        "engine":          engine,
        "safety":          safety,
        "context_builder": builder,
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
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── Main router ───────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    if not st.session_state.authenticated:
        _render_passphrase_gate()
        return

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
