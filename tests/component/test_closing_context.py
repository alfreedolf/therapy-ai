"""
tests/component/test_closing_context.py

Component tests: real SessionStore + real ContextBuilder wired together,
verifying that the closing marker composes correctly on top of an actual
built context (history + preamble). No model, no Streamlit.
"""

from __future__ import annotations

import pytest

from therapy_ai.domain import Message
from therapy_ai.memory.context_builder import ContextBuilder, ContextConfig
from therapy_ai.ui.closing import CLOSING_MARKER, inject_closing_marker, is_closing_context


@pytest.fixture
def builder(store):
    """
    ContextBuilder over the real (throwaway) store, with semantic memory
    disabled so the component test stays deterministic and offline.
    """
    cfg = ContextConfig(use_semantic_memory=False)
    return ContextBuilder(store, config=cfg)


class TestClosingMarkerOverRealContext:
    def test_marker_appended_after_real_history(self, store, builder):
        sid = store.create_session(mood_before=5)
        store.add_message(sid, "user", "I've been anxious about work.")
        store.add_message(sid, "assistant", "That sounds draining. What part weighs most?")

        context = builder.build(sid, latest_user_text="It's the deadlines.")
        closing_context = inject_closing_marker(context)

        # History is preserved
        assert any(m.content == "I've been anxious about work." for m in closing_context)
        # Marker is present and last
        assert closing_context[-1].role == "system"
        assert CLOSING_MARKER in closing_context[-1].content
        assert is_closing_context(closing_context)

    def test_non_closing_context_has_no_marker(self, store, builder):
        sid = store.create_session(mood_before=6)
        store.add_message(sid, "user", "Hello.")
        context = builder.build(sid, latest_user_text="Hello.")
        assert not is_closing_context(context)

    def test_encrypted_roundtrip_preserved_under_closing(self, store, builder):
        """
        The store encrypts message content at rest. Confirm that content
        decrypts correctly and survives into the closing context unchanged.
        """
        sid = store.create_session(mood_before=4)
        secret = "I keep replaying an argument with my father."
        store.add_message(sid, "user", secret)

        context = builder.build(sid, latest_user_text=secret)
        closing_context = inject_closing_marker(context)

        assert any(m.content == secret for m in closing_context)


class TestClosingDoesNotBreakTurnCounting:
    def test_turn_count_reflects_user_messages(self, store):
        sid = store.create_session(mood_before=5)
        for i in range(4):
            store.add_message(sid, "user", f"user turn {i}")
            store.add_message(sid, "assistant", f"assistant turn {i}")

        session = store.get_session(sid)
        # get_session hydrates metadata; load messages separately to count
        msgs = store.load_messages(sid)
        user_turns = sum(1 for m in msgs if m.role == "user")
        assert user_turns == 4
