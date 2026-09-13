"""
tests/e2e/test_session_lifecycle.py

End-to-end test of a full therapy session lifecycle driven into closing mode.

Streamlit itself is not exercised (it needs a browser runtime). Instead this
test reproduces the exact orchestration chat.py performs — start session, loop
turns, evaluate closing decisions, inject the marker, stream through the engine,
persist messages, close session — using the REAL SessionStore, ContextBuilder,
and closing logic, plus the FakeInferenceEngine.

If chat.py's turn loop and closing wiring are correct, this driver mirrors it
and asserts the behaviour a user would experience.
"""

from __future__ import annotations

import pytest

from therapy_ai.domain import Message
from therapy_ai.memory.context_builder import ContextBuilder, ContextConfig
from therapy_ai.ui.closing import (
    ClosingConfig,
    inject_closing_marker,
    is_closing_context,
    should_auto_close,
    should_show_wrapup_button,
)


class SessionDriver:
    """
    Headless re-creation of chat.py's per-turn orchestration.
    Holds the same state chat.py keeps in st.session_state.
    """

    def __init__(self, store, builder, engine, config: ClosingConfig):
        self.store = store
        self.builder = builder
        self.engine = engine
        self.config = config
        self.session_id: str | None = None
        self.turn_count = 0
        self.closing_mode = False
        self.wrapup_button_seen_at: int | None = None

    def start(self, mood_before: int = 5) -> None:
        self.session_id = self.store.create_session(mood_before=mood_before)
        self.turn_count = 0
        self.closing_mode = False

    def user_says(self, text: str) -> str:
        """Process one user turn exactly as chat.py does; return assistant reply."""
        assert self.session_id is not None

        # 1. Auto-trigger closing when threshold reached
        if should_auto_close(self.turn_count, self.config) and not self.closing_mode:
            self.closing_mode = True

        # 2. Record whether the wrap-up button would be shown this turn
        if (
            self.wrapup_button_seen_at is None
            and should_show_wrapup_button(self.turn_count, self.config, self.closing_mode)
        ):
            self.wrapup_button_seen_at = self.turn_count

        # 3. Persist user message
        self.store.add_message(self.session_id, "user", text)

        # 4. Build context, injecting closing marker when active
        context = self.builder.build(self.session_id, latest_user_text=text)
        if self.closing_mode:
            context = inject_closing_marker(context)
        context.append(Message(role="user", content=text))

        # 5. Stream through the engine (collect full reply)
        tokens: list[str] = []
        gen = self.engine.stream(context)
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration:
            pass
        reply = "".join(tokens).strip()

        # 6. Persist assistant reply + advance turn counter
        self.store.add_message(self.session_id, "assistant", reply)
        self.turn_count += 1
        return reply

    def close(self, mood_after: int = 7, rating: int = 5) -> None:
        assert self.session_id is not None
        self.store.close_session(self.session_id, mood_after=mood_after, rating=rating)


@pytest.fixture
def driver(store, fake_engine):
    builder = ContextBuilder(store, config=ContextConfig(use_semantic_memory=False))
    # Short session so the test runs fast: close after 5 turns, warn 2 early (turn 3)
    config = ClosingConfig(auto_close_after_turns=5, warn_before_turns=2)
    return SessionDriver(store, builder, fake_engine, config)


class TestFullSessionLifecycle:
    def test_session_runs_and_enters_closing_mode(self, driver, fake_engine):
        driver.start(mood_before=4)

        # Turns 0-4: normal, no closing
        for i in range(5):
            driver.user_says(f"message {i}")

        # After 5 completed turns, the next turn should flip closing on
        assert driver.closing_mode is False  # not yet — flip happens at start of turn 5
        driver.user_says("one more thing")
        assert driver.closing_mode is True

    def test_closing_marker_injected_only_after_threshold(self, driver, fake_engine):
        driver.start()

        driver.user_says("early message")          # turn 0 -> 1
        assert not is_closing_context(fake_engine.last_context)

        # Drive up to and past the threshold
        for i in range(6):
            driver.user_says(f"msg {i}")

        assert is_closing_context(fake_engine.last_context)

    def test_wrapup_button_appears_before_threshold(self, driver):
        driver.start()
        for i in range(6):
            driver.user_says(f"msg {i}")
        # config: auto=5, warn_before=2 -> warn_at_turn = 3
        assert driver.wrapup_button_seen_at == 3

    def test_session_persists_all_turns_and_closes(self, driver):
        driver.start(mood_before=3)
        for i in range(6):
            driver.user_says(f"msg {i}")
        driver.close(mood_after=8, rating=5)

        session = driver.store.get_session(driver.session_id)
        assert session.is_closed
        assert session.mood_before == 3
        assert session.mood_after == 8
        assert session.mood_delta == 5

        msgs = driver.store.load_messages(driver.session_id)
        user_turns = sum(1 for m in msgs if m.role == "user")
        assert user_turns == 6

    def test_disabled_autoclose_never_enters_closing(self, store, fake_engine):
        builder = ContextBuilder(store, config=ContextConfig(use_semantic_memory=False))
        config = ClosingConfig(auto_close_after_turns=0, allow_manual_trigger=True)
        driver = SessionDriver(store, builder, fake_engine, config)
        driver.start()

        for i in range(10):
            driver.user_says(f"msg {i}")

        assert driver.closing_mode is False
        assert not is_closing_context(fake_engine.last_context)
        # But the manual wrap-up button should be available throughout
        assert driver.wrapup_button_seen_at == 0
