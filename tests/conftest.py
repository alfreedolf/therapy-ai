"""
tests/conftest.py

Shared fixtures for the whole suite.

Test layers
───────────
tests/unit/       — pure logic, no I/O, no framework (closing decisions, config)
tests/component/  — real collaborators wired together (SessionStore + ContextBuilder)
tests/e2e/        — full session lifecycle driven through a fake inference engine
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from therapy_ai.domain import GenerationResult, Message
from therapy_ai.memory.session_store import SessionStore


# ── Config file fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def write_config(tmp_path: Path):
    """
    Factory that writes a minimal default_config.yaml with a custom
    session_closing block and returns its path.
    """
    def _write(
        auto_close_after_turns: int = 20,
        warn_before_turns: int = 3,
        allow_manual_trigger: bool = True,
        include_block: bool = True,
    ) -> Path:
        block = ""
        if include_block:
            block = (
                "session_closing:\n"
                f"  auto_close_after_turns: {auto_close_after_turns}\n"
                f"  warn_before_turns: {warn_before_turns}\n"
                f"  allow_manual_trigger: {str(allow_manual_trigger).lower()}\n"
            )
        cfg = "ui:\n  locale: \"EN\"\n" + block
        path = tmp_path / "default_config.yaml"
        path.write_text(cfg)
        return path

    return _write


# ── Storage fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    """A real SessionStore backed by a throwaway encrypted SQLite file."""
    db_path = tmp_path / "sessions.db"
    return SessionStore(passphrase="test-passphrase-1234", db_path=db_path)


# ── Fake inference engine ─────────────────────────────────────────────────────

class FakeInferenceEngine:
    """
    Stand-in for InferenceEngine that records the context it was given and
    streams back a canned reply token-by-token. No model, no MLX, no network.

    It inspects the incoming context so tests can assert whether closing-mode
    was active for a given turn.
    """

    def __init__(self, reply: str = "I hear you. Tell me more."):
        self.reply = reply
        self.received_contexts: list[list[Message]] = []

    def stream(
        self, messages: list[Message], override_system: str | None = None
    ) -> Generator[str, None, GenerationResult]:
        # Deep-ish copy so later mutation of the caller's list can't rewrite history
        self.received_contexts.append(list(messages))
        for token in self.reply.split(" "):
            yield token + " "
        return GenerationResult(
            text=self.reply,
            prompt_tokens=len(messages),
            completion_tokens=len(self.reply.split(" ")),
            stopped_by="eos",
        )

    # Convenience accessors for assertions
    @property
    def last_context(self) -> list[Message]:
        return self.received_contexts[-1]


@pytest.fixture
def fake_engine() -> FakeInferenceEngine:
    return FakeInferenceEngine()
