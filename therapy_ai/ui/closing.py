"""
therapy_ai/ui/closing.py

Pure, framework-free logic for the evidence-based session-closing ritual.

This module deliberately contains NO Streamlit imports so it can be unit-tested
in isolation. The Streamlit page (chat.py) consumes these helpers and owns all
UI/state concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from therapy_ai.domain import Message

DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")

# The exact marker the model watches for. Kept as a constant so the prompt,
# the injector, and the tests all agree on the literal string.
CLOSING_MARKER = "[CLOSING_MODE]"

_CLOSING_INSTRUCTION = (
    f"{CLOSING_MARKER} The session is now in its closing phase. "
    "Follow the SESSION CLOSING protocol: gently signal the end, "
    "summarise what was explored, name one insight, offer a "
    "between-session anchor, and close with warmth. "
    "Do not open any new emotional topics."
)


@dataclass(frozen=True)
class ClosingConfig:
    """Resolved session-closing configuration."""
    auto_close_after_turns: int = 20
    warn_before_turns: int = 3
    allow_manual_trigger: bool = True

    @property
    def warn_at_turn(self) -> int:
        """The turn at which the 'Wrap up' button should first appear."""
        return max(0, self.auto_close_after_turns - self.warn_before_turns)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ClosingConfig":
        raw = yaml.safe_load(Path(path).read_text())
        block = (raw or {}).get("session_closing", {}) or {}
        return cls(
            auto_close_after_turns=int(block.get("auto_close_after_turns", 20)),
            warn_before_turns=int(block.get("warn_before_turns", 3)),
            allow_manual_trigger=bool(block.get("allow_manual_trigger", True)),
        )


def should_auto_close(turn_count: int, config: ClosingConfig) -> bool:
    """
    True when the automatic closing threshold has been reached.
    A threshold of 0 (or less) disables automatic closing entirely.
    """
    if config.auto_close_after_turns <= 0:
        return False
    return turn_count >= config.auto_close_after_turns


def should_show_wrapup_button(
    turn_count: int, config: ClosingConfig, already_closing: bool
) -> bool:
    """
    True when the manual 'Wrap up' button should be visible.
    Hidden once closing mode is already active, or if manual triggering is off.
    """
    if already_closing or not config.allow_manual_trigger:
        return False
    if config.auto_close_after_turns <= 0:
        # No auto-close configured — offer manual wrap-up at all times.
        return True
    return turn_count >= config.warn_at_turn


def inject_closing_marker(context: list[Message]) -> list[Message]:
    """
    Return a new context list with a system-role closing marker appended.
    The original list is not mutated.
    """
    marker = Message(role="system", content=_CLOSING_INSTRUCTION)
    return [*context, marker]


def is_closing_context(context: list[Message]) -> bool:
    """True if any message in the context carries the closing marker."""
    return any(CLOSING_MARKER in m.content for m in context)
