"""
therapy_ai/ui/closing.py

Pure, framework-free logic for the evidence-based session-closing ritual.

This module deliberately contains NO Streamlit imports so it can be unit-tested
in isolation. The Streamlit page (chat.py) consumes these helpers and owns all
UI/state concerns.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from therapy_ai.domain import Message

DEFAULT_CONFIG_PATH = Path("config/default_config.yaml")
DEFAULT_PHRASES_PATH = Path("config/closing_phrases.yaml")

# ── Built-in phrase fallbacks ─────────────────────────────────────────────────
# These are the defaults used when config/closing_phrases.yaml is missing a
# category (or the whole file is absent). The editable source of truth for
# users is that YAML file; these keep the app working if it's deleted.

# Phrases that signal the user wants to wrap up / leave. Deliberately
# conservative — unambiguous "I'm leaving" signals, not topics someone might
# genuinely want to explore (e.g. "I have to leave my partner").
DEFAULT_CLOSING_PHRASES: tuple[str, ...] = (
    "i have to go",
    "i gotta go",
    "i need to go",
    "i have to leave now",
    "i need to leave now",
    "this is over",
    "that's all for today",
    "thats all for today",
    "let's stop here",
    "lets stop here",
    "let's wrap up",
    "lets wrap up",
    "i'm done for today",
    "im done for today",
    "thanks, until next time",
    "thank you, until next time",
    "talk to you soon",
    "talk to you next time",
    "see you next time",
    "goodbye for now",
    "bye for now",
)

# Short acknowledgement phrases that confirm the user accepts the close.
# These are ONLY treated as an end-signal when the session is already in
# closing mode — otherwise a bare "ok" mid-conversation would end sessions
# prematurely. Order does not matter; matched as whole normalised phrases.
DEFAULT_ACK_PHRASES: tuple[str, ...] = (
    "ok",
    "okay",
    "ok thanks",
    "okay thanks",
    "ok thank you",
    "ok talk to you soon",
    "ok talk soon",
    "ok until next time",
    "ok see you next time",
    "ok bye",
    "okay bye",
    "alright",
    "alright thanks",
    "sounds good",
    "got it thanks",
)

# Phrases by which the user EXPLICITLY asks to keep or discard the session.
# These override the start-screen save toggle for this one session.
DEFAULT_SAVE_PHRASES: tuple[str, ...] = (
    "save this session",
    "save the session",
    "please save this",
    "keep this session",
    "keep this conversation",
    "do save this",
)
DEFAULT_DISCARD_PHRASES: tuple[str, ...] = (
    "don't save this session",
    "dont save this session",
    "don't save this",
    "dont save this",
    "don't save the session",
    "dont save the session",
    "discard this session",
    "discard the session",
    "delete this session",
    "do not save this",
    "do not keep this",
)


def _load_phrase_file(path: str | Path = DEFAULT_PHRASES_PATH) -> dict[str, tuple[str, ...]]:
    """
    Load user-editable phrase lists from config/closing_phrases.yaml.

    Returns a dict with keys 'closing', 'acknowledgement', 'save', 'discard'.
    Any category absent from the file (or if the file is missing/unreadable)
    falls back to the built-in DEFAULT_* tuples, so a partial file never wipes
    the other lists.
    """
    fallbacks: dict[str, tuple[str, ...]] = {
        "closing": DEFAULT_CLOSING_PHRASES,
        "acknowledgement": DEFAULT_ACK_PHRASES,
        "save": DEFAULT_SAVE_PHRASES,
        "discard": DEFAULT_DISCARD_PHRASES,
    }

    data: dict = {}
    try:
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        # Malformed file: fall back entirely rather than crash the session.
        data = {}

    resolved: dict[str, tuple[str, ...]] = {}
    for key, default in fallbacks.items():
        values = data.get(key)
        if values:
            resolved[key] = tuple(str(v).strip().lower() for v in values)
        else:
            resolved[key] = default
    return resolved


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

    # ── Timed-session wind-down ───────────────────────────────────────────────
    # Fraction of the chosen duration after which closing begins. Clinical
    # practice gives a time warning in the last ~10–20% of a session, so 0.8
    # (last 20%) is the default. e.g. a 3-min session begins closing at ~2:24.
    time_close_at_fraction: float = 0.80
    # Floor so very short sessions still leave a sensible wind-down window.
    # Closing starts no later than (duration - this many seconds).
    time_close_floor_seconds: int = 30

    # ── Keyphrase intent ──────────────────────────────────────────────────────
    # Phrase lists are loaded from config/closing_phrases.yaml by from_yaml();
    # these field defaults are the built-in fallbacks.
    detect_keyphrases: bool = True
    closing_phrases: tuple[str, ...] = DEFAULT_CLOSING_PHRASES
    ack_phrases: tuple[str, ...] = DEFAULT_ACK_PHRASES
    save_phrases: tuple[str, ...] = DEFAULT_SAVE_PHRASES
    discard_phrases: tuple[str, ...] = DEFAULT_DISCARD_PHRASES

    @property
    def warn_at_turn(self) -> int:
        """The turn at which the 'Wrap up' button should first appear."""
        return max(0, self.auto_close_after_turns - self.warn_before_turns)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = DEFAULT_CONFIG_PATH,
        phrases_path: str | Path = DEFAULT_PHRASES_PATH,
    ) -> ClosingConfig:
        raw = yaml.safe_load(Path(path).read_text())
        block = (raw or {}).get("session_closing", {}) or {}

        # Phrase content lives in its own data file (config/closing_phrases.yaml).
        phrases = _load_phrase_file(phrases_path)

        # Back-compat: an inline closing_phrases list in default_config.yaml,
        # if present, still takes precedence for the closing category.
        inline_closing = block.get("closing_phrases")
        closing_phrases = (
            tuple(p.strip().lower() for p in inline_closing)
            if inline_closing else phrases["closing"]
        )

        return cls(
            auto_close_after_turns=int(block.get("auto_close_after_turns", 20)),
            warn_before_turns=int(block.get("warn_before_turns", 3)),
            allow_manual_trigger=bool(block.get("allow_manual_trigger", True)),
            time_close_at_fraction=float(block.get("time_close_at_fraction", 0.80)),
            time_close_floor_seconds=int(block.get("time_close_floor_seconds", 30)),
            detect_keyphrases=bool(block.get("detect_keyphrases", True)),
            closing_phrases=closing_phrases,
            ack_phrases=phrases["acknowledgement"],
            save_phrases=phrases["save"],
            discard_phrases=phrases["discard"],
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


# ── Keyphrase intent detection ────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """
    Lowercase, strip diacritics, and collapse punctuation/whitespace for
    phrase matching. Accent-insensitive so "così" matches "cosi" — important
    for multilingual closing phrases.
    """
    text = text.lower()
    # Decompose accented chars and drop the combining marks (ì -> i).
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    # Replace any non-alphanumeric run with a single space, then trim.
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_closing_intent(
    user_text: str, config: ClosingConfig
) -> bool:
    """
    True if the user's message signals they want to end the session.

    Matching is on normalised text (punctuation/case-insensitive) and looks for
    any configured closing phrase as a substring of the normalised message, so
    "Thanks, talk to you soon!" matches "talk to you soon".
    """
    if not config.detect_keyphrases:
        return False
    normalised = _normalise(user_text)
    if not normalised:
        return False
    return any(_normalise(phrase) in normalised for phrase in config.closing_phrases)


def detect_acknowledgement(user_text: str, config: ClosingConfig) -> bool:
    """
    True if the message is a short confirmation that the user accepts the close
    (e.g. "ok", "ok talk to you soon").

    IMPORTANT: the caller must only consult this when the session is ALREADY in
    closing mode. A bare "ok" mid-conversation must not end a session, so this
    function is intentionally not gated on its own — the gating is the caller's
    responsibility. Matching is EXACT on the normalised whole message (not a
    substring) so "ok but my mother..." does not count as an acknowledgement.
    """
    if not config.detect_keyphrases:
        return False
    normalised = _normalise(user_text)
    if not normalised:
        return False
    return any(_normalise(phrase) == normalised for phrase in config.ack_phrases)


def detect_save_intent(user_text: str, config: ClosingConfig) -> str | None:
    """
    Detect an explicit save/discard request in the user's message.

    Returns:
        "save"    if the user explicitly asks to keep the session,
        "discard" if the user explicitly asks not to keep it,
        None      if neither is clearly expressed.

    Discard is checked first: "don't save this" contains "save this", so the
    negative intent must win. Matched as a substring of the normalised message.
    """
    normalised = _normalise(user_text)
    if not normalised:
        return None
    if any(_normalise(phrase) in normalised for phrase in config.discard_phrases):
        return "discard"
    if any(_normalise(phrase) in normalised for phrase in config.save_phrases):
        return "save"
    return None


# ── Timed-session wind-down ─────────────────────────────────────────────────--

def _closing_threshold_seconds(duration_min: float, config: ClosingConfig) -> float:
    """
    Elapsed-seconds point at which a timed session should begin closing.

    Uses the configured fraction, but never later than (duration - floor), so
    short sessions still get a wind-down window. For a 3-min session with
    fraction 0.8 and floor 30s: fraction gives 144s, floor gives 150s -> 144s.
    """
    duration_s = duration_min * 60.0
    by_fraction = duration_s * config.time_close_at_fraction
    by_floor = duration_s - config.time_close_floor_seconds
    # Begin closing at whichever comes FIRST (earlier), but never below 0.
    return max(0.0, min(by_fraction, by_floor))


def should_time_close(
    started_at: float | None,
    duration_min: float | None,
    config: ClosingConfig,
    now: float | None = None,
) -> bool:
    """
    True when a timed session has reached its wind-down point.

    Returns False when there is no timed session (started_at or duration unset),
    so untimed sessions are unaffected.
    """
    if started_at is None or not duration_min or duration_min <= 0:
        return False
    now = time.time() if now is None else now
    elapsed = now - started_at
    return elapsed >= _closing_threshold_seconds(duration_min, config)


def time_remaining_seconds(
    started_at: float | None,
    duration_min: float | None,
    now: float | None = None,
) -> float | None:
    """
    Seconds left in a timed session, or None if the session is untimed.
    Never negative (clamped to 0).
    """
    if started_at is None or not duration_min or duration_min <= 0:
        return None
    now = time.time() if now is None else now
    remaining = (duration_min * 60.0) - (now - started_at)
    return max(0.0, remaining)
