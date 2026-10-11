"""
tests/unit/test_closing_logic.py

Unit tests for the pure session-closing logic in therapy_ai/ui/closing.py.
No I/O, no Streamlit, no model.
"""

from __future__ import annotations

import pytest

from therapy_ai.domain import Message
from therapy_ai.ui.closing import (
    CLOSING_MARKER,
    ClosingConfig,
    _load_phrase_file,
    detect_acknowledgement,
    detect_closing_intent,
    detect_save_intent,
    inject_closing_marker,
    is_closing_context,
    should_auto_close,
    should_show_wrapup_button,
    should_time_close,
    time_remaining_seconds,
)

# ── ClosingConfig.from_yaml ───────────────────────────────────────────────────

class TestClosingConfigLoading:
    def test_reads_values_from_yaml(self, write_config):
        path = write_config(auto_close_after_turns=15, warn_before_turns=2)
        cfg = ClosingConfig.from_yaml(path)
        assert cfg.auto_close_after_turns == 15
        assert cfg.warn_before_turns == 2
        assert cfg.allow_manual_trigger is True

    def test_defaults_when_block_missing(self, write_config):
        path = write_config(include_block=False)
        cfg = ClosingConfig.from_yaml(path)
        assert cfg.auto_close_after_turns == 20
        assert cfg.warn_before_turns == 3
        assert cfg.allow_manual_trigger is True

    def test_manual_trigger_false_is_parsed(self, write_config):
        path = write_config(allow_manual_trigger=False)
        cfg = ClosingConfig.from_yaml(path)
        assert cfg.allow_manual_trigger is False

    def test_warn_at_turn_derived_correctly(self):
        cfg = ClosingConfig(auto_close_after_turns=20, warn_before_turns=3)
        assert cfg.warn_at_turn == 17

    def test_warn_at_turn_never_negative(self):
        cfg = ClosingConfig(auto_close_after_turns=2, warn_before_turns=10)
        assert cfg.warn_at_turn == 0


# ── should_auto_close ─────────────────────────────────────────────────────────

class TestShouldAutoClose:
    @pytest.mark.parametrize("turns,expected", [(0, False), (19, False), (20, True), (21, True)])
    def test_threshold_boundary(self, turns, expected):
        cfg = ClosingConfig(auto_close_after_turns=20)
        assert should_auto_close(turns, cfg) is expected

    def test_zero_threshold_disables_auto_close(self):
        cfg = ClosingConfig(auto_close_after_turns=0)
        assert should_auto_close(1000, cfg) is False

    def test_negative_threshold_disables_auto_close(self):
        cfg = ClosingConfig(auto_close_after_turns=-1)
        assert should_auto_close(1000, cfg) is False


# ── should_show_wrapup_button ─────────────────────────────────────────────────

class TestShouldShowWrapupButton:
    def test_hidden_before_warn_turn(self):
        cfg = ClosingConfig(auto_close_after_turns=20, warn_before_turns=3)  # warn at 17
        assert should_show_wrapup_button(16, cfg, already_closing=False) is False

    def test_visible_at_warn_turn(self):
        cfg = ClosingConfig(auto_close_after_turns=20, warn_before_turns=3)
        assert should_show_wrapup_button(17, cfg, already_closing=False) is True

    def test_hidden_when_already_closing(self):
        cfg = ClosingConfig(auto_close_after_turns=20, warn_before_turns=3)
        assert should_show_wrapup_button(18, cfg, already_closing=True) is False

    def test_hidden_when_manual_trigger_disabled(self):
        cfg = ClosingConfig(auto_close_after_turns=20, allow_manual_trigger=False)
        assert should_show_wrapup_button(19, cfg, already_closing=False) is False

    def test_always_visible_when_auto_close_disabled(self):
        cfg = ClosingConfig(auto_close_after_turns=0, allow_manual_trigger=True)
        assert should_show_wrapup_button(0, cfg, already_closing=False) is True


# ── inject_closing_marker / is_closing_context ───────────────────────────────

class TestClosingMarker:
    def test_appends_system_marker(self):
        ctx = [Message(role="user", content="hello")]
        result = inject_closing_marker(ctx)
        assert len(result) == 2
        assert result[-1].role == "system"
        assert CLOSING_MARKER in result[-1].content

    def test_does_not_mutate_original(self):
        ctx = [Message(role="user", content="hello")]
        inject_closing_marker(ctx)
        assert len(ctx) == 1  # original untouched

    def test_marker_instructs_closing_protocol(self):
        result = inject_closing_marker([])
        content = result[-1].content.lower()
        # The marker must reference the core closing steps
        assert "summarise" in content or "summarize" in content
        assert "insight" in content
        assert "new" in content and "topic" in content  # "do not open new topics"

    def test_is_closing_context_true_when_marker_present(self):
        ctx = inject_closing_marker([Message(role="user", content="hi")])
        assert is_closing_context(ctx) is True

    def test_is_closing_context_false_without_marker(self):
        ctx = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
        assert is_closing_context(ctx) is False


# ── System-prompt protocol presence ──────────────────────────────────────────

class TestSystemPromptProtocol:
    """
    Guards that the shipped system prompt actually contains the closing protocol
    the marker refers to. If someone edits the prompt and drops the section,
    this fails loudly.
    """

    def test_system_prompt_declares_closing_section(self):
        from pathlib import Path
        text = Path("config/prompts/system_prompt.txt").read_text().upper()
        assert "SESSION CLOSING" in text
        assert CLOSING_MARKER.upper() in text

    def test_system_prompt_covers_the_five_steps(self):
        from pathlib import Path
        text = Path("config/prompts/system_prompt.txt").read_text().lower()
        # The five clinical closing steps, loosely matched
        assert "close" in text
        assert "summarise" in text or "summarize" in text
        assert "insight" in text
        assert "anchor" in text or "between-session" in text


# ── Keyphrase intent detection ────────────────────────────────────────────────

class TestDetectClosingIntent:
    @pytest.mark.parametrize("text", [
        "I have to go",
        "I have to go now, sorry",
        "Thanks, talk to you soon!",
        "ok, that's all for today",
        "This is over.",
        "Thank you, until next time.",
        "Let's wrap up please",
        "goodbye for now",
    ])
    def test_closing_phrases_detected(self, text):
        assert detect_closing_intent(text, ClosingConfig()) is True

    def test_case_and_punctuation_insensitive(self):
        assert detect_closing_intent("I HAVE TO GO!!!", ClosingConfig()) is True
        assert detect_closing_intent("...talk, to you, soon...", ClosingConfig()) is True

    @pytest.mark.parametrize("text", [
        "I feel resentment toward my parents.",
        "I had to go to work early and it stressed me out.",  # 'had to go', not 'have to go'
        "We talked about my relationship with my mother.",
        "",
        "   ",
    ])
    def test_non_closing_messages_ignored(self, text):
        assert detect_closing_intent(text, ClosingConfig()) is False

    def test_disabled_by_config(self):
        cfg = ClosingConfig(detect_keyphrases=False)
        assert detect_closing_intent("I have to go", cfg) is False

    def test_custom_phrase_list(self):
        cfg = ClosingConfig(closing_phrases=("basta cosi",))
        assert detect_closing_intent("ok, basta così", cfg) is True
        assert detect_closing_intent("I have to go", cfg) is False  # defaults replaced


# ── Timed-session wind-down ────────────────────────────────────────────────---

class TestShouldTimeClose:
    def test_untimed_session_never_closes(self):
        cfg = ClosingConfig()
        assert should_time_close(None, None, cfg) is False
        assert should_time_close(1000.0, None, cfg, now=999999.0) is False
        assert should_time_close(1000.0, 0, cfg, now=999999.0) is False

    def test_closes_at_fraction_point(self):
        cfg = ClosingConfig(time_close_at_fraction=0.80, time_close_floor_seconds=30)
        start = 1000.0
        # 3-min session: fraction gives 144s, floor gives 150s -> earlier (144s) wins
        assert should_time_close(start, 3, cfg, now=start + 143) is False
        assert should_time_close(start, 3, cfg, now=start + 144) is True

    def test_floor_wins_for_short_session(self):
        # 1-min session: fraction 0.8 -> 48s, floor (60-30)=30s -> 30s is earlier
        cfg = ClosingConfig(time_close_at_fraction=0.80, time_close_floor_seconds=30)
        start = 500.0
        assert should_time_close(start, 1, cfg, now=start + 29) is False
        assert should_time_close(start, 1, cfg, now=start + 30) is True

    def test_long_session_uses_fraction(self):
        # 50-min session: fraction 0.8 -> 2400s; floor (3000-30)=2970s -> fraction earlier
        cfg = ClosingConfig(time_close_at_fraction=0.80, time_close_floor_seconds=30)
        start = 0.0
        assert should_time_close(start, 50, cfg, now=2399) is False
        assert should_time_close(start, 50, cfg, now=2400) is True


class TestTimeRemaining:
    def test_none_when_untimed(self):
        assert time_remaining_seconds(None, None) is None
        assert time_remaining_seconds(1000.0, None) is None
        assert time_remaining_seconds(1000.0, 0) is None

    def test_counts_down(self):
        start = 1000.0
        assert time_remaining_seconds(start, 3, now=start) == 180.0
        assert time_remaining_seconds(start, 3, now=start + 60) == 120.0

    def test_clamped_to_zero(self):
        start = 1000.0
        assert time_remaining_seconds(start, 3, now=start + 999) == 0.0


# ── Acknowledgement detection (feature 2a) ─────────────────────────────────────

class TestDetectAcknowledgement:
    @pytest.mark.parametrize("text", [
        "ok",
        "OK!",
        "okay",
        "ok thanks",
        "ok talk to you soon",
        "ok until next time",
        "alright",
        "sounds good",
    ])
    def test_acknowledgements_detected(self, text):
        assert detect_acknowledgement(text, ClosingConfig()) is True

    @pytest.mark.parametrize("text", [
        "ok but my mother called me yesterday",  # 'ok' + more -> NOT an ack
        "okay so here's the thing",
        "I feel sad",
        "",
        "   ",
    ])
    def test_non_acknowledgements_ignored(self, text):
        # Exact-match on the whole normalised message, not a substring.
        assert detect_acknowledgement(text, ClosingConfig()) is False

    def test_disabled_by_config(self):
        cfg = ClosingConfig(detect_keyphrases=False)
        assert detect_acknowledgement("ok", cfg) is False


# ── Save / discard intent (feature 2b) ─────────────────────────────────────────

class TestDetectSaveIntent:
    @pytest.mark.parametrize("text", [
        "please save this session",
        "save the session",
        "keep this conversation",
    ])
    def test_save_intent(self, text):
        assert detect_save_intent(text, ClosingConfig()) == "save"

    @pytest.mark.parametrize("text", [
        "don't save this session",
        "do not keep this",
        "discard this session",
        "delete this session",
    ])
    def test_discard_intent(self, text):
        assert detect_save_intent(text, ClosingConfig()) == "discard"

    def test_discard_wins_over_save(self):
        # "don't save this" contains "save this" — the negative must win.
        assert detect_save_intent("don't save this session", ClosingConfig()) == "discard"

    @pytest.mark.parametrize("text", [
        "I feel anxious about my job",
        "we talked about my mother",
        "",
    ])
    def test_no_intent(self, text):
        assert detect_save_intent(text, ClosingConfig()) is None


# ── Phrase-file loading & fallback ─────────────────────────────────────────────

class TestPhraseFileLoading:
    def test_loads_all_categories_from_file(self):
        phrases = _load_phrase_file("config/closing_phrases.yaml")
        assert set(phrases) == {"closing", "acknowledgement", "save", "discard"}
        assert "ok" in phrases["acknowledgement"]
        assert all(isinstance(v, tuple) for v in phrases.values())

    def test_missing_file_falls_back_to_defaults(self):
        from therapy_ai.ui.closing import DEFAULT_ACK_PHRASES
        phrases = _load_phrase_file("config/does_not_exist.yaml")
        assert phrases["acknowledgement"] == DEFAULT_ACK_PHRASES

    def test_partial_file_fills_missing_categories(self, tmp_path):
        from therapy_ai.ui.closing import DEFAULT_SAVE_PHRASES
        # A file that defines only 'closing' should keep defaults elsewhere.
        p = tmp_path / "partial.yaml"
        p.write_text("closing:\n  - \"basta\"\n")
        phrases = _load_phrase_file(p)
        assert phrases["closing"] == ("basta",)
        assert phrases["save"] == DEFAULT_SAVE_PHRASES

    def test_config_wires_phrase_lists(self):
        cfg = ClosingConfig.from_yaml(
            "config/default_config.yaml", "config/closing_phrases.yaml"
        )
        assert len(cfg.ack_phrases) >= 10
        assert len(cfg.save_phrases) >= 3
        assert len(cfg.discard_phrases) >= 3
