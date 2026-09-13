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
    inject_closing_marker,
    is_closing_context,
    should_auto_close,
    should_show_wrapup_button,
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
