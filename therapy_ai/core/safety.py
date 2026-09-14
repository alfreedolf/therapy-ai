"""
therapy_ai/core/safety.py

SafetyGuard: synchronous safety layer that runs before EVERY LLM call.
Handles:
  - Crisis detection (self-harm, suicidal ideation keywords)
  - Prompt injection / jailbreak screening
  - Output post-processing (strip leaking system tokens)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    SAFE = auto()
    CRISIS = auto()        # Requires emergency resource display
    INJECTION = auto()     # Prompt injection / jailbreak attempt
    BLOCKED = auto()       # Hard block — do not pass to LLM


@dataclass
class SafetyResult:
    level: SafetyLevel
    matched_pattern: Optional[str] = None
    message: str = ""

    @property
    def is_safe(self) -> bool:
        return self.level == SafetyLevel.SAFE


@dataclass
class EmergencyResources:
    """Localised emergency contacts shown on CRISIS detection."""
    DE: dict = field(default_factory=lambda: {
        "Telefonseelsorge": "0800 111 0 111 (24/7, kostenlos)",
        "Krisentelefon": "0800 111 0 222",
        "Website": "https://www.telefonseelsorge.de",
    })
    IT: dict = field(default_factory=lambda: {
        "Telefono Amico": "02 2327 2327",
        "Telefono Azzurro": "19696",
        "Website": "https://www.telefoniamico.it",
    })
    INTERNATIONAL: dict = field(default_factory=lambda: {
        "Crisis Text Line": "Text HOME to 741741",
        "International Directory": "https://findahelpline.com",
    })

    def for_locale(self, locale: str = "DE") -> dict:
        locale = locale.upper()
        return getattr(self, locale, self.INTERNATIONAL)


class SafetyGuard:
    """
    Run .check_input(text) before sending to LLM.
    Run .check_output(text) before displaying to user.

    All patterns are loaded once at construction time.
    """

    # ── Crisis keyword patterns ───────────────────────────────────────────────
    _CRISIS_PATTERNS: list[str] = [
        r"\b(suicid|suizid|togliermi la vita|quitarme la vida)\w*\b",
        r"\b(self.?harm|self.?injur|autolesion)\w*\b",
        r"\b(kill\s+myself|end\s+my\s+life|end\s+it\s+all)\b",
        r"\b(voglio\s+morire|ich\s+will\s+sterben|quiero\s+morir)\b",
        r"\b(wrist|overdose|hanging|noose)\b",
        r"\bnon\s+voglio\s+più\s+vivere\b",
        r"\bich\s+halte\s+es\s+nicht\s+mehr\s+aus\b",
    ]

    # ── Injection / jailbreak patterns ────────────────────────────────────────
    _INJECTION_PATTERNS: list[str] = [
        r"ignore\s+(previous|all|prior)\s+instructions?",
        r"you\s+are\s+now\s+(DAN|an?\s+AI\s+without)",
        r"system\s*:\s*(override|prompt|instruction)",
        r"<\|?(im_start|system|endoftext)\|?>",
        r"pretend\s+(you\s+are|to\s+be)\s+(evil|unrestricted|jailbreak)",
        r"forget\s+(your|all|the)\s+(training|guidelines|rules)",
        r"do\s+anything\s+now",
        r"\[INST\].*override",
    ]

    # ── Hard-block patterns (never pass to LLM) ───────────────────────────────
    _BLOCK_PATTERNS: list[str] = [
        r"how\s+to\s+(make|build|create)\s+(bomb|weapon|poison|drug)",
        r"(csam|child\s+porn|minor\s+explicit)",
    ]

    def __init__(self, locale: str = "DE"):
        self.locale = locale
        self.resources = EmergencyResources()
        self._crisis_re = self._compile(_CRISIS := self._CRISIS_PATTERNS)
        self._inject_re = self._compile(self._INJECTION_PATTERNS)
        self._block_re  = self._compile(self._BLOCK_PATTERNS)

    # ── Public API ────────────────────────────────────────────────────────────

    def check_input(self, text: str) -> SafetyResult:
        """
        Screen user input before it reaches the LLM.
        Returns SafetyResult — caller decides what to do with it.
        """
        lowered = text.lower()

        for pattern, compiled in self._block_re:
            if compiled.search(lowered):
                logger.warning("BLOCKED input matched: %s", pattern)
                return SafetyResult(
                    level=SafetyLevel.BLOCKED,
                    matched_pattern=pattern,
                    message="This input cannot be processed.",
                )

        for pattern, compiled in self._inject_re:
            if compiled.search(lowered):
                logger.warning("INJECTION attempt detected: %s", pattern)
                return SafetyResult(
                    level=SafetyLevel.INJECTION,
                    matched_pattern=pattern,
                    message="Prompt injection detected. Your message was not processed.",
                )

        for pattern, compiled in self._crisis_re:
            if compiled.search(lowered):
                logger.warning("CRISIS keyword detected: %s", pattern)
                return SafetyResult(
                    level=SafetyLevel.CRISIS,
                    matched_pattern=pattern,
                    message=self._crisis_message(),
                )

        return SafetyResult(level=SafetyLevel.SAFE)

    def check_output(self, text: str) -> str:
        """
        Post-process LLM output — strip leaked system/special tokens.
        Returns cleaned text.
        """
        # Strip any leaked ChatML or system tags
        text = re.sub(r"<\|im_(start|end)\|>", "", text)
        text = re.sub(r"<\|system\|>.*?<\|/system\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL)
        return text.strip()

    def crisis_resources(self) -> dict:
        return self.resources.for_locale(self.locale)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _crisis_message(self) -> str:
        lines = ["I hear that you're going through something very painful right now."]
        lines.append("You don't have to face this alone. Here are some immediate resources:")
        for name, contact in self.resources.for_locale(self.locale).items():
            lines.append(f"  • {name}: {contact}")
        lines.append("\nPlease reach out — a real person is ready to listen.")
        return "\n".join(lines)

    @staticmethod
    def _compile(
        patterns: list[str],
    ) -> list[tuple[str, re.Pattern]]:
        return [
            (p, re.compile(p, re.IGNORECASE | re.MULTILINE))
            for p in patterns
        ]
