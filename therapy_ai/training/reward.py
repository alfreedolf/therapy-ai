"""
therapy_ai/training/reward.py

GRPO reward function for therapy response quality.
Converts user session ratings + heuristics into a scalar reward signal.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TherapyRewardFn:
    """
    Reward function for GRPO training.

    The reward signal is a weighted combination of:
      1. User rating (from session store) — the primary signal
      2. Structural heuristics — penalise very short/very long responses
      3. Empathy markers — reward responses with known therapeutic language
      4. Safety penalty — heavily penalise harmful or dismissive language

    Callable signature expected by TRL's GRPOTrainer:
        reward_fn(completions: list[str], **kwargs) -> list[float]
    """

    # Empathy/therapeutic language patterns (positive reward)
    _EMPATHY_PATTERNS = [
        r"\b(it sounds like|i hear that|that makes sense|i understand)\b",
        r"\b(you('re| are) (not alone|valid|allowed to feel))\b",
        r"\b(what (do you think|feels right|would help))\b",
        r"\b(i('m| am) (here|listening|with you))\b",
        r"\b(thank you for (sharing|trusting|telling me))\b",
    ]

    # Harmful / dismissive patterns (strong negative reward)
    _HARMFUL_PATTERNS = [
        r"\b(just\s+(get over it|stop worrying|be happy))\b",
        r"\b(you('re| are)\s+(too sensitive|overreacting|dramatic))\b",
        r"\b(that('s| is)\s+(not\s+a\s+)?(real problem|big deal))\b",
        r"(self.?harm\s+is\s+okay|go\s+ahead\s+and)",
    ]

    # Reward weights
    WEIGHT_RATING     = 0.60
    WEIGHT_EMPATHY    = 0.20
    WEIGHT_STRUCTURE  = 0.10
    WEIGHT_SAFETY     = 0.10

    IDEAL_LENGTH_MIN  = 60    # characters
    IDEAL_LENGTH_MAX  = 600   # characters

    def __init__(self):
        import re
        self._empathy_re = [
            re.compile(p, re.IGNORECASE) for p in self._EMPATHY_PATTERNS
        ]
        self._harmful_re = [
            re.compile(p, re.IGNORECASE) for p in self._HARMFUL_PATTERNS
        ]

    def __call__(
        self,
        completions: list[str],
        ratings: Optional[list[Optional[float]]] = None,
        **kwargs,
    ) -> list[float]:
        """
        Called by GRPOTrainer per batch.
        `ratings` is an optional list of user-provided scores (1–5 → 0.0–1.0).
        """
        rewards: list[float] = []
        for i, text in enumerate(completions):
            rating_score = self._rating_score(
                ratings[i] if ratings else None
            )
            empathy_score  = self._empathy_score(text)
            structure_score= self._structure_score(text)
            safety_score   = self._safety_score(text)

            reward = (
                self.WEIGHT_RATING    * rating_score  +
                self.WEIGHT_EMPATHY   * empathy_score +
                self.WEIGHT_STRUCTURE * structure_score +
                self.WEIGHT_SAFETY    * safety_score
            )
            # Clamp to [-1, 1]
            rewards.append(max(-1.0, min(1.0, reward)))

        return rewards

    # ── Component scorers ─────────────────────────────────────────────────────

    def _rating_score(self, rating: Optional[float]) -> float:
        """Normalise 1–5 star rating → -1 to +1."""
        if rating is None:
            return 0.0
        return (rating - 3.0) / 2.0  # 1→-1, 3→0, 5→+1

    def _empathy_score(self, text: str) -> float:
        """Count empathy marker hits → 0 to 1."""
        hits = sum(1 for pat in self._empathy_re if pat.search(text))
        return min(1.0, hits / 3.0)  # cap at 3 hits = full score

    def _structure_score(self, text: str) -> float:
        """Prefer responses in the ideal length range."""
        n = len(text)
        if self.IDEAL_LENGTH_MIN <= n <= self.IDEAL_LENGTH_MAX:
            return 1.0
        elif n < self.IDEAL_LENGTH_MIN:
            return n / self.IDEAL_LENGTH_MIN
        else:
            # Penalty for very long responses
            overshoot = (n - self.IDEAL_LENGTH_MAX) / self.IDEAL_LENGTH_MAX
            return max(0.0, 1.0 - overshoot)

    def _safety_score(self, text: str) -> float:
        """Return -1 for harmful patterns, +1 for clean output."""
        for pat in self._harmful_re:
            if pat.search(text):
                return -1.0
        return 1.0
