"""
therapy_ai/domain.py

Pure domain dataclasses — NO persistence, NO framework imports.
The rest of the codebase imports from here, not from session_store.py.
This follows Domain-Driven Design: the domain model is independent of
how it is stored or transported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    """A single conversational turn."""
    role: str       # "system" | "user" | "assistant"
    content: str
    timestamp: Optional[datetime] = None
    tokens: Optional[int] = None
    is_flagged: bool = False   # True if a safety pattern was triggered

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TherapySession:
    """An immutable snapshot of a completed or in-progress session."""
    id: str
    created_at: datetime
    ended_at: Optional[datetime] = None
    mood_before: Optional[int] = None    # 1–10
    mood_after:  Optional[int] = None    # 1–10
    rating:      Optional[int] = None    # 1–5
    themes:      list[str] = field(default_factory=list)
    note:        Optional[str] = None
    messages:    list[Message] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        return self.ended_at is not None

    @property
    def mood_delta(self) -> Optional[int]:
        if self.mood_before is not None and self.mood_after is not None:
            return self.mood_after - self.mood_before
        return None

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")


@dataclass
class GenerationResult:
    """Result returned by InferenceEngine after a generation call."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stopped_by: str = "eos"          # "eos" | "max_tokens" | "safety"
    model_name: str = ""
    backend_name: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
