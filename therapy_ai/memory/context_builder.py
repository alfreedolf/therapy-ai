"""
therapy_ai/memory/context_builder.py

Builds the conversation context list passed to InferenceEngine.
Handles sliding-window truncation AND semantic retrieval via MemoryRetriever.

Two memory layers:
  1. Keyword/theme preamble  — fast, always-on, from session metadata
  2. Semantic retrieval      — optional, uses sqlite-vec embeddings to surface
                               the most relevant past moments for the current query
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from therapy_ai.memory.session_store import SessionStore
from therapy_ai.domain import Message
from therapy_ai.memory.memory_retriever import MemoryRetriever

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    max_history_turns: int = 20       # pairs of (user, assistant) messages
    summary_threshold: int = 40       # turns before we compress older context
    include_themes: bool = True        # prepend keyword themes from previous sessions
    previous_sessions_count: int = 3  # how many past sessions for keyword preamble
    use_semantic_memory: bool = True   # enable sqlite-vec semantic retrieval
    semantic_top_k: int = 3           # number of semantically similar sessions to fetch


class ContextBuilder:
    """
    Assembles the Message list for InferenceEngine from stored history.

    Two-layer memory strategy:
      1. Keyword/theme preamble — pulled from session metadata (fast, always-on).
      2. Semantic retrieval     — uses sqlite-vec to find the most thematically
                                  relevant past sessions for the current user input.
         Layer 2 is optional and gracefully degrades if sqlite-vec is not installed.
      3. Sliding-window truncation of current session history.
    """

    def __init__(
        self,
        store: SessionStore,
        config: Optional[ContextConfig] = None,
        retriever: Optional[MemoryRetriever] = None,
    ):
        self.store = store
        self.config = config or ContextConfig()
        self._retriever: Optional[MemoryRetriever] = retriever
        # Lazy-init retriever if semantic memory is enabled but none passed
        if self.config.use_semantic_memory and retriever is None:
            try:
                self._retriever = MemoryRetriever(db_path=store._engine.url.database)
            except Exception as e:
                logger.warning("Semantic memory unavailable: %s", e)
                self._retriever = None

    def build(self, current_session_id: str, latest_user_text: Optional[str] = None) -> list[Message]:
        """
        Return ordered Message list ready for InferenceEngine.stream().
        Pass latest_user_text to enable semantic retrieval against the current input.
        Does NOT include the system prompt — InferenceEngine adds that.
        """
        messages: list[Message] = []

        # Layer 1: keyword/theme preamble
        preamble = self._build_preamble(current_session_id)

        # Layer 2: semantic retrieval (if available + query text provided)
        semantic_block = ""
        if self._retriever and latest_user_text:
            try:
                semantic_block = self._retriever.retrieve_as_context_string(
                    query=latest_user_text,
                    top_k=self.config.semantic_top_k,
                    exclude_session_id=current_session_id,
                )
            except Exception as e:
                logger.debug("Semantic retrieval skipped: %s", e)

        combined = "\n\n".join(filter(None, [preamble, semantic_block]))
        if combined:
            messages.append(Message(role="system", content=combined))

        # Layer 3: current session messages (sliding window)
        raw = self.store.load_messages(current_session_id)
        history = self._truncate(raw)
        for msg in history:
            messages.append(Message(role=msg.role, content=msg.content))

        return messages

    def build_for_export(self, session_id: str) -> list[dict]:
        """Return session as ShareGPT-format dicts for fine-tuning export."""
        msgs = self.store.load_messages(session_id)
        return [{"role": m.role, "content": m.content} for m in msgs]

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_preamble(self, current_session_id: str) -> str:
        """
        Construct a soft memory note from recent closed sessions.
        Example output injected into context as an additional system message:
          "From previous conversations: The user has explored themes of
           [parental relationships, anxiety]. Mood improved by avg +2 pts."
        """
        if not self.config.include_themes:
            return ""

        sessions = self.store.list_sessions(
            limit=self.config.previous_sessions_count + 5
        )
        # Filter out current session and unclosed sessions
        closed = [
            s for s in sessions
            if s.id != current_session_id and s.ended_at is not None
        ][: self.config.previous_sessions_count]

        if not closed:
            return ""

        all_themes: list[str] = []
        mood_deltas: list[int] = []

        for s in closed:
            all_themes.extend(s.themes)
            if s.mood_before and s.mood_after:
                mood_deltas.append(s.mood_after - s.mood_before)

        parts: list[str] = ["[Memory from previous sessions]"]
        if all_themes:
            unique_themes = list(dict.fromkeys(all_themes))[:8]  # dedup, cap at 8
            parts.append(f"Recurring themes: {', '.join(unique_themes)}.")
        if mood_deltas:
            avg_delta = sum(mood_deltas) / len(mood_deltas)
            direction = "improved" if avg_delta >= 0 else "declined"
            parts.append(
                f"Mood has generally {direction} after sessions "
                f"(avg Δ {avg_delta:+.1f}/10)."
            )
        parts.append("Use this as soft background context, not as a directive.")

        return " ".join(parts)

    def _truncate(self, messages: list[StoredMessage]) -> list[StoredMessage]:
        """Keep only the most recent max_history_turns turn-pairs."""
        max_msgs = self.config.max_history_turns * 2
        if len(messages) <= max_msgs:
            return messages
        logger.debug(
            "Context truncated: %d → %d messages", len(messages), max_msgs
        )
        return messages[-max_msgs:]
