"""
therapy_ai/memory/exporter.py

Exports encrypted sessions → JSONL (ShareGPT format) for fine-tuning.
Data is decrypted in-memory only and written to data/finetune/.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from therapy_ai.memory.session_store import SessionStore
from therapy_ai.memory.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PLACEHOLDER = (
    "You are a compassionate AI therapist. "
    "You support the user with empathy, using CBT and motivational techniques."
)


class FinetuneExporter:
    """
    Converts stored sessions into training-ready JSONL files.

    Output format (ShareGPT / ChatML — compatible with mlx-tune and unsloth):
      {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
      ]}

    Usage:
        exporter = FinetuneExporter(store)
        exporter.export(output_dir="data/finetune", val_split=0.15)
    """

    MIN_TURNS_REQUIRED = 4  # skip sessions that are too short to be useful

    def __init__(
        self,
        store: SessionStore,
        system_prompt: Optional[str] = None,
    ):
        self.store = store
        self.system_prompt = system_prompt or SYSTEM_PROMPT_PLACEHOLDER
        self._builder = ContextBuilder(store)

    def export(
        self,
        output_dir: str | Path = "data/finetune",
        val_split: float = 0.15,
        seed: int = 42,
        min_rating: Optional[int] = None,  # only export sessions rated >= N
        skip_flagged: bool = True,          # skip sessions with crisis flags
    ) -> dict[str, int]:
        """
        Export all eligible sessions to train.jsonl and val.jsonl.
        Returns {"train": N, "val": M} record counts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_sessions = self.store.list_sessions(limit=10_000)
        records: list[dict] = []

        for session in all_sessions:
            # Skip incomplete sessions
            if session.ended_at is None:
                continue
            # Rating filter
            if min_rating and (session.rating is None or session.rating < min_rating):
                continue

            messages = self.store.load_messages(session.id)

            # Skip if any message was crisis-flagged and caller wants clean data
            if skip_flagged and any(m.is_flagged for m in messages):
                logger.debug("Skipping flagged session %s", session.id)
                continue

            # Filter roles, skip very short sessions
            turns = [m for m in messages if m.role in ("user", "assistant")]
            if len(turns) < self.MIN_TURNS_REQUIRED:
                continue

            record = self._to_sharegpt(turns)
            records.append(record)

        if not records:
            logger.warning("No eligible sessions found for export.")
            return {"train": 0, "val": 0}

        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(records)
        split_idx = max(1, int(len(records) * (1 - val_split)))
        train_records = records[:split_idx]
        val_records   = records[split_idx:]

        self._write_jsonl(output_dir / "train.jsonl", train_records)
        self._write_jsonl(output_dir / "val.jsonl",   val_records)

        logger.info(
            "Export complete: %d train / %d val records → %s",
            len(train_records), len(val_records), output_dir,
        )
        return {"train": len(train_records), "val": len(val_records)}

    def preview(self, session_id: str) -> str:
        """Pretty-print a single session as it would appear in training data."""
        msgs = self.store.load_messages(session_id)
        turns = [m for m in msgs if m.role in ("user", "assistant")]
        record = self._to_sharegpt(turns)
        return json.dumps(record, indent=2, ensure_ascii=False)

    # ── Private ───────────────────────────────────────────────────────────────

    def _to_sharegpt(self, turns) -> dict:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [{"role": t.role, "content": t.content} for t in turns]
        return {"messages": messages}

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.debug("Wrote %d records → %s", len(records), path)
