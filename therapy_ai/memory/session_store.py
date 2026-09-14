"""
therapy_ai/memory/session_store.py

SQLite-backed encrypted session storage.
All personal data stays local and is encrypted with AES-256-GCM
derived from a user-controlled passphrase via PBKDF2.
"""

from __future__ import annotations

import json
import os
import logging
import hashlib
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import (
    create_engine, Column, String, Text, Integer,
    Float, DateTime, Boolean, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from therapy_ai.domain import TherapySession, Message

logger = logging.getLogger(__name__)

Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class SessionRecord(Base):
    """One complete therapy session."""
    __tablename__ = "sessions"

    id          = Column(String(36), primary_key=True)   # uuid4
    created_at  = Column(DateTime, nullable=False)
    ended_at    = Column(DateTime, nullable=True)
    mood_before = Column(Integer, nullable=True)   # 1–10
    mood_after  = Column(Integer, nullable=True)   # 1–10
    rating      = Column(Integer, nullable=True)   # 1–5 stars
    themes_enc  = Column(Text, nullable=True)       # encrypted JSON list
    note_enc    = Column(Text, nullable=True)       # encrypted free-text note


class MessageRecord(Base):
    """One turn (user or assistant) inside a session."""
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    role       = Column(String(16), nullable=False)   # user | assistant
    content_enc= Column(Text, nullable=False)          # AES-GCM encrypted
    timestamp  = Column(DateTime, nullable=False)
    tokens     = Column(Integer, nullable=True)
    is_flagged = Column(Boolean, default=False)        # crisis keyword hit



# ── Encryption helpers ────────────────────────────────────────────────────────

class Encryptor:
    """
    AES-256-GCM encryption with PBKDF2-derived key.
    The passphrase never leaves memory; the key is re-derived on each app start.
    """

    SALT_FILE = "data/.salt"
    ITERATIONS = 480_000  # OWASP 2024 recommended minimum for PBKDF2-SHA256

    def __init__(self, passphrase: str):
        salt = self._get_or_create_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        self._key = kdf.derive(passphrase.encode("utf-8"))
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ct).decode("utf-8")

    def decrypt(self, ciphertext_b64: str) -> str:
        raw = base64.b64decode(ciphertext_b64)
        nonce, ct = raw[:12], raw[12:]
        return self._aesgcm.decrypt(nonce, ct, None).decode("utf-8")

    @classmethod
    def _get_or_create_salt(cls) -> bytes:
        path = Path(cls.SALT_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return base64.b64decode(path.read_bytes())
        salt = os.urandom(16)
        path.write_bytes(base64.b64encode(salt))
        path.chmod(0o600)
        return salt


# ── Main store class ──────────────────────────────────────────────────────────

class SessionStore:
    """
    Persists and retrieves therapy sessions from a local encrypted SQLite DB.

    Usage:
        store = SessionStore(passphrase="my-secret")
        sid   = store.create_session(mood_before=7)
        store.add_message(sid, role="user", content="I've been feeling anxious...")
        store.add_message(sid, role="assistant", content="That sounds difficult...")
        store.close_session(sid, mood_after=8, rating=5)
        history = store.load_messages(sid)
    """

    DEFAULT_DB_PATH = "data/sessions.db"

    def __init__(self, passphrase: str, db_path: str | Path = DEFAULT_DB_PATH):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._enc = Encryptor(passphrase)
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        # Enforce WAL mode for concurrent reads
        @event.listens_for(self._engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self, mood_before: Optional[int] = None) -> str:
        import uuid
        sid = str(uuid.uuid4())
        with self._Session() as db:
            rec = SessionRecord(
                id=sid,
                created_at=datetime.now(timezone.utc),
                mood_before=mood_before,
            )
            db.add(rec)
            db.commit()
        logger.debug("Session created: %s", sid)
        return sid

    def close_session(
        self,
        session_id: str,
        mood_after: Optional[int] = None,
        rating: Optional[int] = None,
        themes: Optional[list[str]] = None,
        note: Optional[str] = None,
    ) -> None:
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            rec.ended_at = datetime.now(timezone.utc)
            rec.mood_after = mood_after
            rec.rating = rating
            if themes:
                rec.themes_enc = self._enc.encrypt(json.dumps(themes))
            if note:
                rec.note_enc = self._enc.encrypt(note)
            db.commit()

    def list_sessions(self, limit: int = 50) -> list[TherapySession]:
        with self._Session() as db:
            records = (
                db.query(SessionRecord)
                .order_by(SessionRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._hydrate_session(r) for r in records]

    def get_session(self, session_id: str) -> TherapySession:
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            return self._hydrate_session(rec)

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        is_flagged: bool = False,
    ) -> None:
        with self._Session() as db:
            msg = MessageRecord(
                session_id=session_id,
                role=role,
                content_enc=self._enc.encrypt(content),
                timestamp=datetime.now(timezone.utc),
                tokens=tokens,
                is_flagged=is_flagged,
            )
            db.add(msg)
            db.commit()

    def load_messages(self, session_id: str) -> list[Message]:
        with self._Session() as db:
            records = (
                db.query(MessageRecord)
                .filter(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.timestamp.asc())
                .all()
            )
            return [
                Message(
                    role=r.role,
                    content=self._enc.decrypt(r.content_enc),
                    timestamp=r.timestamp,
                    tokens=r.tokens,
                    is_flagged=r.is_flagged,
                )
                for r in records
            ]

    # ── Data sovereignty ──────────────────────────────────────────────────────

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a single session and all its messages."""
        with self._Session() as db:
            db.query(MessageRecord).filter(
                MessageRecord.session_id == session_id
            ).delete()
            db.query(SessionRecord).filter(
                SessionRecord.id == session_id
            ).delete()
            db.commit()
        logger.info("Session %s permanently deleted.", session_id)

    def wipe_all_data(self) -> None:
        """Nuclear option: drop and recreate all tables, destroy salt file."""
        Base.metadata.drop_all(self._engine)
        Base.metadata.create_all(self._engine)
        salt_path = Path(Encryptor.SALT_FILE)
        if salt_path.exists():
            salt_path.unlink()
        logger.warning("ALL local data has been permanently wiped.")

    # ── Mood trend helper ─────────────────────────────────────────────────────

    def mood_series(self) -> list[dict]:
        """Return list of {date, mood_before, mood_after} dicts for charting."""
        with self._Session() as db:
            records = (
                db.query(SessionRecord)
                .filter(SessionRecord.mood_before.isnot(None))
                .order_by(SessionRecord.created_at.asc())
                .all()
            )
            return [
                {
                    "date": r.created_at.strftime("%Y-%m-%d"),
                    "mood_before": r.mood_before,
                    "mood_after": r.mood_after,
                }
                for r in records
            ]

    # ── Private ───────────────────────────────────────────────────────────────

    def _hydrate_session(self, rec: SessionRecord) -> TherapySession:
        themes: list[str] = []
        note: Optional[str] = None
        if rec.themes_enc:
            themes = json.loads(self._enc.decrypt(rec.themes_enc))
        if rec.note_enc:
            note = self._enc.decrypt(rec.note_enc)
        return TherapySession(
            id=rec.id,
            created_at=rec.created_at,
            ended_at=rec.ended_at,
            mood_before=rec.mood_before,
            mood_after=rec.mood_after,
            rating=rec.rating,
            themes=themes,
            note=note,
        )
