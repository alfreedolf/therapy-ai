"""
therapy_ai/memory/session_store.py

SQLite-backed encrypted session storage.
All personal data stays local and is encrypted with AES-256-GCM
derived from a user-controlled passphrase via PBKDF2.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from therapy_ai.domain import Message, TherapySession

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

    # ── Vault mode ────────────────────────────────────────────────────────────
    # A vault session is locked with its OWN passphrase, independent of the app
    # passphrase. Its messages/notes are encrypted with a key derived from the
    # vault passphrase + a per-session random salt (stored here, NOT in the
    # shared data/.salt). Someone who knows the app passphrase still cannot read
    # a vault session without its vault passphrase.
    is_vault       = Column(Boolean, default=False, nullable=False)
    vault_salt     = Column(Text, nullable=True)   # b64 per-session random salt
    vault_verifier = Column(Text, nullable=True)   # known token enc. w/ vault key


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
    AES-256-GCM encryption with a PBKDF2-SHA256-derived key.
    The passphrase never leaves memory; the key is re-derived on each app start.

    Two construction modes:
      * default  — key derived from the shared app salt file (data/.salt).
                   Used for ordinary (non-vault) sessions.
      * explicit — key derived from a caller-supplied salt. Used for vault
                   sessions, where each session has its own random salt stored
                   in the DB row (never in the shared salt file).
    """

    SALT_FILE = "data/.salt"
    ITERATIONS = 480_000  # OWASP 2024 recommended minimum for PBKDF2-SHA256

    def __init__(self, passphrase: str, salt: bytes | None = None):
        if salt is None:
            salt = self._get_or_create_salt()
        self._aesgcm = AESGCM(self._derive_key(passphrase, salt))

    @classmethod
    def _derive_key(cls, passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=cls.ITERATIONS,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    @classmethod
    def for_vault(cls, passphrase: str, salt: bytes) -> Encryptor:
        """Build an encryptor for a vault session from its own salt."""
        return cls(passphrase, salt=salt)

    @staticmethod
    def new_salt() -> bytes:
        """A fresh 16-byte random salt (for a new vault session)."""
        return os.urandom(16)

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

    # Known plaintext encrypted with the vault key so we can verify a vault
    # passphrase without ever storing the passphrase itself.
    _VAULT_VERIFIER_PLAINTEXT = "therapy-ai-vault-ok"

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
        self._migrate_schema()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    def _migrate_schema(self) -> None:
        """
        Lightweight additive migration for existing databases.

        SQLAlchemy's create_all() creates missing *tables* but never alters an
        existing one, so a DB created before a column was added (e.g. the vault
        columns) is missing them and inserts fail. Here we compare each ORM
        table's declared columns against what the DB actually has and ADD any
        that are missing. Additive only — nothing is dropped or rewritten, so
        existing rows and data are preserved.
        """
        from sqlalchemy import inspect, text

        inspector = inspect(self._engine)
        existing_tables = set(inspector.get_table_names())

        with self._engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    continue  # create_all already handled brand-new tables
                have = {c["name"] for c in inspector.get_columns(table.name)}
                for col in table.columns:
                    if col.name in have:
                        continue
                    ddl_type = col.type.compile(dialect=self._engine.dialect)
                    default = ""
                    if col.default is not None and getattr(col.default, "arg", None) is not None:
                        arg = col.default.arg
                        literal = int(arg) if isinstance(arg, bool) else arg
                        default = f" DEFAULT {literal!r}" if isinstance(literal, str) else f" DEFAULT {literal}"
                    elif not col.nullable:
                        # A non-nullable column needs a default to back-fill
                        # existing rows; 0/'' are safe neutral choices.
                        default = " DEFAULT 0"
                    conn.execute(
                        text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}{default}')
                    )
                    logger.info("Schema migration: added %s.%s", table.name, col.name)

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(
        self,
        mood_before: int | None = None,
        vault_passphrase: str | None = None,
    ) -> str:
        """
        Create a session. If `vault_passphrase` is given, the session is a
        VAULT: a per-session random salt is generated, a verifier token is
        stored (encrypted with the vault key), and all its messages/notes are
        encrypted with the vault key instead of the app key.
        """
        import uuid
        sid = str(uuid.uuid4())
        is_vault = vault_passphrase is not None
        vault_salt_b64 = None
        vault_verifier = None

        if is_vault:
            if not vault_passphrase:
                raise ValueError("Vault passphrase must be non-empty.")
            salt = Encryptor.new_salt()
            venc = Encryptor.for_vault(vault_passphrase, salt)
            vault_salt_b64 = base64.b64encode(salt).decode("utf-8")
            vault_verifier = venc.encrypt(self._VAULT_VERIFIER_PLAINTEXT)

        with self._Session() as db:
            rec = SessionRecord(
                id=sid,
                created_at=datetime.now(UTC),
                mood_before=mood_before,
                is_vault=is_vault,
                vault_salt=vault_salt_b64,
                vault_verifier=vault_verifier,
            )
            db.add(rec)
            db.commit()
        logger.debug("Session created: %s (vault=%s)", sid, is_vault)
        return sid

    # ── Vault helpers ──────────────────────────────────────────────────────────

    def is_vault(self, session_id: str) -> bool:
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            return bool(rec.is_vault)

    def verify_vault_passphrase(self, session_id: str, vault_passphrase: str) -> bool:
        """
        True if `vault_passphrase` can decrypt the session's verifier token.
        Returns False for a wrong passphrase (never raises on mismatch).
        """
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            if not rec.is_vault:
                raise ValueError(f"Session {session_id} is not a vault.")
            try:
                venc = self._vault_encryptor(rec, vault_passphrase)
                return venc.decrypt(rec.vault_verifier) == self._VAULT_VERIFIER_PLAINTEXT
            except Exception:
                return False

    def _vault_encryptor(self, rec: SessionRecord, vault_passphrase: str) -> Encryptor:
        salt = base64.b64decode(rec.vault_salt)
        return Encryptor.for_vault(vault_passphrase, salt)

    def _encryptor_for(
        self, rec: SessionRecord, vault_passphrase: str | None
    ) -> Encryptor:
        """
        Pick the correct encryptor for a session: the vault key for a vault
        session (requires a verified passphrase), else the app key.
        """
        if not rec.is_vault:
            return self._enc
        if vault_passphrase is None:
            raise PermissionError(
                "This session is a vault; a vault passphrase is required."
            )
        venc = self._vault_encryptor(rec, vault_passphrase)
        try:
            ok = venc.decrypt(rec.vault_verifier) == self._VAULT_VERIFIER_PLAINTEXT
        except Exception:
            ok = False
        if not ok:
            raise PermissionError("Incorrect vault passphrase.")
        return venc

    def close_session(
        self,
        session_id: str,
        mood_after: int | None = None,
        rating: int | None = None,
        themes: list[str] | None = None,
        note: str | None = None,
    ) -> None:
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            rec.ended_at = datetime.now(UTC)
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
        tokens: int | None = None,
        is_flagged: bool = False,
        vault_passphrase: str | None = None,
    ) -> None:
        """
        Persist a message. For a vault session, `vault_passphrase` is required
        and the content is encrypted with the vault key; a wrong/missing vault
        passphrase raises PermissionError and nothing is written.
        """
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            enc = self._encryptor_for(rec, vault_passphrase)
            msg = MessageRecord(
                session_id=session_id,
                role=role,
                content_enc=enc.encrypt(content),
                timestamp=datetime.now(UTC),
                tokens=tokens,
                is_flagged=is_flagged,
            )
            db.add(msg)
            db.commit()

    def load_messages(
        self, session_id: str, vault_passphrase: str | None = None
    ) -> list[Message]:
        """
        Load a session's messages. For a vault session, `vault_passphrase` is
        required; a wrong/missing one raises PermissionError (the ciphertext is
        never returned without the correct key).
        """
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            enc = self._encryptor_for(rec, vault_passphrase)
            records = (
                db.query(MessageRecord)
                .filter(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.timestamp.asc())
                .all()
            )
            return [
                Message(
                    role=r.role,
                    content=enc.decrypt(r.content_enc),
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
        note: str | None = None
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
