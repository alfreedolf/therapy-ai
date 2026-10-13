"""
therapy_ai/memory/session_store.py

SQLite-backed encrypted session storage for a single confidential workspace.

Security model (post vault-retirement)
───────────────────────────────────────
* One confidential encrypted workspace per database. Every conversation is
  encrypted by default with a workspace key derived from the user's workspace
  passphrase via PBKDF2-HMAC-SHA256; content is sealed with AES-256-GCM.
* Access is VERIFIED: the workspace passphrase is checked against a stored
  verifier (or, for legacy workspaces, against existing authenticated
  ciphertext) BEFORE the workspace is considered unlocked. Wrong credentials
  never permit writes and never (re)initialise metadata.
* There are NO per-conversation passwords. The old "vault" tier is retired.
  Legacy vault rows (is_vault=1) are NOT read as ordinary records: ordinary
  flows exclude them or raise RetirementRequiredError. Their message content is
  never decrypted with the workspace key. Vault columns are retained (not
  dropped) in this phase so a one-time retirement/conversion tool can migrate
  them.

Distinct operations (clear preconditions)
──────────────────────────────────────────
* initialize()        — create verifier/KDF metadata for a NEW/empty workspace.
* unlock()            — verify an EXISTING workspace against its verifier.
* legacy_bootstrap()  — verify a legacy workspace (no verifier) against existing
                        app-key ciphertext, then persist a verifier.
A supported legacy format (verifier simply absent) is distinguished from
corrupt metadata (verifier present but unusable) — the two produce different
states.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from therapy_ai.domain import Message, TherapySession

logger = logging.getLogger(__name__)

Base = declarative_base()

# Current on-disk crypto/format version for a workspace. Bump when the KDF or
# sealing scheme changes so metadata stays interpretable across versions.
WORKSPACE_FORMAT_VERSION = 1


# ── Exceptions ──────────────────────────────────────────────────────────────

class WorkspaceLockedError(RuntimeError):
    """A protected operation was attempted before a verified unlock."""


class AuthenticationError(RuntimeError):
    """The supplied workspace passphrase failed verification."""


class WorkspaceStateError(RuntimeError):
    """Workspace metadata is in an ambiguous/corrupt state needing a decision."""


class RetirementRequiredError(RuntimeError):
    """
    An operation touched a legacy vault record that has not yet been retired.
    Ordinary flows must not treat such records as ordinary.
    """


# ── Workspace state classification ──────────────────────────────────────────

class WorkspaceState(str, Enum):
    EMPTY = "empty"                 # no data, no verifier → initialize()
    VERIFIED_READY = "verified"     # verifier present → unlock()
    LEGACY_NO_VERIFIER = "legacy"   # has app-key ciphertext, no verifier → legacy_bootstrap()
    VAULT_ONLY_LEGACY = "vault_only"# only vault rows, no app-key anchor → recovery/conversion
    CORRUPT = "corrupt"             # verifier present but unusable → stop & report


# ── ORM Models ────────────────────────────────────────────────────────────────

class WorkspaceMeta(Base):
    """
    Workspace-local, versioned crypto/format metadata. Exactly one row (id=1).
    The verifier is a known token sealed with the workspace key; decrypting it
    proves the passphrase without ever storing the passphrase.
    """
    __tablename__ = "workspace_meta"

    id             = Column(Integer, primary_key=True)   # always 1
    format_version = Column(Integer, nullable=False)
    kdf_algo       = Column(String(32), nullable=False)  # e.g. "pbkdf2-sha256"
    kdf_iterations = Column(Integer, nullable=False)
    verifier       = Column(Text, nullable=False)        # app-key-sealed token
    created_at     = Column(DateTime, nullable=False)


class SessionRecord(Base):
    """One complete therapy session."""
    __tablename__ = "sessions"

    id          = Column(String(36), primary_key=True)   # uuid4
    created_at  = Column(DateTime, nullable=False)
    ended_at    = Column(DateTime, nullable=True)
    mood_before = Column(Integer, nullable=True)   # 1–10   (NOT encrypted)
    mood_after  = Column(Integer, nullable=True)   # 1–10   (NOT encrypted)
    rating      = Column(Integer, nullable=True)   # 1–5    (NOT encrypted)
    themes_enc  = Column(Text, nullable=True)       # encrypted JSON list (app key)
    note_enc    = Column(Text, nullable=True)       # encrypted free-text note (app key)

    # ── Memory contribution policy (authoritative, persisted) ──────────────────
    # Whether THIS conversation may CONTRIBUTE to future cross-session context.
    # Default False (private). Does NOT control whether this conversation
    # RECEIVES context from others — that is a separate global recall setting.
    memory_opt_in = Column(Boolean, default=False, nullable=False)

    # ── Legacy vault marker (RETAINED, not dropped) ────────────────────────────
    # Retired feature. A row with is_vault=1 is a legacy record that has not yet
    # been migrated by the one-time retirement tool. Ordinary flows must exclude
    # it or raise RetirementRequiredError; its content is encrypted with a
    # per-row vault key that the workspace key cannot read.
    is_vault       = Column(Boolean, default=False, nullable=False)
    vault_salt     = Column(Text, nullable=True)
    vault_verifier = Column(Text, nullable=True)


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
    The passphrase never leaves memory; the key is re-derived on each unlock.

    The shared workspace salt lives in a file (data/.salt). KDF parameters are
    recorded in WorkspaceMeta so they stay interpretable and versioned.
    """

    SALT_FILE = "data/.salt"
    ITERATIONS = 480_000  # OWASP 2024 recommended minimum for PBKDF2-SHA256
    ALGO = "pbkdf2-sha256"

    def __init__(self, passphrase: str, salt: bytes | None = None, iterations: int | None = None):
        if salt is None:
            salt = self._get_or_create_salt()
        self._iterations = iterations or self.ITERATIONS
        self._aesgcm = AESGCM(self._derive_key(passphrase, salt, self._iterations))

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(passphrase.encode("utf-8"))

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

    @classmethod
    def read_salt(cls) -> bytes | None:
        """Return the existing salt, or None if the salt file is absent."""
        path = Path(cls.SALT_FILE)
        if not path.exists():
            return None
        return base64.b64decode(path.read_bytes())


# ── Verified workspace authentication ──────────────────────────────────────────

_VERIFIER_PLAINTEXT = "therapy-ai-workspace-ok"


@dataclass
class UnlockResult:
    state: WorkspaceState
    unlocked: bool


class WorkspaceAuth:
    """
    Owns the verifier/KDF metadata and the initialize vs unlock vs
    legacy-bootstrap distinction. Does NOT hold the passphrase beyond deriving
    the key; the caller (SessionStore) keeps the live Encryptor only while
    unlocked.
    """

    def __init__(self, engine):
        self._engine = engine
        self._Session = sessionmaker(bind=engine, expire_on_commit=False)

    # ── State inspection ────────────────────────────────────────────────────
    def classify(self) -> WorkspaceState:
        """
        Determine the workspace state WITHOUT any passphrase. Distinguishes a
        supported legacy format (verifier absent) from corrupt metadata.
        """
        with self._Session() as db:
            meta = db.get(WorkspaceMeta, 1)
            n_sessions = db.query(SessionRecord).count()
            n_app_msgs = (
                db.query(MessageRecord)
                .join(SessionRecord, SessionRecord.id == MessageRecord.session_id)
                .filter(SessionRecord.is_vault == False)  # noqa: E712
                .count()
            )
            n_vault = db.query(SessionRecord).filter(SessionRecord.is_vault == True).count()  # noqa: E712

        if meta is not None:
            # Verifier present. Basic structural sanity (base64, length) only;
            # correctness is proven at unlock. Unusable structure → CORRUPT.
            try:
                raw = base64.b64decode(meta.verifier)
                if len(raw) < 13:  # 12-byte nonce + >=1 ct byte
                    return WorkspaceState.CORRUPT
            except Exception:
                return WorkspaceState.CORRUPT
            return WorkspaceState.VERIFIED_READY

        # No verifier.
        if n_sessions == 0:
            return WorkspaceState.EMPTY
        if n_app_msgs > 0:
            return WorkspaceState.LEGACY_NO_VERIFIER
        if n_vault > 0:
            return WorkspaceState.VAULT_ONLY_LEGACY
        # Sessions exist but no app-key message ciphertext and no vault rows:
        # nothing to anchor verification against → ambiguous, require decision.
        return WorkspaceState.CORRUPT

    # ── Operations ────────────────────────────────────────────────────────────
    def initialize(self, enc: Encryptor) -> None:
        """
        Create the verifier/KDF metadata for a NEW/EMPTY workspace. Refuses if a
        verifier already exists (never overwrites).
        """
        with self._Session() as db:
            if db.get(WorkspaceMeta, 1) is not None:
                raise WorkspaceStateError("Workspace already initialised.")
            meta = WorkspaceMeta(
                id=1,
                format_version=WORKSPACE_FORMAT_VERSION,
                kdf_algo=Encryptor.ALGO,
                kdf_iterations=enc._iterations,
                verifier=enc.encrypt(_VERIFIER_PLAINTEXT),
                created_at=datetime.now(UTC),
            )
            db.add(meta)
            db.commit()
        logger.info("Workspace initialised (format v%d).", WORKSPACE_FORMAT_VERSION)

    def verify(self, enc: Encryptor) -> bool:
        """
        True iff `enc` can decrypt the stored verifier to the known token.
        Never raises on a wrong passphrase. Requires a verifier to exist.
        """
        with self._Session() as db:
            meta = db.get(WorkspaceMeta, 1)
        if meta is None:
            raise WorkspaceStateError("No workspace verifier present.")
        try:
            return enc.decrypt(meta.verifier) == _VERIFIER_PLAINTEXT
        except Exception:
            return False

    def legacy_bootstrap(self, enc: Encryptor) -> bool:
        """
        Verify a legacy workspace (no verifier) by decrypting an EXISTING
        app-key ciphertext (a real non-vault message, note, or themes blob).
        Only on success is a verifier persisted. We NEVER fabricate a verifier
        from the entered passphrase and treat that as proof of correctness.
        Returns True on success (verifier now stored), False otherwise.
        """
        with self._Session() as db:
            if db.get(WorkspaceMeta, 1) is not None:
                raise WorkspaceStateError(
                    "Verifier already present; use unlock(), not legacy_bootstrap()."
                )
            # Find a non-vault authenticated ciphertext to anchor on.
            anchor = (
                db.query(MessageRecord.content_enc)
                .join(SessionRecord, SessionRecord.id == MessageRecord.session_id)
                .filter(SessionRecord.is_vault == False)  # noqa: E712
                .first()
            )
            anchor_ct = anchor[0] if anchor else None
            if anchor_ct is None:
                # Try notes/themes as a secondary anchor.
                srec = (
                    db.query(SessionRecord)
                    .filter(SessionRecord.is_vault == False)  # noqa: E712
                    .filter(
                        (SessionRecord.note_enc.isnot(None))
                        | (SessionRecord.themes_enc.isnot(None))
                    )
                    .first()
                )
                if srec is not None:
                    anchor_ct = srec.note_enc or srec.themes_enc

        if anchor_ct is None:
            # No app-key ciphertext exists to verify against.
            return False

        try:
            enc.decrypt(anchor_ct)  # authenticated: wrong key → exception
        except Exception:
            return False

        # Verified against real ciphertext. Persist a verifier for next time.
        with self._Session() as db:
            if db.get(WorkspaceMeta, 1) is None:
                meta = WorkspaceMeta(
                    id=1,
                    format_version=WORKSPACE_FORMAT_VERSION,
                    kdf_algo=Encryptor.ALGO,
                    kdf_iterations=enc._iterations,
                    verifier=enc.encrypt(_VERIFIER_PLAINTEXT),
                    created_at=datetime.now(UTC),
                )
                db.add(meta)
                db.commit()
        logger.info("Legacy workspace bootstrapped: verifier persisted.")
        return True


# ── Main store class ──────────────────────────────────────────────────────────

class SessionStore:
    """
    Persists and retrieves therapy sessions from a single encrypted SQLite
    workspace. Must be unlocked (verified) before protected operations.

    Construction does NOT unlock. Call:
        store = SessionStore(db_path=...)
        state = store.classify()            # decide which op applies
        store.unlock(passphrase)            # or initialize() / legacy_bootstrap()
    A convenience `open_verified(passphrase)` performs initialize-or-unlock for
    the common EMPTY / VERIFIED_READY cases and raises on anything ambiguous.
    """

    DEFAULT_DB_PATH = "data/sessions.db"

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self._engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self._engine)
        self._migrate_schema()
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._auth = WorkspaceAuth(self._engine)

        # Locked until a verified unlock. These are the ONLY gates for protected
        # operations.
        self._enc: Encryptor | None = None
        self._unlocked = False
        # Lifecycle token: changes on every unlock; results produced under an
        # old token are refused (late-generation-after-lock protection).
        self._lifecycle_token: str | None = None

    # ── Lifecycle / authentication ──────────────────────────────────────────

    def classify(self) -> WorkspaceState:
        return self._auth.classify()

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked and self._enc is not None

    @property
    def lifecycle_token(self) -> str | None:
        """Opaque token identifying the current unlocked lifecycle (or None)."""
        return self._lifecycle_token if self._unlocked else None

    @property
    def active_encryptor(self) -> Encryptor | None:
        """
        The live workspace Encryptor while unlocked, else None. Used by derived
        stores (MemoryRetriever) to seal/unseal protected source summaries.
        """
        return self._enc if self._unlocked else None

    def initialize(self, passphrase: str) -> None:
        """Initialise a NEW/EMPTY workspace and leave it unlocked."""
        state = self._auth.classify()
        if state != WorkspaceState.EMPTY:
            raise WorkspaceStateError(
                f"initialize() requires an EMPTY workspace; state is {state.value}."
            )
        enc = Encryptor(passphrase)
        self._auth.initialize(enc)
        self._set_unlocked(enc)

    def unlock(self, passphrase: str) -> None:
        """
        Verify an EXISTING workspace against its verifier. Fail-closed on wrong
        credentials (no writes, no metadata init) and on missing/corrupt
        metadata. Does NOT handle legacy (no-verifier) workspaces — use
        legacy_bootstrap() for those.
        """
        state = self._auth.classify()
        if state == WorkspaceState.CORRUPT:
            raise WorkspaceStateError(
                "Workspace metadata is corrupt or ambiguous; refusing to unlock. "
                "Manual decision required."
            )
        if state in (WorkspaceState.LEGACY_NO_VERIFIER, WorkspaceState.VAULT_ONLY_LEGACY):
            raise WorkspaceStateError(
                f"Workspace is in legacy state '{state.value}'; use the explicit "
                "legacy bootstrap / recovery path, not ordinary unlock."
            )
        if state == WorkspaceState.EMPTY:
            raise WorkspaceStateError(
                "Workspace is empty; call initialize() to create it."
            )
        # VERIFIED_READY
        salt = Encryptor.read_salt()
        if salt is None:
            # Verifier exists but salt file is gone → cannot derive key. Corrupt.
            raise WorkspaceStateError(
                "Workspace verifier present but salt is missing; fail-closed."
            )
        enc = Encryptor(passphrase, salt=salt)
        if not self._auth.verify(enc):
            raise AuthenticationError("Incorrect workspace passphrase.")
        self._set_unlocked(enc)

    def legacy_bootstrap(self, passphrase: str) -> bool:
        """
        Explicit one-time path for a workspace that has app-key ciphertext but
        no verifier. Verifies against real ciphertext, persists a verifier, and
        unlocks on success. Returns False (and stays locked) if the passphrase
        does not decrypt existing ciphertext.
        """
        state = self._auth.classify()
        if state != WorkspaceState.LEGACY_NO_VERIFIER:
            raise WorkspaceStateError(
                f"legacy_bootstrap() requires LEGACY_NO_VERIFIER; state is {state.value}."
            )
        salt = Encryptor.read_salt()
        if salt is None:
            raise WorkspaceStateError("Legacy workspace has no salt file; fail-closed.")
        enc = Encryptor(passphrase, salt=salt)
        if not self._auth.legacy_bootstrap(enc):
            return False
        self._set_unlocked(enc)
        return True

    def open_verified(self, passphrase: str) -> WorkspaceState:
        """
        Convenience: initialise an EMPTY workspace or unlock a VERIFIED_READY
        one. Raises WorkspaceStateError for legacy/corrupt states so the caller
        routes them to the explicit recovery path. Returns the state acted on.
        """
        state = self._auth.classify()
        if state == WorkspaceState.EMPTY:
            self.initialize(passphrase)
        elif state == WorkspaceState.VERIFIED_READY:
            self.unlock(passphrase)
        else:
            raise WorkspaceStateError(
                f"Workspace requires explicit handling for state '{state.value}'."
            )
        return state

    def lock(self) -> None:
        """
        Service-level access revocation. Drops the key, revokes the lifecycle
        token, and refuses subsequent protected operations until a new unlock.
        Caller is responsible for clearing UI caches/indexes (see UI layer).
        """
        self._enc = None
        self._unlocked = False
        self._lifecycle_token = None

    def _set_unlocked(self, enc: Encryptor) -> None:
        self._enc = enc
        self._unlocked = True
        self._lifecycle_token = uuid.uuid4().hex

    def _require_unlocked(self) -> Encryptor:
        if not self._unlocked or self._enc is None:
            raise WorkspaceLockedError("Workspace is locked; unlock() first.")
        return self._enc

    def check_lifecycle(self, token: str | None) -> None:
        """
        Raise if `token` is not the current unlocked lifecycle. Used to refuse
        persisting/displaying a generation that began under a now-revoked
        unlock (e.g. a stream that finished after the user locked the app).
        """
        if not self._unlocked or token is None or token != self._lifecycle_token:
            raise WorkspaceLockedError(
                "Result is from a revoked workspace lifecycle; discarded."
            )

    # ── Schema migration (additive) ────────────────────────────────────────────

    def _migrate_schema(self) -> None:
        """
        Lightweight additive migration: ADD any ORM-declared column missing from
        an existing table. Additive only — nothing is dropped or rewritten, so
        existing rows survive. (This is how legacy DBs gain memory_opt_in /
        workspace_meta without data loss. Vault columns are retained.)
        """
        from sqlalchemy import inspect, text

        inspector = inspect(self._engine)
        existing_tables = set(inspector.get_table_names())

        with self._engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name not in existing_tables:
                    continue
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
                        default = " DEFAULT 0"
                    conn.execute(
                        text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}{default}')
                    )
                    logger.info("Schema migration: added %s.%s", table.name, col.name)

    # ── Transition guard helpers ────────────────────────────────────────────

    @staticmethod
    def _is_vault_row(rec: SessionRecord) -> bool:
        return bool(rec.is_vault)

    def _guard_not_vault(self, rec: SessionRecord) -> None:
        if self._is_vault_row(rec):
            raise RetirementRequiredError(
                f"Session {rec.id} is a legacy vault record and must be retired "
                "before it can be used. It is not an ordinary conversation."
            )

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(
        self,
        mood_before: int | None = None,
        memory_opt_in: bool = False,
    ) -> str:
        """
        Create an ordinary (workspace-encrypted) session. No per-conversation
        passwords. `memory_opt_in` (default False) controls whether this
        conversation may CONTRIBUTE to future cross-session context.
        """
        self._require_unlocked()
        sid = str(uuid.uuid4())
        with self._Session() as db:
            rec = SessionRecord(
                id=sid,
                created_at=datetime.now(UTC),
                mood_before=mood_before,
                memory_opt_in=memory_opt_in,
                is_vault=False,
            )
            db.add(rec)
            db.commit()
        logger.debug("Session created: %s (memory_opt_in=%s)", sid, memory_opt_in)
        return sid

    def set_memory_opt_in(self, session_id: str, opt_in: bool) -> None:
        """Set a conversation's cross-session CONTRIBUTION consent."""
        self._require_unlocked()
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            self._guard_not_vault(rec)
            rec.memory_opt_in = opt_in
            db.commit()

    def memory_opt_in(self, session_id: str) -> bool:
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            self._guard_not_vault(rec)
            return bool(rec.memory_opt_in)

    def is_vault(self, session_id: str) -> bool:
        """Transition-guard read: True if this is an unretired legacy vault row."""
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            return bool(rec.is_vault)

    def close_session(
        self,
        session_id: str,
        mood_after: int | None = None,
        rating: int | None = None,
        themes: list[str] | None = None,
        note: str | None = None,
    ) -> None:
        enc = self._require_unlocked()
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            self._guard_not_vault(rec)
            rec.ended_at = datetime.now(UTC)
            rec.mood_after = mood_after
            rec.rating = rating
            if themes:
                rec.themes_enc = enc.encrypt(json.dumps(themes))
            if note:
                rec.note_enc = enc.encrypt(note)
            db.commit()

    def list_sessions(
        self, limit: int = 50, include_vault: bool = False
    ) -> list[TherapySession]:
        """
        List ordinary sessions. Unretired vault rows are EXCLUDED by default
        (safe exclusion) so progress/history never treat them as ordinary.
        """
        with self._Session() as db:
            q = db.query(SessionRecord)
            if not include_vault:
                q = q.filter(SessionRecord.is_vault == False)  # noqa: E712
            records = q.order_by(SessionRecord.created_at.desc()).limit(limit).all()
            return [self._hydrate_session(r) for r in records]

    def contributing_session_ids(self) -> list[str]:
        """
        Ids of ordinary (non-vault) sessions that opted in to CONTRIBUTING to
        cross-session memory. Used by preamble/indexing. Excludes vault rows.
        """
        with self._Session() as db:
            rows = (
                db.query(SessionRecord.id)
                .filter(SessionRecord.is_vault == False)  # noqa: E712
                .filter(SessionRecord.memory_opt_in == True)  # noqa: E712
                .all()
            )
            return [r[0] for r in rows]

    def list_vault_session_ids(self) -> list[str]:
        """Ids of unretired legacy vault records (for the retirement tool)."""
        with self._Session() as db:
            rows = (
                db.query(SessionRecord.id)
                .filter(SessionRecord.is_vault == True)  # noqa: E712
                .all()
            )
            return [r[0] for r in rows]

    def get_session(self, session_id: str) -> TherapySession:
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            self._guard_not_vault(rec)
            return self._hydrate_session(rec)

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int | None = None,
        is_flagged: bool = False,
        lifecycle_token: str | None = None,
    ) -> None:
        """
        Persist a message, encrypted with the workspace key. If `lifecycle_token`
        is provided it must match the current unlocked lifecycle (used to refuse
        a late write from a revoked lifecycle).
        """
        enc = self._require_unlocked()
        if lifecycle_token is not None:
            self.check_lifecycle(lifecycle_token)
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            self._guard_not_vault(rec)
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

    def load_messages(self, session_id: str) -> list[Message]:
        """
        Load a session's messages (workspace-decrypted). Raises
        RetirementRequiredError for an unretired vault record — its content is
        NEVER decrypted with the workspace key.
        """
        enc = self._require_unlocked()
        with self._Session() as db:
            rec = db.get(SessionRecord, session_id)
            if rec is None:
                raise ValueError(f"Session not found: {session_id}")
            self._guard_not_vault(rec)
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
        """
        Permanently delete a single session and all its messages. Allowed for
        both ordinary and vault rows (deletion of a vault row does not require
        decrypting it). Derived memory cleanup is handled by the caller/retriever.
        """
        self._require_unlocked()
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
        self._require_unlocked()
        Base.metadata.drop_all(self._engine)
        Base.metadata.create_all(self._engine)
        salt_path = Path(Encryptor.SALT_FILE)
        if salt_path.exists():
            salt_path.unlink()
        self.lock()
        logger.warning("ALL local data has been permanently wiped.")

    # ── Mood trend helper ─────────────────────────────────────────────────────

    def mood_series(self) -> list[dict]:
        """
        {date, mood_before, mood_after} dicts for charting. Excludes unretired
        vault rows.
        """
        with self._Session() as db:
            records = (
                db.query(SessionRecord)
                .filter(SessionRecord.mood_before.isnot(None))
                .filter(SessionRecord.is_vault == False)  # noqa: E712
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
        """
        Build a domain TherapySession. For an unretired vault row we do NOT
        attempt to decrypt themes/note with the workspace key; such a row should
        have been excluded by callers, but if one reaches here we return a
        minimal, content-free snapshot rather than risk a wrong-key decrypt.
        """
        themes: list[str] = []
        note: str | None = None
        if not self._is_vault_row(rec):
            enc = self._require_unlocked()
            if rec.themes_enc:
                themes = json.loads(enc.decrypt(rec.themes_enc))
            if rec.note_enc:
                note = enc.decrypt(rec.note_enc)
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
