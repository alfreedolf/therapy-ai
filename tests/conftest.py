"""
tests/conftest.py

Shared fixtures.

Test layers
───────────
tests/unit/       — pure logic, no I/O (safety regex screening)
tests/component/  — real collaborators with I/O (encrypted SessionStore)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from therapy_ai.core.safety import SafetyGuard
from therapy_ai.memory.session_store import Encryptor, SessionStore


@pytest.fixture(autouse=True)
def isolated_salt(tmp_path, monkeypatch):
    """
    Redirect the encryptor's salt file into the test's tmp dir.

    Encryptor.SALT_FILE is a hardcoded module constant ("data/.salt"). Without
    this redirect, tests would read/write the user's real salt file and could
    interfere with their actual encrypted data. Each test gets a fresh salt.
    """
    salt_path = tmp_path / ".salt"
    monkeypatch.setattr(Encryptor, "SALT_FILE", str(salt_path))
    return salt_path


@pytest.fixture
def guard() -> SafetyGuard:
    """A SafetyGuard with default (DE) locale."""
    return SafetyGuard(locale="DE")


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    """A real SessionStore backed by a throwaway encrypted SQLite file."""
    db_path = tmp_path / "sessions.db"
    return SessionStore(passphrase="correct-horse-battery-staple", db_path=db_path)
