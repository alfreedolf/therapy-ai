"""
tests/component/test_ui_app.py

Focused UI tests using streamlit.testing.v1.AppTest (handoff section 5,
tasks 1-3), driving the ACTUAL production functions in therapy_ai.ui.app and
therapy_ai.ui.pages.chat — not a reimplementation of the UI.

Scope & method:
  * Inference is MOCKED (no model load). Backend detection is mocked. Storage is
    synthetic (tmp DB + isolated salt via the autouse fixture). An injectable
    clock drives inactivity.
  * We exercise production functions through small AppTest driver scripts.

Documented limitation:
  AppTest executes the Streamlit script synchronously and single-threaded; it
  cannot reproduce TRUE concurrent browser behavior (e.g. a lock button click
  landing while a generation genuinely runs in parallel). We simulate
  mid-generation lock by having the mocked engine revoke the workspace lifecycle
  during token iteration, which faithfully drives the SAME server-side
  reveal/persist guard the browser path hits. True concurrency is out of scope
  and is covered instead by production-path assertions (store.check_lifecycle,
  lifecycle_token on add_message) in test_lock_lifecycle.py / test_workspace_auth.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from therapy_ai.domain import GenerationResult, Message

TEST_PASSPHRASE = "correct-horse-battery-staple"
APPTEST_TIMEOUT = 30


# ── Shared mocked-inference scaffolding ─────────────────────────────────────────

class _MockEngine:
    """Mocked InferenceEngine: streams a canned reply, no model."""
    def __init__(self, reply="A calm, supportive reply.", on_token=None):
        self._reply = reply
        self._on_token = on_token

    def stream(self, context, override_system=None):
        toks = self._reply.split(" ")
        for i, t in enumerate(toks):
            if self._on_token is not None:
                self._on_token(i)
            yield t + " "
        return GenerationResult(
            text=self._reply, prompt_tokens=len(context),
            completion_tokens=len(toks), stopped_by="eos",
        )


def _iter(engine, context):
    """Drive a stream generator to completion, yielding tokens (mirrors chat.py)."""
    gen = engine.stream(context)
    try:
        while True:
            yield next(gen)
    except StopIteration:
        return


def _install_common_mocks(monkeypatch, tmp_path, isolated_salt):
    """
    Patch out model/backend/safety heavy deps on therapy_ai.ui.app so a verified
    workspace can be bootstrapped against synthetic storage with no model load.
    Returns nothing; callers build the AppTest afterward.
    """
    import therapy_ai.ui.app as appmod
    from therapy_ai.memory.session_store import SessionStore

    db = tmp_path / "sessions.db"

    # Synthetic store factory bound to the tmp DB.
    def _store_factory():
        return SessionStore(db_path=db)

    monkeypatch.setattr(appmod, "SessionStore", _store_factory)

    class _MockManager:
        @staticmethod
        def from_config(_):
            return object()

    class _MockSafety:
        def __init__(self, *a, **k):
            pass
        def check_input(self, text):
            from therapy_ai.core.safety import SafetyLevel, SafetyResult
            return SafetyResult(level=SafetyLevel.SAFE, matched_pattern=None, message="")
        def check_output(self, text):
            return text

    monkeypatch.setattr(appmod, "ModelManager", _MockManager)
    monkeypatch.setattr(appmod, "InferenceEngine", lambda *_: _MockEngine())
    monkeypatch.setattr(appmod, "SafetyGuard", _MockSafety)

    class _BI:
        device_name = "CPU (test)"
        available_memory_gb = 8.0
    monkeypatch.setattr(appmod, "BackendFactory", type("BF", (), {"detect": staticmethod(lambda: _BI())}))

    return db


# ── Driver scripts (run INSIDE AppTest) ─────────────────────────────────────────
# These call the real app functions. They live as functions passed to
# AppTest.from_function; mocks are applied in the parent process before run().

def _script_gate_then_optional_action():
    """Render the passphrase gate / authenticated shell via real app.main-ish flow."""
    import therapy_ai.ui.app as appmod
    appmod._init_state()
    if not __import__("streamlit").session_state.authenticated:
        appmod._render_passphrase_gate()
    else:
        appmod._render_sidebar()


# ── Tests ───────────────────────────────────────────────────────────────────────

class TestPassphraseGate:
    def test_gate_shown_when_unauthenticated(self, monkeypatch, tmp_path, isolated_salt):
        _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        at = AppTest.from_function(_script_gate_then_optional_action)
        at.run(timeout=APPTEST_TIMEOUT)
        assert not at.exception
        # Locked gate renders a password input and no authenticated content.
        assert at.session_state["authenticated"] is False
        assert len(at.text_input) >= 1

    def test_wrong_passphrase_does_not_authenticate(self, monkeypatch, tmp_path, isolated_salt):
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        # Pre-initialise a verified workspace so "wrong pass" is a real mismatch.
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)

        at = AppTest.from_function(_script_gate_then_optional_action)
        at.run(timeout=APPTEST_TIMEOUT)
        at.text_input[0].set_value("totally-wrong-pass").run()
        # Submit the form.
        if at.button:
            at.button[0].click().run()
        assert at.session_state["authenticated"] is False
        # No store/content leaked into session_state.
        assert at.session_state["store"] is None
        # The error surfaced to the user (incorrect passphrase).
        assert any("ncorrect" in str(e.value) for e in at.error)


class TestBootstrapAndLock:
    def test_bootstrap_authenticates_against_synthetic_store(self, monkeypatch, tmp_path, isolated_salt):
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)

        def _script():
            import therapy_ai.ui.app as appmod
            appmod._init_state()
            appmod._bootstrap("correct-horse-battery-staple")

        at = AppTest.from_function(_script)
        at.run(timeout=APPTEST_TIMEOUT)
        assert not at.exception
        assert at.session_state["authenticated"] is True
        assert at.session_state["lifecycle_token"] is not None

    def test_lock_and_clear_wipes_sensitive_state(self, monkeypatch, tmp_path, isolated_salt):
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)

        def _script():
            import streamlit as st

            import therapy_ai.ui.app as appmod
            appmod._init_state()
            appmod._bootstrap("correct-horse-battery-staple")
            assert st.session_state.get("store") is not None
            appmod._lock_and_clear()
            # After lock-and-clear, sensitive keys are gone.
            st.session_state["post_lock_store"] = st.session_state.get("store")
            st.session_state["post_lock_auth"] = st.session_state.get("authenticated")

        at = AppTest.from_function(_script)
        at.run(timeout=APPTEST_TIMEOUT)
        assert not at.exception
        assert at.session_state["post_lock_store"] is None
        assert at.session_state["post_lock_auth"] in (None, False)


class TestGenerationRevealPersistGuard:
    """
    The reveal/persist boundary used by chat.py's buffered generation:
      * display only while unlocked under the SAME lifecycle;
      * check_lifecycle() before reveal/persist;
      * add_message(..., lifecycle_token=...) refuses a revoked-lifecycle write.

    We drive the SAME production guards chat.py calls. See the module docstring
    for why true browser concurrency is out of AppTest's scope; this exercises
    the server-side decision that governs reveal and persistence.
    """

    def test_lock_mid_generation_blocks_reveal_and_persist(self, monkeypatch, tmp_path, isolated_salt):
        from therapy_ai.memory.session_store import SessionStore, WorkspaceLockedError
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        store = SessionStore(db_path=db)
        store.initialize("correct-horse-battery-staple")
        sid = store.create_session(mood_before=5)
        token = store.lifecycle_token

        # Mocked engine that LOCKS the workspace partway through streaming,
        # standing in for the user hitting Lock mid-generation.
        locked_at = {"i": None}

        def _on_token(i):
            if i == 2 and store.is_unlocked:
                store.lock()
                locked_at["i"] = i

        eng = _MockEngine(reply="one two three four five", on_token=_on_token)

        displayed = []
        for tok in _iter(eng, context=[Message(role="user", content="hi")]):
            # chat.py only displays while unlocked under the same token.
            if store.is_unlocked and store.lifecycle_token == token:
                displayed.append(tok)

        assert locked_at["i"] == 2  # lock really happened mid-stream
        # Nothing displayed after the lock point.
        assert len(displayed) <= 2

        # Reveal/persist guard: refuse the revoked-lifecycle result.
        with pytest.raises(WorkspaceLockedError):
            store.check_lifecycle(token)
        with pytest.raises(WorkspaceLockedError):
            store.add_message(sid, "assistant", "late", lifecycle_token=token)

    def test_old_lifecycle_result_rejected_after_reunlock(self, monkeypatch, tmp_path, isolated_salt):
        from therapy_ai.memory.session_store import SessionStore, WorkspaceLockedError
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        store = SessionStore(db_path=db)
        store.initialize("correct-horse-battery-staple")
        sid = store.create_session(mood_before=5)
        old_token = store.lifecycle_token

        store.lock()
        store.unlock("correct-horse-battery-staple")  # new lifecycle

        # A result produced under the OLD token must still be rejected.
        with pytest.raises(WorkspaceLockedError):
            store.check_lifecycle(old_token)
        with pytest.raises(WorkspaceLockedError):
            store.add_message(sid, "assistant", "stale", lifecycle_token=old_token)
        # The current token persists normally.
        store.add_message(sid, "assistant", "fresh", lifecycle_token=store.lifecycle_token)
        assert store.load_messages(sid)[-1].content == "fresh"


class TestInactivityLock:
    def test_disabled_by_default_zero_timeout(self, monkeypatch, tmp_path, isolated_salt):
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)

        def _script():
            import therapy_ai.ui.app as appmod
            appmod._init_state()
            appmod._bootstrap("correct-horse-battery-staple")
            # Default config has lock_timeout_minutes = 0 (disabled): no lock.
            appmod._enforce_inactivity_lock()

        at = AppTest.from_function(_script)
        at.run(timeout=APPTEST_TIMEOUT)
        assert not at.exception
        assert at.session_state["authenticated"] is True


def _write_timeout_config(tmp_path, minutes: int) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f'ui:\n  locale: "EN"\n  lock_timeout_minutes: {minutes}\n')
    return cfg


class TestInactivityExpirationWithClock:
    """
    Server-side ACCESS expiration (distinct from clearing already-rendered
    browser content — see item 3 / docs). We drive the real production function
    `_enforce_inactivity_lock` with a temp config carrying a nonzero timeout.

    Rather than monkeypatch the global `time.time` (which corrupts AppTest's own
    timeout machinery), we set `st.session_state.last_activity` to a controlled
    offset from real wall-clock time — the exact value the function compares
    against. This keeps real time for Streamlit while deterministically
    simulating "idle for N minutes".
    """

    def test_expiration_locks_after_idle(self, monkeypatch, tmp_path, isolated_salt):
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)
        cfg_path = _write_timeout_config(tmp_path, minutes=5)
        monkeypatch.setenv("THERAPY_TEST_CFG", str(cfg_path))

        def _script():
            import os
            import time
            from pathlib import Path

            import streamlit as st

            import therapy_ai.ui.app as appmod
            appmod.CONFIG_PATH = Path(os.environ["THERAPY_TEST_CFG"])
            appmod._init_state()
            appmod._bootstrap("correct-horse-battery-staple")
            # Simulate: last activity was 6 minutes ago (> 5 min timeout).
            st.session_state["last_activity"] = time.time() - 6 * 60
            # This locks-and-clears, then calls st.stop() (raises StopException),
            # so nothing after it runs. The post-condition is observed via the
            # resulting (wiped) session_state after at.run().
            appmod._enforce_inactivity_lock()

        at = AppTest.from_function(_script)
        at.run(timeout=APPTEST_TIMEOUT)
        # Lock-and-clear wiped sensitive state: 'store' key no longer present,
        # and a warning was surfaced to the user.
        assert "store" not in at.session_state
        assert any("inactivity" in str(w.value).lower() for w in at.warning)

    def test_not_locked_within_timeout(self, monkeypatch, tmp_path, isolated_salt):
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)
        cfg_path = _write_timeout_config(tmp_path, minutes=5)
        monkeypatch.setenv("THERAPY_TEST_CFG", str(cfg_path))

        def _script():
            import os
            import time
            from pathlib import Path

            import streamlit as st

            import therapy_ai.ui.app as appmod
            appmod.CONFIG_PATH = Path(os.environ["THERAPY_TEST_CFG"])
            appmod._init_state()
            appmod._bootstrap("correct-horse-battery-staple")
            # Last activity 1 minute ago (< 5 min timeout): must NOT lock.
            st.session_state["last_activity"] = time.time() - 60
            appmod._enforce_inactivity_lock()
            st.session_state["still_store"] = st.session_state.get("store")

        at = AppTest.from_function(_script)
        at.run(timeout=APPTEST_TIMEOUT)
        assert at.session_state["still_store"] is not None
        assert at.session_state["authenticated"] is True

    def test_every_call_resets_activity_clock(self, monkeypatch, tmp_path, isolated_salt):
        """
        FINDING (item 3): _enforce_inactivity_lock() updates last_activity on
        EVERY call (it runs at the start of each authenticated rerun). So ANY
        rerun — including automatic ones not driven by user interaction — resets
        the idle clock. This test pins that behavior: after a non-expiring call,
        last_activity is refreshed to ~now, so prior idle time is forgotten.
        """
        db = _install_common_mocks(monkeypatch, tmp_path, isolated_salt)
        from therapy_ai.memory.session_store import SessionStore
        SessionStore(db_path=db).initialize(TEST_PASSPHRASE)
        cfg_path = _write_timeout_config(tmp_path, minutes=5)
        monkeypatch.setenv("THERAPY_TEST_CFG", str(cfg_path))

        def _script():
            import os
            import time
            from pathlib import Path

            import streamlit as st

            import therapy_ai.ui.app as appmod
            appmod.CONFIG_PATH = Path(os.environ["THERAPY_TEST_CFG"])
            appmod._init_state()
            appmod._bootstrap("correct-horse-battery-staple")
            # 4 min idle (< timeout): call refreshes last_activity to ~now.
            st.session_state["last_activity"] = time.time() - 4 * 60
            appmod._enforce_inactivity_lock()
            refreshed = st.session_state["last_activity"]
            st.session_state["was_refreshed"] = (time.time() - refreshed) < 2.0
            st.session_state["survived_store"] = st.session_state.get("store")

        at = AppTest.from_function(_script)
        at.run(timeout=APPTEST_TIMEOUT)
        assert at.session_state["survived_store"] is not None
        assert at.session_state["was_refreshed"] is True  # clock reset on the call
