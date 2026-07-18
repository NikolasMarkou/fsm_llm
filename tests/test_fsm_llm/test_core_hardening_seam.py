"""Integration-seam tests for the core-hardening set.

Each test in this file pins a defect that was reproduced by probe before the
fix landed. They exercise public seams (``FileSessionStore.save``,
``setup_logging``) rather than private helpers.
"""

import os
from unittest.mock import patch

import pytest

from fsm_llm.session import FileSessionStore, SessionState


def _make_state(conversation_id: str = "conv-1") -> SessionState:
    return SessionState(
        conversation_id=conversation_id,
        fsm_id="fsm-1",
        current_state="start",
        context_data={"name": "Ada"},
    )


class TestSessionTempFileCleanup:
    """B2 — ``FileSessionStore.save`` must not leak a temp file on ANY failure.

    Before the fix the write+rename was wrapped in ``except OSError:`` only, so
    a non-OSError (RuntimeError, MemoryError, a TypeError out of json.dumps)
    skipped the unlink and left a ``*.tmp`` file behind permanently.
    """

    def test_non_oserror_failure_leaves_no_temp_file(self, tmp_path):
        store = FileSessionStore(tmp_path)

        with patch(
            "os.replace", side_effect=RuntimeError("simulated non-OSError failure")
        ):
            with pytest.raises(RuntimeError, match="simulated non-OSError failure"):
                store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == [], (
            "save() leaked a temp file after a non-OSError failure"
        )

    def test_oserror_failure_leaves_no_temp_file(self, tmp_path):
        """The pre-existing OSError path must keep working."""
        store = FileSessionStore(tmp_path)

        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == []

    def test_successful_save_writes_file_and_leaves_no_temp(self, tmp_path):
        """Positive control: the fix must not be satisfiable by writing nothing."""
        store = FileSessionStore(tmp_path)
        store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == []
        assert (tmp_path / "sess-1.json").exists()

        loaded = store.load("sess-1")
        assert loaded is not None
        assert loaded.conversation_id == "conv-1"
        assert loaded.current_state == "start"
        assert loaded.context_data == {"name": "Ada"}

    def test_write_failure_leaves_no_temp_file(self, tmp_path):
        """A failure BEFORE the rename must also clean up."""
        store = FileSessionStore(tmp_path)

        with patch("json.dumps", side_effect=TypeError("unserializable")):
            with pytest.raises(TypeError, match="unserializable"):
                store.save("sess-1", _make_state())

        assert list(tmp_path.glob("*.tmp")) == []
        assert not os.path.exists(tmp_path / "sess-1.json")
