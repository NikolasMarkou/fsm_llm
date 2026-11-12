"""FileSessionStore id validation, write and list behaviour, and the logging state its writes touch."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest


def _c4_state(conv_id: str = "c") -> Any:
    from fsm_llm.session import SessionState

    return SessionState(conversation_id=conv_id, fsm_id="f", current_state="s")


@pytest.fixture
def _c5_clean_logging():
    """Snapshot and restore every piece of global logging state C5 touches."""
    from loguru import logger

    import fsm_llm.logging as log_module

    before_ids = list(log_module._library_handler_ids)
    before_streams = dict(log_module._stream_handler_ids)
    before_flag = log_module._file_handler_initialized
    log_module._file_handler_initialized = False
    added: list[int] = []
    yield log_module, added
    for hid in added:
        try:
            logger.remove(hid)
        except ValueError:
            pass
    for hid in list(log_module._library_handler_ids):
        if hid not in before_ids:
            try:
                logger.remove(hid)
            except ValueError:
                pass
    log_module._library_handler_ids[:] = before_ids
    log_module._stream_handler_ids.clear()
    log_module._stream_handler_ids.update(before_streams)
    log_module._file_handler_initialized = before_flag
    logger.disable("fsm_llm")


class TestStep13C4C5:
    def test_c4_trailing_newline_id_rejected(self, tmp_path):
        from fsm_llm.session import FileSessionStore

        store = FileSessionStore(tmp_path)
        with pytest.raises(ValueError):
            store.save("abc\n", _c4_state())
        assert list(tmp_path.iterdir()) == []

    def test_c4_enametoolong_load_exists_delete_soft(self, tmp_path):
        from fsm_llm.session import FileSessionStore

        store = FileSessionStore(tmp_path)
        long_id = "a" * 5000  # regex-valid, but the path is ENAMETOOLONG
        assert store.load(long_id) is None
        assert store.exists(long_id) is False
        assert store.delete(long_id) is False

    def test_c4_list_sessions_filters_invalid_stems(self, tmp_path):
        from fsm_llm.session import FileSessionStore

        store = FileSessionStore(tmp_path)
        store.save("good", _c4_state("good"))
        (tmp_path / "bad.id.json").write_text("{}", encoding="utf-8")
        (tmp_path / "has space.json").write_text("{}", encoding="utf-8")
        listed = store.list_sessions()
        assert listed == ["good"]
        for sid in listed:  # list and load agree
            assert store.load(sid) is not None

    def test_c4_delete_missing_file_returns_false(self, tmp_path):
        # The file vanishes between an existence check and unlink (TOCTOU).
        from pathlib import Path

        from fsm_llm.session import FileSessionStore

        store = FileSessionStore(tmp_path)
        assert store.delete("never_saved") is False
        with patch.object(Path, "exists", return_value=True):
            assert store.delete("never_saved") is False
        store.save("real", _c4_state("real"))
        assert store.delete("real") is True
        assert store.exists("real") is False

    def test_c4_save_fsyncs_before_replace(self, tmp_path):
        import os as _os

        from fsm_llm.session import FileSessionStore

        store = FileSessionStore(tmp_path)
        calls: list[str] = []
        real_fsync, real_replace = _os.fsync, _os.replace

        def _fsync(fd):
            calls.append("fsync")
            return real_fsync(fd)

        def _replace(src, dst):
            calls.append("replace")
            return real_replace(src, dst)

        with (
            patch("fsm_llm.session.os.fsync", side_effect=_fsync),
            patch("fsm_llm.session.os.replace", side_effect=_replace),
        ):
            store.save("durable", _c4_state("durable"))
        assert calls == ["fsync", "replace"]
        assert store.load("durable") is not None

    def test_c5_failed_add_does_not_disable_file_logging(
        self, tmp_path, _c5_clean_logging
    ):
        from fsm_llm.logging import setup_logging

        log_module, added = _c5_clean_logging
        with pytest.raises(ValueError):
            setup_logging(sink="file", level="NOT_A_LEVEL", log_dir=str(tmp_path))
        assert log_module._file_handler_initialized is False
        hid = setup_logging(sink="file", level="INFO", log_dir=str(tmp_path))
        added.append(hid)
        assert hid != -1
        assert log_module._file_handler_initialized is True

    def test_c5_json_line_not_duplicated_into_second_sink(
        self, tmp_path, _c5_clean_logging
    ):
        import io

        from loguru import logger

        from fsm_llm.logging import _make_json_sink, prepare_log_record, setup_logging

        _, added = _c5_clean_logging
        added.append(
            setup_logging(
                sink="file", format="json", level="INFO", log_dir=str(tmp_path)
            )
        )
        buf = io.StringIO()
        added.append(
            logger.add(
                _make_json_sink(buf),
                level="INFO",
                filter=prepare_log_record,
                colorize=False,
            )
        )
        logger.bind(conversation_id="c5").info("c5 probe")
        logger.complete()
        stream_entry = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert stream_entry["message"] == "c5 probe"
        assert "_jsonl" not in stream_entry
        file_lines = [
            line
            for f in tmp_path.iterdir()
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        file_entry = json.loads(file_lines[-1])
        assert file_entry["message"] == "c5 probe"
        assert file_entry["conversation_id"] == "c5"
        assert "_jsonl" not in file_entry
