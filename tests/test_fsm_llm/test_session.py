"""FileSessionStore id validation, write and list behaviour, and the logging state its writes touch."""

from __future__ import annotations

import datetime
import decimal
import json
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from fsm_llm.api import API
from tests.test_fsm_llm.support import _blocked_fsm, _mock_llm


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


class _Leaky:
    """An object whose ``__str__`` exposes a secret (the D-008 leak path)."""

    def __str__(self) -> str:
        return "password=zzS3CRETzz"

    __repr__ = __str__


class _SubDatetime(datetime.datetime):
    """Not the exact scalar type, so it may override ``__str__``."""

    def __str__(self) -> str:
        return "sub=zzS3CRETzz"


def _save_with(tmp_path: Any, values: dict[str, Any]) -> tuple[API, str, str]:
    """Start a conversation, put *values* into its context, save to disk."""
    from fsm_llm.session import FileSessionStore

    store = FileSessionStore(tmp_path)
    api = API.from_definition(
        _blocked_fsm(), llm_interface=_mock_llm(), session_store=store
    )
    conv_id, _ = api.start_conversation()
    api.fsm_manager.instances[conv_id].context.data.update(values)
    api.save_session(conv_id)
    text = (tmp_path / f"{conv_id}.json").read_text(encoding="utf-8")
    return api, conv_id, text


class TestStep6SessionSaveDefaultHook:
    """P1 session: the ``default=`` hook redacts every non-scalar object."""

    def test_leaky_object_saved_as_placeholder(self, tmp_path):
        _, _, text = _save_with(
            tmp_path, {"profile": _Leaky(), "nested": {"deep": [_Leaky()]}}
        )
        assert "zzS3CRETzz" not in text
        saved = json.loads(text)["context_data"]
        assert saved["profile"] == "<redacted:_Leaky>"
        assert saved["nested"] == {"deep": ["<redacted:_Leaky>"]}

    def test_set_bytes_frozenset_and_scalar_subclass_are_placeholders(self, tmp_path):
        _, _, text = _save_with(
            tmp_path,
            {
                "tags": {"a"},
                "raw": b"zzS3CRETzz",
                "frozen": frozenset({"b"}),
                "sub": _SubDatetime(2026, 1, 2),
            },
        )
        assert "zzS3CRETzz" not in text
        saved = json.loads(text)["context_data"]
        assert saved["tags"] == "<redacted:set>"
        assert saved["raw"] == "<redacted:bytes>"
        assert saved["frozen"] == "<redacted:frozenset>"
        assert saved["sub"] == "<redacted:_SubDatetime>"

    def test_exact_value_scalars_byte_identical_to_default_str(self, tmp_path):
        from fsm_llm.session import SessionState

        scalars = {
            "when": datetime.datetime(2026, 1, 2, 3, 4, 5),
            "day": datetime.date(2026, 1, 2),
            "at": datetime.time(3, 4, 5),
            "span": datetime.timedelta(seconds=90),
            "amount": decimal.Decimal("9.99"),
            "uid": uuid.UUID("12345678-1234-5678-1234-567812345678"),
        }
        _, _, text = _save_with(tmp_path, scalars)
        state = SessionState.model_validate_json(text)
        assert state.context_data == {k: str(v) for k, v in scalars.items()}
        # The whole file is exactly what the old `default=str` would write.
        state.context_data.update(scalars)
        old_bytes = json.dumps(state.model_dump(), indent=2, default=str)
        assert text == old_bytes

    def test_restore_round_trip(self, tmp_path):
        from fsm_llm.session import FileSessionStore

        when = datetime.datetime(2026, 1, 2, 3, 4, 5)
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        _, conv_id, _ = _save_with(
            tmp_path,
            {
                "when": when,
                "amount": decimal.Decimal("9.99"),
                "uid": uid,
                "profile": _Leaky(),
            },
        )
        api2 = API.from_definition(
            _blocked_fsm(),
            llm_interface=_mock_llm(),
            session_store=FileSessionStore(tmp_path),
        )
        rid, _ = api2.restore_session(conv_id)
        data = api2.get_data(rid)
        assert data["when"] == str(when)
        assert data["amount"] == "9.99"
        assert data["uid"] == str(uid)
        assert data["profile"] == "<redacted:_Leaky>"
