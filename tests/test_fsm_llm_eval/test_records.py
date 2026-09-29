"""Tests for fsm_llm.eval.records: rows, JSON, git metadata, run dirs."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from fsm_llm.definitions import FSMError
from fsm_llm.eval import (
    EvalConfigError,
    EvalDatasetError,
    EvalError,
    append_row,
    git_commit,
    git_short_hash,
    make_run_dir,
    model_slug,
    read_rows,
    utc_now,
    write_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOW = datetime(2026, 9, 29, 9, 30, 45)


class TestRows:
    def test_append_then_read_round_trips_in_order(self, tmp_path: Path):
        path = tmp_path / "rows.jsonl"
        rows = [{"case": "a", "passed": True}, {"case": "b", "passed": False}]
        for row in rows:
            append_row(path, row)
        assert read_rows(path) == rows

    def test_append_never_truncates_existing_rows(self, tmp_path: Path):
        path = tmp_path / "rows.jsonl"
        append_row(path, {"i": 1})
        append_row(path, {"i": 2})
        assert path.read_text(encoding="utf-8").count("\n") == 2

    def test_missing_file_reads_as_empty(self, tmp_path: Path):
        assert read_rows(tmp_path / "absent.jsonl") == []

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / "rows.jsonl"
        path.write_text('{"i": 1}\n\n  \n{"i": 2}\n', encoding="utf-8")
        assert read_rows(path) == [{"i": 1}, {"i": 2}]


class TestWriteJson:
    def test_sorted_indented_with_trailing_newline(self, tmp_path: Path):
        path = tmp_path / "out.json"
        write_json(path, {"b": 1, "a": [1, 2]})
        text = path.read_text(encoding="utf-8")
        assert text.endswith("}\n")
        assert text.index('"a"') < text.index('"b"')
        assert json.loads(text) == {"a": [1, 2], "b": 1}
        assert '\n  "a"' in text


class TestTimeAndGit:
    def test_utc_now_format(self):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", utc_now())

    def test_git_commit_is_full_head_hash_in_repo(self):
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert git_commit(_REPO_ROOT) == head

    def test_git_commit_raises_outside_a_repo(self, tmp_path: Path):
        with pytest.raises((subprocess.CalledProcessError, OSError)):
            git_commit(tmp_path)

    def test_git_short_hash_prefixes_head(self):
        short = git_short_hash(_REPO_ROOT)
        assert short != "unknown"
        assert git_commit(_REPO_ROOT).startswith(short)

    def test_git_short_hash_falls_back_outside_a_repo(self, tmp_path: Path):
        assert git_short_hash(tmp_path) == "unknown"


class TestModelSlug:
    @pytest.mark.parametrize(
        ("model", "slug"),
        [
            ("ollama_chat/qwen3.5:4b", "qwen3.5-4b"),
            ("ollama_chat/qwen3.5:9b-q8_0", "qwen3.5-9b-q8_0"),
            ("gpt-4o-mini", "gpt-4o-mini"),
            ("openrouter/meta/llama:70b", "llama-70b"),
        ],
    )
    def test_old_eval_rule(self, model: str, slug: str):
        assert model_slug(model) == slug


class TestMakeRunDir:
    def test_name_layout(self, tmp_path: Path):
        run_dir = make_run_dir(
            tmp_path, "ollama_chat/qwen3.5:4b", cwd=tmp_path, now=_NOW
        )
        assert run_dir.is_dir()
        assert run_dir.name == "2026-09-29_09-30_unknown_qwen3.5-4b"

    def test_collision_gets_numeric_suffix_and_never_reuses(self, tmp_path: Path):
        first = make_run_dir(tmp_path, "m", cwd=tmp_path, now=_NOW)
        (first / "keep.txt").write_text("x", encoding="utf-8")
        second = make_run_dir(tmp_path, "m", cwd=tmp_path, now=_NOW)
        third = make_run_dir(tmp_path, "m", cwd=tmp_path, now=_NOW)
        assert second.name == f"{first.name}_2"
        assert third.name == f"{first.name}_3"
        assert (first / "keep.txt").read_text(encoding="utf-8") == "x"
        assert not any(second.iterdir())

    def test_creates_missing_parents(self, tmp_path: Path):
        run_dir = make_run_dir(tmp_path / "a" / "b", "m", cwd=tmp_path, now=_NOW)
        assert run_dir.parent == tmp_path / "a" / "b"

    def test_uses_repo_short_hash(self, tmp_path: Path):
        run_dir = make_run_dir(tmp_path, "m", cwd=_REPO_ROOT, now=_NOW)
        assert run_dir.name == f"2026-09-29_09-30_{git_short_hash(_REPO_ROOT)}_m"

    def test_unwritable_root_raises_eval_error(self, tmp_path: Path):
        blocker = tmp_path / "file"
        blocker.write_text("not a dir", encoding="utf-8")
        with pytest.raises(EvalError):
            make_run_dir(blocker, "m", cwd=tmp_path, now=_NOW)


class TestExceptions:
    def test_hierarchy_roots_at_fsm_error(self):
        assert issubclass(EvalError, FSMError)
        assert issubclass(EvalConfigError, EvalError)
        assert issubclass(EvalDatasetError, EvalError)

    def test_details_are_kept(self):
        err = EvalDatasetError("bad", details={"line": 3})
        assert err.details == {"line": 3}
        assert "bad" in str(err)
