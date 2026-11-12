"""``fsm_llm.agents.meta_output.save_artifact``.

Atomic writes: a failed write keeps the old file whole.
"""

from __future__ import annotations

import json

import pytest

from fsm_llm.agents.exceptions import OutputError
from fsm_llm.agents.meta_output import save_artifact


class TestSaveArtifactIsAtomic:
    """F-AM-16: a failed write leaves the old file whole."""

    def test_a_failed_replace_keeps_the_old_file_and_leaves_no_temp(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "bot.json"
        target.write_text('{"old": true}', encoding="utf-8")

        def refuse(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", refuse)
        with pytest.raises(OutputError):
            save_artifact({"name": "new"}, target)
        assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
        assert [p.name for p in tmp_path.iterdir()] == ["bot.json"]

    def test_an_unserialisable_artifact_creates_no_directory(self, tmp_path):
        target = tmp_path / "new" / "dir" / "bot.json"
        with pytest.raises(OutputError):
            save_artifact({"name": object()}, target)
        assert not (tmp_path / "new").exists()
