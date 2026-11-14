"""`docs/testing.md` is executable: every ```python block runs offline.

The blocks run in document order in ONE shared namespace (a later block uses
the FSM and helpers of an earlier one), from a temporary working directory, with
the model server addresses pointed at a closed port as a belt on top of the
fact that nothing in `fsm_llm.testing` builds a real interface. A block fenced
```python {.notest}``` (or ```diff {.notest}```) is not run; any other language is
ignored. The guide's FSM dicts also go through `test_docs_snippets.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "testing.md"
_RUNNABLE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_ALL_FENCES = re.compile(r"```([A-Za-z]*)([^\n]*)\n")


def _blocks() -> list[str]:
    return _RUNNABLE.findall(_GUIDE.read_text(encoding="utf-8"))


class TestGuideIsExecutable:
    def test_the_guide_has_runnable_blocks(self):
        # a vacuous pass guard: a fence style change must not empty the test
        assert len(_blocks()) >= 8

    def test_every_python_block_runs_offline_in_order(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:9")
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
        namespace: dict[str, object] = {"__name__": "docs_testing_guide"}
        for number, block in enumerate(_blocks(), start=1):
            code = compile(block, f"docs/testing.md#python{number}", "exec")
            exec(code, namespace)

    def test_the_only_unrunnable_fences_are_marked_notest(self):
        text = _GUIDE.read_text(encoding="utf-8")
        opening = [
            (lang, rest.strip())
            for lang, rest in _ALL_FENCES.findall(text)
            if lang  # a closing fence has no language
        ]
        for lang, rest in opening:
            if rest:
                assert rest == "{.notest}", f"unknown fence option {lang} {rest!r}"
        assert any(rest == "{.notest}" for _, rest in opening)

    def test_the_guide_names_every_public_helper(self):
        import fsm_llm.testing as public

        text = _GUIDE.read_text(encoding="utf-8")
        missing = [name for name in public.__all__ if name not in text]
        assert missing == []

    @pytest.mark.parametrize(
        "needle",
        [
            "register_assert_rewrite",
            "deep-copy",
            "tests.conftest",
            "python -O",
        ],
    )
    def test_the_guide_covers_the_notes_it_promises(self, needle):
        assert needle in _GUIDE.read_text(encoding="utf-8")
