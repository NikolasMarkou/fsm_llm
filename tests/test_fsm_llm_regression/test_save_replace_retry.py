"""Windows: a refused ``os.replace`` of an atomic save is retried, a few times, there only.

CI run 43959038258 (Windows 3.15): 367 of 1,200 concurrent saves of one target raised
``PermissionError: [WinError 5]`` because another writer's replace held it. The three
save sites of the lockstep family (``FileSessionStore.save``, ``save_working_memory``,
``SemanticMemoryStore._save_locked``) carry a local bounded retry (plan d7719b03, D-010:
no shared helper, the anchors of those sites refuse one). The tests simulate the
platform (``sys.platform``) and the refusal (a patched ``os.replace``) on any OS; the
real contention only exists on Windows and is checked by the CI leg.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from fsm_llm.agents import SemanticMemoryStore, load_working_memory, save_working_memory
from fsm_llm.engine.session import FileSessionStore, SessionState
from fsm_llm.memory import BUFFER_CORE, WorkingMemory

#: The refusals a save may absorb on Windows, and the longest backoff it may add.
ATTEMPTS = 12
MAX_TOTAL_SLEEP_SECONDS = 0.5


def _embed(text: str) -> list[float]:
    return [float(len(text))]


def _session(tmp_path: Path) -> tuple[Callable[[], None], Path, Callable[[], object]]:
    store = FileSessionStore(tmp_path / "sessions")
    state = SessionState(
        conversation_id="c1", fsm_id="f", current_state="s", context_data={"k": "v"}
    )
    target = tmp_path / "sessions" / "c1.json"
    return (lambda: store.save("c1", state), target, lambda: store.load("c1"))


def _memory(tmp_path: Path) -> tuple[Callable[[], None], Path, Callable[[], object]]:
    memory = WorkingMemory()
    memory.set(BUFFER_CORE, "name", "Ada")
    target = tmp_path / "wm.json"
    return (
        lambda: save_working_memory(memory, str(target)),
        target,
        lambda: load_working_memory(str(target)).get(BUFFER_CORE, "name"),
    )


def _semantic(tmp_path: Path) -> tuple[Callable[[], None], Path, Callable[[], object]]:
    target = tmp_path / "semantic.json"
    store = SemanticMemoryStore(persist_path=str(target), embed_fn=_embed)
    return (
        store.save,
        target,
        lambda: SemanticMemoryStore.load(str(target), embed_fn=_embed).to_dict()[
            "embedding_model"
        ],
    )


_SITES = pytest.mark.parametrize(
    "build",
    [_session, _memory, _semantic],
    ids=["session", "working-memory", "semantic"],
)


class _Replace:
    """``os.replace`` that refuses *fail* times with *error*, then really replaces."""

    def __init__(self, fail: int, error: type[OSError] = PermissionError) -> None:
        self.fail, self.error, self.calls = fail, error, 0
        self.real = os.replace

    def __call__(self, src: object, dst: object) -> None:
        self.calls += 1
        assert self.calls <= 5 * ATTEMPTS, "the retry is unbounded"
        if self.calls <= self.fail:
            raise self.error(13, "Access is denied")
        self.real(src, dst)  # type: ignore[arg-type]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


def _leftovers(target: Path) -> list[str]:
    return sorted(p.name for p in target.parent.glob("*.tmp"))


@_SITES
class TestWindows:
    def test_a_refused_replace_is_retried_and_the_save_lands(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        build: Callable,
    ) -> None:
        save, target, load = build(tmp_path)
        replace = _Replace(fail=3)
        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(sys, "platform", "win32")

        save()

        assert replace.calls == 4
        assert sleeps == [0.001, 0.002, 0.004]
        assert load() is not None
        assert _leftovers(target) == []

    def test_the_retry_is_bounded_and_the_error_is_the_original(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        build: Callable,
    ) -> None:
        save, target, _ = build(tmp_path)
        replace = _Replace(fail=10**6)
        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(PermissionError, match="Access is denied"):
            save()

        assert replace.calls == ATTEMPTS
        assert len(sleeps) == ATTEMPTS - 1
        assert sum(sleeps) <= MAX_TOTAL_SLEEP_SECONDS
        assert _leftovers(target) == []  # the temp file is removed on failure too

    def test_another_error_is_not_retried(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        build: Callable,
    ) -> None:
        save, target, _ = build(tmp_path)
        replace = _Replace(fail=1, error=FileNotFoundError)
        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(FileNotFoundError):
            save()

        assert replace.calls == 1 and sleeps == []
        assert _leftovers(target) == []


@_SITES
class TestElsewhere:
    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_a_refused_replace_raises_at_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        build: Callable,
        platform: str,
    ) -> None:
        """On POSIX a PermissionError is a real permission problem: no retry, no sleep."""
        save, target, _ = build(tmp_path)
        replace = _Replace(fail=1)
        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(sys, "platform", platform)

        with pytest.raises(PermissionError):
            save()

        assert replace.calls == 1 and sleeps == []
        assert _leftovers(target) == []

    def test_a_plain_save_is_one_replace_and_no_sleep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sleeps: list[float],
        build: Callable,
    ) -> None:
        save, _, load = build(tmp_path)
        replace = _Replace(fail=0)
        monkeypatch.setattr(os, "replace", replace)

        save()

        assert replace.calls == 1 and sleeps == []
        assert load() is not None
