"""The async twins of ``FSMSynthesizer``: ``acompile``, ``aobtain``, ``aapproval`` (plan ace7a8c8 X6.2, D-066).

One ``async def`` body per public name; ``compile``, ``obtain`` and ``approval`` are thin ``run_sync``
wrappers over it. The strong model is reached through ``acall`` (async mode awaits ``acomplete``), a simulator
and a library search through ``acall`` (a worker thread in async mode, there is no twin), the reviewer through
``aoffload`` (a bare callable: a worker thread in async mode, inline in sync mode). The D6 rows of the sync
suite hold in the async body: a cut reply is never parsed, an outage is raised chained and not retried,
approval fails closed on ``is not True``, a refused compile is never stored, and a cancel stores and approves
nothing. Cancel rows use events and bounded waits.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions import CompletionResponse, LLMResponseError
from fsm_llm.synthesis import (
    FSMLibrary,
    SynthesisError,
    SynthesisPolicy,
    SynthesisRefusedError,
)
from fsm_llm.synthesis.synthesizer import FSMSynthesizer
from fsm_llm.testing import ScriptedLLM

RET = ("answer",)
GOAL = "collect an answer from the user"
STRICT = SynthesisPolicy(require_approval=True)
_BOUND = 10  # seconds: a wait that would hang fails instead


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


def _text(fsm: dict[str, Any]) -> str:
    return json.dumps(fsm)


def _empty(synth: FSMSynthesizer) -> bool:
    assert synth.library is not None
    return not list(synth.library.root.rglob("*.json"))


def _synth(
    replies: list[Any], tmp_path: Path | None = None, **kw: Any
) -> tuple[FSMSynthesizer, ScriptedLLM]:
    llm = ScriptedLLM(replies)
    if tmp_path is not None:
        kw.setdefault("library", FSMLibrary(tmp_path / "lib"))
    return FSMSynthesizer(llm, **kw), llm


class _Parked(ScriptedLLM):
    """A strong model whose ``acomplete`` parks until cancelled; ``entered`` says it is inside."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.entered = asyncio.Event()
        self.cancelled = 0

    async def acomplete(self, request):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class _Threads(ScriptedLLM):
    """Records the thread of every ``complete`` (the native twin calls it on the calling thread)."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.threads: list[int] = []

    def complete(self, request):
        self.threads.append(threading.get_ident())
        return super().complete(request)


class TestTheWrappers:
    @pytest.mark.parametrize(
        ("sync", "twin"),
        [("compile", "acompile"), ("obtain", "aobtain"), ("approval", "aapproval")],
    )
    def test_the_sync_name_is_a_thin_run_sync_over_the_async_body(
        self, sync: str, twin: str
    ) -> None:
        assert inspect.iscoroutinefunction(getattr(FSMSynthesizer, twin))
        assert not inspect.iscoroutinefunction(getattr(FSMSynthesizer, sync))
        source = re.sub(
            r'""".*?"""',
            "",
            inspect.getsource(getattr(FSMSynthesizer, sync)),
            flags=re.S,
        )
        body = re.split(r"\)\s*->\s*[^:\n]+:\n", source, maxsplit=1)[1]
        assert re.fullmatch(
            rf"\s*return run_sync\(\s*self\.{twin}\([^;]*\)\s*\)\s*", body
        )


class TestCompile:
    async def test_acompile_equals_compile(self, valid_fsm) -> None:
        sync_synth, sync_llm = _synth([_text(valid_fsm)])
        async_synth, async_llm = _synth([_text(valid_fsm)])
        one = sync_synth.compile("g", return_keys=RET)
        two = await async_synth.acompile("g", return_keys=RET)
        assert (one.data, one.rounds) == (two.data, two.rounds)
        assert one.definition == two.definition
        assert [r.messages for r in sync_llm.requests] == [
            r.messages for r in async_llm.requests
        ]

    async def test_a_bad_round_is_repaired_with_the_same_messages(
        self, valid_fsm
    ) -> None:
        bad = {**valid_fsm, "persona": "a pirate"}
        sync_synth, sync_llm = _synth([_text(bad), _text(valid_fsm)])
        async_synth, async_llm = _synth([_text(bad), _text(valid_fsm)])
        sync_synth.compile("g", return_keys=RET)
        result = await async_synth.acompile("g", return_keys=RET)
        assert result.rounds == 2
        assert [r.messages for r in sync_llm.requests] == [
            r.messages for r in async_llm.requests
        ]

    async def test_bad_arguments_raise_before_any_call(self) -> None:
        synth, llm = _synth([])
        with pytest.raises(ValueError, match="goal"):
            await synth.acompile("  ", return_keys=RET)
        assert llm.requests == []

    async def test_a_truncated_reply_is_never_parsed(self, valid_fsm) -> None:
        """D-005: a cut reply that is valid JSON still fails the round."""
        cut = CompletionResponse(
            kind="final", text=_text(valid_fsm), finish_reason="length"
        )
        synth, llm = _synth([cut, _text(valid_fsm)])
        result = await synth.acompile("g", return_keys=RET)
        assert result.rounds == 2
        assert "reply-truncated" in llm.requests[1].messages[-1]["content"]

    async def test_a_provider_outage_is_chained_and_not_retried(
        self, valid_fsm
    ) -> None:
        outage = LLMResponseError("connection refused")
        synth, llm = _synth([outage, _text(valid_fsm)])
        with pytest.raises(SynthesisError) as info:
            await synth.acompile("g", return_keys=RET)
        assert info.value.__cause__ is outage
        assert len(llm.requests) == 1

    async def test_exhausted_rounds_raise_with_the_codes_of_every_round(
        self, valid_fsm
    ) -> None:
        synth, llm = _synth(
            ["prose", "prose"], policy=SynthesisPolicy(max_repair_rounds=1)
        )
        with pytest.raises(SynthesisError) as info:
            await synth.acompile("g", return_keys=RET)
        assert len(llm.requests) == 2
        assert info.value.details["rounds"] == [
            ("reply-not-json-object",),
            ("reply-not-json-object",),
        ]

    async def test_the_model_is_awaited_natively_on_the_loop_thread(
        self, valid_fsm
    ) -> None:
        llm = _Threads([_text(valid_fsm)])
        await FSMSynthesizer(llm).acompile("g", return_keys=RET)
        assert llm.threads == [threading.get_ident()]

    async def test_the_loop_keeps_running_while_the_model_is_awaited(
        self, valid_fsm
    ) -> None:
        class Slow(ScriptedLLM):
            async def acomplete(self, request):
                for _ in range(20):
                    await asyncio.sleep(0)
                return await super().acomplete(request)

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        task = asyncio.ensure_future(ticker())
        try:
            await FSMSynthesizer(Slow([_text(valid_fsm)])).acompile(
                "g", return_keys=RET
            )
        finally:
            task.cancel()
        assert ticks >= 10

    def test_a_patch_of_the_sync_complete_still_intercepts_compile(
        self, valid_fsm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth, llm = _synth([])
        seen: list[Any] = []

        def fake(request):
            seen.append(request)
            return CompletionResponse(kind="final", text=_text(valid_fsm))

        monkeypatch.setattr(llm, "complete", fake)
        assert synth.compile("g", return_keys=RET).rounds == 1
        assert len(seen) == 1

    async def test_a_cancel_during_the_model_call_ends_the_compile(self) -> None:
        llm = _Parked([])
        task = asyncio.ensure_future(FSMSynthesizer(llm).acompile("g", return_keys=RET))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert llm.cancelled == 1  # a real cancel at the model await


class _Sim:
    """A simulator double that records the thread of ``run`` and passes."""

    def __init__(self) -> None:
        self.threads: list[int] = []

    def run(self, definition, **kwargs):
        self.threads.append(threading.get_ident())
        return SimpleNamespace(passed=True, findings=(), scenarios=[])


class TestTheSimulatorHop:
    async def test_async_mode_runs_the_simulator_on_a_worker_thread(
        self, valid_fsm
    ) -> None:
        sim = _Sim()
        synth, _ = _synth([_text(valid_fsm)], simulator=sim)
        result = await synth.acompile("g", return_keys=RET)
        assert result.simulation is not None
        assert sim.threads and sim.threads[0] != threading.get_ident()

    def test_sync_mode_runs_it_inline_on_the_caller_thread(self, valid_fsm) -> None:
        sim = _Sim()
        synth, _ = _synth([_text(valid_fsm)], simulator=sim)
        synth.compile("g", return_keys=RET)
        assert sim.threads == [threading.get_ident()]


def _reviewer(seen: list[Any], verdict: Any = True):
    def review(request):
        seen.append((threading.get_ident(), request))
        return verdict

    return review


class TestObtain:
    async def test_aobtain_equals_obtain_for_a_compile_and_a_hit(
        self, tmp_path, valid_fsm
    ) -> None:
        one, _ = _synth([_text(valid_fsm)], tmp_path / "a")
        two, llm = _synth([_text(valid_fsm), _text(valid_fsm)], tmp_path / "b")
        first = one.obtain(GOAL, return_keys=RET)
        second = await two.aobtain(GOAL, return_keys=RET)
        assert (first.digest, first.source) == (second.digest, second.source)
        assert second.record is not None and second.record.lifecycle == "verified"
        hit = await two.aobtain(GOAL, return_keys=RET)
        assert (hit.source, hit.digest) == ("library", first.digest)
        assert len(llm.requests) == 1  # the hit made no strong call

    async def test_a_library_search_runs_on_a_worker_in_async_mode(
        self, tmp_path, valid_fsm
    ) -> None:
        synth, _ = _synth([_text(valid_fsm)], tmp_path)
        lib = synth.library
        assert lib is not None
        threads: list[int] = []
        real = lib.search

        def spy(*args, **kwargs):
            threads.append(threading.get_ident())
            return real(*args, **kwargs)

        lib.search = spy  # type: ignore[method-assign]
        await synth.aobtain(GOAL, return_keys=RET)
        assert threads and threads[0] != threading.get_ident()

    def test_in_sync_mode_the_search_is_inline_and_a_patch_intercepts(
        self, tmp_path, valid_fsm
    ) -> None:
        synth, _ = _synth([_text(valid_fsm)], tmp_path)
        lib = synth.library
        assert lib is not None
        threads: list[int] = []
        real = lib.search

        def spy(*args, **kwargs):
            threads.append(threading.get_ident())
            return real(*args, **kwargs)

        lib.search = spy  # type: ignore[method-assign]
        synth.obtain(GOAL, return_keys=RET)
        assert threads == [threading.get_ident()]

    async def test_a_compile_that_needs_no_approval_asks_no_reviewer(
        self, tmp_path, valid_fsm
    ) -> None:
        seen: list[Any] = []
        synth, _ = _synth([_text(valid_fsm)], tmp_path, reviewer=_reviewer(seen))
        proc = await synth.aobtain(GOAL, return_keys=RET)
        assert seen == [] and proc.record is not None
        assert proc.record.provenance["approved_by"] is None


class TestApproval:
    async def test_the_reviewer_runs_on_a_worker_thread_in_async_mode(
        self, tmp_path, valid_fsm
    ) -> None:
        """Named mutant: the reviewer called inline blocks the loop on a human."""
        seen: list[Any] = []
        synth, _ = _synth(
            [_text(valid_fsm)], tmp_path, policy=STRICT, reviewer=_reviewer(seen)
        )
        proc = await synth.aobtain(GOAL, return_keys=RET)
        assert proc.record is not None
        assert proc.record.provenance["approved_by"] == "reviewer"
        assert len(seen) == 1 and seen[0][0] != threading.get_ident()

    def test_in_sync_mode_the_reviewer_runs_inline(self, tmp_path, valid_fsm) -> None:
        seen: list[Any] = []
        synth, _ = _synth(
            [_text(valid_fsm)], tmp_path, policy=STRICT, reviewer=_reviewer(seen)
        )
        synth.obtain(GOAL, return_keys=RET)
        assert seen[0][0] == threading.get_ident()

    async def test_nothing_is_written_before_the_reviewer_answers(
        self, tmp_path, valid_fsm
    ) -> None:
        """Named mutant: a ``put`` before approval leaves a record while the reviewer is asked."""
        synth, _ = _synth([_text(valid_fsm)], tmp_path, policy=STRICT)
        stored_when_asked: list[bool] = []

        def review(request) -> bool:
            stored_when_asked.append(_empty(synth))
            return True

        synth.reviewer = review
        await synth.aobtain(GOAL, return_keys=RET)
        assert stored_when_asked == [True]
        assert not _empty(synth)  # and it is stored once approved

    @pytest.mark.parametrize("verdict", ["yes", 1, 1.0, [True], None, False])
    async def test_a_verdict_that_is_not_true_is_a_denial_and_stores_nothing(
        self, tmp_path, valid_fsm, verdict
    ) -> None:
        synth, _ = _synth(
            [_text(valid_fsm)],
            tmp_path,
            policy=STRICT,
            reviewer=_reviewer([], verdict),
        )
        with pytest.raises(SynthesisRefusedError) as info:
            await synth.aobtain(GOAL, return_keys=RET)
        assert info.value.code == "approval_denied"
        assert _empty(synth)

    async def test_an_awaitable_verdict_is_a_denial_not_an_approval(
        self, tmp_path, valid_fsm
    ) -> None:
        async def review(request) -> bool:
            return True

        synth, _ = _synth([_text(valid_fsm)], tmp_path, policy=STRICT, reviewer=review)
        with pytest.raises(SynthesisRefusedError) as info:
            await synth.aobtain(GOAL, return_keys=RET)
        assert info.value.code == "approval_denied"
        assert _empty(synth)

    async def test_a_raising_reviewer_is_reviewer_error_chained(
        self, tmp_path, valid_fsm
    ) -> None:
        boom = RuntimeError("no human")

        def review(request):
            raise boom

        synth, _ = _synth([_text(valid_fsm)], tmp_path, policy=STRICT, reviewer=review)
        with pytest.raises(SynthesisRefusedError) as info:
            await synth.aobtain(GOAL, return_keys=RET)
        assert info.value.code == "reviewer_error" and info.value.__cause__ is boom
        assert _empty(synth)

    async def test_no_reviewer_is_approval_required(self, tmp_path, valid_fsm) -> None:
        synth, _ = _synth([_text(valid_fsm)], tmp_path, policy=STRICT)
        with pytest.raises(SynthesisRefusedError) as info:
            await synth.aobtain(GOAL, return_keys=RET)
        assert info.value.code == "approval_required" and _empty(synth)

    async def test_aapproval_returns_the_label_or_none(self, valid_fsm) -> None:
        plain, _ = _synth([_text(valid_fsm)])
        result = await plain.acompile("g", return_keys=RET)
        assert await plain.aapproval("g", RET, result) is None
        strict, _ = _synth([_text(valid_fsm)], policy=STRICT, reviewer=_reviewer([]))
        assert await strict.aapproval("g", RET, result) == "reviewer"
        assert strict.approval("g", RET, result) == "reviewer"  # the sync name agrees

    def test_a_patch_of_approval_still_intercepts_obtain(
        self, tmp_path, valid_fsm
    ) -> None:
        synth, _ = _synth([_text(valid_fsm)], tmp_path, policy=STRICT)
        calls: list[str] = []

        def fake(goal, keys, result):
            calls.append(goal)
            return "patched"

        synth.approval = fake  # type: ignore[method-assign]
        proc = synth.obtain(GOAL, return_keys=RET)
        assert calls == [GOAL]
        assert proc.record is not None
        assert proc.record.provenance["approved_by"] == "patched"


class TestSubclassBypass:
    """D-068: an override of a sync name without its twin is warned about and is bypassed async."""

    @pytest.mark.parametrize(
        ("sync", "twin"),
        [("compile", "acompile"), ("obtain", "aobtain"), ("approval", "aapproval")],
    )
    def test_a_sync_override_without_its_twin_warns_at_the_class_statement(
        self, sync: str, twin: str
    ) -> None:
        with pytest.warns(
            RuntimeWarning, match=f"overrides {sync} but not {twin}"
        ) as w:
            type("Sub", (FSMSynthesizer,), {sync: lambda self, *a, **k: None})
        assert len(w) == 1
        assert Path(w[0].filename).name == Path(__file__).name

    def test_a_subclass_with_the_twin_or_the_twin_alone_or_neither_is_silent(
        self,
    ) -> None:
        async def twin(self, *a, **k):
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            type("Both", (FSMSynthesizer,), {"approval": twin, "aapproval": twin})
            type("TwinOnly", (FSMSynthesizer,), {"aapproval": twin})
            type("Neither", (FSMSynthesizer,), {"x": 1})

    def test_the_base_class_itself_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _synth([])

    async def test_an_async_caller_of_a_sync_only_approval_override_gets_the_base_approval(
        self, tmp_path, valid_fsm
    ) -> None:
        # The pinned limit: the stricter sync override never runs under aobtain (it does under
        # the sync name). The base G7 still holds: the reviewer must say exactly True.
        def deny(self, goal, return_keys, result):
            raise SynthesisRefusedError("never", "approval_denied")

        with pytest.warns(RuntimeWarning, match="overrides approval but not aapproval"):
            Strict = type("Strict", (FSMSynthesizer,), {"approval": deny})
        seen: list[Any] = []
        strict = Strict(
            ScriptedLLM([_text(valid_fsm), _text(valid_fsm)]),
            policy=STRICT,
            library=FSMLibrary(tmp_path / "lib"),
            reviewer=_reviewer(seen),
        )
        proc = await strict.aobtain(GOAL, return_keys=RET)
        assert proc.record is not None and len(seen) == 1  # the base G7 ran
        denying = Strict(
            ScriptedLLM([_text(valid_fsm)]),
            policy=STRICT,
            library=FSMLibrary(tmp_path / "lib2"),
            reviewer=_reviewer([], verdict="yes"),
        )
        with pytest.raises(SynthesisRefusedError) as info:
            await denying.aobtain(GOAL, return_keys=RET)
        assert info.value.code == "approval_denied"  # the base fail-closed rule
        with pytest.raises(SynthesisRefusedError, match="never"):
            strict.obtain(
                GOAL + " other", return_keys=RET
            )  # sync mode runs the override

    async def test_an_override_of_the_async_twin_is_honoured_by_aobtain(
        self, tmp_path, valid_fsm
    ) -> None:
        class Strict(FSMSynthesizer):
            async def aapproval(self, goal, return_keys, result):
                raise SynthesisRefusedError("never", "approval_denied")

        seen: list[Any] = []
        synth = Strict(
            ScriptedLLM([_text(valid_fsm)]),
            policy=STRICT,
            library=FSMLibrary(tmp_path / "lib"),
            reviewer=_reviewer(seen),
        )
        with pytest.raises(SynthesisRefusedError, match="never"):
            await synth.aobtain(GOAL, return_keys=RET)
        assert seen == [] and _empty(synth)


class TestCancel:
    async def test_a_cancel_during_the_compile_stores_nothing_and_asks_nobody(
        self, tmp_path
    ) -> None:
        seen: list[Any] = []
        llm = _Parked([])
        synth = FSMSynthesizer(
            llm,
            policy=STRICT,
            library=FSMLibrary(tmp_path / "lib"),
            reviewer=_reviewer(seen),
        )
        task = asyncio.ensure_future(synth.aobtain(GOAL, return_keys=RET))
        await asyncio.wait_for(llm.entered.wait(), _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert seen == [] and _empty(synth)

    async def test_a_cancel_during_the_review_stores_nothing_even_if_it_says_yes_late(
        self, tmp_path, valid_fsm
    ) -> None:
        entered, release, done = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def review(request) -> bool:
            entered.set()
            assert release.wait(_BOUND)
            done.set()
            return True  # a late approval, after the caller left

        synth, _ = _synth([_text(valid_fsm)], tmp_path, policy=STRICT, reviewer=review)
        task = asyncio.ensure_future(synth.aobtain(GOAL, return_keys=RET))
        assert await asyncio.to_thread(entered.wait, _BOUND)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _BOUND)
        assert not done.is_set()  # the cancel did not wait for the human
        release.set()
        assert await asyncio.to_thread(done.wait, _BOUND)  # the worker ran on
        for _ in range(5):
            await asyncio.sleep(0)
        assert _empty(synth)  # and its late verdict stored and approved nothing
