"""``SynthesizeExecutor`` (plan baa37f40, step 20.2, D-038): what only the executor does.

The rows every driver must pass (the results of a run, ``synthesis_result``, the trace events, the
reasons, nesting, outages) are ``test_driver_rows.py`` on the ``executor`` driver of ``rig.py``.
This file holds what the runner never did and the census rows of the executor: a Pass 2 failure
after a successful obtain leaves nothing pending and no run counted; the child greeting reaches
the arrival turn's reply and the next message goes to the child; the compile runs under the
parent's turn gate; a cancel during ``obtain`` commits nothing; the record is evidence, never an
instruction (a forged running record pushes nothing, the child sees only the declared inputs and
no grant, a restart proves what ran by the frame's ``fsm_id``); the persona rule; the layering.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions import (
    ConversationBusyError,
    LLMResponseError,
    definition_digest,
    fsm_id_of,
)
from fsm_llm.engine.api.conversation import API
from fsm_llm.engine.kinds import ConversationExecutor, StateExecutor
from fsm_llm.synthesis import (
    FSMSynthesizer,
    RunTrace,
    SynthesizeExecutor,
    TraceRecorder,
)
from fsm_llm.synthesis.gates import GATES
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm.test_fsm_lock_order import (
    _RecLock,
    _Recording,
    order_problems,
)
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM
from tests.test_fsm_llm_synthesis.rig import (
    CONTEXT,
    GOAL,
    PERSONA,
    REPLIES,
    Rig,
    make_host,
    make_procedure,
    make_rig,
)

_BOUND = 10  # seconds: a wait that would hang fails instead
_ROOT = Path(fsm_llm.__file__).resolve().parent


def _rig(valid_fsm: dict[str, Any], tmp_path: Any = None, **kw: Any) -> Rig:
    return make_rig(valid_fsm, tmp_path, **kw)


def _speaking(host: dict[str, Any]) -> None:
    host["states"]["solve"]["response_instructions"] = "Say that you are working"


class _SolveFails(ScriptedRunnerLLM):
    """The cheap model whose reply for the host's ``solve`` state fails while ``failing``."""

    failing = True

    def generate_response(self, request: Any) -> Any:
        if self.failing and self._state_of(request) == "solve":
            raise LLMResponseError("the provider went away")
        return super().generate_response(request)


class _Gated(ScriptedLLM):
    """A strong model whose ``acomplete`` waits for ``release`` (``entered`` says it is inside)."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def acomplete(self, request: Any) -> Any:
        self.entered.set()
        await self.release.wait()
        return await super().acomplete(request)


def _child_id(rig: Rig) -> str:
    return rig.api.conversation_stacks[rig.conv][-1].conversation_id


class TestAPassTwoFailureAfterAnObtain:
    """The record rolls back with the turn: no pending request, no frame, no run counted."""

    def test_nothing_is_left_pending_and_no_run_is_counted(self, valid_fsm, tmp_path):
        cheap = _SolveFails({"answer": "42"}, dict(REPLIES))
        rig = _rig(valid_fsm, tmp_path, cheap=cheap, host_edit=_speaking)

        with pytest.raises(LLMResponseError):
            rig.api.converse("hello", rig.conv)

        assert (rig.state, rig.depth) == ("start", 1)
        assert "synthesis_result" not in rig.data
        digest = rig.digest  # the compile did happen (the `obtained` event)
        assert rig.library is not None
        assert rig.library.get_record(digest).stats == {"runs": 0, "successes": 0}
        # nothing pending: the next entry pushes nothing by itself
        rig.api.get_data(rig.conv)
        assert rig.api.get_stack_depth(rig.conv) == 1

    def test_the_retry_obtains_again_runs_once_and_counts_once(
        self, valid_fsm, tmp_path
    ):
        cheap = _SolveFails({"answer": "42"}, dict(REPLIES))
        rig = _rig(valid_fsm, tmp_path, cheap=cheap, host_edit=_speaking)
        with pytest.raises(LLMResponseError):
            rig.api.converse("hello", rig.conv)
        cheap.failing = False

        rig.converse("hello")
        rig.converse("forty-two")

        assert (rig.state, rig.depth) == ("booked", 1)
        assert rig.result["status"] == "ok"
        assert len(rig.strong.requests) == 1  # the second obtain was a library hit
        assert rig.names.count("obtained") == 2 and rig.names.count("pushed") == 1
        assert rig.library is not None
        assert rig.library.get_record(rig.digest).stats == {"runs": 1, "successes": 1}


class TestTheChildGreetingAndTheNextMessage:
    def test_converse_the_greeting_is_the_arrival_reply_and_the_next_message_is_the_childs(
        self, valid_fsm
    ):
        rig = _rig(valid_fsm)

        reply = rig.api.converse("hello", rig.conv)

        assert (
            reply == "What is the answer?"
        )  # the child's greeting, in the arrival turn
        assert (rig.state, rig.depth) == ("ask", 2)
        # the next message goes to the child: the silent synthesize state eats nothing
        assert rig.api.converse("forty-two", rig.conv) == "Thanks"
        assert rig.extractions() == 1
        assert (rig.state, rig.depth) == ("solve", 1)
        assert rig.result["status"] == "ok" and rig.data["answer"] == "42"

    def test_run_the_greeting_is_in_the_step_result_and_the_next_step_is_the_childs(
        self, valid_fsm
    ):
        rig = _rig(valid_fsm)

        first = rig.api.advance(rig.conv)
        assert first.response == "What is the answer?"
        assert (first.state_after, rig.depth) == ("solve", 2)

        second = rig.api.advance(rig.conv)
        assert second.state_after == "done" and second.ended
        assert rig.depth == 1  # popped by the end settle of the step that ended it
        assert rig.result["status"] == "ok"

    def test_a_run_ends_in_the_parents_terminal_state(self, valid_fsm):
        rig = _rig(valid_fsm)
        results = rig.run(20)
        assert results[-1].ended and rig.state == "booked"


class TestTheCompileRunsUnderTheParentsGate:
    """D-038 (6): ``obtain`` runs in Pass 1 under the parent's ``TurnGate``, so a reader on a
    sibling task of the loop is refused for the whole compile (a known limit, written)."""

    async def test_a_sibling_reader_gets_conversation_busy_error_until_the_compile_ends(
        self, valid_fsm
    ):
        rig = _rig(valid_fsm)
        strong = _Gated([json.dumps(valid_fsm)])
        rig.synth.llm = strong
        turn = asyncio.create_task(rig.api.aconverse("hello", rig.conv))
        await asyncio.wait_for(strong.entered.wait(), _BOUND)

        with pytest.raises(ConversationBusyError):
            rig.api.get_data(rig.conv)
        strong.release.set()
        reply = await asyncio.wait_for(turn, _BOUND)

        assert reply == "What is the answer?"
        assert rig.api.get_stack_depth(rig.conv) == 2

    async def test_a_cancel_during_the_compile_commits_nothing(
        self, valid_fsm, tmp_path
    ):
        rig = _rig(valid_fsm, tmp_path)
        strong = _Gated([json.dumps(valid_fsm)])
        rig.synth.llm = strong
        turn = asyncio.create_task(rig.api.aconverse("hello", rig.conv))
        await asyncio.wait_for(strong.entered.wait(), _BOUND)

        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert (rig.state, rig.depth) == ("start", 1)
        assert "synthesis_result" not in rig.data
        assert rig.library is not None and not list(rig.library.root.rglob("*.json"))
        assert rig.names == []  # no `obtained`: the compile never finished
        # the gate is free and the conversation runs on: the cancel left no half run
        strong.release.set()
        assert await rig.api.aconverse("hello", rig.conv) == "What is the answer?"
        assert rig.depth == 2

    def test_an_interrupt_during_obtain_propagates_and_writes_no_record(
        self, valid_fsm, monkeypatch
    ):
        """The runner wrote a final `failed/unexpected_error` on Ctrl-C; the executor writes no
        record at all (D-038 (5), D-039): nothing is counted and the parent, which the engine
        does not roll back for an interrupt (only for a cancel), sits in the synthesize state
        with no record, so its next turn obtains again."""
        rig = _rig(valid_fsm)
        real = rig.synth.obtain

        def interrupted(*args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

        monkeypatch.setattr(rig.synth, "obtain", interrupted)
        with pytest.raises(KeyboardInterrupt):
            rig.api.converse("hello", rig.conv)

        assert rig.depth == 1 and "synthesis_result" not in rig.data
        assert rig.names == []
        monkeypatch.setattr(rig.synth, "obtain", real)
        rig.api.advance(rig.conv)  # the next turn begins in the state: Pass 1 obtains
        assert (rig.state, rig.depth) == ("ask", 2)  # the child, on top
        assert rig.names == ["obtained", "pushed"]


class TestTheRecordIsEvidenceNeverAnInstruction:
    """D1, D3, D6: what settle pushes comes from the executor's own table or library, judged
    again against the state's own contract; the child gets the declared inputs and nothing else."""

    def _forged(self, **fields: Any) -> dict[str, Any]:
        return {
            "status": "running",
            "state": "solve",
            "mode": "frame",
            "procedure_id": "0" * 64,
            **fields,
        }

    def test_a_forged_record_naming_no_known_procedure_pushes_nothing(self, valid_fsm):
        rig = _rig(
            valid_fsm,
            replies=[],
            context={**CONTEXT, "synthesis_result": self._forged()},
        )

        rig.api.advance(
            rig.conv
        )  # arrives at `solve`: the record belongs, so Pass 1 skips

        assert rig.depth == 1 and rig.strong.requests == []
        assert rig.result == {
            "status": "failed",
            "procedure_id": None,
            "version": None,
            "reason": "push_failed",
        }

    def test_a_forged_record_cannot_run_a_procedure_below_the_states_lifecycle(
        self, valid_fsm, tmp_path
    ):
        first = _rig(valid_fsm, tmp_path)
        first.run(20)
        digest = first.digest

        rig = _rig(
            valid_fsm,
            tmp_path,
            replies=[],
            config={"min_lifecycle": "trusted"},
            context={**CONTEXT, "synthesis_result": self._forged(procedure_id=digest)},
        )
        rig.api.advance(rig.conv)

        assert rig.depth == 1 and rig.strong.requests == []
        assert rig.result["reason"] == "lifecycle_too_low"
        assert rig.result["procedure_id"] == digest

    def test_a_forged_record_of_another_mode_fails_and_never_waits(self, valid_fsm):
        rig = _rig(
            valid_fsm,
            replies=[],
            context={**CONTEXT, "synthesis_result": self._forged(mode="delegated")},
        )
        rig.api.advance(rig.conv)
        assert rig.depth == 1 and rig.result["status"] == "failed"
        assert rig.result["reason"] == "push_failed"

    def test_an_api_with_a_passive_executor_ignores_a_running_record(self, valid_fsm):
        """An executor that serves no request is never asked to push: a forged running record is
        left alone and the state stays (a bare API refuses the state now, D-041, so the
        conversation body stands in under the kind's name; the rule is settle's, D3)."""
        api = API.from_definition(
            definition=make_host(),
            llm_interface=ScriptedRunnerLLM({"answer": "42"}, dict(REPLIES)),
            executors={"synthesize": ConversationExecutor()},
        )
        conv, _ = api.start_conversation(
            {**CONTEXT, "synthesis_result": self._forged(procedure_id="a" * 64)}
        )
        api.advance(conv)
        api.advance(conv)

        assert api.get_stack_depth(conv) == 1 and api.get_current_state(conv) == "solve"
        assert api.get_data(conv)["synthesis_result"]["status"] == "running"

    def test_the_child_sees_the_declared_inputs_only_and_never_a_grant(self, valid_fsm):
        rig = _rig(
            valid_fsm,
            context={**CONTEXT, "_approval_granted": True, "other_key": "other"},
        )
        rig.api.advance(rig.conv)

        child = rig.api.fsm_manager.copy_raw_context_data(_child_id(rig)) or {}

        assert child["topic"] == "tables"
        for key in ("_approval_granted", "other_key", "noise_note", "db_password"):
            assert key not in child, key
        # the parent keeps its own seed: the child got a copy of the declared keys, no more
        parent = rig.api.fsm_manager.copy_raw_context_data(rig.conv) or {}
        assert parent["_approval_granted"] is True and "synthesis_result" in parent

    def test_a_restart_proves_the_run_by_the_frames_fsm_id(self, valid_fsm, tmp_path):
        rig = _rig(valid_fsm, tmp_path)
        rig.api.advance(rig.conv)
        digest = rig.digest
        # a restart: the executor lost its memo of the child
        rig.executor._runs.clear()
        rig.executor._depths.clear()

        rig.api.converse("forty-two", rig.conv)

        assert rig.result == {
            "status": "ok",
            "procedure_id": digest,
            "version": "1.0",
            "reason": None,
        }
        assert rig.library is not None
        assert rig.library.get_record(digest).stats == {"runs": 1, "successes": 1}

    def test_a_restart_with_a_forged_digest_proves_nothing_and_counts_no_run(
        self, valid_fsm, tmp_path
    ):
        """The record is public: after a restart it may name ANOTHER stored procedure than the
        one the child frame runs. The frame's own `fsm_id` is what proves a run (the D-027 rule),
        so the forged digest counts no run for either procedure and merges nothing."""
        rig = _rig(valid_fsm, tmp_path)
        rig.api.advance(rig.conv)
        digest = rig.digest
        other_fsm = json.loads(json.dumps(valid_fsm))
        other_fsm["states"]["done"]["response_instructions"] = "Say another thanks"
        other = make_procedure(other_fsm)
        assert rig.library is not None
        rig.library.put(
            other.definition,
            goal=GOAL,
            inputs_required=("topic",),
            return_keys=("answer",),
            gates_passed=GATES,
            lifecycle="verified",
        )
        assert other.digest != digest
        rig.executor._runs.clear()  # a restart: the memo of the child is gone
        record = dict(
            rig.api.fsm_manager.copy_raw_context_data(rig.conv)["synthesis_result"]
        )
        record["procedure_id"] = other.digest
        rig.api.fsm_manager.update_conversation_context(
            rig.conv, {"synthesis_result": record}
        )

        rig.api.converse("forty-two", rig.conv)

        assert rig.result == {
            "status": "failed",
            "procedure_id": None,
            "version": None,
            "reason": "push_failed",
        }
        assert "answer" not in rig.data  # nothing merged
        for known in (digest, other.digest):
            assert rig.library.get_record(known).stats == {"runs": 0, "successes": 0}


class TestThePersonaRule:
    """P7 (D-038 (3)): the child runs the parent's persona; ``procedure_id`` is the library
    digest and ``instance_identity`` the digest of what was pushed."""

    def test_the_pushed_digest_differs_from_the_library_digest_under_a_persona(
        self, valid_fsm
    ):
        rig = _rig(valid_fsm)
        rig.api.advance(rig.conv)
        record = rig.api.fsm_manager.copy_raw_context_data(rig.conv)["synthesis_result"]
        procedure = make_procedure(valid_fsm)
        pushed = procedure.definition.model_copy(update={"persona": PERSONA})
        child = rig.api.conversation_stacks[rig.conv][-1]

        assert record["procedure_id"] == rig.digest == procedure.digest
        assert record["template_identity"] == rig.digest
        assert record["instance_identity"] == definition_digest(pushed)
        assert record["instance_identity"] != record["template_identity"]
        assert child.fsm_definition.persona == PERSONA
        assert child.fsm_id == fsm_id_of(pushed)
        assert (record["state"], record["mode"]) == ("solve", "frame")

    def test_the_two_digests_are_equal_when_the_host_has_no_persona(self, valid_fsm):
        def no_persona(host: dict[str, Any]) -> None:
            host.pop("persona")

        rig = _rig(valid_fsm, host_edit=no_persona)
        rig.api.advance(rig.conv)
        record = rig.api.fsm_manager.copy_raw_context_data(rig.conv)["synthesis_result"]
        assert record["instance_identity"] == record["template_identity"] == rig.digest

    def test_a_compile_that_brings_a_persona_never_runs_it(self, valid_fsm):
        bad = json.loads(json.dumps(valid_fsm))
        bad["persona"] = "an evil persona"
        rig = _rig(valid_fsm, replies=[json.dumps(bad), json.dumps(valid_fsm)])
        rig.run(20)
        assert "persona-set" in str(rig.strong.requests[1].messages)
        assert rig.result["status"] == "ok"
        assert rig.cheap.shown("generate_response", "ask").count("evil") == 0


class TestTraceMode:
    """``RunTrace.mode`` is ``converse`` for a run a user message started and ``run`` for one a
    message-free step started (the ``step`` of ``pushed`` and ``failed``, D-038 (7))."""

    def _traces(self, valid_fsm: dict[str, Any], how: str) -> list[RunTrace]:
        traces: list[RunTrace] = []
        rig = _rig(valid_fsm, sink=TraceRecorder(traces.append))
        if how == "run":
            rig.run(20)
        else:
            rig.converse("hello")
            rig.converse("forty-two")
        return traces

    @pytest.mark.parametrize("how", ["run", "converse"])
    def test_the_mode_and_the_step_count(self, valid_fsm, how):
        (trace,) = self._traces(valid_fsm, how)
        assert trace.mode == how
        assert trace.status == "ok" and trace.steps == 1  # the child's one counted turn


class TestTheLockRules:
    async def test_a_whole_synthesize_flow_keeps_the_lock_rules(self, valid_fsm):
        """One row beside the call one (D7): the request, the push, the child's turn and the pop
        take no manager lock under a conversation lock."""
        rig = _rig(valid_fsm)
        manager = rig.api.fsm_manager
        events: list[tuple[str, str, int]] = []
        manager._lock = _RecLock("lock", manager._lock, events)  # type: ignore[assignment]
        manager._fsm_cache_lock = _RecLock("cache", manager._fsm_cache_lock, events)  # type: ignore[assignment]
        manager._conversation_locks = _Recording(manager._conversation_locks, events)  # type: ignore[assignment]

        await rig.api.arun_until_terminal(rig.conv, max_steps=20)

        assert rig.result["status"] == "ok" and rig.state == "booked"
        assert order_problems(events) == []
        assert sum(1 for n, w, _ in events if n == "conv" and w == "acquire-nb") >= 3


class TestTheExecutorIsAnEngineKindClient:
    """D4: the executor reaches the engine through ``engine.kinds`` only."""

    def test_it_imports_no_api_turn_or_manager_module(self):
        tree = ast.parse(
            (_ROOT / "synthesis" / "executor.py").read_text(encoding="utf-8")
        )
        names = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ] + [
            a.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for a in node.names
        ]
        engine = [n for n in names if n.startswith("fsm_llm.engine")]
        assert engine, "the executor must name the engine kinds it extends"
        for name in engine:
            assert name.startswith("fsm_llm.engine.kinds"), name

    def test_it_is_a_state_executor_of_the_synthesize_kind(self):
        executor = SynthesizeExecutor(FSMSynthesizer(ScriptedLLM([])))
        assert isinstance(executor, StateExecutor)
        assert (executor.kind, executor.serves_requests, executor.watches_outage) == (
            "synthesize",
            True,
            True,
        )
        assert executor.cleans_extracted is False

    def test_a_trace_sink_must_be_callable(self):
        with pytest.raises(TypeError, match="callable"):
            SynthesizeExecutor(FSMSynthesizer(ScriptedLLM([])), trace_sink="no")  # type: ignore[arg-type]

    def test_the_other_executors_ask_settle_for_nothing(self):
        from fsm_llm.engine.kinds import (
            CallExecutor,
            CompletionExecutor,
            ConversationExecutor,
        )

        for executor in (CallExecutor(), CompletionExecutor(), ConversationExecutor()):
            assert (executor.serves_requests, executor.watches_outage) == (False, False)
