"""The rig of the synthesis driver rows: a real ``API`` on a host with one ``synthesize`` state
and the ``SynthesizeExecutor`` of plan baa37f40, a driver that is the ``API`` plus the host's two
steps, and what a row reads back.

The driver rows (``test_driver_rows.py``) say what a host sees whoever drives it: the results of a
run, the ``synthesis_result`` the parent sees and the trace events. Since step 22 of plan baa37f40
the executor is the only driver (``ProcedureRunner``, the second driver that served as the parity
oracle through steps 20 and 21, is gone with its rows).

Interface of ``ExecutorDriver``: ``run(conversation_id, *, max_steps)`` returns the list of
``AdvanceResult`` of a bounded run (``RunBudgetExceededError`` on a spent budget);
``converse(conversation_id, message)`` returns the reply of one user turn;
``end_conversation(conversation_id)`` ends the conversation. It is a thin wrapper over the
``API``: the engine runs no parent turn after a pop and starts no Pass 1 for a child whose initial
state is a synthesize state, so after each ``converse`` it advances the parent once the result is
final (and a child whose result is not yet written) (D-038 (9)).

Interface of ``make_rig(valid_fsm, tmp_path=None, **options)``: ``valid_fsm`` is the compile the
strong model returns (unless ``replies`` is given); ``tmp_path`` makes a library when given;
``extraction``, ``policy``, ``config`` (keys of the host's ``synthesize`` state), ``context``,
``sink``, ``cheap`` (a replacement cheap model), ``event_log`` and ``host_edit`` (a callable that
edits the host definition dict before the API is built) shape the run. It returns a ``Rig``.
"""

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from fsm_llm.core.definitions import LLMResponseError, LLMUsage
from fsm_llm.engine.api.conversation import API
from fsm_llm.engine.kinds import StateExecutor
from fsm_llm.runtime import EventLog
from fsm_llm.synthesis import (
    FSMLibrary,
    FSMSynthesizer,
    Procedure,
    SynthesisPolicy,
    SynthesizeExecutor,
    digest_of,
    run_gates,
)
from fsm_llm.testing import ScriptedLLM
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM, host_fsm

GOAL = "Book a table for two"
PERSONA = "A calm hotel concierge"
SECRET = "SECRET-TEXT"
NOISE = "PARENT-ONLY-NOISE"
CONTEXT = {"topic": "tables", "noise_note": NOISE, "db_password": "hunter2"}
REPLIES = {
    "start": "Hello",
    "ask": "What is the answer?",
    "done": "Thanks",
    "booked": "Booked!",
    "apology": "Sorry",
}
CANARIES = ("CANARY-CONTEXT", "CANARY-USER", "CANARY-MODEL")

TraceSink = Callable[[str, dict[str, Any]], None]


class ExecutorDriver:
    """The executor's driver: the ``API`` itself, plus the host's two steps the engine does not
    take (see the module docstring)."""

    _FINAL = ("ok", "failed", "aborted")

    def __init__(self, api: API) -> None:
        self.api = api

    def run(self, conversation_id: str, *, max_steps: int) -> list[Any]:
        return list(self.api.run_until_terminal(conversation_id, max_steps=max_steps))

    def converse(self, conversation_id: str, message: str) -> str:
        replies = [self.api.converse(message, conversation_id)]
        replies += self._step_on(conversation_id)
        return "\n".join(r for r in replies if r)

    def end_conversation(self, conversation_id: str) -> None:
        self.api.end_conversation(conversation_id)

    def _step_on(self, conversation_id: str) -> list[str]:
        """Advance while the frame on top sits in a synthesize state that nothing is running:
        its result is final (a refusal, a pop) or not written at all (a child's initial state)."""
        api, replies = self.api, []
        for _ in range(40):
            if api.has_conversation_ended(conversation_id):
                break
            frame = api.conversation_stacks[conversation_id][-1]
            state = frame.fsm_definition.states[api.get_current_state(conversation_id)]
            if state.synthesize is None:
                break
            record = api.get_data(conversation_id).get(state.synthesize.result_key)
            if isinstance(record, dict) and record.get("status") not in self._FINAL:
                break
            replies.append(api.advance(conversation_id).response or "")
        return replies


def make_host(**synthesize: Any) -> dict[str, Any]:
    """greet -> solve (synthesize) -> booked on ok, apology on failed/aborted."""
    return host_fsm(
        name="Concierge",
        description="Host conversation",
        persona=PERSONA,
        states=("start", "solve", "booked", "apology"),
        goal=GOAL,
        inputs=["topic"],
        return_keys=["answer"],
        greet_text="Say hello",
        work_description="run the procedure",
        ok_text="Say booked",
        fail_text="Apologise",
        edge_prefix="go to ",
        synthesize=synthesize,
    )


@dataclass
class Rig:
    api: API
    driver: ExecutorDriver
    conv: str
    cheap: ScriptedRunnerLLM
    strong: ScriptedLLM
    synth: FSMSynthesizer
    library: FSMLibrary | None
    stream: list[tuple[str, dict[str, Any]]]
    executors: list[StateExecutor]

    @property
    def executor(self) -> SynthesizeExecutor:
        """The executor of the executor driver, for the rows that look inside it."""
        (executor,) = self.executors
        assert isinstance(executor, SynthesizeExecutor)
        return executor

    def run(self, max_steps: int = 20) -> list[Any]:
        """A bounded run of the conversation by the driver."""
        return self.driver.run(self.conv, max_steps=max_steps)

    def converse(self, message: str, conversation_id: str | None = None) -> str:
        """One user turn through the driver (the rig's conversation unless one is named)."""
        return self.driver.converse(conversation_id or self.conv, message)

    @property
    def events(self) -> list[tuple[str, dict[str, Any]]]:
        """The lifecycle events; the per-step ``stepped`` ones are ``steps``."""
        return [e for e in self.stream if e[0] != "stepped"]

    @property
    def steps(self) -> list[dict[str, Any]]:
        return [f for name, f in self.stream if name == "stepped"]

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    @property
    def state(self) -> str:
        return self.api.get_current_state(self.conv)

    @property
    def depth(self) -> int:
        return self.api.get_stack_depth(self.conv)

    @property
    def data(self) -> dict[str, Any]:
        return self.api.get_data(self.conv)

    @property
    def result(self) -> dict[str, Any]:
        return self.data["synthesis_result"]

    @property
    def digest(self) -> str:
        return next(f["digest"] for name, f in self.events if name == "obtained")

    def to_solve(self) -> None:
        """One message-free step: start -> solve (a synthesize state)."""
        self.api.advance(self.conv)
        assert self.state == "solve"

    def extractions(self) -> int:
        return sum(1 for kind, _ in self.cheap.call_history if kind == "extract_field")


def make_rig(
    valid_fsm: dict[str, Any],
    tmp_path: Any = None,
    *,
    replies: list[Any] | None = None,
    extraction: dict[str, Any] | None = None,
    policy: SynthesisPolicy | None = None,
    config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    sink: Any = None,
    cheap: ScriptedRunnerLLM | None = None,
    event_log: EventLog | None = None,
    host_edit: Callable[[dict[str, Any]], None] | None = None,
) -> Rig:
    cheap = cheap or ScriptedRunnerLLM(
        {"answer": "42"} if extraction is None else extraction, dict(REPLIES)
    )
    strong = ScriptedLLM([json.dumps(valid_fsm)] if replies is None else replies)
    library = FSMLibrary(tmp_path / "lib") if tmp_path is not None else None
    synth = FSMSynthesizer(strong, policy=policy, library=library)
    definition = make_host(**(config or {}))
    if host_edit is not None:
        host_edit(definition)
    stream: list[tuple[str, dict[str, Any]]] = []
    trace = sink or (lambda e, f: stream.append((e, dict(f))))
    executors: list[StateExecutor] = [SynthesizeExecutor(synth, trace_sink=trace)]
    api = API.from_definition(
        definition=definition,
        llm_interface=cheap,
        event_log=event_log,
        executors=executors,
    )
    drive = ExecutorDriver(api)
    conv, _ = api.start_conversation(CONTEXT if context is None else context)
    return Rig(api, drive, conv, cheap, strong, synth, library, stream, executors)


class Metered(ScriptedRunnerLLM):
    """A cheap model whose extraction calls fail ``outages`` times (Pass 2 still
    answers) and whose meter counts the failures, as a real interface's does."""

    def __init__(self, outages: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.outages, self.errors = outages, 0

    def usage(self) -> LLMUsage:
        return LLMUsage(calls=len(self.call_history), errors=self.errors)

    def extract_field(self, request: Any) -> Any:
        if self.errors < self.outages:
            self.errors += 1
            raise LLMResponseError("provider timed out")
        return super().extract_field(request)


def make_procedure(data: dict[str, Any], **kw: Any) -> Procedure:
    """A gate-passed ``Procedure`` built by hand (no strong model)."""
    kw.setdefault("inputs", ("topic",))
    kw.setdefault("return_keys", ("answer",))
    report = run_gates(copy.deepcopy(data), kw.pop("policy", SynthesisPolicy()), **kw)
    assert report.definition is not None, report.findings
    return Procedure(digest_of(report.definition), report.definition, None, "compiled")


def serve_procedure(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, served: Procedure
) -> None:
    """Make ``obtain`` hand back ``served`` (the compile path is tested elsewhere)."""
    monkeypatch.setattr(rig.synth, "obtain", lambda *a, **k: served)
