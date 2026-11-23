"""The rig of the synthesis driver rows: a real ``API`` on a host with one ``synthesize`` state,
a driver with the call surface of ``ProcedureRunner``, and what a row reads back.

The runner rows (``test_driver_rows.py``) say what a driver must do, whoever the driver is: the
results of a run, the ``synthesis_result`` the parent sees and the trace events. They run on every
entry of ``DRIVERS``; the runner is the only driver today, and the executor of plan baa37f40
(step 20) adds its own entry, so its parity with the runner is a row-by-row comparison, not a
second suite. Rows that pin the runner's own bookkeeping (``_runs``, ``_refused``, ``hook``, the
stream settle on close, a signal during the compile) stay on ``test_runner.py`` and use
``Rig.runner``.

Interface of a driver (``Driver``): ``run(conversation_id, *, max_steps)`` returns the list of
``AdvanceResult`` of a bounded run (``RunBudgetExceededError`` on a spent budget);
``converse(conversation_id, message)`` returns the reply of one user turn;
``end_conversation(conversation_id)`` drops what the driver holds for a conversation. A factory
takes ``(api, host_definition, synthesizer, trace_sink)`` and returns a driver; ``trace_sink`` is
``None`` or a callable ``(event, fields)``.

Interface of ``make_rig(valid_fsm, tmp_path=None, **options)``: ``valid_fsm`` is the compile the
strong model returns (unless ``replies`` is given); ``tmp_path`` makes a library when given;
``extraction``, ``policy``, ``config`` (keys of the host's ``synthesize`` state), ``context``,
``sink``, ``cheap`` (a replacement cheap model), ``event_log`` and ``driver`` (a key of
``DRIVERS``, default ``DEFAULT_DRIVER``) shape the run. It returns a ``Rig``.
"""

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pytest

from fsm_llm.core.definitions import LLMResponseError, LLMUsage
from fsm_llm.engine.api.conversation import API
from fsm_llm.runtime import EventLog
from fsm_llm.synthesis import (
    FSMLibrary,
    FSMSynthesizer,
    Procedure,
    ProcedureRunner,
    SynthesisPolicy,
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


class Driver(Protocol):
    def run(self, conversation_id: str, *, max_steps: int) -> list[Any]: ...

    def converse(self, conversation_id: str, message: str) -> str: ...

    def end_conversation(self, conversation_id: str) -> None: ...


DriverFactory = Callable[[API, Any, FSMSynthesizer, TraceSink | None], Driver]


def runner_driver(
    api: API, host_definition: Any, synth: FSMSynthesizer, sink: TraceSink | None
) -> Driver:
    return ProcedureRunner(api, host_definition, synth, trace_sink=sink)


DRIVERS: dict[str, DriverFactory] = {"runner": runner_driver}
DEFAULT_DRIVER = "runner"


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
    driver: Driver
    conv: str
    cheap: ScriptedRunnerLLM
    strong: ScriptedLLM
    synth: FSMSynthesizer
    library: FSMLibrary | None
    stream: list[tuple[str, dict[str, Any]]]

    @property
    def runner(self) -> ProcedureRunner:
        """The driver as the runner, for the rows that pin the runner's own bookkeeping."""
        assert isinstance(self.driver, ProcedureRunner)
        return self.driver

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
    driver: str | None = None,
) -> Rig:
    cheap = cheap or ScriptedRunnerLLM(
        {"answer": "42"} if extraction is None else extraction, dict(REPLIES)
    )
    strong = ScriptedLLM([json.dumps(valid_fsm)] if replies is None else replies)
    library = FSMLibrary(tmp_path / "lib") if tmp_path is not None else None
    synth = FSMSynthesizer(strong, policy=policy, library=library)
    definition = make_host(**(config or {}))
    api = API.from_definition(
        definition=definition, llm_interface=cheap, event_log=event_log
    )
    stream: list[tuple[str, dict[str, Any]]] = []
    drive = DRIVERS[driver or DEFAULT_DRIVER](
        api, definition, synth, sink or (lambda e, f: stream.append((e, dict(f))))
    )
    conv, _ = api.start_conversation(CONTEXT if context is None else context)
    return Rig(api, drive, conv, cheap, strong, synth, library, stream)


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
