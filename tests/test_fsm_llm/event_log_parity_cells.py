"""The parity cells of the event log (not collected: no ``test_`` prefix; plan 3f4530d3 step 9).

A CELL is one scripted product run on a REAL ``LiteLLMInterface`` whose ``completion`` (and
``acompletion``) is the scripted ``Provider`` below, so the bytes the provider receives are the
bytes a host would send. Five cells: ``greeting`` (a speaking and a silent start, a pushed child's
greeting, a pop), ``tool`` (registry tools run by a handler inside a turn: ok, read-only, raising,
unknown, truncated), ``react`` (a ``ReactAgent`` run that calls a registry tool), ``approval`` (a
``ReactAgent`` whose gated tool is approved, then denied) and ``call`` (a ``call`` state that runs a
child frame through a resolver). Each runs through the SYNC entry points (no event loop) or the
ASYNC ones (``asyncio.run``), with a log or without.

Interface contract:
    - ``CELLS``: ``{name: cell}``; ``cell(log, use_async, work) -> Run``. ``log`` is an ``EventLog``
      or ``None`` (log off: the cell passes no ``event_log=`` argument at all, so the same cell
      runs on a tree that predates the log keyword); ``work`` is an empty directory for the
      session store.
    - ``Run``: ``outcome`` (what a caller can observe, scrubbed of clocks and ids, JSON-able:
      replies, data, history, state, usage, the saved snapshot, one record per provider request
      with the sha256 of its bytes), ``received`` (the raw provider kwargs, for the rebuild test)
      and ``cid`` (the root conversation id, ``None`` for an agent run).
    - ``golden_document()`` runs every cell log-off in both modes and returns the document that
      ``--write`` stores in ``fixtures/event_log_parity_golden.json``. WRITE IT ONLY ON A
      ``git archive`` COPY OF THE PARENT (the plan base 7b1206e3 for step 9, where no event
      producer exists): the file then says what the product did before the producers, and a tree
      with the log off must say the same, byte for byte. A failure of that comparison is a
      producer that moved a byte or a value, never a fixture to refresh.
    - Failure mode: a cell never catches; an unscripted provider call raises ``KeyError``.

Usage on a copy of the parent: ``PYTHONPATH=<copy>/src python -m tests.test_fsm_llm.event_log_parity_cells
--write PATH`` (run from the copy; the module asserts ``fsm_llm`` is imported from it).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Coroutine
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from litellm.types.utils import Choices, Message, ModelResponse

from fsm_llm import API, FileSessionStore, HandlerTiming, create_handler
from fsm_llm.llm import LiteLLMInterface
from tests.test_fsm_llm.kind_corpus import _frozen_today

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "event_log_parity_golden.json"
)
COMPLETION = "fsm_llm.llm.providers.litellm.completion"
ACOMPLETION = "fsm_llm.llm.providers.litellm.acompletion"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_STAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)
_VOLATILE = {
    "timestamp", "_timestamp", "_conversation_start", "saved_at", "conversation_id",
    "_conversation_id", "execution_time_ms", "elapsed_ms", "created_at", "updated_at",
}  # fmt: skip
_FIELD = re.compile(r"Determine the value of the field '(\w+)'")
Cell = Callable[..., "Run"]


@dataclass
class Run:
    """One cell run: the comparable ``outcome``, the raw provider kwargs, the root id."""

    outcome: dict[str, Any]
    received: list[dict[str, Any]]
    cid: str | None


def scrub(value: Any) -> Any:
    """*value* as plain JSON without clocks, ids or the keys two identical runs never share."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return _STAMP.sub("<ts>", _UUID.sub("<uuid>", value))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub(repr(value))


def request_record(kwargs: dict[str, Any]) -> dict[str, Any]:
    """A provider request as the golden holds it: the sha256 of its scrubbed bytes and a head."""
    text = json.dumps(
        scrub(json.loads(json.dumps(kwargs, default=repr))), sort_keys=True
    )
    system = str(kwargs["messages"][0]["content"])
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "messages": len(kwargs["messages"]),
        "head": scrub(system[:60]),
    }


def _reply(text: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=text), finish_reason="stop")]
    )


class Provider:
    """A scripted litellm: one router, one list of the kwargs it received, sync and async."""

    def __init__(self, fields: dict[str, Callable[[int], Any]] | None = None) -> None:
        self.fields = fields or {}
        self.received: list[dict[str, Any]] = []
        self.seen: Counter[str] = Counter()

    def _route(self, kwargs: dict[str, Any]) -> str:
        found = _FIELD.match(str(kwargs["messages"][0]["content"]))
        if found:
            name = found.group(1)
            self.seen[name] += 1
            value = self.fields[name](self.seen[name])
            return json.dumps(
                {
                    "field_name": name,
                    "value": value,
                    "confidence": 0.9,
                    "reasoning": "r",
                }
            )
        self.seen["reply"] += 1
        return json.dumps({"message": f"reply {self.seen['reply']}", "reasoning": "r"})

    def __call__(self, **kwargs: Any) -> ModelResponse:
        self.received.append(copy.deepcopy(kwargs))
        return _reply(self._route(kwargs))

    async def acall(self, **kwargs: Any) -> ModelResponse:
        return self(**kwargs)


def new_interface() -> LiteLLMInterface:
    return LiteLLMInterface(model="gpt-4o", api_key="sk-parity")


def drive(body: Coroutine[Any, Any, Any], use_async: bool) -> Any:
    """Run *body*: under ``asyncio.run`` when *use_async*, else to its end with NO event loop (a
    sync cell never awaits anything that suspends, so one ``send`` finishes it)."""
    if use_async:
        return asyncio.run(body)
    try:
        body.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError("a sync cell suspended")


async def step(
    use_async: bool, owner: Any, name: str, *args: Any, **kwargs: Any
) -> Any:
    """``owner.name(...)`` through the sync name, or ``await owner.aname(...)`` when async."""
    if use_async:
        return await getattr(owner, "a" + name)(*args, **kwargs)
    return getattr(owner, name)(*args, **kwargs)


def patched(provider: Provider) -> ExitStack:
    """The scripted provider at both litellm entry points, with the clock frozen."""
    stack = ExitStack()
    stack.enter_context(mock.patch(COMPLETION, provider))
    stack.enter_context(mock.patch(ACOMPLETION, provider.acall))
    stack.enter_context(_frozen_today())
    return stack


def _logged(log: Any) -> dict[str, Any]:
    """The ``event_log=`` keyword, absent with the log off."""
    return {} if log is None else {"event_log": log}


def observe(
    api: API,
    cid: str,
    llm: LiteLLMInterface,
    provider: Provider,
    work: Path,
    **extra: Any,
) -> dict[str, Any]:
    """Everything a caller can see of a conversation, plus the provider requests."""
    api.save_session(cid)
    snapshot = json.loads((work / f"{cid}.json").read_text("utf-8"))
    return scrub(
        {
            "data": api.get_data(cid),
            "history": api.get_conversation_history(cid),
            "state": api.get_current_state(cid),
            "usage": llm.usage().model_dump(),
            "snapshot": snapshot,
            "requests": [request_record(k) for k in provider.received],
            **extra,
        }
    )


def _never() -> list[dict[str, Any]]:
    return [{"description": "never", "logic": {"==": [1, 2]}}]


def _state(name: str, says: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": name,
        "description": f"The {name} state",
        "purpose": f"Do the {name} work",
        "response_instructions": says,
        **extra,
    }


def _greeting_fsm(speaking: bool) -> dict[str, Any]:
    go = {"target_state": "talk", "description": "go", "priority": 1}
    stop = {
        "target_state": "end", "description": "done", "priority": 1,
        "conditions": [{"description": "d", "logic": {"==": [1, 2]}}],
    }  # fmt: skip
    return {
        "name": "Greeter",
        "description": "Greets",
        "initial_state": "start",
        "persona": "Pia",
        "states": {
            "start": _state(
                "start", "Greet the user" if speaking else "", transitions=[go]
            ),
            "talk": _state("talk", "Answer the user", transitions=[stop]),
            "end": _state("end", "Say bye"),
        },
    }


_CHILD = {
    "name": "Child",
    "description": "A pushed child",
    "initial_state": "c1",
    "states": {
        "c1": _state("c1", "Child hello", transitions=[
            {"target_state": "c2", "description": "never", "priority": 1,
             "conditions": _never()}
        ]),
        "c2": _state("c2", "Child bye"),
    },
}  # fmt: skip


def greeting_cell(log: Any, use_async: bool, work: Path) -> Run:
    """A speaking start, a turn, a pushed child (its greeting, a turn), a pop, a silent start."""
    llm, provider = new_interface(), Provider()
    kw = {"llm_interface": llm, "session_store": FileSessionStore(work), **_logged(log)}
    speaking, silent = API(_greeting_fsm(True), **kw), API(_greeting_fsm(False), **kw)

    async def body() -> tuple[str, list[str]]:
        cid, hello = await step(use_async, speaking, "start_conversation")
        replies = [hello, await step(use_async, speaking, "converse", "hi", cid)]
        replies.append(
            await step(use_async, speaking, "push_fsm", cid, copy.deepcopy(_CHILD))
        )
        replies.append(await step(use_async, speaking, "converse", "to the child", cid))
        speaking.pop_fsm(cid)
        replies.append(await step(use_async, speaking, "converse", "back", cid))
        quiet, nothing = await step(use_async, silent, "start_conversation")
        replies += [nothing, await step(use_async, silent, "converse", "hello", quiet)]
        assert silent.get_data(quiet) is not None
        return cid, replies

    with patched(provider):
        cid, replies = drive(body(), use_async)
    outcome = observe(speaking, cid, llm, provider, work, replies=replies)
    return Run(outcome, provider.received, cid)


def _tool_registry() -> Any:
    from fsm_llm.tools import READ_ONLY_ANNOTATIONS, ToolRegistry

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def peek(topic: str) -> str:
        """Read a topic."""
        return f"peeked {topic}"

    def boom(n: int) -> str:
        """Always fails."""
        raise ValueError("boom text")

    def big() -> str:
        """A result over the observation cap."""
        return "x" * 5000

    registry = ToolRegistry()
    registry.register_function(add, name="add")
    registry.register_function(peek, name="peek", annotations=READ_ONLY_ANNOTATIONS)
    registry.register_function(boom, name="boom")
    registry.register_function(big, name="big")
    return registry


_TOOL_CALLS = [
    ("add", {"a": 2, "b": 3}), ("peek", {"topic": "x"}), ("boom", {"n": 1}),
    ("big", {}), ("nope", {}), ("add", {"zzz": 1}),
]  # fmt: skip


def tool_cell(log: Any, use_async: bool, work: Path) -> Run:
    """Registry tools run by a handler of a chat turn: ok, read-only, raising, truncated, an
    unknown name and a bad argument, in two turns (the handler runs before every reply)."""
    from fsm_llm.tools import ToolCall

    registry, results = _tool_registry(), []

    def run_them(_: dict[str, Any]) -> dict[str, Any]:
        for name, params in _TOOL_CALLS:
            r = registry.execute(
                ToolCall(tool_name=name, parameters=copy.deepcopy(params))
            )
            results.append(
                [
                    name,
                    r.status,
                    r.success,
                    r.observation[:40],
                    len(r.observation),
                    r.error,
                ]
            )
        return {"tools_ran": len(results)}

    handler = create_handler("tools").at(HandlerTiming.PRE_PROCESSING).do(run_them)
    fsm = {
        "name": "Tools", "description": "Runs tools", "initial_state": "chat",
        "states": {"chat": _state("chat", "Reply kindly", transitions=[
            {"target_state": "end", "description": "never", "priority": 1,
             "conditions": _never()}]), "end": _state("end", "Bye")},
    }  # fmt: skip
    llm, provider = new_interface(), Provider()
    api = API(
        fsm, llm_interface=llm, handlers=[handler],
        session_store=FileSessionStore(work), **_logged(log),
    )  # fmt: skip

    async def body() -> tuple[str, list[str]]:
        cid, hello = await step(use_async, api, "start_conversation")
        replies = [hello, await step(use_async, api, "converse", "go", cid)]
        replies.append(await step(use_async, api, "converse", "again", cid))
        return cid, replies

    with patched(provider):
        cid, replies = drive(body(), use_async)
    outcome = observe(api, cid, llm, provider, work, replies=replies, results=results)
    return Run(outcome, provider.received, cid)


def call_cell(log: Any, use_async: bool, work: Path) -> Run:
    """A ``call`` state: the parent pushes a child frame through a resolver, the child runs on the
    message-free step loop (one speaking state, a handler writes its answer), the frame pops."""
    from tests.test_fsm_llm.test_call_settle import (
        _TARGET,
        _child_definition,
        _Resolver,
    )
    from tests.test_fsm_llm.test_call_state import _parent

    resolver = _Resolver({_TARGET: _child_definition(2, None, "c-speaks")})
    answer = (
        create_handler("answer")
        .on_state_entry("c3")
        .do(lambda ctx: {"solution": f"solved {ctx.get('topic')}"})
    )
    llm, provider = new_interface(), Provider()
    api = API.from_definition(
        _parent(None, response_instructions=""), llm_interface=llm, targets=resolver,
        handlers=[answer], session_store=FileSessionStore(work), **_logged(log),
    )  # fmt: skip

    async def body() -> str:
        cid, _ = await step(use_async, api, "start_conversation", {"topic": "why"})
        await step(use_async, api, "advance", cid)
        await step(use_async, api, "run_until_terminal", cid, max_steps=10)
        return cid

    with patched(provider):
        cid = drive(body(), use_async)
    outcome = observe(api, cid, llm, provider, work, asked=resolver.asked)
    return Run(outcome, provider.received, cid)


def _agent_fields(tool: str, stop_after: int) -> dict[str, Callable[[int], Any]]:
    """The field answers of a ``ReactAgent``: select *tool* until the *stop_after*-th think."""
    return {
        "tool_name": lambda n: tool if n < stop_after else "none",
        "tool_input": lambda n: {"a": 2, "b": 3} if tool == "add" else {"x": "y"},
        "reasoning": lambda n: "because",
        "should_terminate": lambda n: n >= stop_after,
        "final_answer": lambda n: "the answer",
        "evaluation_passed": lambda n: True,
        "evaluation_score": lambda n: 0.9,
    }


def _agent_outcome(
    result: Any, llm: LiteLLMInterface, provider: Provider, **extra: Any
) -> dict[str, Any]:
    return scrub(
        {
            "answer": result.answer,
            "status": result.status.value,
            "stop_reason": result.stop_reason.value,
            "iterations": result.iterations_used,
            "final_context": result.final_context,
            "usage": llm.usage().model_dump(),
            "requests": [request_record(k) for k in provider.received],
            **extra,
        }
    )


def react_cell(log: Any, use_async: bool, work: Path) -> Run:
    """A ``ReactAgent`` that calls the registry tool ``add`` once and then answers."""
    from fsm_llm.agents import AgentConfig
    from fsm_llm.agents.react import ReactAgent
    from fsm_llm.tools import ToolRegistry

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    registry = ToolRegistry().register_function(add, name="add", description="adds")
    llm, provider = new_interface(), Provider(_agent_fields("add", 2))
    agent = ReactAgent(
        tools=registry, config=AgentConfig(max_iterations=4, model="gpt-4o"),
        llm_interface=llm, **_logged(log),
    )  # fmt: skip
    with patched(provider):
        result = (
            asyncio.run(agent.arun("add 2 and 3"))
            if use_async
            else agent.run("add 2 and 3")
        )
    return Run(
        _agent_outcome(result, llm, provider, trace=result.trace),
        provider.received,
        None,
    )


def approval_cell(log: Any, use_async: bool, work: Path) -> Run:
    """A ``ReactAgent`` whose gated tool is asked for twice: the first ask is approved, the
    second denied. The callback sees the request RAW; the grant is the driver's alone."""
    from fsm_llm.agents import AgentConfig
    from fsm_llm.agents.react import ReactAgent
    from fsm_llm.policy import HumanInTheLoop
    from fsm_llm.tools import ToolRegistry

    runs: list[str] = []
    asks: list[Any] = []

    def danger(params: dict[str, Any]) -> str:
        runs.append("danger")
        return "danger done"

    registry = ToolRegistry().register_function(
        danger, name="danger", description="A gated tool", requires_approval=True,
        parameter_schema={"properties": {"x": {"type": "string"}}},
    )  # fmt: skip

    def decide(request: Any) -> bool:
        asks.append(request.model_dump())
        return len(asks) % 2 == 1

    hitl = HumanInTheLoop(
        approval_policy=lambda call, ctx: True, approval_callback=decide
    )
    llm, provider = new_interface(), Provider(_agent_fields("danger", 4))
    agent = ReactAgent(
        tools=registry, config=AgentConfig(max_iterations=3, model="gpt-4o"),
        llm_interface=llm, hitl=hitl, **_logged(log),
    )  # fmt: skip
    with patched(provider):
        result = asyncio.run(agent.arun("do it")) if use_async else agent.run("do it")
    outcome = _agent_outcome(
        result, llm, provider, asks=asks, runs=runs, trace=result.trace
    )
    return Run(outcome, provider.received, None)


# --------------------------------------------------------------
# The product scenario of the crash sweeps
# --------------------------------------------------------------

SWEEP_VIEWS = "_sweep_views"
_INTENTS = [
    {"name": "buy", "description": "wants to purchase"},
    {"name": "browse", "description": "is looking around"},
]


def product_fsm() -> dict[str, Any]:
    """``ask`` extracts ``city`` and classifies ``user_intent``; a city moves it to ``talk``."""
    has_city = {
        "target_state": "talk", "description": "city known", "priority": 1,
        "conditions": [{"description": "c", "requires_context_keys": ["city"],
                        "logic": {"has_context": "city"}}],
    }  # fmt: skip
    return {
        "name": "Products",
        "description": "Every path the sweeps kill",
        "initial_state": "ask",
        "states": {
            "ask": _state(
                "ask", "Ask", required_context_keys=["city"], transitions=[has_city],
                classification_extractions=[{
                    "field_name": "user_intent", "intents": _INTENTS,
                    "fallback_intent": "browse", "confidence_threshold": 0.7,
                }],
            ),
            "talk": _state("talk", "Talk", transitions=[
                {"target_state": "end", "description": "never", "priority": 1,
                 "conditions": _never()}
            ]),
            "end": _state("end", "Bye"),
        },
    }  # fmt: skip


def product_llm() -> Any:
    """The scripted engine-level model: no city yet (a miss), a fallback intent every classify."""
    from fsm_llm.testing import ScriptedLLM, final_reply

    reply = json.dumps({"reasoning": "r", "intent": "browse", "confidence": 0.9})
    return ScriptedLLM(
        [final_reply(reply)] * 2,
        extraction_data={},
        responses={"ask": "reply", "talk": "talking"},
    )


def product_api(log: Any, llm: Any, **kwargs: Any) -> API:
    """The API of the scenario: a handler that runs two registry tools and asks an approval."""
    from fsm_llm.policy import HumanInTheLoop
    from fsm_llm.tools import ToolCall

    registry = _tool_registry()
    hitl = HumanInTheLoop(approval_callback=lambda request: True)

    def run_them(_: dict[str, Any]) -> dict[str, Any]:
        for name, params in (("add", {"a": 2, "b": 3}), ("peek", {"topic": "x"})):
            registry.execute(ToolCall(tool_name=name, parameters=params))
        hitl.request_approval(ToolCall(tool_name="send", parameters={"to": "a"}), {})
        return {"tools_ran": True}

    handler = (
        create_handler("tools")
        .at(HandlerTiming.PRE_PROCESSING)
        .on_state("ask")
        .do(run_them)
    )
    return API(
        product_fsm(), llm_interface=llm, handlers=[handler], **_logged(log), **kwargs
    )


def product_view(api: API, cid: str) -> dict[str, Any]:
    snapshot = api.fsm_manager.get_conversation_snapshot(cid)
    return scrub(
        {
            "current_state": snapshot["current_state"],
            "data": api.get_data(cid),
            "history": snapshot["conversation_history"],
        }
    )


def product_play(
    log: Any, llm: Any, record: Callable[[API, str], None], use_async: bool
) -> None:
    """Greeting, a turn with a miss and a fallback intent, a turn that extracts and moves, a
    turn in ``talk``; *record* runs after each (the boundaries of the live view)."""
    api = product_api(log, llm)

    async def body() -> None:
        cid, _ = await step(use_async, api, "start_conversation")
        record(api, cid)
        await step(use_async, api, "converse", "I live somewhere", cid)
        record(api, cid)
        llm.extraction_data = {"city": "Rome"}
        await step(use_async, api, "converse", "Rome", cid)
        record(api, cid)
        await step(use_async, api, "converse", "more", cid)
        record(api, cid)

    drive(body(), use_async)


def golden_views() -> dict[str, list[Any]]:
    """The views the PARENT archive showed at each boundary of the scenario (log off)."""
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))[SWEEP_VIEWS]


def sweep_views(use_async: bool) -> list[Any]:
    """The live view at each boundary of the scenario with the log OFF (what ``--write`` stores)."""
    views: list[Any] = []
    product_play(
        None,
        product_llm(),
        lambda api, cid: views.append(product_view(api, cid)),
        use_async,
    )
    return views


CELLS: dict[str, Cell] = {
    "greeting": greeting_cell,
    "tool": tool_cell,
    "react": react_cell,
    "approval": approval_cell,
    "call": call_cell,
}
MODES = {"sync": False, "async": True}


def golden_document() -> dict[str, Any]:
    """Every cell, log off, in both modes (what ``--write`` stores)."""
    document: dict[str, Any] = {}
    for name, cell in CELLS.items():
        for mode, use_async in MODES.items():
            with TemporaryDirectory() as work:
                document.setdefault(name, {})[mode] = cell(
                    None, use_async, Path(work)
                ).outcome
    document[SWEEP_VIEWS] = {
        mode: sweep_views(use_async) for mode, use_async in MODES.items()
    }
    return document


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def main(argv: list[str]) -> int:
    import fsm_llm

    if not Path(fsm_llm.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(
            f"fsm_llm is imported from {fsm_llm.__file__}, not from {ROOT}"
        )
    if argv[:1] == ["--write"] and len(argv) == 2:
        Path(argv[1]).write_text(
            dumps(golden_document()), encoding="utf-8", newline="\n"
        )
        return 0
    print(
        "usage: python -m tests.test_fsm_llm.event_log_parity_cells --write PATH",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
