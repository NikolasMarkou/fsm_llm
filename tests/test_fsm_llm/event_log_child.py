"""The kill-mid-turn child and the scripted tool conversation it shares with the crash sweep.

Plan 21e6f00d step 11. Not collected (no ``test_`` prefix). Run as
``python -m tests.test_fsm_llm.event_log_child <directory> <ready_file>`` from the repository
root; ``test_event_log_crash.py`` also imports ``drive_exchange``, ``tool_fsm`` and ``view``
so the child and the in-process sweep drive the same conversation.

Interface contract:
    - ``tool_fsm()`` -> the definition dict: ``intake -> call_model (completion, tools) ->
      run_tools -> call_model -> ... -> done``. No handler runs a tool: the consumer does, around
      ``EventLog.tool_call``.
    - ``ChildLLM(replies, *, block_at=None)``: a ``ScriptedLLM`` (a custom interface, so the
      engine fallback writes the request events) whose ``complete`` call number *block_at*
      (0-based) calls ``on_block`` and then never returns. A safety bound ends it after
      ``BLOCK_SECONDS`` so an orphan cannot outlive a crashed parent.
    - ``view(api, cid)`` -> a JSON-able dict of what a restore must give back: ``current_state``,
      ``data`` (public context), ``history`` and the completion ``transcript``.
    - ``drive_exchange(api, log, cid, mark=None)``: advance to ``call_model``, let the model ask
      for a tool, run it inside ``log.tool_call`` (a side-effecting checkpoint), append the
      paired exchange with ``update_context`` and advance back to ``call_model``. *mark* (no
      arguments) is called after every step that wrote events, so a caller can record a view at
      each boundary. Raises whatever the API raises.
    - ``leave_a_tool_in_flight(log, cid)``: a side-effecting ``tool/call`` with no result, the log a
      kill inside a tool leaves (no product path writes ``tool/call`` yet, plan 3f4530d3 step 3):
      the SETUP of a scenario whose product event is the ``unknown`` ``tool/result`` of ``API.resume``.
    - ``registry_main(argv)`` (``main`` with ``registry`` as its third argument): a tool of a
      ``ToolRegistry`` run by a handler of the ``run_tools`` state, so the product path
      (``ToolRegistry.execute`` -> ``core.tool_observer`` -> the log scope) writes ``tool/call``;
      the tool BODY announces itself by writing *ready_file* and blocks, and the parent kills it
      there (plan 3f4530d3 step 3, the done-when of roadmap row 1.3).
    - ``main(argv)``: the child. Drives the exchange, takes ``view`` as the expected restore,
      then calls ``api.advance`` inside a second ``tool_call`` named ``in-flight``; the model
      double announces itself by writing *ready_file* (``conversation_id`` and ``expected``,
      through a temp file and ``os.replace`` so the parent never reads half of it) and blocks.

Plan 5a6a1f80 step 6 (the stack): the same conversation one frame up. ``chat_fsm()`` is the root
definition, ``stack_llm(replies, **kw)`` a ``ChildLLM`` that also answers the root, ``top_id`` the
manager id of the top frame, ``stack_view(api, cid)`` the LIVE ``(depth, [(state, public data)])``
(the golden view of the stack sweep; JSON turns its tuples into lists). ``stacked_start(api, log)``
plays a root turn, a push with options and a tool exchange at frame 1 and returns the root id;
``stacked_finish(api, cid)`` the answer, the pop with a merge, a root turn and a second push left
open. ``main`` with ``async`` as its third argument runs ``async_main`` (a kill after the request hop of an async turn); ``main`` with ``stacked`` as its third argument runs ``stacked_main``: role ``run`` blocks at
frame 1 inside an in-flight tool call; role ``resume <conversation_id>`` resumes that log,
announces and holds the writer lock until its stdin is closed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fsm_llm import API, HandlerTiming, create_handler, tool_exchange
from fsm_llm.constants import (
    DEFAULT_COMPLETION_MESSAGES_KEY,
    DEFAULT_COMPLETION_RESULT_KEY,
)
from fsm_llm.core.definitions import CompletionResponse
from fsm_llm.runtime import EventLog, EventType
from fsm_llm.testing import ScriptedLLM, final_reply, tool_call_reply
from fsm_llm.tools import ToolCall, ToolRegistry

BOUND_ENV = "EVENT_LOG_TEST_BOUND"


def open_log(directory: Path) -> EventLog:
    """The child's ``EventLog``: ``max_open_writers`` from ``EVENT_LOG_TEST_BOUND`` ("none" or
    a whole number) when the parent set it (``writer_bound``), else the default."""
    bound = os.environ.get(BOUND_ENV)
    if bound is None:
        return EventLog(directory)
    return EventLog(directory, max_open_writers=None if bound == "none" else int(bound))


MESSAGES = DEFAULT_COMPLETION_MESSAGES_KEY
RESULT = DEFAULT_COMPLETION_RESULT_KEY
QUESTION = "What is 2 + 3?"
BLOCK_SECONDS = 600

_ADD: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


def _state(state_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": state_id,
        "description": f"The {state_id} state",
        "purpose": f"Do the {state_id} work",
        "response_instructions": "",
        **extra,
    }


def _kind_is(kind: str) -> list[dict[str, Any]]:
    return [
        {
            "description": f"The model turn was {kind}",
            "requires_context_keys": [RESULT],
            "logic": {"==": [{"var": f"{RESULT}.kind"}, kind]},
        }
    ]


def tool_fsm() -> dict[str, Any]:
    """intake -> call_model (completion) -> run_tools -> call_model -> ... -> done."""
    return {
        "name": "CrashTurns",
        "description": "A model turn with a tool, a tool turn, a final answer",
        "initial_state": "intake",
        "states": {
            "intake": _state(
                "intake",
                transitions=[
                    {"target_state": "call_model", "description": "Ask the model"}
                ],
            ),
            "call_model": _state(
                "call_model",
                completion={"tools": [_ADD], "instructions": "Use the tools."},
                transitions=[
                    {
                        "target_state": "run_tools",
                        "description": "The model asked for tools",
                        "priority": 10,
                        "conditions": _kind_is("calls"),
                    },
                    {
                        "target_state": "done",
                        "description": "The model answered",
                        "priority": 20,
                        "conditions": _kind_is("final"),
                    },
                ],
            ),
            "run_tools": _state(
                "run_tools",
                transitions=[
                    {"target_state": "call_model", "description": "Back to the model"}
                ],
            ),
            "done": _state("done"),
        },
    }


class ChildLLM(ScriptedLLM):
    """A scripted interface whose ``complete`` call number ``block_at`` never returns.

    *script* is passed to ``ScriptedLLM`` (``extraction_data``, ``responses``).
    """

    def __init__(
        self,
        replies: list[CompletionResponse],
        *,
        block_at: int | None = None,
        on_block: Callable[[], None] | None = None,
        **script: Any,
    ) -> None:
        super().__init__(replies, **script)
        self.block_at = block_at
        self.on_block = on_block

    def complete(self, request: Any) -> CompletionResponse:
        if self.block_at is not None and len(self.requests) == self.block_at:
            if self.on_block is not None:
                self.on_block()
            for _ in range(BLOCK_SECONDS):
                time.sleep(1)
            raise SystemExit("nobody killed the blocked child")
        return super().complete(request)


def scripted_replies() -> list[CompletionResponse]:
    return [tool_call_reply(("add", {"a": 2, "b": 3})), final_reply("5")]


def start(api: API) -> str:
    cid, greeting = api.start_conversation(
        initial_context={MESSAGES: [{"role": "user", "content": QUESTION}]}
    )
    assert greeting == ""
    return cid


def leave_a_tool_in_flight(log: EventLog, cid: str) -> None:
    log.append(
        cid,
        EventType.TOOL_CALL,
        frame=0,
        name="charge",
        call_id="c-1",
        side_effecting=True,
        arguments={"n": 5},
    )


def view(api: API, cid: str) -> dict[str, Any]:
    snapshot = api.fsm_manager.get_conversation_snapshot(cid)
    return {
        "current_state": snapshot["current_state"],
        "data": api.get_data(cid),
        "history": snapshot["conversation_history"],
        "transcript": api.fsm_manager.copy_raw_context_data(cid).get(MESSAGES),
    }


def drive_exchange(
    api: API,
    log: EventLog,
    cid: str,
    mark: Callable[[], None] | None = None,
    *,
    frame: int = 0,
) -> None:
    """The exchange of the module docstring, at stack frame *frame* (the top one, root is 0)."""

    def done() -> None:
        if mark is not None:
            mark()

    api.advance(cid)  # intake -> call_model, silent
    done()
    api.advance(cid)  # the model asks for a tool: call_model -> run_tools
    done()
    result = api.get_data(cid)[RESULT]
    outputs: list[str] = []
    for call in result["calls"]:
        with log.tool_call(
            cid,
            name=call["name"],
            call_id=call["id"],
            arguments=call["arguments"],
            frame=frame,
        ):
            outputs.append(str(call["arguments"]["a"] + call["arguments"]["b"]))
    done()
    transcript = api.fsm_manager.copy_raw_context_data(top_id(api, cid))[MESSAGES]
    paired = tool_exchange(result["text"], result["calls"], outputs)
    api.update_context(cid, {MESSAGES: [*transcript, *paired], RESULT: None})
    done()
    api.advance(cid)  # run_tools -> call_model, silent
    done()


def chat_fsm() -> dict[str, Any]:
    """The root of the stacked scenario: one chat state that extracts ``name`` and never ends."""
    never = [{"description": "never", "logic": {"==": [1, 2]}}]
    return {
        "name": "Stacker",
        "description": "A root that chats and takes a child",
        "initial_state": "chat",
        "states": {
            "chat": _state(
                "chat",
                required_context_keys=["name"],
                response_instructions="Root reply",
                transitions=[
                    {"target_state": "end", "description": "never", "conditions": never}
                ],
            ),
            "end": _state("end"),
        },
    }


def stack_llm(replies: list[CompletionResponse], **kw: Any) -> ChildLLM:
    """A ``ChildLLM`` that also answers the root's extraction and reply."""
    return ChildLLM(
        replies,
        extraction_data={"name": "Ann"},
        responses={"chat": "root reply"},
        **kw,
    )


def top_id(api: API, cid: str) -> str:
    """The manager id of the top frame of the stack of root *cid* (*cid* itself when alone)."""
    with api._stack_lock:
        return api.conversation_stacks[cid][-1].conversation_id


def stack_view(api: API, cid: str) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
    """What the LIVE api shows now: the depth and, per frame, the state and the public data."""
    with api._stack_lock:
        ids = [frame.conversation_id for frame in api.conversation_stacks[cid]]
    manager = api.fsm_manager
    return len(ids), [
        (manager.get_conversation_state(i), manager.get_conversation_data(i))
        for i in ids
    ]


PUSH_OPTIONS: dict[str, Any] = {
    "context_to_pass": {"topic": "x"},
    "return_context": {"answer": 42},
    "shared_context_keys": ["name"],
    "preserve_history": True,
}


def stacked_start(api: API, log: EventLog) -> str:
    """A root turn, a push with options, the transcript and a tool exchange at frame 1."""
    cid, _ = api.start_conversation()
    assert api.converse("Ann", cid) == "root reply"
    api.push_fsm(cid, tool_fsm(), **PUSH_OPTIONS)
    api.update_context(cid, {MESSAGES: [{"role": "user", "content": QUESTION}]})
    drive_exchange(api, log, cid, frame=1)
    return cid


def stacked_finish(api: API, cid: str) -> None:
    """The model answers, the child pops with a merge, a root turn, a second push left open."""
    api.advance(cid)  # call_model -> done, at frame 1
    api.pop_fsm(cid, context_to_return={"result": 5})
    assert api.converse("again", cid) == "root reply"
    api.push_fsm(cid, tool_fsm())


def stacked_main(argv: list[str]) -> None:
    directory, ready, role = Path(argv[1]), Path(argv[2]), argv[4]
    with open_log(directory) as log:
        if role == "resume":
            api = API(chat_fsm(), llm_interface=stack_llm([]), event_log=log)
            cid, _ = api.resume(argv[5])
            _announce(ready, conversation_id=cid, expected=stack_view(api, cid))
            sys.stdin.readline()  # hold the writer lock until the parent lets go
            return
        llm = stack_llm(scripted_replies(), block_at=1)
        api = API(chat_fsm(), llm_interface=llm, event_log=log)
        cid = stacked_start(api, log)
        expected = stack_view(api, cid)
        llm.on_block = lambda: _announce(ready, conversation_id=cid, expected=expected)
        with log.tool_call(
            cid, name="add", call_id="in-flight", arguments={"a": 1, "b": 1}, frame=1
        ):
            api.advance(cid)


def registry_main(argv: list[str]) -> None:
    directory, ready = Path(argv[1]), Path(argv[2])
    where: dict[str, str] = {}

    def charge(n: int) -> str:
        """Charge n units (blocks forever: the parent kills the process here)."""
        _announce(ready, conversation_id=where["cid"])
        for _ in range(BLOCK_SECONDS):
            time.sleep(1)
        raise SystemExit("nobody killed the blocked child")

    registry = ToolRegistry().register_function(charge, name="charge")
    run_it = (
        create_handler("run-registry-tool")
        .at(HandlerTiming.PRE_PROCESSING)
        .on_state("run_tools")
        .do(
            lambda ctx: (
                registry.execute(ToolCall(tool_name="charge", parameters={"n": 5}))
                and {}
            )
        )
    )
    with open_log(directory) as log:
        api = API.from_definition(
            tool_fsm(),
            llm_interface=ChildLLM(scripted_replies()),
            event_log=log,
            handlers=[run_it],
        )
        where["cid"] = cid = start(api)
        api.advance(cid)  # intake -> call_model, silent
        api.advance(cid)  # the model asks for a tool: call_model -> run_tools
        api.advance(cid)  # the handler of run_tools runs the registry tool and blocks


def async_main(argv: list[str]) -> None:
    """The async role (plan 3f4530d3 step 8b): an ``aadvance`` whose model call blocks on the
    loop thread AFTER the request hop, so the ``llm/request`` the parent finds was written and
    fsynced by a lane job (the loop never wrote it). ``main`` with ``async`` as third argument."""
    import asyncio

    directory, ready = Path(argv[1]), Path(argv[2])
    where: dict[str, str] = {}
    llm = ChildLLM(
        scripted_replies(),
        block_at=0,
        on_block=lambda: _announce(ready, conversation_id=where["cid"]),
    )

    async def run() -> None:
        with open_log(directory) as log:
            api = API.from_definition(tool_fsm(), llm_interface=llm, event_log=log)
            where["cid"], _ = await api.astart_conversation(
                initial_context={MESSAGES: [{"role": "user", "content": QUESTION}]}
            )
            await api.aadvance(where["cid"])  # intake -> call_model, silent
            await api.aadvance(
                where["cid"]
            )  # the model call blocks after its request hop

    asyncio.run(run())


def _announce(ready: Path, **payload: Any) -> None:
    pending = ready.with_name(ready.name + ".tmp")
    pending.write_text(
        json.dumps(payload),
        encoding="utf-8",
        newline="",
    )
    os.replace(pending, ready)


def main(argv: list[str]) -> None:
    if argv[3:4] == ["stacked"]:
        return stacked_main(argv)
    if argv[3:4] == ["registry"]:
        return registry_main(argv)
    if argv[3:4] == ["async"]:
        return async_main(argv)
    directory, ready = Path(argv[1]), Path(argv[2])
    log = open_log(directory)
    llm = ChildLLM(scripted_replies(), block_at=1)
    api = API.from_definition(tool_fsm(), llm_interface=llm, event_log=log)
    cid = start(api)
    drive_exchange(api, log, cid)
    expected = view(api, cid)
    llm.on_block = lambda: _announce(ready, conversation_id=cid, expected=expected)
    # The tool is in flight while the second model call is made: a tool whose body asks the model.
    with log.tool_call(
        cid, name="add", call_id="in-flight", arguments={"a": 1, "b": 1}
    ):
        api.advance(cid)


if __name__ == "__main__":
    main(sys.argv)
