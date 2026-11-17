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
    - ``main(argv)``: the child. Drives the exchange, takes ``view`` as the expected restore,
      then calls ``api.advance`` inside a second ``tool_call`` named ``in-flight``; the model
      double announces itself by writing *ready_file* (``conversation_id`` and ``expected``,
      through a temp file and ``os.replace`` so the parent never reads half of it) and blocks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fsm_llm import API, tool_exchange
from fsm_llm.constants import (
    DEFAULT_COMPLETION_MESSAGES_KEY,
    DEFAULT_COMPLETION_RESULT_KEY,
)
from fsm_llm.core.definitions import CompletionResponse
from fsm_llm.runtime import EventLog
from fsm_llm.testing import ScriptedLLM, final_reply, tool_call_reply

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
    """A scripted interface whose ``complete`` call number ``block_at`` never returns."""

    def __init__(
        self,
        replies: list[CompletionResponse],
        *,
        block_at: int | None = None,
        on_block: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(replies)
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


def view(api: API, cid: str) -> dict[str, Any]:
    snapshot = api.fsm_manager.get_conversation_snapshot(cid)
    return {
        "current_state": snapshot["current_state"],
        "data": api.get_data(cid),
        "history": snapshot["conversation_history"],
        "transcript": api.fsm_manager.copy_raw_context_data(cid).get(MESSAGES),
    }


def drive_exchange(
    api: API, log: EventLog, cid: str, mark: Callable[[], None] | None = None
) -> None:
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
            cid, name=call["name"], call_id=call["id"], arguments=call["arguments"]
        ):
            outputs.append(str(call["arguments"]["a"] + call["arguments"]["b"]))
    done()
    transcript = api.fsm_manager.copy_raw_context_data(cid)[MESSAGES]
    paired = tool_exchange(result["text"], result["calls"], outputs)
    api.update_context(cid, {MESSAGES: [*transcript, *paired], RESULT: None})
    done()
    api.advance(cid)  # run_tools -> call_model, silent
    done()


def _announce(ready: Path, cid: str, expected: dict[str, Any]) -> None:
    pending = ready.with_name(ready.name + ".tmp")
    pending.write_text(
        json.dumps({"conversation_id": cid, "expected": expected}),
        encoding="utf-8",
        newline="",
    )
    os.replace(pending, ready)


def main(argv: list[str]) -> None:
    directory, ready = Path(argv[1]), Path(argv[2])
    log = EventLog(directory)
    llm = ChildLLM(scripted_replies(), block_at=1)
    api = API.from_definition(tool_fsm(), llm_interface=llm, event_log=log)
    cid = start(api)
    drive_exchange(api, log, cid)
    expected = view(api, cid)
    llm.on_block = lambda: _announce(ready, cid, expected)
    # The tool is in flight while the second model call is made: a tool whose body asks the model.
    with log.tool_call(
        cid, name="add", call_id="in-flight", arguments={"a": 1, "b": 1}
    ):
        api.advance(cid)


if __name__ == "__main__":
    main(sys.argv)
