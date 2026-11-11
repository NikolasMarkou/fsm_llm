"""
Agent tools over a synthesizer: ``obtain_procedure`` and ``run_procedure``.

A tool cannot push a child on the agent's stack, so ``run_procedure`` runs a
stored procedure on its own sub-``API`` (the cheap model, message-free) and
hands back only its return values. Approval stays inside the synthesizer (G7).

# DECISION plan-2026-11-10T194139-2cefeb61/D-030
# Descriptions and parameter schemas are the module constants below, never built
# from a record, a goal or a model reply; every reply starts with the fixed label
# and carries codes, ids and capped values only (never exception or generated
# text). Do NOT flag a tool ``requires_approval`` (it raises on four agent
# patterns), do NOT set ``timeout_s`` (a timed-out compile cannot be killed and
# could still write the library), and do NOT size a reply from a flat number:
# ``_room`` counts the echoed input like the agent loop does.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from fsm_llm.agents.constants import OBSERVATION_ENTRY, Defaults
from fsm_llm.agents.definitions import ToolAnnotations
from fsm_llm.agents.exceptions import require_int_limit
from fsm_llm.agents.tools import ToolRegistry
from fsm_llm.api import API
from fsm_llm.constants import SKILL_DEFANG
from fsm_llm.definitions import (
    SYNTHESIZE_GOAL_MAX_LENGTH,
    LLMResponseError,
    RunBudgetExceededError,
)
from fsm_llm.expressions import is_missing
from fsm_llm.llm import LLMInterface
from fsm_llm.logging import logger
from fsm_llm.okf import capped_line

from .constants import (
    LIFECYCLE_VERIFIED,
    MAX_TOOL_NAMES,
    PROCEDURE_UNTRUSTED_LABEL,
    REASON_BUDGET_SPENT,
    REASON_INPUTS_MISSING,
    REASON_LIBRARY_ERROR,
    REASON_PROVIDER_ERROR,
    REASON_RETURN_KEYS_MISSING,
)
from .exceptions import LibraryError, SynthesisError
from .gates import verify_stored
from .runner import reason_of, runnable_reason
from .synthesizer import FSMSynthesizer

__all__ = ["TOOL_DESCRIPTIONS", "TOOL_PARAMETERS", "create_procedure_tools"]

TOOL_DESCRIPTIONS: dict[str, str] = {
    "obtain_procedure": (
        "Have a strong model write, check and store a reusable conversation "
        "procedure for a goal, or reuse a stored one. Returns a short receipt "
        "with a procedure_id for run_procedure. Slow and costly: call it once "
        "per goal."
    ),
    "run_procedure": (
        "Run a stored procedure by procedure_id on a cheap model, with no user "
        "messages, and return its values. A failure code is returned instead "
        "when it cannot finish alone (needs_user_input)."
    ),
}
TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "obtain_procedure": {
        "properties": {
            "goal": {
                "type": "string",
                "description": "What the procedure must accomplish, in a sentence or two.",
            },
            "return_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of the values it must hand back (letters, digits, _).",
            },
            "inputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of the values you will give it when you run it.",
            },
        },
        "required": ["goal", "return_keys"],
    },
    "run_procedure": {
        "properties": {
            "procedure_id": {
                "type": "string",
                "description": "The procedure_id from obtain_procedure (64 lowercase hex).",
            },
            "inputs": {
                "type": "object",
                "description": "Values for the procedure's inputs, by name.",
            },
        },
        "required": ["procedure_id"],
    },
}

_BAD_ARGUMENTS = "bad_arguments"  # reply codes local to the tools; the rest are
_NEEDS_USER_INPUT = "needs_user_input"  # `constants.REASONS`
_MARGIN = 324  # the agent loop's wrapper slack and model-added keys (D-016, D-017)
_MIN_ROOM = 600  # when the echoed input alone nearly fills the cap
_MORE = 40  # room kept for the "... N more" line
_VALUE_CHARS = 300  # one return value


def _room(tool: str, **params: Any) -> int:
    """Characters a *tool* reply may use so its observation entry is not cut:
    the loop stores ``[Step n] Tool: .. | Input: <input> | Result: <reply>``."""
    used = len(OBSERVATION_ENTRY.format(step=999, tool=tool, input=params, result=""))
    return max(Defaults.MAX_OBSERVATION_LENGTH - used - _MARGIN, _MIN_ROOM)


def _safe(value: object, cap: int) -> str:
    """One control-stripped line of at most *cap* characters, ``[ ] < >`` defanged."""
    text = value if isinstance(value, str) else str(value)
    return capped_line(text, cap).translate(SKILL_DEFANG)


def _reply(lines: list[str], room: int) -> str:
    """The label, then the lines that fit in *room* (and a count of the rest)."""
    out, used = [PROCEDURE_UNTRUSTED_LABEL], len(PROCEDURE_UNTRUSTED_LABEL)
    for i, line in enumerate(lines):
        if used + len(line) + 1 > room - _MORE:
            out.append(f"... {len(lines) - i} more line(s) not shown")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def create_procedure_tools(
    synthesizer: FSMSynthesizer,
    *,
    runner_llm: LLMInterface | None = None,
    registry: ToolRegistry | None = None,
    max_obtains: int = 20,
) -> ToolRegistry:
    """Register ``obtain_procedure`` and ``run_procedure``; return the registry.

    Interface contract:
        - ``obtain_procedure(goal, return_keys, inputs=())`` calls
          ``synthesizer.obtain`` and replies with a receipt: ``procedure_id``,
          ``version`` and ``lifecycle`` (``-`` without a library), ``source``,
          ``states`` (abort terminal included) and the ``G1`` to ``G6a``
          verdicts. No generated text.
        - ``run_procedure(procedure_id, inputs={})`` needs the synthesizer's
          library. It loads by id (64 lowercase hex), applies
          ``runnable_reason``, needs every declared input, runs the procedure on
          its own ``API`` over *runner_llm* (message-free,
          ``max_steps=policy.max_turns``, declared inputs only, no persona,
          conversation always ended) and replies ``procedure_id``,
          ``status: ok`` and one defanged ``key: value`` line per return key.
          ``needs_user_input`` (budget spent) and ``return_keys_missing`` count
          one failed run; ``provider_error`` counts none.
        - Every reply starts with ``PROCEDURE_UNTRUSTED_LABEL`` and fits the
          agent loop's cap with its echo margin; a failure is ``failed: <code>``
          (``bad_arguments`` or ``constants.REASONS``), never an echoed argument
          or exception text. A bug propagates to ``registry.execute``, which
          returns it as a failed ``ToolResult``.
        - Bounds, refused with ``bad_arguments`` and never clamped: at most
          ``MAX_TOOL_NAMES`` entries in ``inputs`` or ``return_keys`` (each name
          is already capped by the synthesize field's key rule). Per registry,
          ``max_obtains`` (an int from 1, else ``SynthesisError``) calls of
          ``obtain_procedure`` that reach ``synthesizer.obtain``, library hits
          included and argument refusals not, then ``failed: budget_spent``;
          ``run_procedure`` is not counted. ``run_procedure`` closes its
          sub-``API`` on every path.
        - ``runner_llm`` is required (``ValueError``); *registry* is a new one
          when ``None`` and a name already in it is replaced.
    """
    if runner_llm is None:
        raise ValueError("runner_llm is required: run_procedure runs on it")
    max_obtains = require_int_limit(
        "max_obtains", max_obtains, minimum=1, error=SynthesisError
    )
    tools = registry if registry is not None else ToolRegistry()
    policy, library = synthesizer.policy, synthesizer.library
    lock, obtains = Lock(), 0

    def failed(code: str, room: int) -> str:
        return _reply([f"failed: {code}"], room)

    def obtain_procedure(
        goal: Any = None, return_keys: Any = None, inputs: Any = None
    ) -> str:
        room = _room(
            "obtain_procedure", goal=goal, return_keys=return_keys, inputs=inputs
        )
        nonlocal obtains
        names = () if inputs is None else inputs
        if (
            not isinstance(goal, str)
            or len(goal) > SYNTHESIZE_GOAL_MAX_LENGTH
            or any(
                isinstance(v, (list, tuple)) and len(v) > MAX_TOOL_NAMES
                for v in (names, return_keys)
            )
        ):
            return failed(_BAD_ARGUMENTS, room)
        with lock:
            if obtains >= max_obtains:
                return failed(REASON_BUDGET_SPENT, room)
            obtains += 1
        try:
            procedure = synthesizer.obtain(goal, inputs=names, return_keys=return_keys)
        except ValueError:
            with lock:  # a refused argument costs nothing
                obtains -= 1
            return failed(_BAD_ARGUMENTS, room)
        except SynthesisError as e:
            return failed(reason_of(e), room)
        record = procedure.record
        gates = verify_stored(
            procedure.definition, policy, inputs=names, return_keys=return_keys
        ).verdicts
        return _reply(
            [
                f"procedure_id: {procedure.digest}",
                f"version: {record.version if record else '-'}",
                f"lifecycle: {record.lifecycle if record else '-'}",
                f"source: {procedure.source}",
                f"states: {len(procedure.definition.states)}",
                "gates: " + " ".join(f"{g}={v}" for g, v in gates.items()),
            ],
            room,
        )

    def run_procedure(procedure_id: Any = None, inputs: Any = None) -> str:
        room = _room("run_procedure", procedure_id=procedure_id, inputs=inputs)
        given = {} if inputs is None else inputs
        if not isinstance(given, dict):
            return failed(_BAD_ARGUMENTS, room)
        if library is None:
            return failed(REASON_LIBRARY_ERROR, room)
        try:
            record = library.get_record(procedure_id)  # validates the id itself
            definition = library.get_definition(procedure_id)
        except LibraryError:
            return failed(REASON_LIBRARY_ERROR, room)
        needs, keys = (
            record.signature["inputs_required"],
            record.signature["return_keys"],
        )
        refusal = runnable_reason(
            definition,
            record.lifecycle,
            min_lifecycle=LIFECYCLE_VERIFIED,
            policy=policy,
            inputs=needs,
            return_keys=keys,
            gates_passed=record.provenance["gates_passed"],
        )
        if refusal:
            return failed(refusal, room)
        if missing := [n for n in needs if is_missing(given, n)]:
            return failed(
                f"{REASON_INPUTS_MISSING} ({_safe(', '.join(missing), 100)})", room
            )
        ok, values, reason = False, {}, _NEEDS_USER_INPUT
        try:
            with API.from_definition(definition, llm_interface=runner_llm) as api:
                # close() ends the sub-conversation on every path
                conv, _ = api.start_conversation({n: given[n] for n in needs})
                api.run_until_terminal(conv, max_steps=policy.max_turns)
                data = api.get_data(conv)
                if any(is_missing(data, k) for k in keys):
                    reason = REASON_RETURN_KEYS_MISSING
                else:
                    ok, values = True, {k: data[k] for k in keys}
        except RunBudgetExceededError:
            pass  # the budget is spent: close() ended the sub-conversation
        except LLMResponseError:
            return failed(REASON_PROVIDER_ERROR, room)  # not the procedure's fault
        try:
            library.record_run(
                procedure_id, ok, trust_after_successes=policy.trust_after_successes
            )
        except SynthesisError:  # the run happened; a library fault must not undo it
            logger.warning(f"could not count the run of {procedure_id[:12]}")
        if not ok:
            return failed(reason, room)
        return _reply(
            [
                f"procedure_id: {procedure_id}",
                "status: ok",
                *(f"{_safe(k, 64)}: {_safe(values[k], _VALUE_CHARS)}" for k in keys),
            ],
            room,
        )

    annotations = ToolAnnotations(
        read_only=False, destructive=False, idempotent=False, open_world=True
    )
    for fn in (obtain_procedure, run_procedure):
        tools.register_function(
            fn,
            name=fn.__name__,
            description=TOOL_DESCRIPTIONS[fn.__name__],
            parameter_schema=TOOL_PARAMETERS[fn.__name__],
            annotations=annotations,
        )
    return tools
