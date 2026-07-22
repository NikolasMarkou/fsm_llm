"""
The six protocol roles and the default LLM-backed worker factory.

``harness.py`` dispatches one worker per state entry through the
``WorkerFactory`` seam; this module is the default implementation of that seam.
Each protocol state gets a :class:`RoleSpec` naming the agent pattern that
backs it, the workspace tools it may call, and the pydantic schema its reply
must satisfy.  :func:`build_default_worker_factory` turns those specs into a
callable the driver can use unchanged.

Three properties matter more than feature breadth here, because the harness's
default model is a 4B one:

1. **Tool scope is structural, not advisory, and DERIVED from ownership.**  A
   role's plan-directory scope comes from ``rules.OWNERSHIP``
   (:func:`_plan_scope`), and a role that owns no artifact is handed a registry
   containing no write tool at all (invariant I7 / decisions.md D-008, D-047).
   Telling a small model not to write is not a control -- and, as review C2
   showed, telling it to write with no tool to do so is not an instruction.
2. **Output schemas are DERIVED, never restated.**  Every role's schema is
   built from ``harness.py``'s ``_WORKER_WRITABLE`` table, which stays the one
   place the protocol's writable keys and their exact types are declared
   (decisions.md D-028).  A hand-written mirror of that table would be exactly
   the duplication D-028 exists to prevent.
3. **Prompts are short, imperative, and ask for one decision per turn.**
   ``rules.py`` already caps operative rules at 8 one-sentence bullets for the
   same reason; this module adds position, gate and output-shape lines and
   nothing else.

Composition, not reimplementation: ``create_agent``/``BaseAgent`` for the
agents, :mod:`fsm_llm_harness.tools` for the confined actions,
:mod:`fsm_llm_harness.hardening` for parsing / coercion / retry, and
:mod:`fsm_llm_harness.rules` for every line of protocol prose.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, Field, create_model

from fsm_llm.definitions import StateNotFoundError
from fsm_llm.logging import logger
from fsm_llm.utilities import _match_brace_partners
from fsm_llm_agents import create_agent
from fsm_llm_agents.definitions import AgentConfig, AgentResult
from fsm_llm_agents.native_fc import NativeFunctionCallingReactAgent
from fsm_llm_agents.tools import ToolRegistry

from .constants import ArtifactNames, ContextKeys, Defaults, HarnessStates
from .hardening import coerce_worker_output, parse_role_output, retry
from .harness import _WORKER_WRITABLE, RoleRequest, WorkerFactory
from .rules import ROLE_BY_STATE, artifacts_writable_by, explore_topic
from .tools import (
    _PER_PLAN_DIRS,
    DISK_DERIVED_COUNTS,
    PLAN_READ_TOOLS,
    PLAN_WRITE_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    WRITE_TOOLS,
    PlanMemory,
    PlanTools,
    Workspace,
    WorkspaceTools,
    build_plan_tools,
    build_workspace_tools,
    derive_disk_counts,
    gate_files,
    has_bytes,
)

__all__ = [
    "AgentBuilder",
    "ROLE_SPECS",
    "RoleSpec",
    "build_default_worker_factory",
    "build_role_prompt",
    "build_role_system_prompt",
    "build_role_task_prompt",
    "count_top_level_json_objects",
    "get_role_spec",
    "held_tools",
]


# ---------------------------------------------------------------------------
# Output schemas (derived from the one writable-key table)
# ---------------------------------------------------------------------------

#: The EXECUTE role's report-only field.
#:
#: ``_WORKER_WRITABLE[EXECUTE]`` is deliberately EMPTY -- the driver derives
#: ``execute_complete`` / ``step_number`` / ``fix_attempts`` from the dispatch
#: itself so a worker cannot understate its own attempt count.  A schema with
#: zero properties would force the model to emit ``{}``, which makes an
#: unparseable reply indistinguishable from a correct one.  ``summary`` gives
#: the executor something answerable that is NOT a gate key: it is not in the
#: allowlist, so ``coerce_worker_output`` drops it before context.
_SUMMARY_FIELD = "summary"

#: The prose field EVERY role's schema carries, for core's own rescue path.
#:
#: Not a protocol key and never writable -- it exists so a reply that IS a bare
#: JSON envelope survives ``fsm_llm/llm.py``'s response-generation parser.  See
#: the D-004 block on :func:`_build_output_schema`.
_MESSAGE_FIELD = "message"

#: Short, imperative field descriptions rendered into the JSON schema.
#:
#: These are the FIELD-level prose; the protocol rules stay in ``rules.py``.
#: Sourced per context key so a key used by two states describes itself
#: identically in both.
_FIELD_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        ContextKeys.FINDINGS_COUNT: (
            "Advisory only: how many findings/<topic>.md files you have "
            "written. The driver counts them on disk."
        ),
        ContextKeys.NEEDS_EXPLORE: "True if more research is required first.",
        ContextKeys.TOTAL_STEPS: "How many steps the plan contains.",
        ContextKeys.ALL_CRITERIA_PASS: (
            "True only if every success criterion is verified PASS with evidence."
        ),
        ContextKeys.NEEDS_PIVOT: "True if the approach itself failed.",
        ContextKeys.COMPLETION_FIX: (
            "True if small fixes in this same iteration finish the work."
        ),
        ContextKeys.CRITERIA_PASS_COUNT: "How many success criteria passed.",
        ContextKeys.CRITERIA_TOTAL: "How many success criteria exist.",
        ContextKeys.PIVOT_RESOLVED: "True once a new direction is chosen and logged.",
        ContextKeys.PIVOT_REASON: "One sentence naming what failed and why.",
        ContextKeys.HALT_REASON: "One sentence recording why the run ended.",
        _SUMMARY_FIELD: "One sentence describing what this step actually did.",
        _MESSAGE_FIELD: "One short sentence stating the result, in plain prose.",
    }
)

#: Python type -> the word used for it in the prompt's output line.
_TYPE_WORDS: Mapping[type, str] = MappingProxyType(
    {int: "integer", bool: "true or false", str: "string"}
)


def _schema_fields(state: str) -> dict[str, type]:
    """Return the field name -> exact type map for *state*'s output schema.

    Derived from ``_WORKER_WRITABLE`` so the schema and the driver's fail-closed
    allowlist can never disagree.  Only EXECUTE differs, and only by adding the
    non-gate ``summary`` field explained above.
    """
    # DECISION plan-2026-07-21T125237-191b2eb2/D-035
    # Two things here look wrong and are deliberate:
    #   1. This reads `harness.py`'s PRIVATE `_WORKER_WRITABLE`. Do NOT fix that
    #      by copying the table here. decisions.md D-028 pins the table's home in
    #      `harness.py` (protocol DATA) while the algorithms live elsewhere, and
    #      a copy is precisely the drift that split exists to prevent. Step 11
    #      owns the tidy-up (it already rewires `harness.py` onto
    #      `hardening.py`'s coercers) -- promote the name there, in one commit,
    #      with every reader migrated.
    #   2. The EXECUTE fallback is NOT a missing case. Its allowlist is
    #      intentionally empty (the driver derives execute_complete /
    #      step_number / fix_attempts from the dispatch, so a worker cannot
    #      understate its own attempt count) -- but a pydantic model with ZERO
    #      properties forces the reply `{}`, and then a correct executor and a
    #      totally garbled one are indistinguishable, so the fix leash never
    #      engages. `summary` is answerable and is NOT a gate key:
    #      `coerce_worker_output` drops it against the same empty allowlist, so
    #      it cannot reach context. Do NOT "complete" the table by adding
    #      `summary` to `_WORKER_WRITABLE` -- that would make it writable.
    # See decisions.md D-035.
    allowed = dict(_WORKER_WRITABLE[state])
    if not allowed:
        return {_SUMMARY_FIELD: str}
    return allowed


def _build_output_schema(state: str, fields: Mapping[str, type]) -> type[BaseModel]:
    """Build the pydantic model a *state*'s worker must answer with.

    Every field is REQUIRED: a reply missing one fails validation, leaves
    ``structured_output`` as ``None``, and falls back to text parsing which then
    reports ``missing-keys`` (invariant I8, fail closed).  ``message`` is added
    to every role's schema; it is prose for core, never a protocol key.
    """
    definitions: dict[str, Any] = {
        name: (
            annotation,
            Field(..., description=_FIELD_DESCRIPTIONS.get(name, name)),
        )
        for name, annotation in (*fields.items(), (_MESSAGE_FIELD, str))
    }
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-004
    # `message` is appended to EVERY role schema, and it is a [REUSE] of a core
    # path -- NOT a weakening of the apology guard. `llm.py`'s
    # `_parse_response_generation_response` reads `data.get("message")` on its
    # FIRST rung; with no such key a brace-wrapped role reply falls two rungs
    # through to the terminal guard, which substitutes
    # `_GENERIC_FALLBACK_MESSAGE` ("I'm sorry, I couldn't generate a proper
    # response.") -- MEASURED at step 7f of the predecessor plan, where a
    # perfectly valid `{"findings_count": 3, "needs_explore": false}` still
    # parsed as `unparseable`. D-022 of plan-2026-07-18T162030-a02151fe decided
    # PERMANENTLY not to loosen that guard (no text-shape discriminator can
    # separate a mistaken envelope from prose that quotes JSON), so the harness
    # supplies the key the guard looks for instead of asking core to look away.
    # Do NOT add `message` to `_WORKER_WRITABLE`: it follows `summary`'s rule
    # (D-035) -- schema-visible, dropped by `coerce_worker_output`, and absent
    # from `expected_keys` so `parse_role_output` never requires it. Making it
    # writable would let a worker smuggle prose into a gate key.
    # See decisions.md D-004.
    model: type[BaseModel] = create_model(
        f"{state.capitalize()}Output",
        __doc__=f"Structured reply required from the {state} role.",
        **definitions,
    )
    return model


# ---------------------------------------------------------------------------
# Role specs
# ---------------------------------------------------------------------------

#: state -> the WORKSPACE tools that state's role may call.
#:
#: The workspace holds the code the protocol is changing, so EXECUTE is the only
#: state whose role receives ``WRITE_TOOLS``, and REFLECT is the only one
#: receiving ``SHELL_TOOLS`` (inert unless the caller enabled shell access)
#: because verification is what needs to run a test command.  Every other role
#: is strictly read-only *here* -- their writes go to the plan directory
#: instead, through :data:`_PLAN_SCOPE_BY_STATE` below.
_TOOL_SCOPE_BY_STATE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        HarnessStates.EXPLORE: READ_ONLY_TOOLS,
        HarnessStates.PLAN: READ_ONLY_TOOLS,
        HarnessStates.EXECUTE: READ_ONLY_TOOLS + WRITE_TOOLS,
        HarnessStates.REFLECT: READ_ONLY_TOOLS + SHELL_TOOLS,
        HarnessStates.PIVOT: READ_ONLY_TOOLS,
        HarnessStates.CLOSE: READ_ONLY_TOOLS,
    }
)


def _plan_scope(role: str) -> tuple[str, ...]:
    """Return the plan-directory tools *role* may call, DERIVED from ownership.

    This function IS invariant I7 at the tool layer.  A role that owns at least
    one artifact gets the write tools (which then refuse every artifact it does
    not own); a role that owns nothing gets read tools only, so it cannot write
    protocol memory at all.
    """
    # DECISION plan-2026-07-21T125237-191b2eb2/D-047
    # DERIVED from `rules.OWNERSHIP`, never hand-listed. The defect this repairs
    # (review C2) was precisely a hand-maintained scope table drifting from the
    # ownership table it was supposed to encode: five of six roles were ORDERED
    # by their operative rules to write artifacts `OWNERSHIP` grants them, while
    # holding only read tools -- so the live spike's "workspace byte-identical"
    # result was structural, not a model ceiling. Do NOT reintroduce a literal
    # per-state plan-tool table "for clarity": clarity is what produced the bug.
    # The ownership check itself lives in `PlanMemory.authorise`, so this
    # coarse grant can never be more permissive than the table.
    # See decisions.md D-047.
    if artifacts_writable_by(role):
        return PLAN_READ_TOOLS + PLAN_WRITE_TOOLS
    return PLAN_READ_TOOLS


#: state -> the agent loop budget for one dispatch.
#:
#: Sized so a role can read a few files, WRITE the artifacts it owns, and still
#: have a turn left to stop with a final answer.  Still small: a 4B model that
#: has not answered by then is looping, not thinking, and every extra turn is
#: seconds of wall clock plus a longer prompt on the next one.
# DECISION plan-2026-07-21T191807-bf7ffe24/D-013
# These numbers were RAISED from 8/6/10/8/6/6 on measurement, and the old row
# must not come back as a "tightening". Live n=5 on `ollama_chat/qwen3.5:4b`
# with the real EXPLORE spec: 5/5 dispatches EXHAUSTED the 8-turn budget on
# READ tools (11-20 calls/run, only list_dir/read_file/read_plan_file), wrote
# ZERO bytes to the plan directory, and 5/5 still returned findings_count: 3 --
# one reply even named the three findings files it had not written. `concluded`
# and therefore D-005's `success` were False 5/5 despite a schema-valid payload.
# A budget that cannot fit "read, write, stop" makes the write structurally
# unreachable, which is the same defect class as review C2's missing write tool.
# The budget is only HALF the repair -- `_finish_line` below is the other half,
# and neither works alone: raising turns without a stopping condition just buys
# more reading. Do NOT tune these down for wall clock without re-running the
# EXPLORE bench; the per-dispatch guard against a runaway loop is
# `AgentConfig.timeout_seconds`, not this table.
# See decisions.md D-013.
_MAX_ITERATIONS_BY_STATE: Mapping[str, int] = MappingProxyType(
    {
        HarnessStates.EXPLORE: 14,
        HarnessStates.PLAN: 10,
        HarnessStates.EXECUTE: 14,
        HarnessStates.REFLECT: 12,
        HarnessStates.PIVOT: 10,
        HarnessStates.CLOSE: 10,
    }
)


@dataclass(frozen=True)
class RoleSpec:
    """How one protocol role is realised as an agent.

    Part of the worker-seam data group (``RoleSpec`` / ``RoleRequest`` /
    ``RoleOutput`` / ``WorkerFactory``), which the Complexity Budget counts as a
    single abstraction slot.

    Attributes:
        role: The worker role dispatched for ``state``.
        state: The protocol state this spec serves.
        pattern: The ``create_agent`` pattern name backing the role.
        tool_scope: Workspace tool names this role may call.  Enforced by
            construction -- the role's registry holds nothing else.
        plan_tool_scope: Plan-directory tool names this role may call, derived
            from ``rules.OWNERSHIP``.  Registered only when the dispatch knows
            a plan directory (``RoleRequest.plan_dir``).
        owned_artifacts: Every artifact ``OWNERSHIP`` lets this role write.
            Empty means the role reports its result and writes nothing.
        output_schema: The pydantic model set as ``AgentConfig.output_schema``.
        expected_keys: Keys ``parse_role_output`` requires; identical to the
            schema's fields.
        writable_keys: The subset of ``expected_keys`` the driver will accept
            into context, with their exact required types.
        max_iterations: Agent loop budget for one dispatch.
    """

    role: str
    state: str
    pattern: str
    tool_scope: tuple[str, ...]
    plan_tool_scope: tuple[str, ...]
    owned_artifacts: tuple[str, ...]
    output_schema: type[BaseModel]
    expected_keys: tuple[str, ...]
    writable_keys: Mapping[str, type]
    max_iterations: int


def _build_spec(state: str) -> RoleSpec:
    """Assemble the :class:`RoleSpec` for one state from the shared tables."""
    fields = _schema_fields(state)
    role = ROLE_BY_STATE[state]
    return RoleSpec(
        role=role,
        state=state,
        # Every role uses the ReAct loop: it is the only pattern in
        # `create_agent`'s registry that both takes a tool registry and drives
        # a read-act-conclude cycle, which is what all six roles do. Patterns
        # like debate/self_consistency multiply LLM calls per turn -- on a 4B
        # model that is minutes of wall clock for no protocol gain.
        pattern="react",
        tool_scope=_TOOL_SCOPE_BY_STATE[state],
        plan_tool_scope=_plan_scope(role),
        owned_artifacts=artifacts_writable_by(role),
        output_schema=_build_output_schema(state, fields),
        expected_keys=tuple(fields),
        writable_keys=MappingProxyType(dict(_WORKER_WRITABLE[state])),
        max_iterations=_MAX_ITERATIONS_BY_STATE[state],
    )


#: state id -> its frozen role spec.  Covers all 6 protocol states.
ROLE_SPECS: Mapping[str, RoleSpec] = MappingProxyType(
    {state: _build_spec(state) for state in HarnessStates.ALL}
)


def get_role_spec(state: str) -> RoleSpec:
    """Return the role spec for *state*.

    Raises:
        StateNotFoundError: If *state* is not one of the 6 protocol states.
    """
    try:
        return ROLE_SPECS[state]
    except KeyError:
        raise StateNotFoundError(
            f"Unknown harness state '{state}'; expected one of "
            f"{', '.join(HarnessStates.ALL)}",
            state_id=state,
        ) from None


# ---------------------------------------------------------------------------
# Reply observation (decisions.md D-031)
# ---------------------------------------------------------------------------

#: Context keys never rendered into a role prompt: they are the driver's own
#: bookkeeping and are large, so they would crowd out the actual question.
_PROMPT_CONTEXT_SKIP: frozenset[str] = frozenset(
    {
        ContextKeys.DISPATCH_LEDGER,
        ContextKeys.ROLE_RESULTS,
        ContextKeys.CURRENT_ROLE_RESULT,
        ContextKeys.GOAL,
    }
)

#: Bounds on the context snapshot rendered into a role prompt.
_MAX_PROMPT_CONTEXT_KEYS = 12
_MAX_PROMPT_VALUE_CHARS = 120

#: Every ``{`` in a reply, for the top-level object COUNT below.
_OPEN_BRACE_RE = re.compile(r"\{")


def count_top_level_json_objects(text: Any) -> int:
    """Count the balanced, non-nested JSON objects in a model's reply.

    Interface contract (observation only -- 2 call sites: the default worker
    factory below, and the live spike):
        - Parameter: any object; a non-``str`` counts 0.
        - Returns how many top-level ``{...}`` spans balance.  Nested objects
          are not counted, and an unbalanced ``{`` is ignored.
        - Never raises.

    A count of 2+ is the observable signature of the one measured FAIL-OPEN
    path in the hardening layer (decisions.md D-031): a reply that drafts one
    object and then corrects it yields the DRAFT, because
    ``extract_json_from_text`` is deliberately FIRST-wins.
    """
    # DECISION plan-2026-07-21T125237-191b2eb2/D-033
    # This is a COUNTING PASS over core's brace scanner, NOT a second scanner --
    # writing one is a named Complexity-Budget BREACH, and D-031 names the count
    # as the non-breaching escalation. `_match_brace_partners`
    # (utilities.py:119) is the repo's only string/escape-aware brace matcher;
    # do NOT replace this call with a local depth counter or a `text.count("{")`
    # heuristic. A local depth counter mishandles a `}` inside a quoted string
    # (the still-open M5 defect in llm.py:704-720), and `count("{")` counts
    # nested objects. The `skip_until` walk below mirrors
    # `extract_json_from_text` Strategy 3's own top-level walk, so this count
    # and that extractor agree about what "top level" means.
    # This is OBSERVATION ONLY and must stay that way until step 7's live spike
    # says otherwise: it does not reject a reply, because a 2-object reply may
    # still be perfectly correct (an answer plus a restated schema).
    # See decisions.md D-031, D-033.
    if not isinstance(text, str) or "{" not in text:
        return 0
    starts = [match.start() for match in _OPEN_BRACE_RE.finditer(text)]
    partners = _match_brace_partners(text, starts)

    count = 0
    covered_until = -1
    for start in starts:
        if start <= covered_until:
            continue
        end = partners.get(start)
        if end is None:
            continue
        count += 1
        covered_until = end
    return count


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def held_tools(request: RoleRequest, spec: RoleSpec) -> tuple[str, ...]:
    """Return the exact tool names this dispatch will hold.

    Interface contract (shared helper, 2 call sites: the default worker factory
    registers exactly these, and :func:`build_role_prompt` names exactly these):
        - Parameters: the driver's ``RoleRequest`` and the matching
          :class:`RoleSpec`.
        - Returns workspace tools plus -- only when ``request.plan_dir`` is set
          -- the role's plan-directory tools.
        - Never raises; performs no I/O.

    One function so the registry and the prompt cannot disagree.  The prompt
    naming a tool the role does not hold is the review-C2 defect in its other
    direction, and it is just as unexecutable.
    """
    if request.plan_dir is None:
        return tuple(spec.tool_scope)
    return (*spec.tool_scope, *spec.plan_tool_scope)


def _holds_write_tool(names: Iterable[str]) -> bool:
    """Whether this dispatch holds a tool that can put bytes anywhere.

    Interface contract (shared predicate, 2 call sites: :func:`build_role_prompt`
    decides the prompt's verb and write obligation from it, and the default
    worker factory decides whether a completion claim needs write EVIDENCE from
    it):
        - Parameter: the tool names the dispatch actually holds
          (:func:`held_tools`).
        - Returns ``True`` when any workspace or plan-directory write tool is
          among them.
        - Never raises; performs no I/O.

    One predicate so the prompt's promise and the mechanical check are the same
    fact: a role is asked to write exactly when it will be held to having done
    so.
    """
    return any(name in WRITE_TOOLS or name in PLAN_WRITE_TOOLS for name in names)


def _writes_line(request: RoleRequest, spec: RoleSpec) -> str:
    """Render what this dispatch may write, or that it may write nothing."""
    if request.plan_dir is not None and spec.owned_artifacts:
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-013
        # A directory artifact must be rendered as a writable PATH, not as its
        # bare name. Measured, step 4b attempt 1: told it may write `findings`,
        # `:4b` called the WORKSPACE `path_exists("findings")`, got "no", and
        # answered "cannot write findings without the directory existing" --
        # while `PlanMemory.write_text` creates parents and would have accepted
        # `findings/<topic>.md` on the first try. Naming the tool and the root
        # closes the other half of the same confusion (the model reached for a
        # workspace tool for a plan-directory path). The dir/file split is READ
        # from `tools._PER_PLAN_DIRS`, the same set `PlanMemory._classify`
        # authorises against, so this text cannot claim a shape the tool
        # refuses -- do NOT re-derive it here from a `.md` suffix test.
        # See decisions.md D-013.
        targets = ", ".join(
            f"{name}/<topic>.md" if name in _PER_PLAN_DIRS else name
            for name in spec.owned_artifacts
        )
        return (
            f"YOU MAY WRITE these protocol files with "
            f"{PlanTools.WRITE_PLAN_FILE}, and no others: {targets}. "
            "These paths are relative to the plan directory, NOT the "
            "workspace, and any missing folder is created for you. A write to "
            "anything else is refused."
        )
    return (
        "YOU MAY WRITE no protocol file. Report your result in the JSON "
        "object below; the driver records it."
    )


def _topic_line(request: RoleRequest) -> str:
    """Render this dispatch's ONE assigned findings file, or ``""``.

    Interface contract (1 call site, :func:`_prompt_blocks`):
        - ``request``: the driver's ``RoleRequest``; only ``assigned_topic`` is
          read.
        - Returns ``""`` when the driver assigned no topic -- so every state
          but EXPLORE, and an EXPLORE dispatch with no plan directory, receives
          byte-for-byte the prompt it received before.
        - Never raises; performs no I/O.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-035
    # A PATH, not a count. The exit gate two blocks up already asks for three
    # distinct files, and the coverage line at the end of the task message
    # already reports how many exist -- restating the number here would be the
    # fifth refutation of a mechanism refuted four times (decisions.md D-027).
    # What is new is that this dispatch is told WHICH ONE FILE is its job, which
    # is the shape EXECUTE already succeeds at (write one named file, 10/10).
    # Do NOT move this into the SYSTEM half: the assignment changes between two
    # dispatches of the same role, and the system half is the text that does not
    # (D-021). Do NOT let the role choose or negotiate the path -- the driver
    # picked it against the files really on disk, so a role that writes
    # somewhere else re-opens the collision the assignment exists to prevent.
    # See decisions.md D-035.
    if not request.assigned_topic:
        return ""
    topic = explore_topic(request.assigned_topic)
    return (
        f"YOUR TOPIC THIS DISPATCH: {topic.label} -- {topic.brief}.\n"
        f"WRITE IT TO: {ArtifactNames.FINDINGS_DIR}/{topic.slug}.md\n"
        "That exact path, nothing else. One topic, one file. The other topics "
        "belong to other dispatches; do not write them here."
    )


def _finish_line(spec: RoleSpec, can_write: bool) -> str:
    """Render the terminal instruction: what to do, and when to stop doing it."""
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-013
    # This section exists because a live n=5 EXPLORE bench measured the model
    # reading until its budget ran out and never writing: 11-20 read-tool calls
    # per run, zero bytes on disk, and a final payload claiming three findings
    # were indexed. Nothing in the prompt had ever told it WHEN to stop reading
    # or that the write was the deliverable -- the operative rules say what to
    # write, never that reading more is not a substitute. The last sentence
    # (do not claim a write a tool did not confirm) targets the exact fail-open
    # reply that was measured, not a hypothetical one. Do NOT delete this as
    # prompt bloat: `rules.py` caps operative rules at 8 bullets precisely so
    # the driver can add the few lines a dispatch needs, and this is one of
    # them. See decisions.md D-013.
    budget = (
        f"HOW TO FINISH: you have at most {spec.max_iterations} turns, and "
        "reading more is not a substitute for finishing."
    )
    if not can_write:
        return f"{budget} Read only what you need, then stop calling tools and answer."
    return (
        f"{budget} Read only what you need, then WRITE -- the write is the "
        "deliverable. As soon as it is written, stop calling tools and answer. "
        "Never state that you wrote a file unless a write tool reported success."
    )


def _output_line(spec: RoleSpec) -> str:
    """Render the one-object output instruction for *spec*."""
    # Read off the BUILT schema, not `_schema_fields`, so the shape the prompt
    # asks for and the shape constrained decoding enforces are one fact --
    # including the `message` field D-004 appends.
    shape = ", ".join(
        f'"{name}": <{_TYPE_WORDS.get(info.annotation or str, "value")}>'
        for name, info in spec.output_schema.model_fields.items()
    )
    # The two sentences after the shape are the ONLY available mitigation for
    # the draft-then-correction fail-open path (decisions.md D-031): constrained
    # decoding plus an explicit "one object, no drafts" instruction. They are
    # load-bearing prompt text, not politeness.
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-004
    # This asks for a BARE object, and the "one short sentence first" wording
    # that stood here until step 20 must not come back. It was correct when it
    # was written (D-061 of plan-2026-07-21T125237-191b2eb2): a reply that was
    # nothing but a JSON object was destroyed by core's response-generation
    # envelope guard, which substitutes `_GENERIC_FALLBACK_MESSAGE` for any
    # brace-wrapped text carrying no `message` key. Step 4 removed that
    # precondition -- `_build_output_schema` now appends `message: str` to EVERY
    # role schema (see the D-004 block there), so core's FIRST rung recovers the
    # sentence from inside the object and the bare shape survives verbatim.
    # The old text therefore made two false statements to every dispatch, and
    # the `response_format` repair turn (D-002) structurally cannot obey it --
    # a constrained-decoding turn emits the object and nothing else.
    # Do NOT re-add the prose prefix without ALSO removing `message` from the
    # schema; the two are one fact, and the control for it is the "no-`message`
    # reply still gets the apology" test in `test_roles_and_tools.py`.
    # See decisions.md D-004.
    return (
        f"Finish by returning exactly ONE JSON object and nothing else: "
        f"{{{shape}}}\n"
        "Put your sentence in the object's message field, not around it. "
        "Do not show drafts, corrections or alternatives. "
        "Do not repeat the object."
    )


def _context_snapshot(context: Mapping[str, Any]) -> str:
    """Render a bounded, readable slice of the FSM context."""
    lines: list[str] = []
    for key, value in context.items():
        if key.startswith("_") or key in _PROMPT_CONTEXT_SKIP:
            continue
        if value is None:
            continue
        text = str(value)
        if len(text) > _MAX_PROMPT_VALUE_CHARS:
            text = text[:_MAX_PROMPT_VALUE_CHARS] + "..."
        lines.append(f"- {key}: {text}")
        if len(lines) >= _MAX_PROMPT_CONTEXT_KEYS:
            break
    return "\n".join(lines)


def _prompt_blocks(request: RoleRequest, spec: RoleSpec) -> list[tuple[bool, str]]:
    """Return every prompt block as ``(standing, text)``, in prompt order.

    Interface contract (3 call sites: :func:`build_role_system_prompt`,
    :func:`build_role_task_prompt` and :func:`build_role_prompt`, which are
    three FILTERS over this one list):
        - ``standing`` is ``True`` for text that is true on EVERY turn of the
          dispatch -- who the role is, its gate, its rules, the tools it holds,
          what it may write, when to stop, the reply shape.  It is ``False``
          for what this dispatch in particular is about: the goal, the position
          and the context snapshot.
        - The order is the prompt's order, so joining every block reproduces
          the single-string prompt exactly.
        - Never raises.

    One list, three readers, so the system message and the task message cannot
    between them drop or duplicate a block.
    """
    rules = "\n".join(f"- {rule}" for rule in request.operative_rules)
    snapshot = _context_snapshot(request.context)
    names = held_tools(request, spec)
    tools = ", ".join(names) or "none"
    can_write = _holds_write_tool(names)
    # The verb is DERIVED from the tools actually registered for this dispatch.
    # Hardcoding "inspect and change" is what told five read-only roles to write
    # files they had no tool for (review C2); a prompt that describes capability
    # the role does not have is an unexecutable instruction, not encouragement.
    verb = "inspect and change" if can_write else "inspect"

    blocks: list[tuple[bool, str]] = [
        (
            True,
            f"You are the {request.role} for the {request.state.upper()} phase "
            "of an iterative planning protocol.",
        ),
        (False, f"GOAL: {request.goal}"),
    ]
    topic = _topic_line(request)
    if topic:
        # Directly under the GOAL, because it is what this dispatch is FOR: the
        # goal says what the run wants, this says which one slice of it this
        # dispatch owns and where that slice goes.
        blocks.append((False, topic))
    blocks.extend(
        [
            (
                False,
                f"POSITION: iteration {request.iteration}, step "
                f"{request.step_number} of {request.total_steps}, fix attempts "
                f"used {request.fix_attempts} of {Defaults.MAX_FIX_ATTEMPTS}.",
            ),
            (True, f"EXIT GATE: {request.gate_summary}"),
            (True, f"RULES:\n{rules}"),
        ]
    )
    if snapshot:
        blocks.append((False, f"CURRENT STATE:\n{snapshot}"))
    blocks.append(
        (
            True,
            f"TOOLS: {tools}. Use them to {verb} real files; "
            "do not guess file contents.",
        )
    )
    blocks.append((True, _writes_line(request, spec)))
    blocks.append((True, _finish_line(spec, can_write)))
    blocks.append((True, _output_line(spec)))
    return blocks


def build_role_system_prompt(request: RoleRequest, spec: RoleSpec) -> str:
    """Build the STANDING half of a role dispatch: policy, not task.

    Interface contract (2 call sites: the default worker factory, for agents
    that can hold a system policy, and the live bench):
        - Parameters: the driver's ``RoleRequest`` and the matching
          :class:`RoleSpec`.
        - Returns the blocks that are true on every turn -- identity, exit gate,
          operative rules, held tools, write scope, stop rule, reply shape.
        - Never raises.  Together with :func:`build_role_task_prompt` it
          partitions :func:`build_role_prompt`; neither drops a block.
    """
    return "\n\n".join(
        text for standing, text in _prompt_blocks(request, spec) if standing
    )


def build_role_task_prompt(request: RoleRequest, spec: RoleSpec) -> str:
    """Build the per-dispatch half of a role dispatch: task, not policy.

    Interface contract (2 call sites: the default worker factory, for agents
    that can hold a system policy, and the live bench):
        - Parameters: as :func:`build_role_system_prompt`.
        - Returns the goal, the position and the bounded context snapshot -- the
          things that differ between two dispatches of the SAME role.
        - Never raises.
    """
    return "\n\n".join(
        text for standing, text in _prompt_blocks(request, spec) if not standing
    )


def build_role_prompt(request: RoleRequest, spec: RoleSpec) -> str:
    """Build the whole prompt for one role dispatch as a single string.

    Interface contract (2+ call sites: the default worker factory, for agents
    that cannot hold a system policy, and any caller writing its own factory
    against the same specs):
        - Parameters: the driver's ``RoleRequest`` and the matching
          :class:`RoleSpec`.
        - Returns a single prompt string: role, goal, position, gate, rules,
          bounded context, the tools this dispatch actually holds
          (:func:`held_tools`), the artifacts it may write, and the
          one-JSON-object output instruction.
        - Never raises.

    Short and imperative on purpose.  Every line here is read by a 4B model on
    every turn; prose costs accuracy, not just tokens.
    """
    return "\n\n".join(text for _, text in _prompt_blocks(request, spec))


# ---------------------------------------------------------------------------
# Mechanical verification: the filesystem answers, the worker only claims
# ---------------------------------------------------------------------------

#: The two confined roots, as labels for :func:`_verified_writes`' report.
_WORKSPACE_ROOT = "workspace"
_PLAN_ROOT = "plan"

#: Write tool -> the root a successful call leaves bytes under.
#:
#: ``delete_file`` is deliberately ABSENT: a delete is a write tool but it
#: removes bytes, so counting it as evidence of work would let "I deleted a
#: file" stand in for "I implemented the step".
_BYTE_WRITING_TOOLS: Mapping[str, str] = MappingProxyType(
    {
        WorkspaceTools.WRITE_FILE: _WORKSPACE_ROOT,
        WorkspaceTools.APPEND_FILE: _WORKSPACE_ROOT,
        PlanTools.WRITE_PLAN_FILE: _PLAN_ROOT,
        PlanTools.APPEND_PLAN_FILE: _PLAN_ROOT,
    }
)


def _verified_writes(
    result: AgentResult,
    *,
    workspace: Workspace,
    memory: PlanMemory | None,
) -> tuple[str, ...]:
    """Return the writes this dispatch both CALLED and left bytes for.

    Interface contract (1 call site today, the default worker factory; exported
    shape kept narrow on purpose):
        - Reads ``result.trace.tool_calls`` -- the agent loop's own record of
          what it dispatched -- and re-reads each target through the confined
          root that tool writes to.
        - Returns ``"<root>:<path>"`` labels, one per VERIFIED write.  A call
          whose target is missing, empty or refused is not in the result.
        - Never raises, whatever shape the trace has.
    """
    calls = getattr(getattr(result, "trace", None), "tool_calls", None) or []
    verified: list[str] = []
    for call in calls:
        root = _BYTE_WRITING_TOOLS.get(getattr(call, "tool_name", ""))
        if root is None:
            continue
        path = (getattr(call, "parameters", None) or {}).get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        reader = workspace.read_text if root == _WORKSPACE_ROOT else None
        if reader is None and memory is not None:
            reader = memory.read_text
        if reader is not None and has_bytes(reader, path):
            verified.append(f"{root}:{path}")
    return tuple(verified)


def _reconcile_structured(structured: Any, derived: Mapping[str, Any]) -> Any:
    """Rebuild a worker's structured payload with the DERIVED values in it.

    Correcting ``final_context`` alone would leave the fix depending on
    ``harness._apply_role_result`` merging ``structured_output`` FIRST -- a real
    ordering, but an accident rather than a guarantee.  Both channels are
    corrected so neither can carry the claim.  A payload that cannot be rebuilt
    is dropped entirely (``None``), never returned uncorrected.
    """
    if not derived or not isinstance(structured, BaseModel):
        return structured
    dumped = structured.model_dump()
    if not any(key in dumped for key in derived):
        return structured
    try:
        return type(structured)(**{**dumped, **derived})
    except Exception:  # unrebuildable: fail closed rather than pass the claim on
        logger.warning(
            f"Could not reconcile {type(structured).__name__} with the "
            f"filesystem-derived values {dict(derived)!r}; dropping the payload."
        )
        return None


# ---------------------------------------------------------------------------
# Default worker factory
# ---------------------------------------------------------------------------

#: Builds the agent that executes one role dispatch.
#:
#: Injectable so tests and the live spike can substitute a recording or mocked
#: agent without an LLM; the returned object needs only ``run(task) ->
#: AgentResult``.
AgentBuilder = Callable[[RoleSpec, ToolRegistry, AgentConfig], Any]


def _default_agent_builder(
    *, native_function_calling: bool, seed: int | None = None
) -> AgentBuilder:
    """Return the stock agent builder.

    ``native_function_calling`` selects ``NativeFunctionCallingReactAgent``
    (the default since D-049) or the prompt-parsed ReAct loop.  Both are
    existing agents; this is a choice between them, not a third implementation.
    """

    def build(spec: RoleSpec, registry: ToolRegistry, config: AgentConfig) -> Any:
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-049
        # The DEFAULT is `NativeFunctionCallingReactAgent`. This SUPERSEDES the
        # predecessor plan's D-034, which kept the prompt-parsed ReAct loop, and
        # the replacement reason is not the one D-034 pre-registered. Read both
        # halves before touching this line.
        #
        # (1) D-034's PREMISE IS GONE, not overruled. It kept ReAct because
        # `create_agent("react")` routes through `BaseAgent._init_context`,
        # turning `AgentConfig.output_schema` into a `response_format` for
        # CONSTRAINED DECODING, while `NativeFunctionCallingReactAgent.run` set
        # no `response_format` at all. That second clause stopped being true at
        # `6c3da51`: native_fc now makes exactly ONE post-loop repair completion
        # carrying `response_format=` and no `tools=` (native_fc.py, D-002),
        # measured 5/5 schema-valid on `:4b`. The native arm no longer trades a
        # single-object guarantee for tool-calling reliability, so D-034's
        # gate-safety tiebreak has nothing left to weigh.
        #
        # (2) THE NATIVE ARM DID NOT PASS ITS BAR, and this flip does not claim
        # it did. On the L4 EXECUTE bench (`ollama_chat/qwen3.5:4b`) it measured
        # "write tool issued AND bytes on disk" at 3/5 and, on a second n=5
        # block taken at this commit, 2/5 -- **5/10 pooled** -- against
        # plan.md's >=4/5, and 0/20 on the strict content-hash form. Criterion 1
        # is NOT met. The flip is authorised by the OTHER arm of the same two
        # benches: ReAct measured 0/5 and 0/5, **0/10 pooled**, issuing ONE tool
        # call in twenty dispatches, most of them dying with `Stall detected: 3
        # consecutive iterations with no tool selected` and `success=False`.
        # 5/10 vs 0/10 is Fisher two-sided p = 0.033. The decision is "5/10
        # beats 0/10 on the only executor users actually get", NOT "the native
        # arm cleared 4/5". Do NOT launder that 5/10 into a pass anywhere.
        #
        # Do NOT delete the ReAct branch below or drop the keyword: it is the
        # control arm criterion 1 is measured against (both arms, every run --
        # see `test_live_ollama.py::TestL4ToolCallAndFileWrite`), and a default
        # chosen against 0/5 is a default that must stay falsifiable.
        # See decisions.md D-049 (this plan), superseding D-034 (predecessor).
        if native_function_calling:
            return NativeFunctionCallingReactAgent(
                tools=registry, config=config, seed=seed
            )
        return create_agent(pattern=spec.pattern, tools=registry, config=config)

    return build


def _address(agent: Any, request: RoleRequest, spec: RoleSpec) -> str:
    """Hand *agent* its standing policy where it can hold one; return the task.

    Interface contract (1 call site, the worker below; a function rather than
    four inline lines so the fallback is testable on its own):
        - Sets ``agent.system_policy`` and returns ONLY the task text when the
          agent exposes that attribute.
        - Returns the whole single-string prompt otherwise, leaving the agent
          untouched.  An agent that cannot hold a system policy must still be
          told everything, so this fallback is the no-loss path, not a
          degradation.
        - Never raises.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-021
    # WHERE the standing policy is delivered is the whole of this change, and
    # it is measured, not stylistic. Live `ollama_chat/qwen3.5:4b`, EXECUTE
    # role, n=5 per arm, bytes stat'd on disk before/after: the entire prompt
    # in the user turn wrote 0/5 (5/5 of those runs still CLAIMED the edit);
    # the identical text with every standing block moved to the system message
    # wrote 4/5 to the workspace and 4/5 to the plan directory. Two content
    # ablations refute the obvious alternative explanations -- repairing the
    # writes line so it stops reading as "you may not touch the code" left it
    # at 0/5, and deleting the 730-char rules block outright reached only 2/5.
    # Do NOT re-merge the two halves "for simplicity": the merged form IS
    # `build_role_prompt`, it is still here, and it is the measured 0/5 arm.
    # Do NOT switch the duck-typed check to `isinstance(...)`: the fallback is
    # what keeps `create_agent("react")` (the pre-D-049 default, still one
    # keyword away) and every injected
    # `agent_builder` receiving the complete prompt they receive today, so this
    # change is inert for them by construction rather than by promise.
    # See decisions.md D-021.
    if not hasattr(agent, "system_policy"):
        return build_role_prompt(request, spec)
    agent.system_policy = build_role_system_prompt(request, spec)
    return build_role_task_prompt(request, spec)


def _coverage_line(memory: PlanMemory | None, spec: RoleSpec) -> str:
    """What EARLIER dispatches already put on disk, for a gate-counted role.

    Interface contract (1 call site, the worker below):
        - ``memory``: this dispatch's role-scoped :class:`PlanMemory`, or
          ``None`` when there is no plan directory.
        - Returns ``""`` unless the role owns a disk-derived gate count AND that
          directory already holds at least one file -- so the FIRST dispatch of
          a run receives byte-for-byte the prompt it receives today, and every
          role that owns no such count is untouched.
        - Never raises.
    """
    # DECISION plan-2026-07-21T191807-bf7ffe24/D-028
    # This is a PER-DISPATCH line, and it is the half of bounded re-dispatch
    # that stops N explorers writing N copies of one topic. It states what is
    # already covered, not how many findings the protocol wants -- the exit gate
    # already says that, twice, in the system message, and step 22 measured that
    # the model reads the count back accurately and stops at 1 anyway. Do NOT
    # "strengthen" it into a fifth restatement of the 3-file requirement: that
    # mechanism has now been refuted four separate times (decisions.md D-027).
    # The NAMES are the payload; the count is context for them.
    # Do NOT move it into `_prompt_blocks`: the names come from the filesystem
    # through `gate_files`, and `_prompt_blocks` takes a `RoleRequest` -- which
    # is a context snapshot, not a plan directory. Reading the directory in the
    # factory, where the confined role-scoped `PlanMemory` already exists, is
    # what keeps the ONE derivation (D-027) unduplicated.
    if memory is None:
        return ""
    for key, (directory, threshold) in DISK_DERIVED_COUNTS.items():
        if key not in spec.writable_keys:
            continue
        names = gate_files(memory, directory)
        if not names:
            return ""
        return (
            f"ALREADY ON DISK -- {directory}/ holds {len(names)} of the "
            f"{threshold} distinct files the exit gate requires: "
            f"{', '.join(names)}. Earlier dispatches wrote those. Write ONE "
            f"file this list does not have; rewriting one of them leaves the "
            f"count at {len(names)}."
        )
    return ""


def _payload_from(result: AgentResult) -> Any:
    """Prefer a validated structured output; fall back to the raw answer."""
    structured = result.structured_output
    if isinstance(structured, BaseModel):
        return structured.model_dump()
    if isinstance(structured, Mapping):
        return structured
    return result.answer


def build_default_worker_factory(
    workspace: Workspace,
    *,
    model: str = Defaults.MODEL,
    temperature: float = Defaults.TEMPERATURE,
    max_tokens: int = Defaults.MAX_TOKENS,
    timeout_seconds: float = Defaults.LLM_TIMEOUT_SECONDS,
    retry_attempts: int = Defaults.RETRY_ATTEMPTS,
    native_function_calling: bool = True,
    seed: int | None = None,
    agent_builder: AgentBuilder | None = None,
    observer: Callable[[Mapping[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> WorkerFactory:
    """Build the stock ``worker_factory`` for :class:`~fsm_llm_harness.HarnessAgent`.

    Interface contract:
        - ``workspace``: the confined root the CODE-editing tools act on.  The
            protocol's own artifacts live under a second root, the plan
            directory, which arrives per dispatch as ``RoleRequest.plan_dir``
            (driver-owned) and is confined and ownership-scoped by
            :class:`~fsm_llm_harness.tools.PlanMemory`.  A dispatch without a
            plan directory holds no plan-file tool at all.
        - ``model`` / ``temperature`` / ``max_tokens`` / ``timeout_seconds``:
            the per-dispatch ``AgentConfig``.
        - ``retry_attempts``: total attempts per dispatch, retried only on
            transient LLM/transport faults (``hardening.retry``'s strict default
            allowlist -- a garbled reply is NOT retried, it fails closed).
        - ``native_function_calling``: back roles with
            ``NativeFunctionCallingReactAgent`` (the default) or, when
            ``False``, with the prompt-parsed ReAct loop.  The default was
            flipped by D-049 because the ReAct arm measured **0/10 write tools
            issued** live, not because the native arm cleared its 4/5 bar (it
            measured 5/10); read the D-049 block above before changing it back.
        - ``seed``: optional sampling seed for the NATIVE arm's completions
            (``None`` sends no seed key). Live-probed as honored by Ollama on
            `qwen3.5:4b` (D-008); ignored by ReAct and injected builders.
        - ``agent_builder``: override the agent construction entirely (tests,
            live spikes, custom backends).
        - ``observer``: called with one observation ``dict`` per dispatch.  The
            hook exists so a live run can measure the
            ``parse_role_output`` failure distribution and the top-level-object
            count without changing behaviour.
        - ``sleep``: injected into the retry backoff so tests run instantly.
        - Returns a ``Callable[[RoleRequest], AgentResult]`` whose
            ``final_context`` carries ONLY the dispatching state's writable keys,
            already exact-type filtered.
        - Raises nothing itself; an exception from the agent propagates so the
            driver can record it (and so ``HarnessReentrancyError`` is never
            swallowed).

    Example::

        ws = Workspace("/tmp/scratch")
        agent = HarnessAgent(worker_factory=build_default_worker_factory(ws))
        agent.run(
            "add a retry to the uploader",
            initial_context={
                ContextKeys.PLAN_DIR: "plans/plan-2026-07-21T125237-191b2eb2",
                ContextKeys.WORKSPACE_ROOT: "/tmp/scratch",
            },
        )
    """
    build_agent = agent_builder or _default_agent_builder(
        native_function_calling=native_function_calling, seed=seed
    )

    def _observe(record: dict[str, Any]) -> None:
        logger.info(
            f"role dispatch {record['role']}/{record['state']}: "
            f"success={record['success']} reason={record['failure_reason']} "
            f"objects={record['top_level_objects']}"
        )
        if record["top_level_objects"] > 1:
            logger.warning(
                f"{record['role']} returned {record['top_level_objects']} "
                "top-level JSON objects; first-wins extraction may have taken "
                "a draft (see decisions.md D-031)"
            )
        if observer is not None:
            observer(record)

    def worker(request: RoleRequest) -> AgentResult:
        spec = get_role_spec(request.state)
        registry = build_workspace_tools(workspace, allowed=spec.tool_scope)
        memory: PlanMemory | None = None
        if request.plan_dir is None:
            logger.warning(
                f"{spec.role} dispatched without a plan directory; it holds no "
                "plan-file tools and can write no protocol artifact. Pass "
                f"initial_context={{'{ContextKeys.PLAN_DIR}': ...}} to run()."
            )
        else:
            # One PlanMemory per dispatch: it is scoped to exactly this role,
            # so the ownership refusal is a property of the tool the role
            # holds, not a check the role could route around.  It is kept
            # because the post-dispatch verification below re-reads the plan
            # directory through the same confined, role-scoped object the tools
            # wrote through -- not through a second raw filesystem path.
            memory = PlanMemory(request.plan_dir, role=spec.role)
            build_plan_tools(
                memory,
                allowed=spec.plan_tool_scope,
                registry=registry,
            )
        config = AgentConfig(
            model=model,
            max_iterations=spec.max_iterations,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
            output_schema=spec.output_schema,
        )
        agent = build_agent(spec, registry, config)
        prompt = _address(agent, request, spec)
        coverage = _coverage_line(memory, spec)
        if coverage:
            # Appended to the TASK half deliberately: what is already covered is
            # exactly what differs between two dispatches of the same role, so
            # it belongs where the goal and the position are, not in the
            # standing policy `_address` may have just installed (D-021).
            prompt = f"{prompt}\n\n{coverage}"

        started = time.monotonic()
        try:
            result = retry(
                lambda: agent.run(prompt),
                attempts=retry_attempts,
                sleep=sleep,
                description=f"{spec.role} dispatch",
            )
        except Exception as exc:
            _observe(
                {
                    "role": spec.role,
                    "state": spec.state,
                    "success": False,
                    "failure_reason": f"exception:{type(exc).__name__}",
                    "missing_keys": (),
                    "top_level_objects": 0,
                    "agent_success": False,
                    "answer_chars": 0,
                    "elapsed_s": time.monotonic() - started,
                    # Same keys as the success path: an observer that reads the
                    # verification fields must not have to branch on which
                    # record it got.
                    "write_evidence": 0,
                    "write_required": False,
                    "claimed_findings_count": None,
                    "derived_findings_count": None,
                }
            )
            raise

        elapsed = time.monotonic() - started
        answer = result.answer or ""
        output = parse_role_output(
            _payload_from(result), expected_keys=spec.expected_keys
        )
        accepted = coerce_worker_output(
            output.payload, spec.writable_keys, where=spec.role
        )

        # DECISION plan-2026-07-21T191807-bf7ffe24/D-015
        # A gate key listed in `tools.DISK_DERIVED_COUNTS` is REMOVED from the
        # worker's payload and replaced by a count of the files that really
        # exist -- through `derive_disk_counts`, which is the ONE derivation the
        # write tools also report back to the model (D-027) and the one the
        # DRIVER re-derives for itself after every dispatch (D-032); a second
        # count here would be a gate and a progress report that can disagree.
        # Do NOT "optimise" this into "trust the worker unless the
        # filesystem disagrees", and do NOT skip the pop when there is no plan
        # directory: review C1 reproduced the fail-open with the repo's own
        # objects -- an EXPLORE dispatch that made one read call, answered in
        # prose and wrote ZERO bytes returned `findings_count: 3`, which
        # SATISFIED the EXPLORE->PLAN hard gate. The worker's integer is kept
        # only as `claimed_findings_count` in the observation, so a fabricated
        # claim is COUNTABLE rather than merely blocked. Review C3 is the other
        # half: the explorer is forbidden to touch the `findings.md` index the
        # count used to be defined against, so a rule-compliant role could
        # never report it truthfully at all. See decisions.md D-015.
        for key in DISK_DERIVED_COUNTS:
            if key in spec.writable_keys:
                accepted.pop(key, None)
        derived: dict[str, int] = (
            derive_disk_counts(memory, spec.writable_keys) if memory is not None else {}
        )
        accepted.update(derived)

        # DECISION plan-2026-07-21T191807-bf7ffe24/D-016
        # A role that HOLDS a write tool must show a verified write, or its
        # reply is not a success. This is mechanical on purpose and must not be
        # replaced by (or reduced back to) prompt wording: the instruction
        # "Never state that you wrote a file unless a write tool reported
        # success" is already in `_finish_line` (D-013), it was in the prompt on
        # all 5 runs of the RCA bench that claimed "implemented retry-with-
        # backoff in uploader.py" having called `write_file` 0/5 times, and it
        # had already been strengthened once. A third strengthening would be the
        # third strike on the same mechanism.
        # Two properties are load-bearing:
        #   1. The trigger is `_holds_write_tool`, the SAME predicate that puts
        #      the write obligation in the prompt. A role is held to exactly
        #      what it was asked for -- REFLECT's verifier owns no artifact, so
        #      it is never asked and never checked.
        #   2. Evidence is BYTES, not the tool name. A `write_plan_file` call
        #      the ownership layer refused appears in the trace and leaves
        #      nothing on disk, so the name alone would re-open the same hole
        #      one layer down.
        # Do NOT parse the answer text for write claims instead: prose is the
        # thing that lies here. See decisions.md D-016.
        write_required = _holds_write_tool(held_tools(request, spec))
        verified = _verified_writes(result, workspace=workspace, memory=memory)
        unverified_claim = write_required and not verified
        success = output.success and not unverified_claim
        failure_reason = output.failure_reason
        if unverified_claim:
            failure_reason = failure_reason or "unverified-write"
            logger.warning(
                f"{spec.role} reported a result but no write tool left bytes on "
                f"disk, though it held {', '.join(WRITE_TOOLS + PLAN_WRITE_TOOLS)}"
                "-class tools. Recording the dispatch as FAILED: a completion "
                "claim with no verified write is not evidence of work."
            )

        claimed = output.payload.get(ContextKeys.FINDINGS_COUNT)
        _observe(
            {
                "role": spec.role,
                "state": spec.state,
                "success": success,
                "failure_reason": failure_reason,
                "missing_keys": output.missing_keys,
                "top_level_objects": count_top_level_json_objects(answer),
                "agent_success": result.success,
                "answer_chars": len(answer),
                "elapsed_s": elapsed,
                "write_evidence": len(verified),
                "write_required": write_required,
                "claimed_findings_count": claimed,
                "derived_findings_count": derived.get(ContextKeys.FINDINGS_COUNT),
            }
        )

        return AgentResult(
            answer=answer,
            success=success,
            trace=result.trace,
            # ONLY the state's writable keys, already exact-type filtered.
            # Anything else the agent left in its own final_context -- task
            # text, tool observations, iteration counters -- is discarded here
            # rather than at the driver, so a worker cannot smuggle a gate key
            # from a state that does not own it.
            final_context=accepted,
            structured_output=_reconcile_structured(result.structured_output, derived),
        )

    return worker
