"""The equality corpus for the state-kind registry (plan d454fd62, step 45, D-004, D-015).

Row 0.14 moves every kind-specific rule (completion, synthesize) behind one
registry. The refactor must change no behaviour, so this module records what the
PARENT code answers for a fixed corpus of definitions and three scripted turns,
and ``test_kind_equality.py`` compares every later state of the code with that
record (``fixtures/kind_equality_baseline.json``).

Per FSM the record holds: whether it loads (and the loader's messages when it
does not), the kind of every state, ``fsm_id``, the sha256 of the full
``model_dump()`` and of ``model_dump(exclude_defaults=True)`` (the ``fsm_id``
input), the ``FSMValidator`` errors, warnings and infos, the findings of
``analyze_data_flow``, ``analyze_liveness`` and ``analyze_dryrun``, the
``edge_witnesses``, ``handler_only_names`` and ``declared_available``. Per
scenario it holds the bytes of every request an interface double received,
the visited states, the replies and the final context.

Interface contract (the only callers are ``test_kind_equality.py`` and the
regenerate command below):

- ``corpus()``: ``{case name: (definition dict, options)}``, built fresh on every
  call (mutating one never reaches another); ``options`` may hold ``inputs`` and
  ``return_keys`` for the analyses. Case names are stable ids, sorted.
- ``describe(definition, options)``: the full, unpacked record of one case.
  Never raises on a definition that fails to load: the failure is recorded.
- ``scenarios()``: ``{scenario name: record}``, each run against doubles from
  ``fsm_llm.testing`` (no network, no model, no clock, no randomness).
- ``build_baseline()``: the whole packed record; ``pack`` keeps a value inline
  when its JSON is at most ``INLINE_LIMIT`` characters and else stores its
  count and sha256 (``python -m tests.test_fsm_llm.kind_corpus --show NAME`` prints
  the unpacked record of one case).
- Failure mode: every group imports with the base install plus the ``agents``
  and ``reasoning`` subpackages (``.venv`` has them); a missing import raises
  ``ImportError`` and the test fails, it never skips a group.

Regenerate (never implicit, never in CI): run, from the repository root,
``.venv/bin/python -m tests.test_fsm_llm.kind_corpus --write`` on the PARENT of
a kind refactor, review the diff and commit it with the reason. The file is
written with ``\\n`` line ends, UTF-8, sorted keys, no timestamp and no path
outside the repository.

The corpus reads ``examples/`` and ``scripts/synthesis_bench.py`` and never
changes them.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "kind_equality_baseline.json"
)

#: A packed value whose JSON is at most this long stays in the baseline as is.
INLINE_LIMIT = 400

_CompletionTools: list[dict[str, Any]] = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a word up in the dictionary.",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        },
    },
]
_Format: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}


# --------------------------------------------------------------
# Small definition helpers (the corpus owns its fixtures; it imports no test module)
# --------------------------------------------------------------


def _state(sid: str, **fields: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "id": sid,
        "description": f"The {sid} state",
        "purpose": f"Do the {sid} work",
    }
    state.update(fields)
    return state


def _go(target: str, **fields: Any) -> dict[str, Any]:
    return {"target_state": target, "description": f"Go to {target}", **fields}


def _when(key: str, logic: Any, description: str = "gate") -> list[dict[str, Any]]:
    return [
        {
            "description": description,
            "requires_context_keys": [key],
            "logic": logic,
        }
    ]


def _fsm(
    name: str, initial: str, *states: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"The {name} fixture",
        "initial_state": initial,
        "states": {s["id"]: s for s in states},
        **extra,
    }


def _kind_is(kind: str, result_key: str = "completion_result") -> list[dict[str, Any]]:
    return _when(
        result_key,
        {"==": [{"var": f"{result_key}.kind"}, kind]},
        f"The model turn was {kind}",
    )


def _tool_fsm(
    *, result_key: str = "completion_result", **completion: Any
) -> dict[str, Any]:
    """intake -> call_model (completion) -> run_tools -> call_model ... -> done."""
    config: dict[str, Any] = {
        "tools": _CompletionTools,
        "instructions": "You answer arithmetic and word questions with the tools you have.",
    }
    config.update(completion)
    if result_key != "completion_result":
        config["result_key"] = result_key
    return _fsm(
        "ToolTurns",
        "intake",
        _state("intake", transitions=[_go("call_model")]),
        _state(
            "call_model",
            completion=config,
            transitions=[
                _go("run_tools", priority=10, conditions=_kind_is("calls", result_key)),
                _go("done", priority=20, conditions=_kind_is("final", result_key)),
                _go(
                    "failed", priority=30, conditions=_kind_is("malformed", result_key)
                ),
            ],
        ),
        _state("run_tools", transitions=[_go("call_model")]),
        _state("done", response_instructions="Tell the user the answer."),
        _state("failed"),
    )


def _synth_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "goal": "Book a table for two",
        "inputs": ["topic"],
        "return_keys": ["answer"],
        "on_failure": "apology",
    }
    config.update(overrides)
    return config


def _synth_edges(
    result_key: str = "synthesis_result", *, also_reads: str | None = None
) -> list[dict[str, Any]]:
    """The two edges a synthesize state needs; the ok edge may also read *also_reads*."""
    status = {"var": f"{result_key}.status"}
    ok = _when(result_key, {"==": [status, "ok"]}, "status ok")
    if also_reads:
        ok[0]["requires_context_keys"].append(also_reads)
    return [
        _go("done", priority=1, conditions=ok),
        _go(
            "apology",
            priority=2,
            conditions=_when(
                result_key, {"in": [status, ["failed", "aborted"]]}, "status failed"
            ),
        ),
    ]


def _synth_fsm(
    config: dict[str, Any] | None = None,
    *,
    result_key: str = "synthesis_result",
    silent: bool = True,
    also_reads: str | None = None,
) -> dict[str, Any]:
    """A host: welcome, the synthesize state (silent like the bench host, or one that
    leaves ``response_instructions`` out), then ``done`` or ``apology``."""
    config = _synth_config() if config is None else config
    if result_key != "synthesis_result":
        config = {**config, "result_key": result_key}
    return _fsm(
        "SynthHost",
        "welcome",
        _state(
            "welcome",
            response_instructions="Greet the user.",
            transitions=[_go("solve")],
        ),
        _state(
            "solve",
            synthesize=config,
            transitions=_synth_edges(result_key, also_reads=also_reads),
            **({"response_instructions": ""} if silent else {}),
        ),
        _state("done", response_instructions="Tell the user it is done."),
        _state("apology", response_instructions="Apologise briefly."),
    )


def _plain_fsm() -> dict[str, Any]:
    """Field, bulk and classification extraction, two transitions, no kind field."""
    return _fsm(
        "PlainKinds",
        "greet",
        _state(
            "greet",
            response_instructions="Greet the user by name.",
            field_extractions=[
                {
                    "field_name": "name",
                    "field_type": "str",
                    "extraction_instructions": "The user's first name",
                }
            ],
            transitions=[
                _go(
                    "choose",
                    conditions=_when(
                        "name", {"!=": [{"var": "name"}, ""]}, "name known"
                    ),
                )
            ],
        ),
        _state(
            "choose",
            extraction_instructions="Extract what the user wants to buy as item.",
            required_context_keys=["item"],
            response_instructions="Ask what the user wants.",
            classification_extractions=[
                {
                    "field_name": "user_intent",
                    "intents": [
                        {"name": "buy", "description": "The user wants to purchase"},
                        {"name": "browse", "description": "The user is looking around"},
                    ],
                    "fallback_intent": "browse",
                    "confidence_threshold": 0.5,
                }
            ],
            transitions=[
                _go(
                    "confirm",
                    priority=1,
                    conditions=_when(
                        "user_intent", {"==": [{"var": "user_intent"}, "buy"]}, "buying"
                    ),
                ),
                _go("choose", priority=2),
            ],
        ),
        _state("confirm", response_instructions="Confirm the order."),
        persona="A friendly shop assistant",
    )


# --------------------------------------------------------------
# The corpus
# --------------------------------------------------------------


def _example_cases() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Every FSM JSON under ``examples/`` and the evaluation sample (read only)."""
    paths = sorted(
        [
            *(ROOT / "examples").rglob("*.json"),
            *(ROOT / "evaluation").rglob("*fsm*.json"),
        ],
        key=lambda p: p.relative_to(ROOT).as_posix(),
    )
    cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "initial_state" in data:
            cases[f"file/{path.relative_to(ROOT).as_posix()}"] = (data, {})
    return cases


class _Report(BaseModel):
    """The ``output_schema`` of the react builder variants."""

    answer: str
    confidence: float


def _registry() -> Any:
    from fsm_llm.agents import ToolRegistry
    from fsm_llm.agents.definitions import ToolDefinition

    def _weather(city: str) -> str:
        return f"sunny in {city}"

    def _note(text: str) -> str:
        return f"saved {len(text)}"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="weather",
            description="Get the weather for a city.",
            parameter_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City"}},
                "required": ["city"],
            },
            execute_fn=_weather,
        )
    )
    registry.register(
        ToolDefinition(
            name="write_note",
            description="Save a note.",
            parameter_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Note"}},
                "required": ["text"],
            },
            execute_fn=_note,
        )
    )
    return registry


def _builder_cases() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """The agent pattern builders, ``native_fc`` and the meta builder."""
    from fsm_llm.agents import fsm_definitions as fd
    from fsm_llm.agents.definitions import ChainStep

    registry = _registry()
    schemas = registry.get_json_schemas()
    chain = [
        ChainStep(
            step_id="draft",
            name="Draft",
            extraction_instructions="Extract the topic.",
            response_instructions="Draft a paragraph.",
        ),
        ChainStep(
            step_id="polish",
            name="Polish",
            extraction_instructions="Extract the draft.",
            response_instructions="Polish the draft.",
        ),
    ]
    built: dict[str, dict[str, Any]] = {
        "agents/orchestrator": fd.build_orchestrator_fsm("Plan a trip", ["city"]),
        "agents/adapt": fd.build_adapt_fsm("Solve it", 3, ["city"]),
        "agents/reflexion": fd.build_reflexion_fsm(registry, "Fix the bug"),
        "agents/reflexion-approval": fd.build_reflexion_fsm(
            registry, "Fix the bug", include_approval_state=True
        ),
        "agents/plan-execute": fd.build_plan_execute_fsm(registry, "Plan and run"),
        "agents/react": fd.build_react_fsm(registry, "Find the weather"),
        "agents/react-approval": fd.build_react_fsm(
            registry, "Find the weather", include_approval_state=True
        ),
        "agents/react-classification": fd.build_react_fsm(
            registry, "Find the weather", use_classification=True
        ),
        "agents/react-schema": fd.build_react_fsm(
            registry, "Find the weather", output_schema=_Report, context_keys=["city"]
        ),
        "agents/parallel-react": fd.build_parallel_react_fsm(
            registry, "Compare cities"
        ),
        "agents/prompt-chain": fd.build_prompt_chain_fsm(chain, "Write it"),
        "agents/self-consistency": fd.build_self_consistency_fsm("Vote on it"),
        "agents/debate": fd.build_debate_fsm("Is it true?", "Pro", "Con", "Judge", 2),
        "agents/rewoo": fd.build_rewoo_fsm(registry, "Plan then solve", ["city"]),
        "agents/evalopt": fd.build_evalopt_fsm("Improve it"),
        "agents/maker-checker": fd.build_maker_checker_fsm(
            "Write the text.", "Check the text.", "Produce a text", 2
        ),
        "agents/meta-builder": fd.build_meta_builder_fsm(),
        "agents/native-fc": fd.build_native_fc_fsm(
            schemas, instructions="Use the tools."
        ),
        "agents/native-fc-final": fd.build_native_fc_fsm(
            schemas, instructions="Use the tools.", force_final_tool="write_note"
        ),
        "agents/native-fc-repair": fd.build_native_fc_fsm(
            schemas, instructions="Use the tools.", repair=True
        ),
        "agents/native-fc-final-repair": fd.build_native_fc_fsm(
            schemas,
            instructions="Use the tools.",
            force_final_tool="write_note",
            repair=True,
        ),
    }
    return {f"builder/{name}": (fsm, {}) for name, fsm in built.items()}


def _reasoning_cases() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS

    return {f"reasoning/{name}": (fsm, {}) for name, fsm in ALL_REASONING_FSMS.items()}


@contextlib.contextmanager
def _bench() -> Iterator[Any]:
    """``scripts/synthesis_bench.py`` loaded by path, leaving ``sys.modules`` as it was."""
    added = {"kind_corpus_synthesis_bench", "bench_common"} - set(sys.modules)
    path = ROOT / "scripts" / "synthesis_bench.py"
    spec = importlib.util.spec_from_file_location("kind_corpus_synthesis_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name in added:  # only what this block put there: a module the bench
            sys.modules.pop(name, None)  # imports for the first time stays imported


def _bench_cases() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """The 16 ``hand_fsm`` and 16 ``host_definition`` of the synthesis bench."""
    cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    with _bench() as bench:
        for task in bench.TASKS:
            cases[f"bench/hand/{task.id}"] = (
                bench.hand_fsm(task),
                {"return_keys": sorted(task.gold)},
            )
            cases[f"bench/host/{task.id}"] = (bench.host_definition(task), {})
    return cases


def _fixture_cases() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Hand-written definitions with completion and synthesize states, valid and refused."""
    cases: dict[str, dict[str, Any]] = {}
    cases["completion-tools"] = _tool_fsm()
    structured = _tool_fsm()
    structured["states"]["call_model"]["completion"] = {
        "response_format_key": "_answer_format",
        "instructions": "Answer in JSON.",
    }
    cases["completion-structured"] = structured
    cases["completion-custom"] = _tool_fsm(
        result_key="answer_turn",
        tool_choice="required",
        messages_key="_my_messages",
    )
    cases["completion-named-choice"] = _tool_fsm(
        tool_choice="lookup", instructions=None
    )
    cases["synthesize-host"] = _synth_fsm()
    cases["synthesize-host-speaking"] = _synth_fsm(silent=False)
    cases["synthesize-edge-reads-other-key"] = _synth_fsm(also_reads="note")
    reads_note = _tool_fsm()
    reads_note["states"]["call_model"]["transitions"][1]["conditions"][0][
        "requires_context_keys"
    ].append("note")
    cases["completion-edge-reads-other-key"] = reads_note
    cases["synthesize-custom"] = _synth_fsm(
        _synth_config(inputs=[], return_keys=["a", "b"], min_lifecycle="trusted"),
        result_key="proc_result",
    )
    both = _synth_fsm()
    both["states"]["done"] = _state(
        "done",
        completion={"tools": _CompletionTools, "instructions": "Say it."},
        transitions=[_go("end", conditions=_kind_is("final"))],
    )
    both["states"]["end"] = _state("end", response_instructions="Bye.")
    cases["synthesize-then-completion"] = both
    cases["plain"] = _plain_fsm()

    # Refused at load.
    for label, field in (
        ("extraction-instructions", {"extraction_instructions": "Extract x."}),
        (
            "field-extractions",
            {
                "field_extractions": [
                    {
                        "field_name": "x",
                        "field_type": "str",
                        "extraction_instructions": "x",
                    }
                ]
            },
        ),
        ("required-context-keys", {"required_context_keys": ["x"]}),
        (
            "classification",
            {
                "classification_extractions": [
                    {
                        "field_name": "c",
                        "intents": [
                            {"name": "a", "description": "A"},
                            {"name": "b", "description": "B"},
                        ],
                        "fallback_intent": "a",
                    }
                ]
            },
        ),
    ):
        bad = _tool_fsm()
        bad["states"]["call_model"].update(field)
        cases[f"refused/completion-with-{label}"] = bad
        host = _synth_fsm()
        host["states"]["solve"].update(field)
        cases[f"refused/synthesize-with-{label}"] = host
    no_edge = _tool_fsm()
    no_edge["states"]["call_model"]["transitions"] = []
    cases["refused/completion-without-transition"] = no_edge
    both_fields = _tool_fsm(response_format_key="_f")
    cases["refused/completion-tools-and-format"] = both_fields
    neither = _tool_fsm()
    del neither["states"]["call_model"]["completion"]["tools"]
    cases["refused/completion-neither-tools-nor-format"] = neither
    unpaired = _synth_fsm()
    unpaired["states"]["solve"]["completion"] = {"tools": _CompletionTools}
    cases["refused/synthesize-and-completion"] = unpaired
    off_target = _synth_fsm(_synth_config(on_failure="nowhere"))
    cases["refused/synthesize-on-failure-not-a-target"] = off_target
    no_transitions = _synth_fsm()
    no_transitions["states"]["solve"]["transitions"] = []
    cases["refused/synthesize-without-transition"] = no_transitions
    ignores = _synth_fsm()
    ignores["states"]["solve"]["transitions"] = [
        _go(
            "done",
            priority=1,
            conditions=_when("topic", {"==": [{"var": "topic"}, "x"]}),
        ),
        _go("apology", priority=2),
    ]
    cases["refused/synthesize-edge-ignores-the-result"] = ignores
    bare = _synth_fsm()
    bare["states"]["solve"]["transitions"][1]["conditions"] = _when(
        "synthesis_result",
        {"==": [{"var": "synthesis_result.status"}, "failed"]},
        "bare failed",
    )
    cases["refused/synthesize-bare-failed-edge"] = bare
    collide = _synth_fsm(result_key="shared_result")
    collide["states"]["done"] = _state(
        "done",
        completion={
            "tools": _CompletionTools,
            "instructions": "Say it.",
            "result_key": "shared_result",
        },
        transitions=[_go("end")],
    )
    collide["states"]["end"] = _state("end")
    cases["shared-result-key"] = collide
    reserved = _tool_fsm(result_key="_hidden")
    cases["refused/completion-private-result-key"] = reserved
    data_key = _tool_fsm()
    data_key["states"]["intake"]["field_extractions"] = [
        {
            "field_name": "completion_result",
            "field_type": "str",
            "extraction_instructions": "Never extracted: the result key",
        }
    ]
    cases["completion-result-key-as-a-field"] = data_key

    # Valid, with findings.
    uncovered = _tool_fsm()
    uncovered["states"]["call_model"]["transitions"] = [
        _go("run_tools", priority=10, conditions=_kind_is("calls")),
        _go("done", priority=20, conditions=_kind_is("final")),
    ]
    uncovered["states"]["run_tools"]["transitions"] = [
        _go("call_model", priority=1),
        _go("failed", priority=5),
    ]
    cases["lint/completion-uncovered-kind"] = uncovered
    bulk = _tool_fsm()
    bulk["states"]["intake"]["extraction_instructions"] = "Extract anything useful."
    cases["lint/completion-then-bulk"] = bulk
    reads = _fsm(
        "Reads",
        "a",
        _state(
            "a",
            transitions=[
                _go("b", conditions=_when("ghost", {"==": [{"var": "ghost"}, 1]}))
            ],
        ),
        _state("b"),
    )
    cases["flow/undeclared-read"] = reads
    trap = _fsm(
        "Trap",
        "a",
        _state(
            "a",
            transitions=[
                _go("b"),
                _go(
                    "end", priority=5, conditions=_when("k", {"==": [{"var": "k"}, 1]})
                ),
            ],
        ),
        _state("b", transitions=[_go("c")]),
        _state("c", transitions=[_go("b")]),
        _state("end"),
    )
    cases["flow/trap"] = trap
    dead = _fsm(
        "Dead",
        "a",
        _state(
            "a",
            field_extractions=[
                {
                    "field_name": "flag",
                    "field_type": "bool",
                    "extraction_instructions": "yes?",
                }
            ],
            transitions=[
                _go("b", conditions=_when("flag", {"==": [{"var": "flag"}, "yes"]})),
                _go("c", priority=5),
            ],
        ),
        _state("b"),
        _state("c"),
    )
    cases["flow/dead-edge"] = dead
    return {f"fixture/{name}": (fsm, {}) for name, fsm in cases.items()}


def corpus() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Every case, name -> (definition dict, analysis options), names sorted."""
    merged: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for group in (
        _example_cases(),
        _builder_cases(),
        _reasoning_cases(),
        _bench_cases(),
        _fixture_cases(),
    ):
        overlap = merged.keys() & group.keys()
        assert not overlap, f"duplicate corpus case names: {sorted(overlap)}"
        merged.update(group)
    return {name: merged[name] for name in sorted(merged)}


# --------------------------------------------------------------
# Records
# --------------------------------------------------------------


def _sha(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    """One JSON text of *value*: sorted keys, ASCII, compact, ``repr`` for the odd type."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=repr
    )


def _plain(value: Any) -> Any:
    """*value* through JSON (tuples become lists, keys strings), the shape the baseline holds."""
    return json.loads(_canonical(value))


def _findings(rows: Any) -> list[list[str]]:
    return [[f.analysis, f.code, f.state, f.message] for f in rows]


_TERMINAL_LIST = re.compile(r"^(.*[Tt]erminal states: )(.*)$")


def _stable_info(lines: list[str]) -> list[str]:
    """The validator's info lines in an order that does not move with ``PYTHONHASHSEED``.

    The validator joins sets ("Terminal states: b, c") and walks them, so both the
    order of the lines and the order inside a joined list change from run to run:
    the joined lists are sorted, then the lines are.
    """
    out = []
    for line in lines:
        match = _TERMINAL_LIST.match(line)
        if match:
            line = match[1] + ", ".join(sorted(match[2].split(", ")))
        out.append(line)
    return sorted(out)


def _recorded(compute: Callable[[], Any]) -> Any:
    """``compute()``, or ``{"raised": ...}`` when it raises: a behaviour change that
    makes an analysis fail then shows as a difference, not as a crashed test."""
    try:
        return compute()
    except Exception as error:
        return {"raised": f"{type(error).__name__}: {error}"}


def _load_errors(error: Exception) -> list[str]:
    if isinstance(error, ValidationError):
        return [str(item["msg"]) for item in error.errors()]
    return [f"{type(error).__name__}: {error}"]


def describe(
    definition: dict[str, Any], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The unpacked record of one case (see the module docstring)."""
    from fsm_llm import API
    from fsm_llm.core.definitions import FSMDefinition, handler_only_names
    from fsm_llm.core.verify import FSMValidator
    from fsm_llm.core.verify.flow import (
        analyze_data_flow,
        analyze_liveness,
        declared_available,
    )
    from fsm_llm.synthesis.dryrun import analyze_dryrun, edge_witnesses

    options = options or {}
    inputs = tuple(options.get("inputs", ()))
    return_keys = tuple(options.get("return_keys", ()))
    raw = copy.deepcopy(definition)
    record: dict[str, Any] = {}

    result = FSMValidator(copy.deepcopy(raw)).validate()
    record["validator_valid"] = result.is_valid
    record["validator_errors"] = result.errors
    record["validator_warnings"] = result.warnings
    record["validator_info"] = _stable_info(result.info)
    try:
        loaded = FSMDefinition(**copy.deepcopy(raw))
    except Exception as error:  # the baseline records every refusal as it is
        record["loads"] = False
        record["load_errors"] = _load_errors(error)
        return record
    record["loads"] = True
    record["state_count"] = len(loaded.states)
    # The kind the parent code gives a state; a state of the default kind is left out.
    record["state_kinds"] = {
        sid: "completion" if state.completion is not None else "synthesize"
        for sid, state in loaded.states.items()
        if state.completion is not None or state.synthesize is not None
    }
    dump = loaded.model_dump(exclude_defaults=True)
    record["fsm_id"] = API.process_fsm_definition(copy.deepcopy(raw))[1]
    record["dump_sha256"] = _sha(_canonical(dump))
    record["full_dump_sha256"] = _sha(_canonical(loaded.model_dump()))
    record["handler_only_names"] = _recorded(lambda: sorted(handler_only_names(loaded)))
    record["data_flow"] = _recorded(
        lambda: _findings(
            analyze_data_flow(loaded, inputs=inputs, return_keys=return_keys)
        )
    )
    if return_keys:
        record["data_flow_unguaranteed"] = _recorded(
            lambda: _findings(
                analyze_data_flow(
                    loaded,
                    inputs=inputs,
                    return_keys=return_keys,
                    require_guaranteed_return_keys=False,
                )
            )
        )
    record["liveness"] = _recorded(
        lambda: _findings(analyze_liveness(loaded, inputs=inputs))
    )
    record["dryrun"] = _recorded(
        lambda: _findings(analyze_dryrun(loaded, inputs=inputs))
    )
    record["edge_witnesses"] = _recorded(
        lambda: {
            sid: {str(i): witness for i, witness in edges.items()}
            for sid, edges in edge_witnesses(loaded, inputs=inputs).items()
        }
    )
    record["declared_available"] = _recorded(
        lambda: {
            sid: sorted(keys)
            for sid, keys in declared_available(loaded, inputs).items()
        }
    )
    return _plain(record)


def pack(value: Any) -> Any:
    """*value* inline when short, else ``{count, sha256}`` of its canonical JSON."""
    text = _canonical(value)
    if len(text) <= INLINE_LIMIT:
        return _plain(value)
    count = len(value) if hasattr(value, "__len__") else None
    return {"count": count, "sha256": _sha(text)}


def pack_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: pack(value) for key, value in record.items()}


# --------------------------------------------------------------
# Scenarios: three turns against doubles, request bytes recorded
# --------------------------------------------------------------


def _request_dump(method: str, request: Any) -> dict[str, Any]:
    body = _canonical(
        request.model_dump(mode="json") if hasattr(request, "model_dump") else request
    )
    return {"method": method, "bytes": len(body.encode("utf-8")), "sha256": _sha(body)}


def _requests(llm: Any, *, inline: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows = []
    for method, request in llm.call_history:
        row = _request_dump(method, request)
        if method in inline:
            row["request"] = _plain(request.model_dump(mode="json"))
        rows.append(row)
    return rows


def _context(api: Any, conversation_id: str, keys: tuple[str, ...]) -> dict[str, Any]:
    data = api.get_data(conversation_id)
    return {key: data.get(key) for key in keys if key in data}


def _scenario_plain() -> dict[str, Any]:
    from fsm_llm import API
    from fsm_llm.testing import ScriptedLLM, final_reply

    classify = final_reply(
        json.dumps(
            {
                "reasoning": "wants a lamp",
                "intent": "buy",
                "confidence": 0.9,
                "entities": {},
            }
        )
    )
    llm = ScriptedLLM(
        [classify, classify],
        extraction_data={"name": "Ada", "item": "lamp"},
        responses={
            "greet": "Hello! Who are you?",
            "choose": "Nice to meet you, Ada. What would you like?",
            "confirm": "One lamp, coming up.",
        },
    )
    api = API.from_definition(_plain_fsm(), llm_interface=llm)
    conversation_id, opening = api.start_conversation()
    replies = [opening]
    states = [api.get_current_state(conversation_id)]
    for message in ("I am Ada", "I want a lamp"):
        replies.append(api.converse(message, conversation_id))
        states.append(api.get_current_state(conversation_id))
    stream_llm = ScriptedLLM(
        [classify],
        extraction_data={"name": "Ada", "item": "lamp"},
        responses={"greet": "Hello!", "choose": "What now?", "confirm": "Done."},
        stream_chunks=("Hel", "lo"),
    )
    stream_api = API.from_definition(_plain_fsm(), llm_interface=stream_llm)
    stream_id, _ = stream_api.start_conversation()
    streamed = "".join(stream_api.converse_stream("I am Ada", stream_id))
    llm.assert_exhausted()
    return {
        "replies": replies,
        "states": states,
        "context": _context(api, conversation_id, ("name", "item", "user_intent")),
        "requests": _requests(llm),
        "stream": {
            "text": streamed,
            "state": stream_api.get_current_state(stream_id),
            "requests": _requests(stream_llm),
        },
    }


def _tool_runner() -> Any:
    from fsm_llm import tool_exchange
    from fsm_llm.handlers import create_handler

    functions = {"add": lambda a, b: a + b, "lookup": lambda word: f"{word}: a word"}

    def run(context: dict[str, Any]) -> dict[str, Any]:
        result = context["completion_result"]
        outputs = [str(functions[c["name"]](**c["arguments"])) for c in result["calls"]]
        transcript = list(context.get("_completion_messages") or [])
        transcript.extend(tool_exchange(result["text"], result["calls"], outputs))
        return {"_completion_messages": transcript, "completion_result": None}

    return create_handler("run_tools").on_state_entry("run_tools").do(run)


def _scenario_completion() -> dict[str, Any]:
    from fsm_llm import API
    from fsm_llm.testing import ScriptedLLM, final_reply, tool_call_reply

    question = [{"role": "user", "content": "What is 2 + 3?"}]
    llm = ScriptedLLM(
        [tool_call_reply(("add", {"a": 2, "b": 3})), final_reply("5")],
        default_response="The answer is 5.",
    )
    api = API.from_definition(_tool_fsm(), llm_interface=llm)
    api.register_handler(_tool_runner())
    conversation_id, opening = api.start_conversation(
        initial_context={"_completion_messages": question}
    )
    steps = [api.advance(conversation_id)]
    while not steps[-1].ended and len(steps) < 8:
        steps.append(api.advance(conversation_id))
    run = [[s.state_before, s.state_after, s.response, s.ended] for s in steps]

    # A turn with a user message: the message is not added to the transcript.
    talk_llm = ScriptedLLM([final_reply("hello")], default_response="Fine.")
    talk = API.from_definition(_tool_fsm(), llm_interface=talk_llm)
    talk_id, _ = talk.start_conversation(
        initial_context={"_completion_messages": question}
    )
    talk.advance(talk_id)
    talk_reply = talk.converse("please hurry", talk_id)

    # A structured turn.
    structured = _tool_fsm()
    structured["states"]["call_model"]["completion"] = {
        "response_format_key": "_answer_format",
        "instructions": "Answer in JSON.",
    }
    form_llm = ScriptedLLM([final_reply('{"answer": "5"}')], default_response="Fine.")
    form = API.from_definition(structured, llm_interface=form_llm)
    form_id, _ = form.start_conversation(
        initial_context={"_completion_messages": question, "_answer_format": _Format}
    )
    form.advance(form_id)
    form.advance(form_id)

    # The result key of a completion state is dropped from a pushed child.
    push_llm = ScriptedLLM([final_reply("5")], default_response="Fine.")
    push = API.from_definition(_tool_fsm(), llm_interface=push_llm)
    push_id, _ = push.start_conversation(
        initial_context={"_completion_messages": question}
    )
    push.advance(push_id)
    push.advance(push_id)
    child = _fsm("Child", "only", _state("only", response_instructions="Hi"))
    push.push_fsm(push_id, child, inherit_context=True)
    inherited = sorted(key for key in push.get_data(push_id) if not key.startswith("_"))

    llm.assert_exhausted()
    talk_llm.assert_exhausted()
    form_llm.assert_exhausted()
    return {
        "opening": opening,
        "run": run,
        "context": _context(
            api, conversation_id, ("completion_result", "_completion_messages")
        ),
        "requests": _requests(llm, inline=("complete",)),
        "converse": {
            "reply": talk_reply,
            "state": talk.get_current_state(talk_id),
            "requests": _requests(talk_llm, inline=("complete",)),
        },
        "structured": {
            "state": form.get_current_state(form_id),
            "context": _context(form, form_id, ("completion_result",)),
            "requests": _requests(form_llm, inline=("complete",)),
        },
        "pushed_child_keys": inherited,
    }


def _scenario_synthesize_host(*, silent: bool) -> dict[str, Any]:
    from fsm_llm import API
    from fsm_llm.testing import ScriptedLLM

    def host() -> tuple[Any, API]:
        llm = ScriptedLLM(
            [],
            default_response="ok",
            responses={"welcome": "Welcome!", "done": "All done.", "apology": "Sorry."},
        )
        host_fsm = _synth_fsm(silent=silent, also_reads=None if silent else "note")
        return llm, API.from_definition(host_fsm, llm_interface=llm)

    llm, api = host()
    conversation_id, opening = api.start_conversation()
    log: list[list[Any]] = [["start", opening, api.get_current_state(conversation_id)]]
    for label, message in (
        ("to-work", "hello"),
        ("bare-converse", "anything"),
        ("again", "more"),
    ):
        reply = api.converse(message, conversation_id)
        log.append([label, reply, api.get_current_state(conversation_id)])
    calls_while_blocked = len(llm.call_history)
    ok = {"status": "ok", "procedure_id": "p1", "version": 1, "reason": ""}
    api.update_context(
        conversation_id, {"synthesis_result": ok, **({} if silent else {"note": "n"})}
    )
    reply = api.converse("finish", conversation_id)
    log.append(["after-result", reply, api.get_current_state(conversation_id)])

    _, failed = host()
    failed_id, _ = failed.start_conversation()
    failed.converse("hello", failed_id)
    aborted = {"status": "aborted", "procedure_id": "", "version": 0, "reason": "no"}
    failed.update_context(failed_id, {"synthesis_result": aborted})
    failed.converse("finish", failed_id)
    llm.assert_exhausted()
    return {
        "log": log,
        "calls_while_blocked": calls_while_blocked,
        "requests": _requests(llm),
        "aborted_state": failed.get_current_state(failed_id),
        "result": _context(api, conversation_id, ("synthesis_result",)),
    }


def scenarios() -> dict[str, dict[str, Any]]:
    """The recorded turns: plain, completion, and the synthesize host (silent and speaking).

    A turn that raises is recorded as ``{"raised": ...}``, never propagated.
    """
    runs: dict[str, Callable[[], dict[str, Any]]] = {
        "completion": _scenario_completion,
        "plain": _scenario_plain,
        "synthesize-host": lambda: _scenario_synthesize_host(silent=True),
        "synthesize-host-speaking": lambda: _scenario_synthesize_host(silent=False),
    }
    return {name: _plain(_recorded(run)) for name, run in runs.items()}


# --------------------------------------------------------------
# The baseline
# --------------------------------------------------------------


def _schema_facts() -> dict[str, Any]:
    from fsm_llm.core.definitions import FSMDefinition, State

    return {
        "state_fields": sorted(State.model_fields),
        "state_schema_properties": sorted(State.model_json_schema()["properties"]),
        "fsm_schema_properties": sorted(
            FSMDefinition.model_json_schema()["properties"]
        ),
    }


def build_baseline() -> dict[str, Any]:
    """The whole packed record of the current code (what ``--write`` stores)."""
    cases = corpus()
    return {
        "schema": _schema_facts(),
        "cases": {
            name: pack_record(describe(definition, options))
            for name, (definition, options) in cases.items()
        },
        "scenarios": scenarios(),
    }


def dumps(baseline: dict[str, Any]) -> str:
    """The file text: one line per case, sorted keys, ASCII, one final ``\\n``.

    Valid JSON; the layout only keeps a diff to the cases that changed.
    """
    cases = ",\n".join(
        f"{json.dumps(name)}: {_canonical(record)}"
        for name, record in sorted(baseline["cases"].items())
    )
    scenarios = json.dumps(baseline["scenarios"], sort_keys=True, indent=1)
    schema = _canonical(baseline["schema"])
    return (
        f'{{\n"cases": {{\n{cases}\n}},\n"scenarios": {scenarios},\n'
        f'"schema": {schema}\n}}\n'
    )


def write_baseline(path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps(build_baseline()))


def read_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    if argv[:1] == ["--write"]:
        write_baseline()
        print(f"wrote {BASELINE_PATH.relative_to(ROOT).as_posix()}")
        return 0
    if len(argv) == 2 and argv[0] == "--show":
        definition, options = corpus()[argv[1]]
        print(json.dumps(describe(definition, options), sort_keys=True, indent=2))
        return 0
    print("usage: python -m tests.test_fsm_llm.kind_corpus (--write | --show CASE)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
