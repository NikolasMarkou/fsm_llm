"""The case table of every built-in FSM and its text fixtures (plan ace7a8c8, F1.2).

``CASES`` is the table of ``test_fsm_definitions_golden.py``, hoisted so the
executable-format work can import it: one argument set per agent builder, and
whether the result is a full FSM or a fragment. ``fsm_cases()`` adds every
reasoning FSM and keeps the full FSMs; each has a text fixture under
``tests/fixtures/executables/`` in INSERTION order (``indent=2``,
``ensure_ascii=True``, one final newline), so a failure shows a diff where the
hash goldens (``sort_keys=True``) cannot see state order. ``index.json`` holds
``sha256`` (the golden recipe), ``fsm_id`` and ``digest64`` per case. Excluded:
``SemanticToolRegistry`` retrieval (an embedding model, not reproducible).
``wrap(case)`` puts a case in its executable envelope (plan ace7a8c8, C7); ``--write`` also
writes the shipped JSON Schema of the envelope. Regenerate with
``python -m tests.executables_corpus --write`` and review the diff.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fsm_llm.agents import build_parallel_react_fsm
from fsm_llm.agents import fsm_definitions as fd
from fsm_llm.agents.constants import ContextKeys
from fsm_llm.agents.definitions import ChainStep
from fsm_llm.agents.fsm_definitions import _common, _loop, native_fc
from fsm_llm.core.definitions import FSMDefinition, definition_digest, fsm_id_of
from fsm_llm.reasoning.reasoning_modes import ALL_REASONING_FSMS
from fsm_llm.reasoning.utilities import get_reasoning_fsm
from fsm_llm.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "executables"
INDEX_PATH = FIXTURES / "index.json"


class _Out(BaseModel):
    answer: str
    score: int


def _tool(params):
    return "ok"


def _registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register_function(_tool, name=name, description=f"Tool {name}")
    return registry


_TOOLS = ("search", "calc")
_TASK = "Find the answer to the golden question"
_KEYS = ("topic", "audience")
_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("lookup", "record")
]


def _chain(*gated: bool) -> list[ChainStep]:
    return [
        ChainStep(
            step_id=f"s{i}",
            name=f"step{i}",
            extraction_instructions=f"extract {i}",
            response_instructions=f"respond {i}" if i % 2 == 0 else "",
            validation_fn=(lambda ctx: True) if gate else None,
        )
        for i, gate in enumerate(gated)
    ]


#: case name -> (callable, is a full FSM definition that core can load)
CASES: dict[str, tuple[Callable[[], Any], bool]] = {
    # orchestrator and ADaPT
    "orchestrator": (lambda: fd.build_orchestrator_fsm(), True),
    "orchestrator_keys": (lambda: fd.build_orchestrator_fsm(_TASK, _KEYS), True),
    "adapt": (lambda: fd.build_adapt_fsm(), True),
    "adapt_depth": (lambda: fd.build_adapt_fsm(max_depth=5), True),
    # reflexion
    "reflexion": (lambda: fd.build_reflexion_fsm(_registry(*_TOOLS)), True),
    "reflexion_approval": (
        lambda: fd.build_reflexion_fsm(_registry(*_TOOLS), _TASK, True, _KEYS),
        True,
    ),
    "reflexion_empty": (lambda: fd.build_reflexion_fsm(ToolRegistry()), True),
    # plan-execute
    "plan_execute": (lambda: fd.build_plan_execute_fsm(), True),
    "plan_execute_empty": (lambda: fd.build_plan_execute_fsm(ToolRegistry()), True),
    "plan_execute_tools": (
        lambda: fd.build_plan_execute_fsm(_registry(*_TOOLS), _TASK),
        True,
    ),
    # react
    "react": (lambda: fd.build_react_fsm(_registry(*_TOOLS)), True),
    "react_task_keys": (
        lambda: fd.build_react_fsm(_registry(*_TOOLS), _TASK, context_keys=_KEYS),
        True,
    ),
    "react_approval": (
        lambda: fd.build_react_fsm(_registry(*_TOOLS), include_approval_state=True),
        True,
    ),
    "react_classification": (
        lambda: fd.build_react_fsm(_registry(*_TOOLS), use_classification=True),
        True,
    ),
    "react_classification_approval": (
        lambda: fd.build_react_fsm(_registry(*_TOOLS), _TASK, True, True, _Out, _KEYS),
        True,
    ),
    "react_schema": (
        lambda: fd.build_react_fsm(_registry("search"), output_schema=_Out),
        True,
    ),
    "react_non_model_schema": (
        lambda: fd.build_react_fsm(_registry("search"), output_schema=dict),
        True,
    ),
    # prompt chain
    "prompt_chain_one": (lambda: fd.build_prompt_chain_fsm(_chain(False)), True),
    "prompt_chain_gates": (
        lambda: fd.build_prompt_chain_fsm(_chain(True, False, True, False), _TASK),
        True,
    ),
    # self-consistency
    "self_consistency": (lambda: fd.build_self_consistency_fsm(), True),
    "self_consistency_task": (lambda: fd.build_self_consistency_fsm(_TASK), True),
    # debate
    "debate": (lambda: fd.build_debate_fsm(), True),
    "debate_personas": (
        lambda: fd.build_debate_fsm(_TASK, "a proposer", "a critic", "a judge", 5),
        True,
    ),
    # rewoo
    "rewoo": (lambda: fd.build_rewoo_fsm(_registry(*_TOOLS)), True),
    "rewoo_task_keys": (
        lambda: fd.build_rewoo_fsm(_registry(*_TOOLS), _TASK, _KEYS),
        True,
    ),
    # evaluator-optimizer
    "evalopt": (lambda: fd.build_evalopt_fsm(), True),
    "evalopt_task": (lambda: fd.build_evalopt_fsm(_TASK), True),
    # maker-checker
    "maker_checker": (lambda: fd.build_maker_checker_fsm("make it", "check it"), True),
    "maker_checker_cap": (
        lambda: fd.build_maker_checker_fsm("make it", "check it", _TASK, 2),
        True,
    ),
    # meta builder and native function calling
    "meta_builder": (lambda: fd.build_meta_builder_fsm(), True),
    "native_fc": (
        lambda: fd.build_native_fc_fsm(_SCHEMAS, instructions="Use the tools."),
        True,
    ),
    "native_fc_forced": (
        lambda: fd.build_native_fc_fsm(
            _SCHEMAS, instructions="Use the tools.", force_final_tool="record"
        ),
        True,
    ),
    "native_fc_repair": (
        lambda: fd.build_native_fc_fsm(
            _SCHEMAS, instructions="Use the tools.", repair=True
        ),
        True,
    ),
    "native_fc_forced_repair": (
        lambda: fd.build_native_fc_fsm(
            _SCHEMAS,
            instructions="Use the tools.",
            force_final_tool="record",
            repair=True,
        ),
        True,
    ),
    "native_fc_undeclared_forced": (
        lambda: fd.build_native_fc_fsm(
            _SCHEMAS, instructions="Use the tools.", force_final_tool="missing"
        ),
        True,
    ),
    # parallel react (exported from the package, not only the module)
    "parallel_react": (lambda: build_parallel_react_fsm(_registry(*_TOOLS)), True),
    "parallel_react_schema_keys": (
        lambda: build_parallel_react_fsm(
            _registry(*_TOOLS), _TASK, output_schema=_Out, context_keys=_KEYS
        ),
        True,
    ),
    "parallel_react_non_model_schema": (
        lambda: build_parallel_react_fsm(_registry("search"), output_schema=dict),
        True,
    ),
    # shared pieces
    "loop_field_context_keys": (lambda: fd.loop_field_context_keys(), False),
    "loop_field_context_keys_extra": (
        lambda: fd.loop_field_context_keys(("a", ContextKeys.TASK, "b", "a")),
        False,
    ),
    "tool_selection": (
        lambda: _loop._tool_selection_field_extractions("pick", context_keys=("k",)),
        False,
    ),
    "tool_selection_no_name": (
        lambda: _loop._tool_selection_field_extractions(
            "pick", include_tool_name=False, context_keys=()
        ),
        False,
    ),
    "think_fields": (lambda: _loop._think_field_extractions("pick"), False),
    "think_fields_no_name": (
        lambda: _loop._think_field_extractions(
            "pick", include_tool_name=False, context_keys=("k",)
        ),
        False,
    ),
    "conclude_logic": (lambda: fd.conclude_on_evidence_logic(), False),
    "conclude_logic_flag": (
        lambda: fd.conclude_on_evidence_logic(ContextKeys.EVALUATION_PASSED),
        False,
    ),
    "await_approval_state": (lambda: _loop._await_approval_state(), False),
    "flag_condition": (lambda: _common._flag_condition("flag", "A flag"), False),
    "reply_kind_condition": (lambda: native_fc._reply_kind_condition("calls"), False),
    "finalize": (
        lambda: _common._finalize_fsm("n", "d" * 400, "dflt", "s", "p", {"s": {}}),
        False,
    ),
    "finalize_default_description": (
        lambda: _common._finalize_fsm("n", "", "dflt", "s", "p", {}),
        False,
    ),
}


def fsm_cases() -> dict[str, dict[str, Any]]:
    """Every full FSM of the corpus, ``"agents/<case>"`` then ``"reasoning/<name>"``.

    Takes nothing, builds each dict fresh (a reasoning FSM through the deep copy
    of ``get_reasoning_fsm``), so the caller may mutate a result.
    """
    cases = {f"agents/{name}": build() for name, (build, full) in CASES.items() if full}
    for name in ALL_REASONING_FSMS:
        cases[f"reasoning/{name}"] = get_reasoning_fsm(name)
    return cases


def render(definition: dict[str, Any]) -> str:
    """The fixture text: insertion order kept, ASCII only, one final newline."""
    return json.dumps(definition, indent=2, ensure_ascii=True) + "\n"


def canonical_sha256(value: Any) -> str:
    """The recipe of the hash goldens: sha256 of ``json.dumps(value, sort_keys=True)``."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixture_path(case: str) -> Path:
    return FIXTURES / f"{case}.json"


def texts_digest() -> str:
    """One sha256 over every rendered text in case order (hash-seed checks)."""
    texts = "".join(render(d) for d in fsm_cases().values())
    return hashlib.sha256(texts.encode()).hexdigest()


def index() -> dict[str, dict[str, str]]:
    """``case -> {sha256, fsm_id, digest64}``, the identity of each document."""
    out = {}
    for case, raw in fsm_cases().items():
        definition = FSMDefinition(**raw)
        out[case] = {
            "sha256": canonical_sha256(raw),
            "fsm_id": fsm_id_of(definition),
            "digest64": definition_digest(definition),
        }
    return out


def index_text() -> str:
    return json.dumps(index(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def wrap(case: str, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """The envelope document of a full-FSM case, as a plain dict.

    ``raw`` is the case's FSM dict when the caller already built it (``fsm_cases()``
    builds all of them); omitted, it is built here.

    ``agents/*`` are ``fsm`` documents (the ``pattern`` kind arrives with E4.2); a
    ``reasoning/*`` case with an entry in ``STRATEGIES`` is a ``strategy`` document
    carrying its aliases, results and solution key. The header comes from the case:
    ``id`` is the case name, ``name`` and ``description`` those of the FSM,
    ``requires.tools`` the tool names its completion states declare, ``capabilities``
    ``tools`` (some tool is declared), ``hitl`` (a state names an approval) and
    ``strategy``; ``lifecycle`` is ``builtin``; the contract is empty.
    """
    from fsm_llm.core.state_kinds import declared_tool_names
    from fsm_llm.reasoning.constants import STRATEGIES

    raw = fsm_cases()[case] if raw is None else raw
    tools = sorted(declared_tool_names(FSMDefinition(**raw).states.values()))
    reasoning = case.startswith("reasoning/")
    spec = STRATEGIES.get(case.removeprefix("reasoning/")) if reasoning else None
    capabilities = (
        (["tools"] if tools else [])
        + (["hitl"] if any("approval" in sid for sid in raw["states"]) else [])
        + (["strategy"] if spec else [])
    )
    body: dict[str, Any] = {"fsm": raw}
    if spec:
        body["strategy"] = {
            "aliases": list(spec.aliases),
            "results": dict(spec.results),
            "solution_key": spec.solution_key,
        }
    return {
        "schema": "fsm-llm/executable",
        "kind": "strategy" if spec else "fsm",
        "id": case,
        "name": raw["name"],
        "description": raw["description"],
        "requires": {"tools": tools},
        "capabilities": capabilities,
        "lifecycle": "builtin",
        "body": body,
    }


def write_all() -> None:
    """Regenerate every fixture, the index and the shipped schema; delete a fixture no case owns."""
    from fsm_llm.core.definitions.executable import SCHEMA_PATH, schema_text

    cases = fsm_cases()
    files = {fixture_path(case): render(d) for case, d in cases.items()}
    files[INDEX_PATH] = index_text()
    files[SCHEMA_PATH] = schema_text()
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    for path in FIXTURES.glob("*/*.json"):
        if path not in files:
            path.unlink()


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python -m tests.executables_corpus --write")
    write_all()
