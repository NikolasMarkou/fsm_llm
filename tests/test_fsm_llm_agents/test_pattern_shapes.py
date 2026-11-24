"""The shape rules of the agent pattern FSMs, as rows over the definition dict (plan a555dcfe, step 8).

Each rule was a ``# DECISION`` anchor in a builder under ``agents/fsm_definitions/``. E4.2 (steps 9
to 18) turns the builders into JSON files, which cannot carry an anchor (``decision_index.py
--check`` refuses an anchor in an unscanned file), and the owner dropped the anchors of replaced
code (D-029 of plan ace7a8c8). The rules must outlive them, so each is a function over a plain
definition dict that returns its problems, with one row on today's dicts (GREEN: the rule is true
now) and one planted mutant on a copy (RED: the rule can fail). After a conversion the rows read
the file instance: ``definitions()`` is ``tests.executables_corpus.fsm_cases()``, which step 9 and
the later ones re-point at the file.

Anchors each rule replaces (the anchors stay in the builders until the builders go):

- ``fallback``: ``plan-2026-09-24T045559-3e4eb3e5/D-002`` (the unconditional lowest-priority edge
  of ``think``, ``generate``, ``check``, ``assess``, ``decompose``).
- ``approval_edge``: ``plan-2026-09-24T091842-c1d5bfbc/D-005`` (the edge to ``await_approval`` at
  150 with one predicate, ``approval_required`` True).
- ``conclude_on_evidence``: ``plan-2026-09-24T091842-c1d5bfbc/D-008`` and
  ``plan-2026-09-29T103145-06a5ec0a/D-051`` (no conclude on the flag alone; a forced stop
  concludes on its own).
- ``await_approval_silent``: ``plan-2026-09-30T062855-07ad3f8c/D-033`` (the state extracts
  nothing and speaks nothing).
- ``strict_flag``: ``plan-2026-10-01T093600-944e2692/D-022`` (a flag gate compares with ``===``).
- ``field_types``: ``plan-2026-09-29T103145-06a5ec0a/D-050`` (``any`` for a whole artifact, ``str``
  for short prose).
- ``context_keys``: ``plan-2026-09-29T103145-06a5ec0a/D-017`` (a field never has ``context_keys``
  None and never lists ``agent_trace``).
- ``no_bulk``: ``plan-2026-09-29T103145-06a5ec0a/D-009`` and ``plan-2026-09-30T062855-07ad3f8c/D-031``
  (no state-level bulk call, except ``think`` under classification, which relies on it).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from fsm_llm.core.expressions import evaluate_logic
from tests.executables_corpus import fsm_cases

_Definition = dict[str, Any]
_Problems = list[str]

#: Every tool loop, with and without approval and classification.
_LOOPS = (
    "react",
    "react_approval",
    "react_classification",
    "react_classification_approval",
    "reflexion",
    "reflexion_approval",
    "parallel_react",
)
_APPROVAL = ("react_approval", "react_classification_approval", "reflexion_approval")


def definitions() -> dict[str, _Definition]:
    """``case -> definition dict``: today the builders, after a conversion the file instances.

    Takes nothing, builds each fresh (a caller may mutate a result). Failure mode: none.
    """
    return {
        k.removeprefix("agents/"): v
        for k, v in fsm_cases().items()
        if k.startswith("agents/")
    }


def _edges(definition: _Definition, state: str) -> list[dict[str, Any]]:
    return list(definition["states"][state].get("transitions", []))


def _logic(edge: dict[str, Any]) -> Any:
    (condition,) = edge["conditions"]
    return condition["logic"]


# ---- the rules: each returns a list of problems, empty when the rule holds -----------------------

#: (state, target, priority) of the unconditional fallback each pattern keeps (3e4eb3e5 D-002).
_FALLBACKS: dict[str, tuple[tuple[str, str, int], ...]] = {
    **{name: (("think", "act", 300),) for name in _LOOPS},
    "evalopt": (("generate", "evaluate", 900),),
    "maker_checker": (("check", "revise", 900),),
    "adapt": (("assess", "combine", 900), ("decompose", "combine", 900)),
}


def fallback_problems(
    definition: _Definition, spec: tuple[tuple[str, str, int], ...]
) -> _Problems:
    """Each ``(state, target, priority)`` is an edge with NO condition, the state's lowest
    priority: a gated one BLOCKS the state on a null selection and no limiter runs on a BLOCKED
    turn (the run then burns the loop ceiling)."""
    problems = []
    for state, target, priority in spec:
        edges = _edges(definition, state)
        hits = [
            e
            for e in edges
            if e["target_state"] == target and e["priority"] == priority
        ]
        if len(hits) != 1:
            problems.append(f"{state} -> {target} at {priority}: {len(hits)} edges")
            continue
        if hits[0].get("conditions"):
            problems.append(f"{state} -> {target} at {priority} is gated")
        if priority != max(e["priority"] for e in edges):
            problems.append(f"{state} -> {target} is not the lowest priority edge")
    return problems


def approval_edge_problems(definition: _Definition) -> _Problems:
    """``think -> await_approval`` at 150 with exactly one condition, the bare predicate
    ``approval_required == True``; it sits between conclude (10) and the fallback (300)."""
    edges = [
        e for e in _edges(definition, "think") if e["target_state"] == "await_approval"
    ]
    if len(edges) != 1:
        return [f"{len(edges)} approval edges on think"]
    edge = edges[0]
    problems = []
    if edge["priority"] != 150:
        problems.append(f"approval edge priority {edge['priority']}")
    if len(edge.get("conditions") or []) != 1:
        return [*problems, "approval edge is not exactly one condition"]
    if _logic(edge) != {"==": [{"var": "approval_required"}, True]}:
        problems.append(f"approval predicate is {_logic(edge)}")
    conclude = [
        e for e in _edges(definition, "think") if e["target_state"] == "conclude"
    ]
    if not all(c["priority"] < 150 for c in conclude):
        problems.append("conclude does not beat the approval edge")
    return problems


def no_approval_problems(definition: _Definition) -> _Problems:
    """A pattern with no approval has no ``await_approval`` state and no edge to it."""
    problems = []
    if "await_approval" in definition["states"]:
        problems.append("await_approval state without approval")
    for sid in definition["states"]:
        if any(e["target_state"] == "await_approval" for e in _edges(definition, sid)):
            problems.append(f"{sid} has an edge to await_approval")
    return problems


#: The context shapes a conclude edge is judged on: ``(context, passes)``. The flag is the
#: edge's own (``should_terminate``, or ``evaluation_passed`` on Reflexion's ``evaluate``).
def _evidence_witnesses(flag: str) -> list[tuple[dict[str, Any], bool]]:
    return [
        ({flag: True, "observation_count": 0}, False),
        ({flag: True}, False),
        ({flag: True, "observation_count": 1}, True),
        ({flag: False, "observation_count": 3}, False),
        ({"observation_count": 3}, False),
        ({"max_iterations_reached": True}, True),
        ({flag: False, "max_iterations_reached": True, "observation_count": 0}, True),
        ({flag: "yes", "observation_count": 2}, False),
    ]


def conclude_problems(
    definition: _Definition, edges: tuple[tuple[str, str], ...]
) -> _Problems:
    """Each ``(state, flag)`` conclude edge passes on evidence (the flag AND a tool run) or on a
    forced stop alone, and never on the flag alone."""
    problems = []
    for state, flag in edges:
        found = [
            e for e in _edges(definition, state) if e["target_state"] == "conclude"
        ]
        if len(found) != 1 or len(found[0].get("conditions") or []) != 1:
            problems.append(f"{state}: no single gated conclude edge")
            continue
        logic = _logic(found[0])
        for context, passes in _evidence_witnesses(flag):
            if bool(evaluate_logic(logic, context)) is not passes:
                problems.append(f"{state}: conclude on {context} is not {passes}")
    return problems


def await_approval_problems(definition: _Definition) -> _Problems:
    """``await_approval`` extracts nothing and speaks nothing: no instructions, no required keys,
    no field, no classification (the driver asks the approver and writes the decision)."""
    state = definition["states"].get("await_approval")
    if state is None:
        return ["no await_approval state"]
    problems = [
        f"await_approval has {key}"
        for key in (
            "extraction_instructions",
            "required_context_keys",
            "field_extractions",
            "classification_extractions",
            "response_instructions",
        )
        if state.get(key)
    ]
    targets = {e["target_state"] for e in state.get("transitions", [])}
    if targets != {"conclude", "act", "think"}:
        problems.append(f"await_approval edges go to {sorted(targets)}")
    return problems


def _flag_conditions(definition: _Definition) -> list[dict[str, Any]]:
    """Every gated condition (one that names ``requires_context_keys``) comparing one variable
    to the boolean True with ``==`` or ``===``."""
    found = []
    for sid in definition["states"]:
        for edge in _edges(definition, sid):
            for condition in edge.get("conditions") or []:
                logic = condition["logic"]
                op = next(iter(logic)) if len(logic) == 1 else None
                args = logic.get(op) if op else None
                if (
                    op in ("==", "===")
                    and isinstance(args, list)
                    and len(args) == 2
                    and args[1] is True
                    and isinstance(args[0], dict)
                    and set(args[0]) == {"var"}
                    and condition.get("requires_context_keys")
                ):
                    found.append(condition)
    return found


def strict_flag_problems(definition: _Definition, minimum: int) -> _Problems:
    """Every gated flag condition (one that names its ``requires_context_keys``) compares with
    ``===``, so a model's ``"true"``, ``"True"``, ``1`` or ``1.0`` never opens it."""
    conditions = _flag_conditions(definition)
    problems = (
        [] if len(conditions) >= minimum else [f"{len(conditions)} flag conditions"]
    )
    for condition in conditions:
        logic = condition["logic"]
        key = logic["==="][0]["var"] if "===" in logic else logic["=="][0]["var"]
        if "===" not in logic:
            problems.append(f"{key} gate uses ==")
            continue
        if not evaluate_logic(logic, {key: True}):
            problems.append(f"{key} gate does not pass True")
        for lookalike in ("true", "True", 1, 1.0):
            if evaluate_logic(logic, {key: lookalike}):
                problems.append(f"{key} gate passes {lookalike!r}")
    return problems


#: (state, field, type) of the typed fields D-050 fixes, per pattern.
_ANY = {
    "evalopt": (("generate", "generated_output"), ("refine", "generated_output")),
    "maker_checker": (("make", "draft_output"), ("revise", "draft_output")),
    "prompt_chain_gates": tuple((f"step_{i}", "chain_step_result") for i in range(4)),
    "prompt_chain_one": (("step_0", "chain_step_result"),),
}
_STR = {
    "maker_checker": (("check", "checker_feedback"),),
    "reflexion": (
        ("evaluate", "evaluation_feedback"),
        ("reflect", "reflection"),
        ("reflect", "lessons"),
    ),
    "plan_execute_tools": (("execute_step", "step_result"),),
    "adapt": (("attempt", "attempt_result"),),
}


def field_type_problems(
    definition: _Definition,
    any_fields: tuple[tuple[str, str], ...],
    str_fields: tuple[tuple[str, str], ...],
) -> _Problems:
    """A whole generated artifact is ``any``, short prose is ``str``."""

    def kind(state: str, field: str) -> str | None:
        for entry in definition["states"][state].get("field_extractions") or []:
            if entry["field_name"] == field:
                return entry["field_type"]
        return None

    problems = [
        f"{s}.{f} is {kind(s, f)}, not any"
        for s, f in any_fields
        if kind(s, f) != "any"
    ]
    problems += [
        f"{s}.{f} is {kind(s, f)}, not str"
        for s, f in str_fields
        if kind(s, f) != "str"
    ]
    return problems


def context_key_problems(definition: _Definition) -> _Problems:
    """Every typed field lists its ``context_keys`` (never ``None``, never empty) and never
    ``agent_trace`` (it is unbounded and feeds ``AgentResult.trace``)."""
    problems = []
    for sid, state in definition["states"].items():
        for entry in state.get("field_extractions") or []:
            keys = entry.get("context_keys")
            where = f"{sid}.{entry['field_name']}"
            if not isinstance(keys, list) or not keys:
                problems.append(f"{where}: context_keys is {keys!r}")
            elif "agent_trace" in keys:
                problems.append(f"{where}: lists agent_trace")
    return problems


#: (case, state): a state that makes no state-level bulk call (``extraction_instructions`` empty).
_SILENT_BULK = (
    ("react", "think"),
    ("react_approval", "think"),
    ("reflexion", "think"),
    ("rewoo", "plan_all"),
    ("orchestrator", "orchestrate"),
    ("orchestrator", "collect"),
)


def no_bulk_problems(definitions_: dict[str, _Definition]) -> _Problems:
    """No state-level bulk call where the typed fields already fill the keys; but
    ``think`` under classification KEEPS its instructions (the bulk fill is the fallback when the
    classifier declines)."""
    problems = []
    for case, state in _SILENT_BULK:
        if definitions_[case]["states"][state].get("extraction_instructions"):
            problems.append(f"{case}.{state} makes a bulk call")
    for case in ("react_classification", "react_classification_approval"):
        if not definitions_[case]["states"]["think"].get("extraction_instructions"):
            problems.append(f"{case}.think lost its bulk instructions")
    return problems


# ---- rows on today's definitions (the rules are true now) ----------------------------------------


@pytest.fixture(scope="module")
def defs() -> dict[str, _Definition]:
    return definitions()


def _copy(defs: dict[str, _Definition], name: str) -> _Definition:
    return copy.deepcopy(defs[name])


class TestRulesHoldToday:
    @pytest.mark.parametrize("name", sorted(_FALLBACKS))
    def test_the_fallback_edge_is_unconditional_and_lowest(self, defs, name):
        assert fallback_problems(defs[name], _FALLBACKS[name]) == []

    @pytest.mark.parametrize("name", _APPROVAL)
    def test_the_approval_edge_is_150_with_one_predicate(self, defs, name):
        assert approval_edge_problems(defs[name]) == []

    @pytest.mark.parametrize("name", [n for n in _LOOPS if n not in _APPROVAL])
    def test_a_loop_without_approval_has_no_approval_state(self, defs, name):
        assert no_approval_problems(defs[name]) == []

    @pytest.mark.parametrize("name", _LOOPS)
    def test_think_and_act_conclude_on_evidence_or_a_forced_stop(self, defs, name):
        edges = (("think", "should_terminate"), ("act", "should_terminate"))
        assert conclude_problems(defs[name], edges) == []

    def test_reflexion_evaluate_concludes_on_evidence_or_a_forced_stop(self, defs):
        edges = (("evaluate", "evaluation_passed"),)
        assert conclude_problems(defs["reflexion"], edges) == []

    @pytest.mark.parametrize("name", _APPROVAL)
    def test_await_approval_extracts_and_speaks_nothing(self, defs, name):
        assert await_approval_problems(defs[name]) == []

    @pytest.mark.parametrize(
        ("name", "minimum"),
        [("meta_builder", 4), ("native_fc_forced_repair", 2)],
    )
    def test_flag_gates_compare_strictly(self, defs, name, minimum):
        assert strict_flag_problems(defs[name], minimum) == []

    @pytest.mark.parametrize("name", sorted(set(_ANY) | set(_STR)))
    def test_artifacts_are_any_and_prose_is_str(self, defs, name):
        assert (
            field_type_problems(defs[name], _ANY.get(name, ()), _STR.get(name, ()))
            == []
        )

    def test_every_pattern_field_lists_its_context_keys(self, defs):
        checked = 0
        for name, definition in defs.items():
            assert context_key_problems(definition) == [], name
            checked += sum(
                len(s.get("field_extractions") or [])
                for s in definition["states"].values()
            )
        assert checked > 50

    def test_no_state_level_bulk_call(self, defs):
        assert no_bulk_problems(defs) == []


# ---- the planted mutants: each rule must be able to fail -----------------------------------------


def _set_logic(definition: _Definition, state: str, target: str, logic: Any) -> None:
    for edge in _edges(definition, state):
        if edge["target_state"] == target and edge.get("conditions"):
            edge["conditions"][0]["logic"] = logic


class TestEachRuleCanFail:
    """A mutant on a copy of the react dict (or the one pattern that holds the rule) turns the
    rule red. A row that cannot fail proves nothing (the lesson of d454fd62)."""

    def test_dropping_the_p300_edge_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        think = copy_["states"]["think"]
        think["transitions"] = [e for e in think["transitions"] if e["priority"] != 300]
        assert fallback_problems(copy_, _FALLBACKS["react"])

    def test_gating_the_fallback_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        fallback = copy_["states"]["think"]["transitions"][-1]
        fallback["conditions"] = [
            {"description": "d", "logic": {"!=": [{"var": "tool_name"}, None]}}
        ]
        assert fallback_problems(copy_, _FALLBACKS["react"])

    def test_a_second_predicate_on_the_approval_edge_is_caught(self, defs):
        copy_ = _copy(defs, "react_approval")
        edge = next(
            e for e in _edges(copy_, "think") if e["target_state"] == "await_approval"
        )
        edge["conditions"].append(
            {"description": "d", "logic": {"==": [{"var": "tool_name"}, "danger"]}}
        )
        assert approval_edge_problems(copy_)

    def test_the_approval_edge_below_the_fallback_is_caught(self, defs):
        copy_ = _copy(defs, "react_approval")
        edge = next(
            e for e in _edges(copy_, "think") if e["target_state"] == "await_approval"
        )
        edge["priority"] = 400
        assert approval_edge_problems(copy_)

    def test_an_approval_state_in_a_pattern_without_approval_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        copy_["states"]["await_approval"] = copy.deepcopy(
            defs["react_approval"]["states"]["await_approval"]
        )
        assert no_approval_problems(copy_)

    def test_concluding_on_the_flag_alone_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        flag = {"==": [{"var": "should_terminate"}, True]}
        _set_logic(copy_, "think", "conclude", flag)
        assert conclude_problems(copy_, (("think", "should_terminate"),))

    def test_gating_the_forced_stop_on_the_flag_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        logic = {
            "and": [
                {"==": [{"var": "should_terminate"}, True]},
                {
                    "or": [
                        {">": [{"var": ["observation_count", 0]}, 0]},
                        {"==": [{"var": "max_iterations_reached"}, True]},
                    ]
                },
            ]
        }
        _set_logic(copy_, "think", "conclude", logic)
        assert conclude_problems(copy_, (("think", "should_terminate"),))

    def test_an_extraction_on_await_approval_is_caught(self, defs):
        for key, value in (
            ("extraction_instructions", "Extract the approval"),
            ("required_context_keys", ["approval_granted"]),
            ("response_instructions", "Say that approval is pending"),
        ):
            copy_ = _copy(defs, "react_approval")
            copy_["states"]["await_approval"][key] = value
            assert await_approval_problems(copy_), key

    def test_a_field_on_await_approval_is_caught(self, defs):
        copy_ = _copy(defs, "react_approval")
        copy_["states"]["await_approval"]["field_extractions"] = [
            copy_["states"]["think"]["field_extractions"][0]
        ]
        assert await_approval_problems(copy_)

    def test_double_equals_for_triple_equals_is_caught(self, defs):
        """The meta builder holds the strict gates (a react loop has no flag gate)."""
        copy_ = _copy(defs, "meta_builder")
        for sid in copy_["states"]:
            for edge in _edges(copy_, sid):
                for condition in edge.get("conditions") or []:
                    if "===" in condition["logic"]:
                        condition["logic"] = {"==": condition["logic"]["==="]}
        assert strict_flag_problems(copy_, 4)

    def test_str_for_any_is_caught(self, defs):
        copy_ = _copy(defs, "evalopt")
        copy_["states"]["generate"]["field_extractions"][0]["field_type"] = "str"
        assert field_type_problems(copy_, _ANY["evalopt"], ())

    def test_any_for_str_is_caught(self, defs):
        copy_ = _copy(defs, "maker_checker")
        copy_["states"]["check"]["field_extractions"][0]["field_type"] = "any"
        assert field_type_problems(copy_, (), _STR["maker_checker"])

    def test_a_none_in_context_keys_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        copy_["states"]["think"]["field_extractions"][0]["context_keys"] = None
        assert context_key_problems(copy_)

    def test_agent_trace_in_context_keys_is_caught(self, defs):
        copy_ = _copy(defs, "react")
        copy_["states"]["think"]["field_extractions"][0]["context_keys"].append(
            "agent_trace"
        )
        assert context_key_problems(copy_)

    def test_a_state_level_bulk_call_is_caught(self, defs):
        mutated = {**defs, "react": _copy(defs, "react")}
        mutated["react"]["states"]["think"]["extraction_instructions"] = (
            "Extract it all"
        )
        assert no_bulk_problems(mutated)

    def test_a_classification_think_without_its_bulk_call_is_caught(self, defs):
        mutated = {**defs, "react_classification": _copy(defs, "react_classification")}
        mutated["react_classification"]["states"]["think"][
            "extraction_instructions"
        ] = ""
        assert no_bulk_problems(mutated)
