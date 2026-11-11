"""
Tests for ``fsm_llm.synthesis.policy``: ``SynthesisPolicy``, the G5 policy gate
(``check_policy``, ``needs_approval``) and abort injection and stripping.

The security claims are proven through the real ``API`` (no network): the
injected definition loads, validates and ends in ``abort`` when the driver sets
``procedure_abort``, and a scripted extraction cannot forge that key.
"""

import copy
import dataclasses
import math
import typing
from typing import Any

import pytest
from pydantic import BaseModel

from fsm_llm import API
from fsm_llm.definitions import ClassificationExtractionConfig, FSMDefinition
from fsm_llm.security import is_forbidden_context_entry
from fsm_llm.synthesis import SynthesisError, run_gates
from fsm_llm.synthesis import policy as policy_module
from fsm_llm.synthesis.policy import (
    PolicyFinding,
    SynthesisPolicy,
    check_policy,
    inject_abort,
    needs_approval,
    strip_abort,
)
from fsm_llm.validator import FSMValidator
from tests.conftest import MockLLM2Interface
from tests.test_fsm_llm_synthesis.conftest import good_fsm as _good


def _codes(data: Any, policy: SynthesisPolicy | None = None) -> list[str]:
    return [f.code for f in check_policy(data, policy or SynthesisPolicy())]


class TestCharacterizeNameOnlySecretCall:
    """`is_forbidden_context_entry(name, "")` is the name-only call G5 uses."""

    @pytest.mark.parametrize("name", ["password", "api_key", "token", "secret_x"])
    def test_secret_names_are_refused_with_or_without_a_value(self, name):
        assert is_forbidden_context_entry(name, "")
        assert is_forbidden_context_entry(name, "plain")

    @pytest.mark.parametrize("name", ["user_key", "name", "order_total", "email"])
    def test_ordinary_names_pass_with_an_empty_value(self, name):
        assert not is_forbidden_context_entry(name, "")
        assert not is_forbidden_context_entry(name, "plain")


class TestSynthesisPolicyLimits:
    def test_defaults(self):
        p = SynthesisPolicy()
        assert (p.max_states, p.max_transitions_per_state, p.max_text_chars) == (
            12,
            6,
            600,
        )
        assert p.max_fields_per_state == 10
        assert (p.max_synthesis_depth, p.max_repair_rounds, p.max_turns) == (1, 3, 40)
        assert (p.max_compile_tokens, p.max_synthesis_seconds) == (8192, 300)
        assert p.reuse_min_score == 0.8 and p.trust_after_successes is None
        assert p.allowed_tools == frozenset() and p.require_approval is False
        assert p.max_traces_per_digest == 200

    @pytest.mark.parametrize(
        "name",
        [
            "max_states",
            "max_transitions_per_state",
            "max_fields_per_state",
            "max_text_chars",
            "max_synthesis_depth",
            "max_turns",
            "max_compile_tokens",
            "max_synthesis_seconds",
            "max_traces_per_digest",
        ],
    )
    @pytest.mark.parametrize(
        "bad", [True, 3.0, "5", None, 0, -1, 10**30, float("nan"), [4]]
    )
    def test_a_bad_limit_is_refused_never_clamped(self, name, bad):
        with pytest.raises(SynthesisError):
            SynthesisPolicy(**{name: bad})

    def test_repair_rounds_accept_zero_and_refuse_below(self):
        assert SynthesisPolicy(max_repair_rounds=0).max_repair_rounds == 0
        for bad in (-1, True, 1.0, "1", 10**30):
            with pytest.raises(SynthesisError):
                SynthesisPolicy(max_repair_rounds=bad)

    @pytest.mark.parametrize("good", [0, 0.0, 0.5, 1, 1.0])
    def test_reuse_min_score_accepts_the_unit_interval(self, good):
        assert SynthesisPolicy(reuse_min_score=good).reuse_min_score == float(good)

    @pytest.mark.parametrize(
        "bad", [True, "0.5", None, -0.1, 1.5, math.nan, math.inf, -math.inf]
    )
    def test_reuse_min_score_refuses_the_rest(self, bad):
        with pytest.raises(SynthesisError):
            SynthesisPolicy(reuse_min_score=bad)

    def test_trust_after_successes(self):
        assert SynthesisPolicy(trust_after_successes=3).trust_after_successes == 3
        for bad in (0, -1, True, 2.0, "3"):
            with pytest.raises(SynthesisError):
                SynthesisPolicy(trust_after_successes=bad)

    def test_allowed_tools_is_a_frozenset_of_names(self):
        assert SynthesisPolicy(allowed_tools=["a", "b", "a"]).allowed_tools == {
            "a",
            "b",
        }
        for bad in ("search", [" "], [""], [1], None, [["x"]]):
            with pytest.raises(SynthesisError):
                SynthesisPolicy(allowed_tools=bad)

    @pytest.mark.parametrize("bad", [1, 0, "yes", None])
    def test_require_approval_is_a_bool(self, bad):
        with pytest.raises(SynthesisError):
            SynthesisPolicy(require_approval=bad)

    def test_frozen_and_replace_revalidates(self):
        p = SynthesisPolicy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.max_states = 99  # type: ignore[misc]
        assert dataclasses.replace(p, max_states=5).max_states == 5
        with pytest.raises(SynthesisError):
            dataclasses.replace(p, max_states=True)


def _nested(**fields: Any) -> dict[str, Any]:
    """A nested ``synthesize`` block (judged at any depth, see ``_key_names``)."""
    block: dict[str, Any] = {
        "goal": "g",
        "return_keys": ["answer"],
        "on_failure": "done",
    }
    return block | fields


def _put(d: dict[str, Any], where: str, name: str) -> None:
    """Place the key name `name` at one of the positions G5 reads."""
    ask = d["states"]["ask"]
    if where == "field":
        ask["field_extractions"][0]["field_name"] = name
    elif where == "field_context_keys":
        ask["field_extractions"][0]["context_keys"] = [name]
    elif where == "classification":
        ask["classification_extractions"] = [{"field_name": name, "intents": []}]
    elif where == "classification_context_keys":
        ask["classification_extractions"] = [
            {"field_name": "mood", "intents": [], "context_keys": [name]}
        ]
    elif where == "required":
        ask["required_context_keys"] = [name]
    elif where == "requires":
        ask["transitions"][0]["conditions"][0]["requires_context_keys"] = [name]
    elif where == "var":
        ask["transitions"][0]["conditions"][0]["logic"] = {"!!": [{"var": name}]}
    elif where in ("read_keys", "write_keys"):
        ask["context_scope"] = {where: [name]}
    elif where == "handler_only":
        d["handler_only_keys"] = [name]
    elif where in ("synth_inputs", "synth_return_keys", "synth_result_key"):
        block = _nested(result_key="synth_out")
        field = where.removeprefix("synth_")
        block[field] = name if field == "result_key" else [name]
        ask["synthesize"] = block
    else:  # pragma: no cover
        raise AssertionError(where)


WHERES = [
    "field",
    "field_context_keys",
    "classification",
    "classification_context_keys",
    "required",
    "requires",
    "var",
    "read_keys",
    "write_keys",
    "handler_only",
    "synth_inputs",
    "synth_return_keys",
    "synth_result_key",
]


class TestCheckPolicyOneHostileCasePerCode:
    def test_the_clean_compile_has_no_finding(self):
        assert check_policy(_good(), SynthesisPolicy()) == []

    @pytest.mark.parametrize(
        ("code", "policy", "mutate"),
        [
            (
                "too-many-states",
                SynthesisPolicy(max_states=1),
                lambda d: None,
            ),
            (
                "too-many-transitions",
                SynthesisPolicy(max_transitions_per_state=1),
                lambda d: d["states"]["ask"]["transitions"].append(
                    {"target_state": "done", "description": "x", "priority": 2}
                ),
            ),
            (
                "text-too-long",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"].update(response_instructions="x" * 601),
            ),
            ("unknown-key", SynthesisPolicy(), lambda d: d.update(handlers="x")),
            (
                "unknown-key",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"].update(on_enter="x"),
            ),
            ("persona-set", SynthesisPolicy(), lambda d: d.update(persona="root")),
            (
                "reserved-state",
                SynthesisPolicy(),
                lambda d: d["states"].update(abort=d["states"]["done"]),
            ),
            (
                "reserved-state",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"]["transitions"][0].update(
                    target_state="abort"
                ),
            ),
            (
                "priority-reserved",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"]["transitions"][0].update(priority=0),
            ),
            (
                "priority-reserved",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"]["transitions"][0].update(priority=-5),
            ),
            (
                "priority-type",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"]["transitions"][0].update(priority=0.0),
            ),
            (
                "model-setting",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"].update(
                    classification_extractions=[_classification(model="openai/x")]
                ),
            ),
            (
                "regex-pattern",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"]["field_extractions"][0].update(
                    validation_rules={"pattern": "^(a+)+$"}
                ),
            ),
            (
                "nested-synthesize",
                SynthesisPolicy(),
                lambda d: d["states"]["ask"].update(synthesize={"goal": "g"}),
            ),
        ],
    )
    def test_each_structural_code(self, code, policy, mutate):
        d = _good()
        mutate(d)
        findings = check_policy(d, policy)
        assert code in [f.code for f in findings]
        assert all(f.gate == "G5" for f in findings)

    def test_depth_above_one_allows_a_nested_synthesize(self):
        d = _good()
        d["states"]["ask"]["synthesize"] = {"goal": "g"}
        assert "nested-synthesize" not in _codes(
            d, SynthesisPolicy(max_synthesis_depth=2)
        )

    @pytest.mark.parametrize("where", WHERES)
    @pytest.mark.parametrize(
        "name", ["_hidden", "system_x", "INTERNAL_y", "__z", "_conversation_id"]
    )
    def test_internal_names_at_every_position(self, where, name):
        d = _good()
        _put(d, where, name)
        assert {"internal-key", "reserved-key"} & set(_codes(d))

    @pytest.mark.parametrize("where", WHERES)
    @pytest.mark.parametrize(
        "name", ["procedure_abort", "synthesis_result", "agent_trace"]
    )
    def test_reserved_names_at_every_position(self, where, name):
        d = _good()
        _put(d, where, name)
        assert "reserved-key" in _codes(d)

    @pytest.mark.parametrize("where", WHERES)
    @pytest.mark.parametrize("name", ["password", "api_key"])
    def test_secret_names_at_every_position(self, where, name):
        d = _good()
        _put(d, where, name)
        assert "secret-key" in _codes(d)

    @pytest.mark.parametrize(
        "text",
        ["\x1b[31mred", "a\x00b", "bad\ud800", "x‮y", "a\rb", "\x7f"],
    )
    def test_control_text_anywhere(self, text):
        for place in ("description", "purpose", "response_instructions"):
            d = _good()
            d["states"]["ask"][place] = text
            assert "control-text" in _codes(d)
        d = _good()
        d["states"]["ask"]["transitions"][0]["description"] = text
        assert "control-text" in _codes(d)

    def test_newline_and_tab_are_ordinary_text(self):
        d = _good()
        d["states"]["ask"]["response_instructions"] = "Line one\n\tline two"
        assert check_policy(d, SynthesisPolicy()) == []

    def test_a_clean_name_in_every_position_is_clean(self):
        for where in WHERES:
            d = _good()
            _put(d, where, "user_note")
            assert "internal-key" not in _codes(d), where


def _fields(extra: int) -> dict[str, Any]:
    """The clean compile with ``extra`` more required keys on ``ask`` (answer is 1)."""
    d = _good()
    d["states"]["ask"]["required_context_keys"] = [f"f{i}" for i in range(extra)]
    return d


class TestFieldsPerStateAreCapped:
    """Every extracted name is one cheap-model call per child step (review pass 2,
    concern 3: 60 required fields made 488 calls in two steps)."""

    def test_the_documented_code_and_default(self):
        assert "too-many-fields" in (check_policy.__doc__ or "")
        assert "max_fields_per_state" in (SynthesisPolicy.__doc__ or "")

    def test_a_state_with_sixty_required_fields_is_refused(self):
        findings = check_policy(_fields(60), SynthesisPolicy())
        assert [(f.code, f.state, f.gate) for f in findings] == [
            ("too-many-fields", "ask", "G5")
        ]
        report = run_gates(
            _fields(60), SynthesisPolicy(), inputs=(), return_keys=("answer",)
        )
        assert "too-many-fields" in [f.code for f in report.findings]

    def test_exactly_at_the_limit_passes_and_one_over_is_refused(self):
        # `answer` (field_extractions and the edge's requires) is 1 name
        assert _codes(_fields(9)) == []
        assert _codes(_fields(10)) == ["too-many-fields"]
        assert _codes(_fields(1), SynthesisPolicy(max_fields_per_state=2)) == []
        assert _codes(_fields(2), SynthesisPolicy(max_fields_per_state=2)) == [
            "too-many-fields"
        ]

    def test_the_same_name_in_every_position_counts_once(self):
        d = _good()
        ask = d["states"]["ask"]
        ask["required_context_keys"] = ["answer"] * 30
        ask["classification_extractions"] = [{"field_name": "answer", "intents": []}]
        assert _codes(d, SynthesisPolicy(max_fields_per_state=1)) == []

    @pytest.mark.parametrize(
        "place", ["field_extractions", "classification", "required", "requires"]
    )
    def test_every_position_that_mints_an_extraction_counts(self, place):
        d = _good()
        ask = d["states"]["ask"]
        names = [f"n{i}" for i in range(11)]
        if place == "field_extractions":
            ask["field_extractions"] += [{"field_name": n} for n in names]
        elif place == "classification":
            ask["classification_extractions"] = [
                {"field_name": n, "intents": []} for n in names
            ]
        elif place == "required":
            ask["required_context_keys"] = names
        else:
            ask["transitions"][0]["conditions"][0]["requires_context_keys"] = names
        assert _codes(d) == ["too-many-fields"]

    def test_the_count_is_per_state(self):
        d = _fields(9)
        d["states"]["done"]["required_context_keys"] = [f"g{i}" for i in range(9)]
        assert _codes(d) == []

    @pytest.mark.parametrize(
        "shape",
        [
            "x",
            3,
            [1, None, {"a": 1}, ["b"]],
            {"k": "v"},
            [{"field_name": ["a"]}, {"field_name": 5}, None],
        ],
    )
    def test_odd_shapes_never_raise_and_count_nothing(self, shape):
        d = _good()
        ask = d["states"]["ask"]
        ask["required_context_keys"] = shape
        ask["classification_extractions"] = shape
        assert "too-many-fields" not in _codes(
            d, SynthesisPolicy(max_fields_per_state=1)
        )


def _completion(*names: str) -> dict[str, Any]:
    tools = [{"type": "function", "function": {"name": n}} for n in names]
    return {"tools": tools}


class TestCompletionTools:
    def _with(self, completion: dict[str, Any]) -> dict[str, Any]:
        d = _good()
        d["states"]["ask"]["completion"] = completion
        return d

    def test_an_unlisted_tool_is_refused(self):
        findings = check_policy(self._with(_completion("web")), SynthesisPolicy())
        assert [f.code for f in findings] == ["completion-tool-not-allowed"]
        assert findings[0].state == "ask"

    @pytest.mark.parametrize(
        "completion",
        [
            {"tools": ["web"]},
            {"tools": [{"function": {}}]},
            {"tools": [{"function": {"name": 7}}]},
            {"tools": "web"},
            {"tools": [None]},
        ],
    )
    def test_a_malformed_tool_is_refused(self, completion):
        policy = SynthesisPolicy(allowed_tools={"web"})
        assert "completion-tool-not-allowed" in _codes(self._with(completion), policy)

    def test_listed_tools_pass_and_need_approval(self):
        d = self._with(_completion("web"))
        policy = SynthesisPolicy(allowed_tools={"web"})
        assert check_policy(d, policy) == []
        assert needs_approval(d, policy) is True

    def test_one_unlisted_among_listed_is_refused(self):
        d = self._with(_completion("web", "shell"))
        assert "completion-tool-not-allowed" in _codes(
            d, SynthesisPolicy(allowed_tools={"web"})
        )

    def test_needs_approval_by_policy_or_by_tools_only(self):
        assert needs_approval(_good(), SynthesisPolicy()) is False
        assert needs_approval(_good(), SynthesisPolicy(require_approval=True)) is True
        structured = self._with({"response_format_key": "_shape"})
        assert needs_approval(structured, SynthesisPolicy()) is False
        assert needs_approval(None, SynthesisPolicy()) is False


class TestCheckPolicyNeverRaises:
    @pytest.mark.parametrize(
        "data",
        [
            None,
            [],
            "x",
            7,
            {},
            {"states": []},
            {"states": {"a": 1}},
            {"states": {"a": {"transitions": "x"}}},
            {"states": {"a": {"transitions": [1, None]}}},
            {"states": {"a": {"transitions": [{"conditions": [3]}]}}},
            {"states": {"a": {"field_extractions": [1], "required_context_keys": 4}}},
            {"states": {"a": {"completion": 5}}, "handler_only_keys": "x"},
            {"states": {"a": {"transitions": [{"conditions": [{"logic": {}}]}]}}},
            {"states": {"a": {"transitions": [{"conditions": [{"logic": 3}]}]}}},
            {"states": {1: {}}, "persona": None},
        ],
    )
    def test_garbage_gives_findings_not_an_exception(self, data):
        findings = check_policy(data, SynthesisPolicy())
        assert all(isinstance(f, PolicyFinding) for f in findings)

    def test_deep_nesting_is_judged_without_recursion(self):
        node: Any = {"var": "x"}
        for _ in range(20_000):
            node = {"!": [node]}
        d = _good()
        d["states"]["ask"]["transitions"][0]["conditions"][0]["logic"] = node
        check_policy(d, SynthesisPolicy())

    def test_a_finding_is_frozen_plain_data(self):
        d = _good()
        d["persona"] = "x" * 5000
        findings = check_policy(d, SynthesisPolicy())
        assert [f.code for f in findings] == ["text-too-long", "persona-set"]
        (finding,) = [f for f in findings if f.code == "persona-set"]
        assert finding.gate == "G5" and len(finding.message) < 400
        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.code = "other"  # type: ignore[misc]

    def test_a_hostile_name_is_capped_in_the_message(self):
        d = _good()
        d["states"]["ask"]["required_context_keys"] = ["_" + "k" * 5000]
        assert all(len(f.message) < 400 for f in check_policy(d, SynthesisPolicy()))


def _multi() -> dict[str, Any]:
    d = _good()
    d["states"]["done"]["transitions"] = [
        {"target_state": "ask", "description": "again", "priority": 5}
    ]
    d["states"]["end"] = {"id": "end", "description": "e", "purpose": "p"}
    d["states"]["done"]["transitions"].append(
        {"target_state": "end", "description": "bye", "priority": 2}
    )
    d["handler_only_keys"] = ["note"]
    return d


class TestInjectAndStripAbort:
    @pytest.mark.parametrize(
        "make",
        [
            _good,
            _multi,
            lambda: {
                **_good(),
                "states": {"ask": _good()["states"]["done"]},
                "initial_state": "ask",
            },
        ],
        ids=["linear", "multi-state-with-handler-only", "all-terminal"],
    )
    def test_strip_inverts_inject(self, make):
        d = make()
        before = copy.deepcopy(d)
        injected = inject_abort(d)
        assert d == before  # never mutates its argument
        assert strip_abort(injected) == before
        assert injected == inject_abort(strip_abort(injected))  # type: ignore[arg-type]
        assert injected is not d

    def test_the_injected_shape(self):
        injected = inject_abort(_multi())
        abort = injected["states"]["abort"]
        assert abort["response_instructions"] == ""
        assert not abort.get("transitions")
        assert injected["handler_only_keys"] == ["note", "procedure_abort"]
        for sid in ("ask", "done"):
            edge = injected["states"][sid]["transitions"][-1]
            assert edge["target_state"] == "abort" and edge["priority"] == 0
            assert edge["conditions"] == [
                {
                    "description": edge["conditions"][0]["description"],
                    "requires_context_keys": ["procedure_abort"],
                }
            ]
        for sid in ("end", "abort"):
            assert not injected["states"][sid].get("transitions")

    def test_a_listed_procedure_abort_is_not_duplicated(self):
        d = _good()
        d["handler_only_keys"] = ["procedure_abort"]
        assert inject_abort(d)["handler_only_keys"] == ["procedure_abort"]

    def test_empty_handler_only_list_reads_back_as_absent(self):
        d = _good()
        d["handler_only_keys"] = []
        back = strip_abort(inject_abort(d))
        assert back is not None and "handler_only_keys" not in back

    @pytest.mark.parametrize(
        "data",
        [
            None,
            [],
            {"states": []},
            {"states": {"a": 1}},
            {"states": {"a": {"transitions": "x"}}},
            {"states": {"a": {"transitions": [1]}}},
            {"states": {"abort": {}}},
            {"states": {"a": {}}, "handler_only_keys": "x"},
            {"states": {"a": {}}, "handler_only_keys": [1]},
        ],
    )
    def test_inject_refuses_a_malformed_compile(self, data):
        with pytest.raises(SynthesisError):
            inject_abort(data)

    def test_the_injected_form_loads_and_validates(self):
        for make in (_good, _multi):
            injected = inject_abort(make())
            FSMDefinition(**injected)
            result = FSMValidator(injected).validate()
            assert result.is_valid, result.errors


def _abort_edge(d: dict[str, Any], sid: str = "ask") -> dict[str, Any]:
    return d["states"][sid]["transitions"][-1]


#: name -> an in-place edit of a canonical injected dict that makes it non-canonical.
_TAMPERS: dict[str, Any] = {
    "abort-state-altered": lambda d: d["states"]["abort"].update(
        response_instructions="Say something"
    ),
    "abort-state-purpose-changed": lambda d: d["states"]["abort"].update(purpose="x"),
    "abort-state-has-transitions": lambda d: d["states"]["abort"].update(
        transitions=[{"target_state": "ask", "description": "b"}]
    ),
    "abort-state-missing": lambda d: d["states"].pop("abort"),
    "edge-differs-priority": lambda d: _abort_edge(d).update(priority=1),
    "edge-differs-key": lambda d: _abort_edge(d)["conditions"][0].update(
        requires_context_keys=["other"]
    ),
    "edge-with-logic": lambda d: _abort_edge(d)["conditions"][0].update(
        logic={"var": "x"}
    ),
    "edge-missing-on-one-state": lambda d: d["states"]["done"]["transitions"].pop(),
    "extra-abort-edge": lambda d: d["states"]["ask"]["transitions"].append(
        copy.deepcopy(_abort_edge(d))
    ),
    "abort-edge-not-last": lambda d: d["states"]["ask"]["transitions"].reverse(),
    "abort-edge-on-a-terminal": lambda d: d["states"]["end"].update(
        transitions=[copy.deepcopy(_abort_edge(d))]
    ),
    "handler-only-key-missing": lambda d: d["handler_only_keys"].remove(
        "procedure_abort"
    ),
    "handler-only-key-twice": lambda d: d["handler_only_keys"].append(
        "procedure_abort"
    ),
    "initial-state-is-abort": lambda d: d.update(initial_state="abort"),
    "a-state-is-not-an-object": lambda d: d["states"].update(end=3),
}


class TestStripAbortRefusesNonCanonicalShapes:
    @pytest.mark.parametrize("name", sorted(_TAMPERS))
    def test_returns_none(self, name):
        data = inject_abort(_multi())
        _TAMPERS[name](data)
        before = copy.deepcopy(data)
        assert strip_abort(data) is None
        assert data == before

    @pytest.mark.parametrize("data", [None, [], "x", {}, {"states": []}, _good()])
    def test_other_shapes_are_not_canonical(self, data):
        assert strip_abort(data) is None


def _scripted_api(definition: dict[str, Any], extraction: dict[str, Any]):
    llm = MockLLM2Interface(extraction_data=extraction)
    api = API.from_definition(definition=definition, llm_interface=llm)
    conv_id, _ = api.start_conversation()
    return api, conv_id, llm


class TestAbortThroughTheRealApi:
    def test_the_driver_sets_the_key_and_the_run_ends_in_abort(self):
        api, conv_id, _ = _scripted_api(inject_abort(_good()), {})
        assert api.advance(conv_id).state_after == "ask"  # blocked: no answer
        api.update_context(conv_id, {"procedure_abort": True})
        results = api.run_until_terminal(conv_id, max_steps=5)
        assert api.get_current_state(conv_id) == "abort"
        assert results[-1].ended
        assert api.get_data(conv_id)["procedure_abort"] is True
        assert results[-1].response is None  # the abort state is silent

    def test_the_abort_edge_wins_outright_over_every_compiled_edge(self):
        d = _good()
        d["states"]["ask"]["transitions"][0]["conditions"] = None  # always fires
        api, conv_id, _ = _scripted_api(inject_abort(d), {})
        api.update_context(conv_id, {"procedure_abort": True})
        api.run_until_terminal(conv_id, max_steps=5)
        assert api.get_current_state(conv_id) == "abort"

    def test_a_scripted_extraction_cannot_forge_the_key(self):
        forged = {"procedure_abort": True, "answer": None}
        api, conv_id, llm = _scripted_api(inject_abort(_good()), forged)
        for _ in range(3):
            assert api.advance(conv_id).state_after == "ask"
        assert "procedure_abort" not in api.get_data(conv_id)
        asked = [r.field_name for c, r in llm.call_history if c == "extract_field"]
        assert "procedure_abort" not in asked

    def test_negative_control_without_handler_only_the_forgery_works(self):
        # Proves the test above bites: the same script on a definition missing
        # `handler_only_keys` reaches `abort`.
        weak = inject_abort(_good())
        weak["handler_only_keys"] = []
        api, conv_id, _ = _scripted_api(weak, {"procedure_abort": True})
        api.run_until_terminal(conv_id, max_steps=3)
        assert api.get_current_state(conv_id) == "abort"

    def test_a_normal_run_reaches_done_not_abort(self):
        api, conv_id, _ = _scripted_api(inject_abort(_good()), {"answer": "yes"})
        api.run_until_terminal(conv_id, max_steps=5)
        assert api.get_current_state(conv_id) == "done"


# --- Step 6.1: model settings, coerced priorities, patterns, positions ------


def _classification(**extra: Any) -> dict[str, Any]:
    return {
        "field_name": "mood",
        "intents": [
            {"name": "happy", "description": "glad"},
            {"name": "sad", "description": "unhappy"},
        ],
        "fallback_intent": "happy",
        **extra,
    }


def _with_classification(**extra: Any) -> dict[str, Any]:
    d = _good()
    d["states"]["ask"]["classification_extractions"] = [_classification(**extra)]
    return d


def _model_reachable(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every pydantic model reachable from ``root`` through field annotations."""

    def inside(annotation: Any) -> list[type[BaseModel]]:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        return [m for arg in typing.get_args(annotation) for m in inside(arg)]

    seen: set[type[BaseModel]] = set()
    todo = [root]
    while todo:
        model = todo.pop()
        if model not in seen:
            seen.add(model)
            for info in model.model_fields.values():
                todo.extend(inside(info.annotation))
    return seen


class TestModelSettings:
    """A compile may not pick a model, a provider or any LLM setting: the child
    classifies on the host's own interface (review iter 1, concern 1)."""

    def test_a_clean_classification_entry_passes(self):
        d = _with_classification(
            required=True, confidence_threshold=0.5, context_keys=["answer"]
        )
        assert check_policy(d, SynthesisPolicy()) == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", "openai/gpt-4o"),
            ("model", None),
            ("model", ""),
            ("prompt_config", {"temperature": 2.0, "max_tokens": 9000}),
            ("prompt_config", {}),
        ],
    )
    def test_model_and_prompt_config_are_refused_whatever_the_value(self, field, value):
        findings = check_policy(
            _with_classification(**{field: value}), SynthesisPolicy()
        )
        assert [(f.code, f.state) for f in findings] == [("model-setting", "ask")]

    @pytest.mark.parametrize(
        "key", ["api_key", "api_base", "base_url", "provider", "temperature", "x"]
    )
    def test_any_other_classification_key_is_unknown(self, key):
        assert _codes(_with_classification(**{key: "v"})) == ["unknown-key"]

    def test_an_intent_with_a_stray_key_is_unknown(self):
        d = _with_classification()
        d["states"]["ask"]["classification_extractions"][0]["intents"][0]["model"] = (
            "openai/gpt-4o"
        )
        assert _codes(d) == ["unknown-key"]

    def test_the_allow_list_names_every_field_of_the_core_model(self):
        """A field added to the core model must be judged here, not inherited."""
        allowed = policy_module._CLASSIFICATION_KEYS
        refused = policy_module._MODEL_SETTING_KEYS
        assert not allowed & refused
        assert allowed | refused == set(ClassificationExtractionConfig.model_fields)

    def test_no_other_model_of_the_definition_carries_an_llm_setting(self):
        hints = ("model", "api", "url", "provider", "temperature", "token", "llm")
        found = {
            (m.__name__, name)
            for m in _model_reachable(FSMDefinition)
            for name in m.model_fields
            if any(h in name for h in hints) or name == "prompt_config"
        }
        found.discard(("Transition", "llm_description"))  # prompt text, no setting
        assert found == {
            ("ClassificationExtractionConfig", "model"),
            ("ClassificationExtractionConfig", "prompt_config"),
        }

    def test_run_gates_refuses_a_compile_that_names_a_model(self):
        report = run_gates(
            _with_classification(model="openai/gpt-4o"),
            SynthesisPolicy(),
            inputs=(),
            return_keys=("answer",),
        )
        assert not report.passed
        assert [(f.gate, f.code) for f in report.findings] == [("G5", "model-setting")]


_PRIORITY_CODES: list[tuple[Any, str | None]] = [
    (0, "priority-reserved"),
    (-5, "priority-reserved"),
    (0.0, "priority-type"),
    (-0.0, "priority-type"),
    ("0", "priority-type"),
    ("00", "priority-type"),
    (False, "priority-type"),
    (True, "priority-type"),
    (1.0, "priority-type"),
    ("5", "priority-type"),
    (None, "priority-type"),
    (1, None),
    (2, None),
    (1000, None),
]  # a list: 0, 0.0, False (and 1, 1.0, True) are equal dict keys


class TestPriorityMustBeARealInt:
    """G1's lax load coerces 0.0, "0" and False to 0 (the abort edge's
    priority); G5 reads the raw value, so it must not accept those shapes."""

    @pytest.mark.parametrize(
        ("value", "code"),
        _PRIORITY_CODES,
        ids=[repr(v) for v, _ in _PRIORITY_CODES],
    )
    def test_the_table(self, value, code):
        d = _good()
        d["states"]["ask"]["transitions"][0]["priority"] = value
        found = [f.code for f in check_policy(d, SynthesisPolicy())]
        assert found == ([code] if code else [])

    @pytest.mark.parametrize("value", [[], {}])
    def test_a_container_is_refused(self, value):
        d = _good()
        d["states"]["ask"]["transitions"][0]["priority"] = value
        assert _codes(d) == ["priority-type"]

    def test_an_absent_priority_is_the_loader_default(self):
        d = _good()
        del d["states"]["ask"]["transitions"][0]["priority"]
        assert check_policy(d, SynthesisPolicy()) == []

    @pytest.mark.parametrize("value", [0.0, -0.0, "0", "00", False])
    def test_a_coerced_zero_never_passes_the_gates(self, value):
        d = _good()
        d["states"]["ask"]["transitions"][0]["priority"] = value
        report = run_gates(d, SynthesisPolicy(), inputs=(), return_keys=("answer",))
        assert not report.passed
        assert "priority-type" in [f.code for f in report.findings if f.gate == "G5"]


class TestNoModelAuthoredRegex:
    """A model-written pattern runs on user text in the child: ReDoS."""

    @pytest.mark.parametrize("pattern", ["^(a+)+$", "^[a-z]+$", "", None, 5])
    def test_a_pattern_is_refused_whatever_it_says(self, pattern):
        d = _good()
        d["states"]["ask"]["field_extractions"][0]["validation_rules"] = {
            "pattern": pattern
        }
        assert _codes(d) == ["regex-pattern"]

    def test_the_other_rules_stay_allowed(self):
        d = _good()
        d["states"]["ask"]["field_extractions"][0]["validation_rules"] = {
            "allowed_values": ["a", "b"],
            "min_length": 1,
            "max_length": 9,
            "min_value": 0,
            "max_value": 9,
        }
        assert check_policy(d, SynthesisPolicy()) == []

    def test_the_gates_refuse_the_redos_compile(self):
        d = _good()
        d["states"]["ask"]["field_extractions"][0]["validation_rules"] = {
            "pattern": "^(a+)+$"
        }
        report = run_gates(d, SynthesisPolicy(), inputs=(), return_keys=("answer",))
        assert not report.passed
        assert [(f.gate, f.code) for f in report.findings] == [("G5", "regex-pattern")]


class TestNestedSynthesizeNames:
    """At depth 2 a compiled synthesize block is judged like any other state."""

    DEEP = SynthesisPolicy(max_synthesis_depth=2)

    def _d(self, **fields: Any) -> dict[str, Any]:
        d = _good()
        d["states"]["ask"]["synthesize"] = _nested(**fields)
        return d

    def test_a_clean_block_passes(self):
        assert check_policy(self._d(result_key="synth_out"), self.DEEP) == []

    @pytest.mark.parametrize("name", ["procedure_abort", "_hidden", "db_password"])
    def test_a_bad_result_key_is_refused(self, name):
        assert {"reserved-key", "internal-key", "secret-key"} & set(
            _codes(self._d(result_key=name), self.DEEP)
        )

    def test_an_absent_result_key_is_the_child_own_default(self):
        assert check_policy(self._d(), self.DEEP) == []

    def test_a_tuple_of_names_is_read_like_a_list(self):
        """A stored, dumped definition holds tuples (``verify_stored``)."""
        d = self._d(result_key="synth_out", inputs=("_hidden",))
        assert "internal-key" in _codes(d, self.DEEP)

    @pytest.mark.parametrize("bad", [None, 5, ["done"], {"x": 1}])
    def test_on_failure_must_be_a_state_id_string(self, bad):
        d = self._d(result_key="synth_out", on_failure=bad)
        assert "bad-shape" in _codes(d, self.DEEP)

    def test_on_failure_may_not_name_the_abort_state(self):
        d = self._d(result_key="synth_out", on_failure="abort")
        assert "reserved-state" in _codes(d, self.DEEP)

    def test_a_non_object_block_is_a_shape_finding(self):
        d = _good()
        d["states"]["ask"]["synthesize"] = ["x"]
        assert "bad-shape" in _codes(d, self.DEEP)


class TestCompletionKeyNames:
    """``messages_key`` and ``response_format_key`` are internal by design, so
    only a reserved or secret-looking name is refused there."""

    def _with(self, **fields: Any) -> dict[str, Any]:
        d = _good()
        ask = d["states"]["ask"]
        del ask["field_extractions"]
        ask["completion"] = {"response_format_key": "_fmt", **fields}
        return d

    def test_the_internal_defaults_pass_the_name_rules(self):
        d = self._with(messages_key="_completion_messages")
        assert not {"internal-key", "reserved-key", "secret-key"} & set(_codes(d))

    @pytest.mark.parametrize("field", ["messages_key", "response_format_key"])
    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("_conversation_id", "reserved-key"),
            ("procedure_abort", "reserved-key"),
            ("_db_password", "secret-key"),
        ],
    )
    def test_a_reserved_or_secret_name_is_refused(self, field, name, code):
        assert code in _codes(self._with(**{field: name}))
