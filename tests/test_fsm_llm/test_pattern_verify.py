"""The load sequence of a ``pattern`` executable: V2 on instances, V3, V4 and V7 (plan a555dcfe D-012).

A pattern is a template, so V2 judges the FSM it would run with every default and with every
variant applied; V3 resolves the names it binds and holds the ``hitl`` capability, the approval
gate binding and the ``await_approval`` state equal; V4 keeps free text out of instruction text
except through a declared ``{{variable}}``; V7 refuses, for a document a model wrote, every field
that binds code or reaches the driver, whatever the document claims. These rows guard the HITL
census invariant (D1) and the fail-closed rule of V7 (D6).
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core.definitions.executable_pattern import (
    APPROVAL_GATE_HANDLER,
    APPROVAL_STATE,
)
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.core.verify.executable_pattern import TEXT_KEYS
from fsm_llm.core.verify.validator import FSMValidator
from tests.test_fsm_llm import kind_corpus as kc

_ROOT = Path(__file__).resolve().parents[2]
_ALL = lambda category, name: True  # noqa: E731  every name resolves


def _fsm() -> dict[str, Any]:
    start = kc._state(
        "start",
        response_instructions="Say hello {{tone}}.",
        transitions=[
            kc._go(
                "done",
                priority={"$hole": "priority"},
                conditions=kc._when(
                    "n", {"==": [{"var": "n"}, {"$hole": "limit"}]}, "n reached"
                ),
            )
        ],
    )
    return kc._fsm("Pat", "start", start, kc._state("done"))


_APPROVAL_VARIANT = [
    {
        "op": "add",
        "path": f"/states/{APPROVAL_STATE}",
        "value": kc._state(
            APPROVAL_STATE, transitions=[kc._go("done", priority={"$hole": "priority"})]
        ),
    },
    {
        "op": "replace",
        "path": "/states/start/transitions/0/target_state",
        "value": APPROVAL_STATE,
    },
]


def _body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "fsm": _fsm(),
        "options_schema": {
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "priority": {"type": "integer", "minimum": 1, "maximum": 900},
            }
        },
        "options": {"limit": 3, "priority": 100},
    }
    body.update(over)
    return body


def _doc(**over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": "fsm-llm/executable",
        "kind": "pattern",
        "id": "demo/pattern",
        "variables": ["tone"],
        "body": _body(),
    }
    doc.update(over)
    return doc


def _lines(result, tag: str, severity: str = "errors") -> list[str]:
    return [m for m in getattr(result, severity) if m.startswith(f"[{tag}]")]


def _loads(doc: object, **kw: Any):
    envelope, result = load_executable(doc, **kw)
    assert envelope is not None and result.is_valid, result.errors
    return envelope


def _refused(doc: object, tag: str, needle: str, **kw: Any) -> str:
    """The document is refused with exactly one line of *tag* holding *needle*."""
    envelope, result = load_executable(doc, **kw)
    assert envelope is None and not result.is_valid
    lines = _lines(result, tag)
    assert len(lines) == 1, result.errors
    assert needle in lines[0], lines[0]
    return lines[0]


class TestV2LoadsTheDefaultInstance:
    def test_a_hole_in_json_logic_loads_on_its_default_instance(self):
        template = _fsm()
        assert not FSMValidator(template).validate().is_valid  # the hole is no operand

        envelope = _loads(_doc())

        assert envelope.kind == "pattern"

    def test_the_default_instance_is_what_the_body_check_reads(self):
        doc = _doc()
        doc["body"]["options"]["limit"] = 3
        doc["body"]["fsm"]["states"]["start"]["required_context_keys"] = {
            "$hole": "limit"
        }  # an integer where a list belongs

        _, result = load_executable(doc)

        assert not result.is_valid
        assert _lines(result, "body"), result.errors

    def test_a_declared_variable_is_left_as_written_without_values_and_filled_with_them(
        self,
    ):
        _, bare = load_executable(_doc())
        _, filled = load_executable(_doc(), variables={"tone": "calmly"})

        assert bare.is_valid and filled.is_valid
        assert any("not filled" in m for m in bare.info)
        assert not any("not filled" in m for m in filled.info)

    def test_a_bad_value_set_refuses_once_and_builds_nothing(self):
        envelope, result = load_executable(_doc(), variables={"other": "x"})

        assert envelope is None
        assert _lines(result, "variables")
        assert not _lines(result, "body")

    def test_an_option_with_no_default_is_one_body_line(self):
        doc = _doc()
        del doc["body"]["options"]["limit"]

        line = _refused(doc, "body", "have no value and no default")

        assert "'limit'" in line and "default instance" in line

    def test_an_instance_with_every_variant_applied_is_checked_too(self):
        doc = _doc()
        doc["body"]["variants"] = {
            "broken": [
                {
                    "op": "replace",
                    "path": "/states/start/transitions/0/target_state",
                    "value": "nowhere",
                }
            ]
        }

        _, result = load_executable(doc)

        assert not result.is_valid
        assert any(
            m.startswith("[body] with every variant applied:") for m in result.errors
        ), result.errors

    def test_a_default_instance_error_is_not_said_twice(self):
        doc = _doc()
        doc["body"]["fsm"]["states"]["orphan"] = kc._state("orphan")
        doc["body"]["variants"] = {
            "noop": [{"op": "test", "path": "/name", "value": "Pat"}]
        }

        _, result = load_executable(doc)

        assert not any("with every variant applied" in m for m in result.errors)
        assert any("orphan" in m for m in _lines(result, "body"))

    def test_a_variant_patch_that_is_refused_is_one_body_line(self):
        doc = _doc()
        doc["body"]["variants"] = {"bad": [{"op": "remove", "path": "/states/missing"}]}

        line = _refused(doc, "body", "cannot be built")

        assert "'bad'" in line and "/states/missing" not in line

    def test_a_good_variant_is_valid_and_the_variants_are_never_checked_alone(self):
        doc = _doc()
        doc["body"]["variants"] = {"approval": _APPROVAL_VARIANT}
        doc["body"]["handlers"] = [
            {
                "handler": APPROVAL_GATE_HANDLER,
                "on": "entry",
                "states": [APPROVAL_STATE],
            }
        ]
        doc["capabilities"] = ["hitl"]

        _loads(doc, resolver=_ALL)

    def test_a_generator_body_has_nothing_to_check_and_needs_a_resolver(self):
        doc = _doc(body={"generator": "agents.prompt_chain"})

        line = _refused(doc, "names", "no resolver was given")
        assert "'agents.prompt_chain'" in line
        envelope, result = load_executable(doc, resolver=_ALL)
        assert envelope is not None and result.is_valid
        assert any("generator body" in m for m in result.info)

    def test_the_validators_result_for_a_template_is_the_loaders(
        self, tmp_path, capsys
    ):
        from fsm_llm.core.verify.cli import main

        good = tmp_path / "good.json"
        good.write_text(json.dumps(_doc()), encoding="utf-8")

        assert main(str(good)) == 0
        assert json.loads(capsys.readouterr().out)["is_valid"] is True


class TestV3Names:
    def _bound(self) -> dict[str, Any]:
        doc = _doc()
        doc["requires"] = {"handlers": ["agents.limiter"]}
        doc["body"]["handlers"] = [
            {"handler": "agents.limiter", "on": "state", "states": ["start"]},
            {"handler": "agents.recorder", "on": "entry", "states": ["done"]},
        ]
        doc["body"]["run"] = {"hooks": [{"hook": "agents.after", "at": "after_run"}]}
        return doc

    def test_a_template_with_no_name_loads_with_no_resolver(self):
        _loads(_doc())

    def test_a_binding_with_no_resolver_is_refused_by_name(self):
        line = _refused(self._bound(), "names", "no resolver was given")

        assert "and 2 more" in line

    def test_each_name_goes_through_the_resolver_once_by_category(self):
        calls: list[tuple[str, str]] = []

        def resolver(category: str, name: str) -> bool:
            calls.append((category, name))
            return True

        doc = self._bound()
        doc["body"]["generator"] = None
        envelope, result = load_executable(doc, resolver=resolver)

        assert envelope is not None and result.is_valid, result.errors
        assert sorted(calls) == [
            ("handlers", "agents.after"),
            ("handlers", "agents.limiter"),
            ("handlers", "agents.recorder"),
        ]

    def test_a_name_the_resolver_does_not_know_is_refused_by_category(self):
        known = {("handlers", "agents.limiter"), ("handlers", "agents.recorder")}

        line = _refused(
            self._bound(),
            "names",
            "handlers name 'agents.after' does not resolve",
            resolver=lambda c, n: (c, n) in known,
        )

        assert line.startswith("[names]")

    def test_a_generator_goes_through_the_generators_category(self):
        seen = []
        _, result = load_executable(
            _doc(body={"generator": "g.one"}),
            resolver=lambda c, n: seen.append((c, n)) or False,
        )

        assert seen == [("generators", "g.one")]
        assert any(
            "generators name 'g.one' does not resolve" in m for m in result.errors
        )

    def test_the_tools_of_an_instance_are_checked_against_requires_tools(self):
        doc = _doc(
            body={"fsm": kc._tool_fsm(), "options_schema": {}},
            requires={"tools": ["add"]},
            variables=[],
        )

        line = _refused(doc, "names", "completion tools")

        assert "'lookup'" in line


def _gate_doc(*, claim: bool, bound: bool, state: str) -> dict[str, Any]:
    """state: 'base' (in the template), 'variant' (added by a variant) or 'none'."""
    doc = _doc()
    body = doc["body"]
    if state == "base":
        body["fsm"]["states"][APPROVAL_STATE] = kc._state(
            APPROVAL_STATE, transitions=[kc._go("done", priority=10)]
        )
        body["fsm"]["states"]["start"]["transitions"][0]["target_state"] = (
            APPROVAL_STATE
        )
    elif state == "variant":
        body["variants"] = {"approval": copy.deepcopy(_APPROVAL_VARIANT)}
    if bound:
        body["handlers"] = [
            {
                "handler": APPROVAL_GATE_HANDLER,
                "on": "entry",
                "states": [APPROVAL_STATE],
            }
        ]
    if claim:
        doc["capabilities"] = ["hitl"]
    return doc


class TestTheHitlRule:
    @pytest.mark.parametrize("state", ["base", "variant"])
    def test_a_claim_a_gate_and_a_state_agree(self, state):
        _loads(_gate_doc(claim=True, bound=True, state=state), resolver=_ALL)

    def test_no_claim_no_gate_and_no_state_agree(self):
        _loads(_gate_doc(claim=False, bound=False, state="none"))

    def test_a_hitl_claim_with_no_gate_binding_is_refused(self):
        doc = _gate_doc(claim=True, bound=False, state="none")

        _refused(doc, "names", "claims hitl")
        # the claim never lifts the rule, with or without a state
        _, result = load_executable(
            _gate_doc(claim=True, bound=False, state="base"), resolver=_ALL
        )
        assert any("claims hitl" in m for m in result.errors)

    def test_a_gate_bound_with_no_claim_is_refused(self):
        _refused(
            _gate_doc(claim=False, bound=True, state="base"),
            "names",
            "does not claim hitl",
            resolver=_ALL,
        )

    def test_a_gate_with_no_await_approval_state_is_refused(self):
        _refused(
            _gate_doc(claim=True, bound=True, state="none"),
            "names",
            f"no instance has an {APPROVAL_STATE!r} state",
            resolver=_ALL,
        )

    @pytest.mark.parametrize("state", ["base", "variant"])
    def test_an_await_approval_state_with_no_gate_is_refused(self, state):
        _refused(
            _gate_doc(claim=False, bound=False, state=state),
            "names",
            "is not bound in body.handlers",
        )

    def test_a_gate_bound_for_a_variant_alone_still_needs_the_state_somewhere(self):
        doc = _gate_doc(claim=True, bound=True, state="variant")
        doc["body"]["variants"]["approval"] = [
            {"op": "test", "path": "/name", "value": "Pat"}
        ]

        _refused(doc, "names", "no instance has", resolver=_ALL)

    def test_the_names_are_the_ones_the_agents_package_uses(self):
        from fsm_llm.agents.constants import AgentStates

        assert APPROVAL_STATE == AgentStates.AWAIT_APPROVAL

    def test_an_operator_and_a_model_document_meet_the_same_rule(self):
        doc = _gate_doc(claim=True, bound=False, state="none")

        for origin in ("operator", "model"):
            _, result = load_executable(doc, origin=origin, resolver=_ALL)
            assert any("claims hitl" in m for m in result.errors)


class TestV4TextChannel:
    def test_an_undeclared_variable_in_a_template_is_an_error(self):
        doc = _doc()
        doc["body"]["fsm"]["states"]["start"]["purpose"] = "Begin {{mood}}."

        line = _refused(doc, "variables", "{{mood}} is not a declared variable")

        assert "only a declared variable" in line

    def test_the_same_text_in_an_fsm_executable_stays_a_warning(self):
        fsm = kc._fsm("F", "a", kc._state("a", purpose="Begin {{mood}}."))
        _, result = load_executable(
            {
                "schema": "fsm-llm/executable",
                "kind": "fsm",
                "id": "plain",
                "body": {"fsm": fsm},
            }
        )

        assert result.is_valid
        assert any("{{mood}} is not a declared variable" in m for m in result.warnings)

    def test_a_caller_input_in_a_variant_is_refused(self):
        doc = _doc(contract={"inputs": ["task"]})
        doc["body"]["variants"] = {
            "x": [
                {
                    "op": "replace",
                    "path": "/states/start/purpose",
                    "value": "Work on {{task}}.",
                }
            ]
        }

        _refused(doc, "inputs", "{{task}}, a caller input")

    @pytest.mark.parametrize("key", TEXT_KEYS)
    def test_a_free_text_hole_under_a_text_key_is_refused(self, key):
        doc = _doc()
        doc["body"]["options_schema"]["properties"]["text"] = {"type": "string"}
        doc["body"]["options"]["text"] = "hello"
        state = doc["body"]["fsm"]["states"]["done"]
        if key == "persona":
            doc["body"]["fsm"]["persona"] = {"$hole": "text"}
        elif (
            key == "instructions"
        ):  # a field extraction carries `extraction_instructions`
            state["field_extractions"] = [
                {
                    "field_name": "f",
                    "field_type": "str",
                    "instructions": {"$hole": "text"},
                }
            ]
        else:
            state[key] = {"$hole": "text"}

        line = _refused(doc, "inputs", f"sits under {key!r} in fsm")

        assert "'text'" in line and "declared {{variable}}" in line

    def test_a_hole_in_a_variant_op_value_is_judged_by_the_path(self):
        doc = _doc()
        doc["body"]["options_schema"]["properties"]["text"] = {"type": "string"}
        doc["body"]["options"]["text"] = "hello"
        doc["body"]["variants"] = {
            "swap": [
                {
                    "op": "replace",
                    "path": "/states/start/response_instructions",
                    "value": {"$hole": "text"},
                }
            ]
        }

        _refused(doc, "inputs", "variant 'swap'")

    def test_other_holes_and_listed_strings_may_sit_there_or_anywhere(self):
        doc = _doc()
        props = doc["body"]["options_schema"]["properties"]
        props["mode"] = {"type": "string", "enum": ["calm", "brisk"]}
        props["target"] = {"type": "string"}
        props["names"] = {"type": "array", "items": {"type": "string"}}
        doc["body"]["options"].update(mode="calm", target="done", names=["n"])
        states = doc["body"]["fsm"]["states"]
        states["done"]["purpose"] = {"$hole": "mode"}  # a listed vocabulary
        states["start"]["transitions"][0]["target_state"] = {"$hole": "target"}
        states["start"]["required_context_keys"] = {"$hole": "names"}

        _loads(doc)


_REFUSED_FIELDS: dict[str, tuple[Any, str]] = {
    "handlers": (
        lambda b: b.update(
            handlers=[{"handler": "evil.exec", "on": "entry", "states": ["start"]}]
        ),
        "body.handlers binds 'evil.exec'",
    ),
    "hooks": (
        lambda b: b.update(run={"hooks": [{"hook": "evil.hook", "at": "before_run"}]}),
        "body.run.hooks binds 'evil.hook'",
    ),
    "variants": (
        lambda b: b.update(
            variants={"v": [{"op": "replace", "path": "/description", "value": "x"}]}
        ),
        "body.variants is set",
    ),
    "generator": (
        lambda b: (b.clear(), b.update(generator="g.evil")),
        "body.generator names 'g.evil'",
    ),
    "options_schema": (
        lambda b: b.update(
            options_schema={"properties": {"knob": {"type": "integer"}}},
            options={"knob": 1},
        ),
        "body.options_schema declares options",
    ),
    "agent_type": (
        lambda b: b.update(run={"agent_type": "react"}),
        "body.run.agent_type names 'react'",
    ),
    "run_output_keys": (
        lambda b: b.update(run={"run_output_keys": ["final_answer"]}),
        "body.run.run_output_keys is set",
    ),
    "context_seed": (
        lambda b: b.update(run={"context_seed": {"k": 1}}),
        "body.run.context_seed is set",
    ),
}


def _plain_doc() -> dict[str, Any]:
    """A pattern with no option and no hole: the control a model could write."""
    fsm = kc._fsm(
        "Plain",
        "start",
        kc._state("start", transitions=[kc._go("done")]),
        kc._state("done"),
    )
    return _doc(variables=[], body={"fsm": fsm})


def _with_field(field: str, **header: Any) -> dict[str, Any]:
    doc = _plain_doc()
    _REFUSED_FIELDS[field][0](doc["body"])
    doc.update(header)
    return doc


class TestV7RefusesWhatBindsCode:
    def test_a_plain_pattern_a_model_wrote_loads(self):
        envelope, result = load_executable(_plain_doc(), origin="model")

        assert envelope is not None and result.is_valid, result.errors

    @pytest.mark.parametrize("field", sorted(_REFUSED_FIELDS))
    def test_a_model_document_with_the_field_is_refused_and_an_operator_one_loads(
        self, field
    ):
        doc = _with_field(field)

        envelope, result = load_executable(doc, origin="model", resolver=_ALL)
        lines = _lines(result, "model")

        assert envelope is None and not result.is_valid
        assert len(lines) == 1 and _REFUSED_FIELDS[field][1] in lines[0], lines
        operator, report = load_executable(doc, origin="operator", resolver=_ALL)
        assert operator is not None and report.is_valid, report.errors

    @pytest.mark.parametrize("field", sorted(_REFUSED_FIELDS))
    @pytest.mark.parametrize(
        "claims",
        [
            {"requires": {"tools": "none"}},
            {"requires": {"tools": []}},
            {"requires": {"tools": "none"}, "lifecycle": "draft", "capabilities": []},
            {"contract": {"inputs": ["k"], "outputs": ["k"]}},
            {"requires": {"tools": "none"}, "capabilities": ["tools"]},
        ],
        ids=["none", "empty", "all-quiet", "contract", "tools-claim"],
    )
    def test_no_claim_of_the_document_lifts_the_refusal(self, field, claims):
        doc = _with_field(field, **claims)

        _, result = load_executable(doc, origin="model", resolver=_ALL)

        assert any(_REFUSED_FIELDS[field][1] in m for m in _lines(result, "model")), (
            result.errors
        )

    def test_a_handler_listed_in_requires_is_refused_without_a_binding(self):
        doc = _plain_doc()
        doc["requires"] = {"handlers": ["evil.exec"]}

        _refused(
            doc,
            "model",
            "requires.handlers names 'evil.exec'",
            origin="model",
            resolver=_ALL,
        )

    def test_a_grant_gate_is_no_exception_for_a_model(self):
        doc = _gate_doc(claim=True, bound=True, state="base")
        doc["variables"] = []
        doc["body"]["fsm"]["states"]["start"].pop("response_instructions")
        doc["body"]["options"] = {}
        doc["body"]["options_schema"] = {}
        del doc["body"]["fsm"]["states"]["start"]["transitions"][0]["priority"]
        del doc["body"]["fsm"]["states"]["start"]["transitions"][0]["conditions"]
        del doc["body"]["fsm"]["states"][APPROVAL_STATE]["transitions"][0]["priority"]

        _, result = load_executable(doc, origin="model", resolver=_ALL)

        assert any(
            f"body.handlers binds {APPROVAL_GATE_HANDLER!r}" in m
            for m in _lines(result, "model")
        ), result.errors

    def test_a_model_document_with_a_non_draft_lifecycle_is_refused_as_before(self):
        doc = _plain_doc()
        doc["lifecycle"] = "verified"

        _refused(doc, "model", "lifecycle", origin="model")

    def test_a_name_in_a_refusal_is_capped_and_stripped(self):
        doc = _with_field("handlers")
        doc["body"]["handlers"][0]["handler"] = "evil\x1b[31m" + "x" * 100

        _, result = load_executable(doc, origin="model", resolver=_ALL)

        line = _lines(result, "model")[0]
        assert "\x1b" not in line and len(line) < 400


class TestTheValidatorIsNeverStricterThanTheLoader:
    @pytest.fixture(autouse=True)
    def _keep_logging_state(self, monkeypatch):
        import fsm_llm.core.verify.cli as cli_module
        from fsm_llm.logging import logger

        monkeypatch.setattr(cli_module, "setup_cli_logging", lambda level: 0)
        yield
        logger.disable("fsm_llm")

    def _cases(self) -> dict[str, dict[str, Any]]:
        bound = _doc()
        bound["body"]["handlers"] = [
            {"handler": "agents.limiter", "on": "state", "states": ["start"]}
        ]
        missing_default = _doc()
        del missing_default["body"]["options"]["limit"]
        return {
            "template": _doc(),
            "plain": _plain_doc(),
            "bound": bound,
            "gate": _gate_doc(claim=True, bound=True, state="base"),
            "claim only": _gate_doc(claim=True, bound=False, state="none"),
            "no default": missing_default,
            "generator": _doc(body={"generator": "g.one"}),
        }

    def test_the_cli_agrees_with_the_loader_on_every_document(self, tmp_path, capsys):
        from fsm_llm.core.verify.cli import main

        for name, doc in self._cases().items():
            path = tmp_path / f"{name.replace(' ', '_')}.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            loader = load_executable(doc)[1]

            code = main(str(path))
            report = json.loads(capsys.readouterr().out)

            assert (code == 0) == loader.is_valid, name
            assert report["is_valid"] is loader.is_valid, name

    def test_nothing_the_cli_refuses_for_a_name_loads_through_the_loader_by_default(
        self,
    ):
        # the loader and the CLI call one function with no resolver: a binding is refused by both
        for name in ("bound", "generator"):
            envelope, _ = load_executable(self._cases()[name])
            assert envelope is None, name


class TestNoRouteBuildsAnExecutableFromAPatternWithoutAnAdapter:
    """Step 6 pinned "no registry serves a pattern yet"; since plan a555dcfe step 9 the agents
    package registers the ``pattern`` kind, so the fail-closed row asks a registry with none."""

    def test_a_registry_with_no_adapter_for_the_kind_refuses_the_document(self):
        from fsm_llm.core.names import NameRegistry
        from fsm_llm.engine.executables.protocol import ExecutableKindError
        from fsm_llm.engine.executables.registry import ExecutableRegistry

        registry = ExecutableRegistry(kinds=NameRegistry("executable_kinds"))

        with pytest.raises(ExecutableKindError):
            registry.add_document(_doc(), variables={"tone": "x"})
        assert registry.resolve_target("demo/pattern") is None


def test_the_pattern_checks_import_no_engine_and_call_no_model():
    path = _ROOT / "src" / "fsm_llm" / "core" / "verify" / "executable_pattern.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    forbidden = ("fsm_llm.engine", "fsm_llm.llm", "fsm_llm.agents", "fsm_llm.synthesis")
    assert [m for m in imported if m.startswith((*forbidden, "litellm"))] == []
