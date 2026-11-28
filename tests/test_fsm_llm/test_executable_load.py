"""The load sequence V1 to V5 and V7, and ``fsm-llm-validate`` reading executables (plan ace7a8c8, C8).

``load_executable`` reports every failure of a document in one call, tagged ``[schema] [body]
[variables] [inputs] [contract] [names] [model]``. The built-ins of the corpus are the real
documents: they must load valid, with their contract findings as warnings only. The module is lazy:
``import fsm_llm`` and ``fsm-llm-validate`` on an FSM file never load it.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.definitions import executable as ex
from fsm_llm.core.definitions import executable_instance as ei
from fsm_llm.core.state_kinds import declared_tool_names
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.core.verify.flow import analyze_data_flow, analyze_liveness
from tests import executables_corpus as corpus
from tests.test_fsm_llm import kind_corpus as kc

_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "src" / "fsm_llm" / "core" / "verify" / "executable.py"
_CASES = corpus.fsm_cases()
_ALLOW = {"tools": ["search"], "callables": [], "executables": []}


def _fsm(**state_a: Any) -> dict[str, Any]:
    return kc._fsm(
        "Mini",
        "a",
        kc._state("a", transitions=[kc._go("b")], **state_a),
        kc._state("b"),
    )


def _doc(fsm: dict[str, Any] | None = None, **header: Any) -> dict[str, Any]:
    return {
        "schema": "fsm-llm/executable",
        "kind": "fsm",
        "id": "mini",
        "body": {"fsm": fsm if fsm is not None else _fsm()},
        **header,
    }


def _tagged(result, tag: str) -> list[str]:
    return [m for m in (*result.errors, *result.warnings) if m.startswith(f"[{tag}]")]


def _errors(result, tag: str) -> list[str]:
    return [m for m in result.errors if m.startswith(f"[{tag}]")]


class TestEveryBuiltInLoads:
    @pytest.mark.parametrize("case", sorted(_CASES))
    def test_it_is_valid_and_its_contract_findings_are_warnings(self, case):
        document = corpus.wrap(case, _CASES[case])

        envelope, result = load_executable(document)

        assert result.is_valid, result.errors
        assert envelope is not None and envelope.id == case
        definition = FSMDefinition(**_CASES[case])
        findings = [*analyze_data_flow(definition), *analyze_liveness(definition)]
        assert len(_tagged(result, "contract")) == len(findings)
        assert all(w.startswith("[contract] ") for w in _tagged(result, "contract"))

    def test_the_contract_analysis_is_not_dropped(self):
        total = sum(
            len(_tagged(load_executable(corpus.wrap(c, r))[1], "contract"))
            for c, r in _CASES.items()
        )

        assert (
            total > 0
        )  # the built-ins do give findings: a loader that drops V5 is seen

    def test_without_a_resolver_the_names_are_not_resolved_and_it_says_so_once(self):
        _, result = load_executable(_doc())

        assert [i for i in result.info if i.startswith("[names]")] == [
            "[names] names are not resolved: no resolver was given"
        ]


class TestAllFailuresAtOnce:
    def _bad_document(self) -> dict[str, Any]:
        fsm = kc._tool_fsm()
        fsm["states"]["orphan"] = kc._state("orphan")  # no edge reaches it
        fsm["states"]["done"]["response_instructions"] = (
            "Answer in a {{tone}} tone about {{task}}."
        )
        return _doc(
            fsm,
            contract={"inputs": ["task"]},
            variables=["tone"],
            requires={"tools": ["something_else"], "executables": ["ghost"]},
        )

    def test_a_document_with_five_defects_reports_all_five_tagged(self):
        envelope, result = load_executable(
            self._bad_document(),
            variables={},
            resolver=lambda category, name: name != "ghost",
        )

        assert envelope is None and not result.is_valid
        assert _errors(result, "body"), "the orphaned state"
        assert any("tone" in m for m in _errors(result, "variables"))
        assert any("task" in m for m in _errors(result, "inputs"))
        names = _errors(result, "names")
        tools = sorted(
            declared_tool_names(FSMDefinition(**kc._tool_fsm()).states.values())
        )
        assert any(tools[0] in m and "requires.tools" in m for m in names)
        assert any("executables" in m and "ghost" in m for m in names)
        assert _tagged(result, "schema") == []

    def test_a_schema_failure_ends_the_call(self):
        envelope, result = load_executable(_doc(kind="bogus"))

        assert envelope is None
        assert result.errors and all(m.startswith("[schema] ") for m in result.errors)

    def test_a_valid_document_returns_its_envelope(self):
        envelope, result = load_executable(_doc())

        assert result.is_valid and isinstance(envelope, ex.FsmEnvelope)

    def test_the_tool_check_reads_a_body_that_does_not_load(self):
        fsm = kc._tool_fsm()
        fsm["states"]["orphan"] = kc._state("orphan")

        _, result = load_executable(_doc(fsm, requires={"tools": ["x"]}))

        assert _errors(result, "body") and _errors(result, "names")

    @pytest.mark.parametrize("tools", ["none", "some"])
    def test_an_operator_bare_tools_claim_skips_the_name_list_check(self, tools):
        # the operator path is unchanged (D-053); a MODEL document is judged by V7 instead,
        # whatever it claims: see TestModelOrigin
        _, result = load_executable(_doc(kc._tool_fsm(), requires={"tools": tools}))

        assert _errors(result, "names") == []


class TestVariables:
    def test_no_variables_declared_changes_nothing(self):
        envelope, result = load_executable(
            corpus.wrap("agents/react", _CASES["agents/react"])
        )

        assert result.is_valid
        definition, _ = ei.instantiate_envelope(envelope)
        assert (
            ei.identity_of(envelope, definition).instance_identity
            == json.loads(corpus.INDEX_PATH.read_text(encoding="utf-8"))[
                "agents/react"
            ]["digest64"]
        )

    def test_a_template_is_checked_as_written_when_no_values_are_given(self):
        fsm = _fsm(response_instructions="Greet in a {{tone}} tone.")

        envelope, result = load_executable(_doc(fsm, variables=["tone"]))

        assert envelope is not None and result.is_valid
        assert any("tone" in i and i.startswith("[variables]") for i in result.info)

    def test_values_fill_the_body_before_the_body_is_checked(self):
        # the variable is the initial state id: only the filled body loads
        fsm = _fsm()
        fsm["initial_state"] = "{{first}}"

        assert not load_executable(_doc(fsm, variables=["first"]))[1].is_valid
        envelope, result = load_executable(
            _doc(fsm, variables=["first"]), variables={"first": "a"}
        )
        assert result.is_valid and envelope is not None

    def test_missing_extra_and_non_text_values_are_errors(self):
        document = _doc(
            _fsm(response_instructions="{{tone}} {{topic}}"),
            variables=["tone", "topic"],
        )

        _, missing = load_executable(document, variables={"tone": "a"})
        _, extra = load_executable(
            document, variables={"tone": "a", "topic": "b", "color": "c"}
        )
        _, bad = load_executable(document, variables={"tone": "a", "topic": 3})

        assert any("topic" in m for m in _errors(missing, "variables"))
        assert any("color" in m for m in _errors(extra, "variables"))
        assert any("topic" in m for m in _errors(bad, "variables"))

    def test_a_value_with_a_placeholder_is_inserted_once_and_not_rescanned(self):
        document = _doc(
            _fsm(response_instructions="Use {{tone}}."), variables=["tone", "topic"]
        )

        envelope, result = load_executable(
            document, variables={"tone": "{{topic}}", "topic": "X"}
        )

        assert result.is_valid
        definition, _ = ei.instantiate_envelope(
            envelope, variables={"tone": "{{topic}}", "topic": "X"}
        )
        assert definition.states["a"].response_instructions == "Use {{topic}}."

    def test_a_placeholder_in_a_key_is_left_alone_and_warned(self):
        fsm = _fsm()
        fsm["notes"] = {"{{tone}}": "x"}  # an unknown top-level key: a key, not a leaf

        _, result = load_executable(
            _doc(fsm, variables=["tone"]), variables={"tone": "calm"}
        )

        assert any(
            "key" in m and "tone" in m
            for m in result.warnings
            if m.startswith("[variables]")
        )

    def test_an_undeclared_token_is_a_warning(self):
        fsm = _fsm(response_instructions="Say {{mystery}}.")

        _, result = load_executable(_doc(fsm))

        assert result.is_valid
        assert any("mystery" in m for m in _tagged(result, "variables"))


class TestInputsNeverRenderIntoInstructions:
    def test_an_input_name_used_as_a_placeholder_is_an_error(self):
        fsm = _fsm(response_instructions="Work on {{task}} now.")

        envelope, result = load_executable(_doc(fsm, contract={"inputs": ["task"]}))

        assert envelope is None
        assert any("task" in m for m in _errors(result, "inputs"))

    def test_an_input_template_placeholder_is_an_error_too(self):
        fsm = _fsm(response_instructions="Review {{file}}.")
        contract = {"inputs": ["task"], "input_templates": {"task": "Review {{file}}"}}

        _, result = load_executable(_doc(fsm, contract=contract))

        assert any("file" in m for m in _errors(result, "inputs"))

    def test_a_clean_body_with_templates_loads(self):
        contract = {"inputs": ["task"], "input_templates": {"task": "Review {{file}}"}}

        envelope, result = load_executable(_doc(contract=contract))

        assert envelope is not None and result.is_valid


class TestNames:
    def test_the_resolver_gets_category_and_name_pairs_and_is_honoured(self):
        calls: list[tuple[str, str]] = []

        def resolver(category: str, name: str) -> bool:
            calls.append((category, name))
            return name != "nope"

        requires = {
            "tools": ["t1"],
            "handlers": ["h1", "nope"],
            "callables": ["c1"],
            "executables": ["e1"],
        }
        _, result = load_executable(_doc(requires=requires), resolver=resolver)

        assert sorted(calls) == [
            ("callables", "c1"),
            ("executables", "e1"),
            ("handlers", "h1"),
            ("handlers", "nope"),
            ("tools", "t1"),
        ]
        assert [m for m in _errors(result, "names") if "nope" in m]
        assert len(_errors(result, "names")) == 1
        assert not any(
            i.startswith("[names] names are not resolved") for i in result.info
        )

    def test_a_bare_tools_claim_has_no_names_to_resolve(self):
        calls: list[tuple[str, str]] = []

        load_executable(
            _doc(requires={"tools": "some"}),
            resolver=lambda c, n: calls.append((c, n)) or True,
        )

        assert calls == []


class TestNeverRaises:
    @pytest.mark.parametrize(
        "data", [None, [], "x", 7, {}, {"schema": "fsm-llm/executable"}]
    )
    def test_a_non_document_is_invalid(self, data):
        envelope, result = load_executable(data)

        assert envelope is None and not result.is_valid
        assert all(m.startswith("[schema] ") for m in result.errors)

    def test_a_deep_document_is_invalid(self):
        deep: dict[str, Any] = {}
        cursor = deep
        for _ in range(100_000):
            cursor["a"] = {}
            cursor = cursor["a"]

        envelope, result = load_executable(_doc(metadata=deep))

        assert envelope is None and not result.is_valid

    def test_a_non_dict_mapping_root_with_a_deep_body_is_invalid_not_a_crash(self):
        deep: Any = "x"
        for _ in range(5_000):
            deep = [deep]
        data = types.MappingProxyType(_doc(_fsm(notes=deep)))

        envelope, result = load_executable(data, variables={})

        assert envelope is None and not result.is_valid
        assert all(m.startswith("[schema] ") for m in result.errors)

    def test_a_document_that_breaks_v1_shows_no_raw_text(self):
        data = _doc(**{"evil\nkey\u202e": 1}, kind="y" * 600 + "\u202e")

        _, result = load_executable(data)

        assert result.errors
        for message in result.errors:
            assert len(message) <= 320 and "\u202e" not in message
            assert all(ch.isprintable() for ch in message)

    def test_an_unknown_origin_is_a_programming_error(self):
        with pytest.raises(ValueError, match="origin"):
            load_executable(_doc(), origin="robot")


class TestModelOrigin:
    def _hostile(self) -> dict[str, Any]:
        return _doc(
            lifecycle="trusted",
            requires={"tools": ["search", "rm_rf"], "handlers": ["bind_everything"]},
        )

    def test_three_defects_are_reported_together(self):
        envelope, result = load_executable(
            self._hostile(), origin="model", allow=_ALLOW
        )

        assert envelope is None
        model = _errors(result, "model")
        assert len(model) == 3, model
        assert any("rm_rf" in m for m in model)
        assert any("trusted" in m for m in model)
        assert any("handler" in m for m in model)
        assert not any("'search'" in m for m in model)  # the listed tool is fine

    def test_the_same_document_from_an_operator_raises_no_model_error(self):
        envelope, result = load_executable(self._hostile(), allow=_ALLOW)

        assert _tagged(result, "model") == []
        assert envelope is not None and result.is_valid

    def test_a_clean_model_document_loads(self):
        document = _doc(requires={"tools": ["search"]})

        envelope, result = load_executable(document, origin="model", allow=_ALLOW)

        assert envelope is not None and result.is_valid
        assert envelope.lifecycle.value == "draft"

    def test_without_an_allow_list_every_name_is_refused(self):
        _, result = load_executable(
            _doc(requires={"tools": ["search"]}), origin="model"
        )

        assert any("search" in m for m in _errors(result, "model"))

    def test_a_bare_some_cannot_be_checked_and_is_refused(self):
        _, result = load_executable(
            _doc(requires={"tools": "some"}), origin="model", allow=_ALLOW
        )

        assert any("some" in m for m in _errors(result, "model"))

    @pytest.mark.parametrize(
        "lifecycle", ["verified", "trusted", "builtin", "deprecated"]
    )
    def test_a_model_may_only_claim_draft(self, lifecycle):
        _, result = load_executable(_doc(lifecycle=lifecycle), origin="model", allow={})

        assert any(lifecycle in m for m in _errors(result, "model"))

    @staticmethod
    def _call_doc(call: dict[str, Any], **header: Any) -> dict[str, Any]:
        return _doc(kc._call_fsm(call), **header)

    def test_a_call_target_off_the_allow_list_is_refused(self):
        envelope, result = load_executable(
            self._call_doc({"target": "evil_exec"}), origin="model", allow=_ALLOW
        )

        assert envelope is None
        assert any("evil_exec" in m for m in _errors(result, "model"))
        assert all(m.startswith("[model] ") for m in _errors(result, "model"))

    def test_every_target_of_a_choose_list_is_judged_and_the_listed_ones_pass(self):
        call = {
            "choose": {"key": "picked", "targets": ["ok", "evil_a", "evil_b"]},
        }
        _, result = load_executable(
            self._call_doc(call, requires={"executables": ["ok"]}),
            origin="model",
            allow={**_ALLOW, "executables": ["ok"]},
        )

        model = _errors(result, "model")
        assert any("evil_a" in m for m in model) and any("evil_b" in m for m in model)
        assert not any("'ok'" in m and "allow-list" in m for m in model)

    def test_a_fallback_is_judged_too(self):
        call = {"choose": {"key": "picked", "targets": ["ok"], "fallback": "evil_fb"}}
        _, result = load_executable(
            self._call_doc(call, requires={"executables": ["ok"]}),
            origin="model",
            allow={**_ALLOW, "executables": ["ok"]},
        )

        assert any("evil_fb" in m for m in _errors(result, "model"))

    def test_the_body_is_judged_whatever_requires_executables_says(self):
        """An empty claim beside a body that calls an executable is a disagreement, and the
        allow-list is read from the body (the claim lists nothing to compare with)."""
        _, off_list = load_executable(
            self._call_doc({"target": "evil_exec"}, requires={"executables": []}),
            origin="model",
            allow=_ALLOW,
        )
        _, undeclared = load_executable(
            self._call_doc({"target": "ok"}, requires={"executables": []}),
            origin="model",
            allow={**_ALLOW, "executables": ["ok"]},
        )

        assert any("allow-list" in m for m in _errors(off_list, "model"))
        assert any(
            "requires.executables" in m and "'ok'" in m
            for m in _errors(undeclared, "model")
        )

    def test_a_claim_that_disagrees_with_the_body_is_refused(self):
        _, result = load_executable(
            self._call_doc({"target": "ok"}, requires={"executables": ["other"]}),
            origin="model",
            allow={**_ALLOW, "executables": ["ok", "other"]},
        )

        model = _errors(result, "model")
        assert any("requires.executables" in m and "'ok'" in m for m in model), model

    def test_an_allowed_and_declared_call_target_loads(self):
        envelope, result = load_executable(
            self._call_doc({"target": "ok"}, requires={"executables": ["ok"]}),
            origin="model",
            allow={**_ALLOW, "executables": ["ok"]},
        )

        assert _errors(result, "model") == []
        assert envelope is not None

    def test_the_same_call_from_an_operator_is_not_judged(self):
        envelope, result = load_executable(
            self._call_doc({"target": "evil_exec"}), allow=_ALLOW
        )

        assert _tagged(result, "model") == []
        assert envelope is not None and result.is_valid

    def test_a_call_target_with_no_readable_name_is_refused(self):
        call = {"choose": {"key": "picked", "targets": ["ok", 7]}}
        _, result = load_executable(
            self._call_doc(call, requires={"executables": ["ok"]}),
            origin="model",
            allow={**_ALLOW, "executables": ["ok"]},
        )

        assert any("no readable name" in m for m in _errors(result, "model"))

    def test_a_call_target_is_capped_and_stripped_in_messages(self):
        name = "\x1b[31m" + "x" * 500 + "\u202e\x00end"
        _, result = load_executable(
            self._call_doc({"target": name}), origin="model", allow=_ALLOW
        )

        for message in _errors(result, "model"):
            assert len(message) <= 320 and "\u202e" not in message
            assert all(ch.isprintable() for ch in message)
        assert any("allow-list" in m for m in _errors(result, "model"))

    def test_model_authored_names_are_capped_and_stripped_in_messages(self):
        name = "\x1b[31m" + "x" * 500 + "‮\x00end"
        _, result = load_executable(
            _doc(requires={"tools": [name[:128]], "callables": [name[:128]]}),
            origin="model",
            allow={},
        )

        model = _errors(result, "model")
        assert len(model) == 2
        for message in model:
            assert len(message) < 200
            assert "x" * 61 not in message  # a name is cut at 60 characters
            assert all(ch.isprintable() for ch in message)
            assert "\x1b" not in message and "‮" not in message

    # D-053: V7 reads the tools the BODY declares, never only what requires.tools says.
    _BODY_TOOLS = ("add", "lookup")

    @pytest.mark.parametrize("claim", ["none", "some", [], ["add"], ["add", "lookup"]])
    def test_a_tools_claim_cannot_hide_the_tools_the_body_declares(self, claim):
        envelope, result = load_executable(
            _doc(kc._tool_fsm(), requires={"tools": claim}),
            origin="model",
            allow={"tools": ["search"]},
        )

        model = _errors(result, "model")
        assert envelope is None
        for name in self._BODY_TOOLS:  # off the allow-list, named by V7 itself
            assert any(repr(name) in m for m in model), (claim, model)

    @pytest.mark.parametrize("claim", ["none", "some"])
    def test_a_bare_claim_that_disagrees_with_the_body_is_refused_even_on_the_list(
        self, claim
    ):
        envelope, result = load_executable(
            _doc(kc._tool_fsm(), requires={"tools": claim}),
            origin="model",
            allow={"tools": ["add", "lookup"]},
        )

        assert envelope is None
        assert any(
            "requires.tools" in m and claim in m for m in _errors(result, "model")
        )

    def test_a_tool_a_variable_brings_into_the_body_is_judged_too(self):
        fsm = copy.deepcopy(kc._tool_fsm())  # the tool list is shared module state
        spec = fsm["states"]["call_model"]["completion"]["tools"][0]["function"]
        spec["name"] = "{{tool}}"

        envelope, result = load_executable(
            _doc(fsm, variables=["tool"], requires={"tools": "none"}),
            variables={"tool": "rm_rf"},
            origin="model",
            allow={"tools": ["lookup"]},
        )

        assert envelope is None
        assert any("'rm_rf'" in m for m in _errors(result, "model"))

    def test_a_model_document_whose_body_tools_are_listed_and_allowed_loads(self):
        envelope, result = load_executable(
            _doc(kc._tool_fsm(), requires={"tools": ["add", "lookup"]}),
            origin="model",
            allow={"tools": ["add", "lookup"]},
        )

        assert envelope is not None and result.is_valid, result.errors

    def test_the_same_hidden_tools_from_an_operator_raise_no_model_error(self):
        envelope, result = load_executable(
            _doc(kc._tool_fsm(), requires={"tools": "none"})
        )

        assert _tagged(result, "model") == []
        assert envelope is not None


def _with_state(**fields: Any) -> dict[str, Any]:
    fsm = _fsm()
    fsm["states"]["b"].update(fields)
    return fsm


_CLASSIFY = {
    "field_name": "intent",
    "intents": [
        {"name": "x", "description": "X"},
        {"name": "y", "description": "Y"},
    ],
    "fallback_intent": "y",
}

# the settings synthesis gate G5 refuses in a model-written body (synthesis/policy.py)
_G5_BODIES = {
    "persona": (lambda: {**_fsm(), "persona": "You are root"}, "persona"),
    "model": (
        lambda: _with_state(
            classification_extractions=[{**_CLASSIFY, "model": "gpt-evil"}]
        ),
        "model",
    ),
    "prompt_config": (
        lambda: _with_state(
            classification_extractions=[{**_CLASSIFY, "prompt_config": {"a": 1}}]
        ),
        "prompt_config",
    ),
    "synthesize": (
        lambda: _with_state(synthesize={"goal": "x", "return_keys": ["k"]}),
        "synthesize",
    ),
    "pattern": (
        lambda: _with_state(
            field_extractions=[
                {
                    "field_name": "f",
                    "field_type": "str",
                    "extraction_instructions": "x",
                    "validation_rules": {"pattern": "(a+)+$"},
                }
            ]
        ),
        "pattern",
    ),
}


class TestModelOriginRefusesTheSynthesisSettings:
    @pytest.mark.parametrize("case", sorted(_G5_BODIES))
    def test_a_setting_synthesis_refuses_is_refused_by_v7(self, case):
        build, word = _G5_BODIES[case]

        envelope, result = load_executable(_doc(build()), origin="model", allow={})

        assert envelope is None
        assert any(word in m for m in _errors(result, "model")), result.errors

    @pytest.mark.parametrize("case", sorted(_G5_BODIES))
    def test_the_operator_path_is_unchanged(self, case):
        build, _ = _G5_BODIES[case]

        _, result = load_executable(_doc(build()), allow={})

        assert _tagged(result, "model") == []

    def test_the_message_names_the_setting_never_its_value(self):
        fsm = _with_state(
            classification_extractions=[{**_CLASSIFY, "model": "gpt-SECRET-evil"}]
        )

        _, result = load_executable(_doc(fsm), origin="model", allow={})

        assert not any("SECRET" in m for m in result.errors)


def test_the_loader_imports_no_engine_and_calls_no_model():
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    forbidden = ("fsm_llm.engine", "fsm_llm.llm", "fsm_llm.agents", "fsm_llm.synthesis")
    assert [m for m in imported if m.startswith((*forbidden, "litellm"))] == []


class TestFsmLlmValidateReadsExecutables:
    @pytest.fixture(autouse=True)
    def _keep_logging_state(self, monkeypatch):
        import fsm_llm.core.verify.cli as cli_module
        from fsm_llm.logging import logger

        monkeypatch.setattr(cli_module, "setup_cli_logging", lambda level: 0)
        yield
        logger.disable("fsm_llm")

    @staticmethod
    def _write(tmp_path: Path, name: str, data: Any) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_a_valid_executable_prints_the_report_and_exits_zero(
        self, tmp_path, capsys
    ):
        from fsm_llm.core.verify.cli import main

        path = self._write(
            tmp_path, "agent.txt", _doc()
        )  # the extension decides nothing

        assert main(str(path)) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["is_valid"] is True
        assert report["fsm_name"] == "mini"

    def test_an_invalid_executable_prints_every_error_and_exits_one(
        self, tmp_path, capsys
    ):
        from fsm_llm.core.verify.cli import main

        fsm = _fsm(response_instructions="Work on {{task}}.")
        fsm["states"]["orphan"] = kc._state("orphan")
        document = _doc(fsm, contract={"inputs": ["task"]})
        path = self._write(tmp_path, "bad.json", document)

        assert main(str(path)) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["is_valid"] is False
        assert any(e.startswith("[body]") for e in report["errors"])
        assert any(e.startswith("[inputs]") for e in report["errors"])

    def test_a_built_in_validates_through_the_cli(self, tmp_path, capsys):
        from fsm_llm.core.verify.cli import main

        path = self._write(
            tmp_path, "react.json", corpus.wrap("agents/react", _CASES["agents/react"])
        )

        assert main(str(path)) == 0
        assert json.loads(capsys.readouterr().out)["is_valid"] is True

    def test_an_fsm_file_takes_the_old_path(self, tmp_path, capsys):
        from fsm_llm.core.verify.cli import main

        path = self._write(tmp_path, "fsm.json", _fsm())

        assert main(str(path)) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["fsm_name"] == "Mini"

    def test_another_schema_header_is_an_fsm_file_with_the_old_message(
        self, tmp_path, capsys
    ):
        from fsm_llm.core.verify.cli import main

        path = self._write(tmp_path, "other.json", {"schema": "something/else"})

        assert main(str(path)) == 1
        assert json.loads(capsys.readouterr().out)["is_valid"] is False

    def test_upgrade_on_an_executable_fails_and_writes_nothing(
        self, tmp_path, capsys, monkeypatch
    ):
        from fsm_llm.core.verify.cli import main_cli
        from fsm_llm.logging import logger

        path = self._write(tmp_path, "agent.json", _doc())
        before = path.read_bytes()
        seen: list[str] = []
        sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        monkeypatch.setattr(
            sys, "argv", ["fsm-llm-validate", "--fsm", str(path), "--upgrade"]
        )
        try:
            with pytest.raises(SystemExit) as exit_info:
                main_cli()
        finally:
            logger.remove(sink)

        assert exit_info.value.code == 1
        assert json.loads(capsys.readouterr().out)["is_valid"] is True
        assert path.read_bytes() == before
        assert [m for m in seen if "executable" in m and "upgrade" in m.lower()]

    def test_the_help_names_executables(self, capsys, monkeypatch):
        from fsm_llm.core.verify.cli import main_cli

        monkeypatch.setattr(sys, "argv", ["fsm-llm-validate", "--help"])
        with pytest.raises(SystemExit):
            main_cli()

        assert "executable" in capsys.readouterr().out.lower()

    def test_the_cli_on_an_fsm_file_does_not_load_the_executable_modules(
        self, tmp_path
    ):
        path = self._write(tmp_path, "fsm.json", _fsm())
        code = (
            "import sys, json\n"
            "from fsm_llm.core.verify.cli import main\n"
            "main(sys.argv[1])\n"
            "print('MODS', json.dumps(sorted(m for m in sys.modules if m.endswith('.executable'))))\n"
        )

        done = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=_ROOT,
        )

        assert done.stdout.strip().splitlines()[-1] == "MODS []"
