"""The load sequence of a ``workflow`` executable: V3, V4, V5, the host-bound rule and V7 (plan 7d9c0955, D-006).

The steps of a workflow are untyped dicts in core, so every rule reads the DOCUMENT through the
one iterative walker. V3 and V7 judge the BODY: a ``requires`` claim, ``capabilities`` or
``lifecycle`` never lifts a refusal (the ace7a8c8 bug skipped V7 when ``requires.tools`` said
"none"). V7 is a closed, tag-keyed list of step types applied to every nested step; a step that
sets ``registry`` or ``callables`` is host-bound and refused in EVERY origin. These rows guard the
HITL census invariant (D1: a workflow document never carries an approval grant or a tool binding)
and the untrusted-text rule (D6: a document's text reaches a line only through ``capped_line``).

Invariants (plan 7d9c0955): 1 (no tool, handler or grant by a document), 2 (a document names
objects and holds no code; a claim is not content), 5 (a line names keys, tags and capped names,
never a value).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fsm_llm.core.definitions import executable as ex
from fsm_llm.core.definitions.executable_workflow import workflow_names
from fsm_llm.core.verify import executable_workflow as ew
from fsm_llm.core.verify.executable import load_executable
from tests import executables_corpus as corpus

_SECRET = "SECRET-VALUE-4711"
_ALLOW_ALL = {"callables": ["act", "cond", "deep", "agg"], "executables": ["tool.one"]}


def _step(tag: str, step_id: str = "a", **fields: Any) -> dict[str, Any]:
    return {"type": tag, "step_id": step_id, **fields}


def _auto(step_id: str = "a", next_step_id: str = "", **fields: Any) -> dict[str, Any]:
    return _step("AutoStep", step_id, next_step_id=next_step_id, **fields)


def _wf(
    steps: dict[str, Any] | None = None,
    *,
    initial: str = "a",
    callables: list[str] | None = None,
    executables: list[str] | None = None,
    requires: dict[str, Any] | None = None,
    **header: Any,
) -> dict[str, Any]:
    needs: dict[str, Any] = {"tools": "none"}
    if callables is not None:
        needs["callables"] = callables
    if executables is not None:
        needs["executables"] = executables
    return {
        "schema": "fsm-llm/executable",
        "kind": "workflow",
        "id": "wf",
        "body": {
            "initial_step_id": initial,
            "steps": {"a": _auto()} if steps is None else steps,
        },
        "requires": needs if requires is None else requires,
        **header,
    }


def _lines(result: Any, tag: str, severity: str = "errors") -> list[str]:
    return [m for m in getattr(result, severity) if m.startswith(f"[{tag}]")]


def _one_model_line(doc: dict[str, Any], needle: str, allow: Any = None) -> str:
    """The document is refused for a model with exactly one ``[model]`` line holding *needle*."""
    envelope, result = load_executable(
        doc, origin="model", allow=_ALLOW_ALL if allow is None else allow
    )
    assert envelope is None and not result.is_valid
    lines = _lines(result, "model")
    assert len(lines) == 1, result.errors
    assert needle in lines[0], lines[0]
    return lines[0]


def _loads(doc: dict[str, Any], **kw: Any) -> Any:
    envelope, result = load_executable(doc, **kw)
    assert envelope is not None and result.is_valid, result.errors
    return result


# ---------------------------------------------------------------- the shipped documents first


class TestTheShippedDocumentsLoad:
    @pytest.mark.parametrize("origin", ["operator", "model"])
    @pytest.mark.parametrize("name", sorted(corpus.workflow_documents()))
    def test_a_preset_shaped_document_loads_for_both_origins(self, origin, name):
        doc = corpus.workflow_documents()[name]
        allow = {"callables": doc["requires"]["callables"]}

        result = _loads(doc, origin=origin, allow=allow)

        assert result.errors == [] and result.warnings == []

    def test_the_structure_info_line_is_said_once(self):
        result = _loads(corpus.workflow_documents()["demo_linear"])

        assert _lines(result, "body", "info") == [
            "[body] workflow structure is checked when a registry loads the document"
        ]

    def test_a_workflow_with_no_names_loads_with_no_resolver(self):
        _loads(_wf())

    def test_the_golden_corpus_documents_load_except_the_host_bound_one(self):
        # the G1 documents written by serialize(callables=): the one that names a `registry` for
        # its ExecutableStep is host-bound and refused, every other loads (the relabel of step 3)
        from tests.test_fsm_llm_workflows import golden_corpus as g1

        cases = g1.snapshot()["cases"]
        verdicts = {}
        for name, case in cases.items():
            doc = json.loads(case["serialize_named"])
            names = workflow_names(doc["steps"])
            wrapped = _wf(
                doc["steps"],
                initial=doc["initial_step_id"],
                callables=list(names.callables),
                executables=list(names.executables),
            )
            envelope, _ = load_executable(wrapped)
            verdicts[name] = envelope is not None
        assert verdicts.pop("executable_step") is False
        assert verdicts and all(verdicts.values())


# ---------------------------------------------------------------- V3: the names of the body


class TestV3NamesAreReadFromTheBody:
    def test_a_callable_the_body_uses_and_requires_lacks_is_refused(self):
        doc = _wf({"a": _auto(action="act")}, callables=[])

        envelope, result = load_executable(doc)

        assert envelope is None
        assert _lines(result, "names") == [
            "[names] the body uses callables name 'act' that requires.callables lacks"
        ]

    def test_a_target_the_body_uses_and_requires_lacks_is_refused(self):
        steps = {"a": _step("ExecutableStep", target="tool.one")}

        _, result = load_executable(_wf(steps, callables=[], executables=[]))

        assert _lines(result, "names") == [
            "[names] the body uses executables name 'tool.one' that requires.executables lacks"
        ]

    def test_a_claimed_name_no_step_uses_is_a_warning_and_never_refuses(self):
        result = _loads(_wf(callables=["act"], executables=["tool.one"]))

        assert _lines(result, "names", "warnings") == [
            "[names] requires.callables names 'act' and no step uses it",
            "[names] requires.executables names 'tool.one' and no step uses it",
        ]

    def test_the_claim_is_not_the_content_requires_none_skips_nothing(self):
        # requires.tools "none" and an empty claim: the body still decides
        doc = _wf({"a": _auto(action="act")}, requires={"tools": "none"})

        envelope, _ = load_executable(doc)

        assert envelope is None

    def test_names_inside_a_retry_and_a_parallel_step_are_found(self):
        inner = _auto("deep", action="deep")
        steps = {
            "a": _step("RetryStep", step=inner),
            "b": _step(
                "ParallelStep",
                "b",
                steps=[_auto("c", action="act")],
                **{"next_step_id": ""},
            ),
        }

        _, result = load_executable(_wf(steps, callables=[]))

        assert sorted(_lines(result, "names")) == [
            "[names] the body uses callables name 'act' that requires.callables lacks",
            "[names] the body uses callables name 'deep' that requires.callables lacks",
        ]

    def test_a_named_field_that_holds_no_text_is_refused_and_shows_no_value(self):
        doc = _wf({"a": _auto(action={"leak": _SECRET})})

        envelope, result = load_executable(doc)

        assert envelope is None
        assert _lines(result, "names") == ["[names] step 'a': action holds no name"]
        assert _SECRET not in json.dumps(result.as_dict())

    def test_a_name_in_a_line_is_capped_and_stripped(self):
        name = "evil\x1b[31m" + "x" * 200
        doc = _wf({"a": _auto(action=name)}, callables=[])

        _, result = load_executable(doc)

        line = _lines(result, "names")[0]
        assert "\x1b" not in line and len(line) < 300

    def test_the_resolver_is_asked_by_category_and_a_refusal_is_one_line(self):
        asked: list[tuple[str, str]] = []

        def resolver(category: str, name: str) -> bool:
            asked.append((category, name))
            return name != "tool.one"

        doc = _wf(
            {
                "a": _auto(action="act"),
                "b": _step("ExecutableStep", "b", target="tool.one"),
            },
            callables=["act"],
            executables=["tool.one"],
        )

        _, result = load_executable(doc, resolver=resolver)

        assert sorted(asked) == [("callables", "act"), ("executables", "tool.one")]
        assert _lines(result, "names") == [
            "[names] executables name 'tool.one' does not resolve"
        ]

    def test_with_no_resolver_one_info_line_says_so(self):
        result = _loads(_wf())

        assert _lines(result, "names", "info") == [
            "[names] names are not resolved: no resolver was given"
        ]


# ---------------------------------------------------------------- V4 and the host-bound rule


class TestV4NoVariablesNoPlaceholder:
    def test_a_placeholder_in_a_step_text_is_refused_by_name(self):
        doc = _wf({"a": _auto(description="say {{task}} now")})

        envelope, result = load_executable(doc)

        assert envelope is None
        assert _lines(result, "variables") == [
            "[variables] the body uses {{task}}: a workflow takes no variables, so nothing fills it"
        ]

    def test_a_placeholder_in_a_nested_step_and_in_a_key_is_found(self):
        steps = {
            "a": _step("RetryStep", step=_auto("in", description="{{one}}")),
            "{{two}}": _auto("{{two}}"),
        }

        _, result = load_executable(_wf(steps))

        assert [x.split(" ")[4] for x in _lines(result, "variables")] == [
            "{{one}}:",
            "{{two}}:",
        ]

    def test_text_that_is_not_a_placeholder_is_left_alone(self):
        _loads(_wf({"a": _auto(description="a {brace} and {{ Spaced }} and {{1x}}")}))


class TestTheHostBoundFieldsAreRefusedInEveryOrigin:
    @pytest.mark.parametrize("field", ["registry", "callables"])
    @pytest.mark.parametrize("origin", ["operator", "model"])
    def test_an_executable_step_that_sets_the_field_is_refused(self, field, origin):
        steps = {"a": _step("ExecutableStep", target="tool.one", **{field: "host"})}
        doc = _wf(steps, callables=["host"], executables=["tool.one"])

        envelope, result = load_executable(doc, origin=origin, allow=_ALLOW_ALL)

        assert envelope is None
        assert (
            f"[body] step 'a' sets {field}: the host binds it at run time"
            in result.errors
        )

    def test_a_nested_executable_step_is_refused_too(self):
        inner = _step("ExecutableStep", "in", target="tool.one", registry=None)
        doc = _wf({"a": _step("RetryStep", step=inner)}, executables=["tool.one"])

        envelope, result = load_executable(doc)

        assert envelope is None
        assert (
            "[body] step 'a' sets registry: the host binds it at run time"
            in result.errors
        )

    def test_a_workflow_that_leaves_them_out_loads(self):
        _loads(
            _wf(
                {"a": _step("ExecutableStep", target="tool.one")},
                executables=["tool.one"],
            )
        )


# ---------------------------------------------------------------- V7: the closed allow-list

_OFF_LIST = [
    "LLMStep",
    "AgentStep",
    "ConversationStep",
    "APIStep",
    "TimerStep",
    "WaitEventStep",
    "NoSuchStep",
    "autostep",
    "AutoStep2",
]


class TestV7TheClosedStepList:
    @pytest.mark.parametrize("tag", _OFF_LIST)
    def test_an_off_list_step_type_is_refused_with_one_line(self, tag):
        doc = _wf({"a": _step(tag)})

        line = _one_model_line(doc, f"has type '{tag}'")

        assert line == (
            f"[model] step 'a' has type '{tag}': not a step type a model may write"
        )

    @pytest.mark.parametrize("tag", _OFF_LIST)
    def test_the_same_document_loads_for_an_operator(self, tag):
        _loads(_wf({"a": _step(tag)}))

    @pytest.mark.parametrize(
        ("wrapper", "key"),
        [("RetryStep", "step"), ("ParallelStep", "steps")],
    )
    @pytest.mark.parametrize(
        "tag", ["LLMStep", "WaitEventStep", "TimerStep", "AgentStep"]
    )
    def test_a_wrapper_around_an_off_list_step_is_refused_at_the_child(
        self, wrapper, key, tag
    ):
        child = _step(tag, "child")
        doc = _wf({"a": _step(wrapper, **{key: child if key == "step" else [child]})})

        _one_model_line(doc, f"has type '{tag}'")

    def test_nested_wrappers_of_allowed_steps_load(self):
        deep = _step(
            "RetryStep",
            "r1",
            step=_step("RetryStep", "r2", step=_auto("c", action="deep")),
        )
        par = _step(
            "ParallelStep", "a", steps=[deep, _auto("d", action="act")], next_step_id=""
        )
        doc = _wf({"a": par}, callables=["act", "deep"])

        result = _loads(doc, origin="model", allow=_ALLOW_ALL)

        assert result.errors == []

    def test_a_deeply_nested_off_list_step_is_found_without_recursion(self):
        node: dict[str, Any] = _step("LLMStep", "leaf")
        for index in range(14):
            node = _step("RetryStep", "a" if index == 13 else f"r{index}", step=node)

        _one_model_line(_wf({"a": node}), "has type 'LLMStep'")

    def test_every_allowed_type_loads_alone(self):
        for tag in sorted(ew.ALLOWED_STEP_TYPES):
            _loads(_wf({"a": _step(tag)}), origin="model", allow=_ALLOW_ALL)

    def test_the_names_of_the_body_must_be_on_the_allow_list(self):
        steps = {
            "a": _auto(action="evil", next_step_id="b"),
            "b": _step("ExecutableStep", "b", target="tool.two"),
        }
        doc = _wf(steps, callables=["evil"], executables=["tool.two"])

        envelope, result = load_executable(doc, origin="model", allow=_ALLOW_ALL)

        assert envelope is None
        assert _lines(result, "model") == [
            "[model] the body uses callables name 'evil': not on the allow-list",
            "[model] the body uses executables name 'tool.two': not on the allow-list",
        ]

    def test_no_allow_list_is_an_empty_one_and_fails_closed(self):
        doc = _wf({"a": _auto(action="act")}, callables=["act"])

        for allow in (None, {}, {"callables": []}):
            envelope, result = load_executable(doc, origin="model", allow=allow)
            assert envelope is None, allow
            assert len(_lines(result, "model")) == 1

    def test_an_allow_list_given_as_one_string_is_one_name_not_its_letters(self):
        doc = _wf({"a": _auto(action="act")}, callables=["act"])

        _loads(doc, origin="model", allow={"callables": "act"})
        envelope, _ = load_executable(doc, origin="model", allow={"callables": "ac"})
        assert envelope is None

    def test_a_claim_in_requires_never_lifts_a_refusal(self):
        # the document claims every name and says it uses no tool: the body is still judged
        doc = _wf(
            {"a": _auto(action="evil"), "b": _step("LLMStep", "b")},
            callables=["evil"],
            capabilities=["hitl"],
            lifecycle="trusted",
        )

        envelope, result = load_executable(doc, origin="model", allow={})

        assert envelope is None
        assert len(_lines(result, "model")) == 3  # the type, the name, the lifecycle

    def test_a_handler_binding_is_refused(self):
        doc = _wf(requires={"tools": "none", "handlers": ["agents.limiter"]})

        _one_model_line(doc, "requires.handlers names 'agents.limiter'")

    @pytest.mark.parametrize("tools", ["some", ["search"], ["a", "b"]])
    def test_a_tool_binding_is_refused_whatever_it_lists(self, tools):
        doc = _wf(requires={"tools": tools})

        _one_model_line(
            doc, "requires.tools is not none", allow={"tools": ["search", "a", "b"]}
        )

    @pytest.mark.parametrize("tools", ["none", []])
    def test_no_tool_loads(self, tools):
        _loads(_wf(requires={"tools": tools}), origin="model", allow={})

    def test_only_draft_may_be_claimed(self):
        _one_model_line(_wf(lifecycle="verified"), "lifecycle 'verified'")

    def test_a_refusal_never_shows_a_value_of_a_step(self):
        steps = {"a": _step("LLMStep", prompt_template=_SECRET, description=_SECRET)}

        _, result = load_executable(_wf(steps), origin="model", allow={})

        assert _SECRET not in json.dumps(result.as_dict())

    def test_a_name_off_the_allow_list_is_capped_and_stripped(self):
        name = "evil\x1b[31m" + "x" * 200
        doc = _wf({"a": _auto(action=name)}, callables=[])

        line = _one_model_line(doc, "not on the allow-list", allow={})

        assert "\x1b" not in line and len(line) < 300

    def test_an_off_list_tag_is_capped_and_stripped(self):
        tag = "Evil" + "x" * 40  # V1 bounds a tag to 64 characters of [A-Za-z0-9_]
        line = _one_model_line(_wf({"a": _step(tag)}), "has type")

        assert len(line) < 200

    def test_the_origin_decides_whether_v7_runs(
        self,
    ):
        doc = _wf({"a": _step("WaitEventStep")})

        assert load_executable(doc, origin="operator")[0] is not None
        assert load_executable(doc, origin="model", allow={})[0] is None


class TestTheAllowListIsHeldToTheLoader:
    def test_it_is_a_subset_of_the_built_in_step_classes(self):
        from fsm_llm.workflows.loader import builtin_step_classes

        assert ew.ALLOWED_STEP_TYPES <= set(builtin_step_classes())

    def test_it_is_exactly_the_six_planned_types_and_holds_no_pausing_step(self):
        assert ew.ALLOWED_STEP_TYPES == {
            "ExecutableStep",
            "SwitchStep",
            "AutoStep",
            "ConditionStep",
            "ParallelStep",
            "RetryStep",
        }

    def test_every_built_in_step_class_is_judged_by_its_exact_tag(self):
        from fsm_llm.workflows.loader import builtin_step_classes

        for tag in builtin_step_classes():
            envelope, _ = load_executable(
                _wf({"a": _step(tag)}), origin="model", allow={}
            )
            assert (envelope is not None) == (tag in ew.ALLOWED_STEP_TYPES), tag

    def test_the_host_bound_fields_are_fields_of_the_executable_step(self):
        from fsm_llm.workflows.steps import ExecutableStep

        assert set(ew.HOST_BOUND_FIELDS) <= set(ExecutableStep.model_fields)


class TestAnotherKindIsNotJudgedByTheWorkflowRules:
    def test_model_problems_is_empty_for_a_non_workflow(self):
        envelope = ex.parse_executable(corpus.wrap(next(iter(corpus.fsm_cases()))))

        assert not isinstance(envelope, ex.WorkflowEnvelope)
        assert ew.model_problems(envelope, {}) == []


# ---------------------------------------------------------------- the validator and the loader


class TestTheValidatorIsNeverStricterThanTheLoader:
    @pytest.fixture(autouse=True)
    def _keep_logging_state(self, monkeypatch):
        import fsm_llm.core.verify.cli as cli_module
        from fsm_llm.logging import logger

        monkeypatch.setattr(cli_module, "setup_cli_logging", lambda level: 0)
        yield
        logger.disable("fsm_llm")

    def _cases(self) -> dict[str, dict[str, Any]]:
        cases = dict(corpus.workflow_documents())
        cases["unclaimed callable"] = _wf({"a": _auto(action="act")}, callables=[])
        cases["placeholder"] = _wf({"a": _auto(description="{{x}}")})
        cases["host bound"] = _wf(
            {"a": _step("ExecutableStep", target="t", registry="r")}
        )
        cases["off list for an operator"] = _wf({"a": _step("LLMStep")})
        cases["pausing for an operator"] = _wf({"a": _step("WaitEventStep")})
        return cases

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

    def test_the_cli_prints_the_structure_info_line(self, tmp_path, capsys):
        from fsm_llm.core.verify.cli import main

        path = tmp_path / "wf.json"
        path.write_text(
            json.dumps(corpus.workflow_documents()["demo_linear"]), encoding="utf-8"
        )

        main(str(path))

        report = json.loads(capsys.readouterr().out)
        assert (
            "[body] workflow structure is checked when a registry loads the document"
            in report["info"]
        )
