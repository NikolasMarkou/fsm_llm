"""The workflow envelope kind: body, one source of names, identity (plan 7d9c0955 step 2, D-005).

A ``workflow`` document is an envelope whose body is ``{initial_step_id, steps}``, the step documents
of ``serialize(callables=)``. Core knows the steps as untyped dicts: V1 judges shape and size, the
two tables of named-object fields live in ``executable_workflow`` once, the instance identity hashes
the body as a tree and never through the FSM loader. The refusals are run against the SHIPPED
documents first (the two preset shapes of ``corpus.workflow_documents`` and every workflow of the G1
corpus), so a refusal that would relabel an honest document is a red row here.

Invariants (plan 7d9c0955): 2 (a document names objects and never holds code; V1 never reads
semantics), 5 (an error line names keys and levels, never a value), 8 (a workflow body never loads
as an FSM; no ``model_dump`` of a free-form tree), 7 (laziness: nothing at ``import fsm_llm``).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions import FSMDefinition, definition_digest
from fsm_llm.core.definitions import executable as ex
from fsm_llm.core.definitions import executable_instance as ei
from fsm_llm.core.definitions import executable_workflow as ew
from fsm_llm.core.verify.executable import load_executable
from tests import executables_corpus as corpus
from tests.conftest import cpu_timer
from tests.test_fsm_llm import kind_corpus as kc

_SRC = Path(fsm_llm.__file__).resolve().parents[1]
_SECRET = "SECRET-VALUE-4711"


def _auto(step_id: str, next_step_id: str = "", **fields: Any) -> dict[str, Any]:
    return {
        "type": "AutoStep",
        "step_id": step_id,
        "name": step_id,
        "next_step_id": next_step_id,
        **fields,
    }


def _doc(
    steps: dict[str, Any] | None = None,
    initial: str = "a",
    **header: Any,
) -> dict[str, Any]:
    body = {
        "initial_step_id": initial,
        "steps": {"a": _auto("a")} if steps is None else steps,
    }
    return {
        "schema": "fsm-llm/executable",
        "kind": "workflow",
        "id": "wf",
        "body": body,
        **header,
    }


def _refused(data: Any) -> ex.ExecutableDocumentError:
    with pytest.raises(ex.ExecutableDocumentError) as info:
        ex.parse_executable(data)
    return info.value


def _chain(depth: int) -> dict[str, Any]:
    """``depth`` RetryStep documents nested through ``step``, innermost an AutoStep."""
    node: dict[str, Any] = {"type": "AutoStep", "step_id": "leaf"}
    for _ in range(depth - 1):
        node = {"type": "RetryStep", "step": node}
    return node


def _g1_documents() -> dict[str, dict[str, Any]]:
    """One envelope document per workflow of the G1 corpus (the SHIPPED step shapes)."""
    from tests.test_fsm_llm_workflows import golden_corpus

    out = {}
    for name, build in golden_corpus.CORPUS.items():
        workflow, registry, _ = build()
        dump = workflow.serialize(callables=registry)
        out[name] = _doc(
            dump["steps"], dump["initial_step_id"], id=name.replace("_", "-")
        )
    return out


_SHIPPED = {**corpus.workflow_documents(), **_g1_documents()}


# ---------------------------------------------------------------- the kind parses


class TestTheKindParses:
    @pytest.mark.parametrize("name", sorted(_SHIPPED))
    def test_every_shipped_document_parses_and_round_trips(self, name):
        envelope = ex.parse_executable(_SHIPPED[name])

        assert isinstance(envelope, ex.WorkflowEnvelope)
        assert envelope.kind == "workflow"
        dumped = json.loads(json.dumps(envelope.model_dump(by_alias=True, mode="json")))
        assert ex.parse_executable(dumped) == envelope
        assert ex.newer_envelope_fields(envelope) == ()

    def test_the_two_preset_shapes_are_not_vacuous(self):
        docs = corpus.workflow_documents()

        assert set(docs) == {"demo_linear", "demo_branching"}
        for doc in docs.values():
            assert len(doc["body"]["steps"]) >= 3
            assert doc["body"]["steps"]["finish"]["next_step_id"] == ""

    def test_a_workflow_takes_no_variables(self):
        problems = _refused(_doc(variables=["task"])).problems

        assert problems == (
            "document: a workflow takes no variables: no placeholder reaches a step",
        )

    def test_the_header_keeps_its_rules_for_a_workflow(self):
        _refused(_doc(surprise=1))
        _refused(_doc(id="bad id"))
        _refused({**_doc(), "body": {**_doc()["body"], "surprise": 1}})
        _refused({**_doc(), "body": {"fsm": {"states": {}}}})


# ---------------------------------------------------------------- one source of names


class TestTheNamesLiveInCoreOnce:
    def test_the_tables_are_the_same_objects_in_core_and_in_workflows(self):
        from fsm_llm.workflows import definitions, loader

        assert definitions.NAMED_OBJECT_FIELDS is ew.NAMED_OBJECT_FIELDS
        assert (
            definitions.STEP_CALLABLE_EXCLUDE_FIELDS is ew.STEP_CALLABLE_EXCLUDE_FIELDS
        )
        assert loader.NAMED_OBJECT_FIELDS is ew.NAMED_OBJECT_FIELDS
        assert not hasattr(definitions, "_NAMED_OBJECT_FIELDS")
        assert not hasattr(definitions, "_STEP_CALLABLE_EXCLUDE_FIELDS")

    def test_the_tables_hold_the_known_fields(self):
        assert ew.STEP_CALLABLE_EXCLUDE_FIELDS == {
            "action",
            "api_function",
            "condition",
            "aggregation_function",
            "llm_interface",
        }
        assert ew.NAMED_OBJECT_FIELDS == ew.STEP_CALLABLE_EXCLUDE_FIELDS | {
            "agent",
            "registry",
            "callables",
        }

    def test_the_names_of_a_body_are_read_from_the_steps_through_every_nesting(self):
        steps = {
            "a": {
                "type": "ParallelStep",
                "steps": [
                    {"type": "RetryStep", "step": _auto("x", action="inner")},
                    _auto("y", action="outer"),
                    {"type": "ExecutableStep", "target": "t.one", "registry": "reg"},
                ],
                "aggregation_function": "agg",
            },
            "b": {"type": "ExecutableStep", "target": "t.two", "callables": "cs"},
            "c": {"type": "AutoStep", "action": 5, "target": "not.a.target"},
        }

        names = ew.workflow_names(steps)

        assert names.callables == ("agg", "cs", "inner", "outer", "reg")
        assert names.executables == ("t.one", "t.two")

    def test_the_names_agree_with_the_object_model_on_every_g1_workflow(self):
        from tests.test_fsm_llm_workflows import golden_corpus

        for name, build in golden_corpus.CORPUS.items():
            workflow, registry, _ = build()
            dump = workflow.serialize(callables=registry)
            expected: set[str] = set()
            stack = list(workflow.steps.values())
            while stack:
                step = stack.pop()
                for field in ew.NAMED_OBJECT_FIELDS & set(type(step).model_fields):
                    value = getattr(step, field, None)
                    if value is not None:
                        expected.add(registry.name_of(value))
                inner = getattr(step, "step", None)
                if inner is not None and hasattr(inner, "execute"):
                    stack.append(inner)
                stack.extend(getattr(step, "steps", None) or [])

            assert set(ew.workflow_names(dump["steps"]).callables) == expected, name

    def test_the_walker_visits_every_nesting_in_document_order(self):
        steps = {
            "a": {"type": "RetryStep", "step": {"type": "AutoStep", "n": 2}},
            "b": {"type": "ParallelStep", "steps": [{"n": 3}, "text", {"n": 4}]},
        }

        seen = [
            (step.get("type", step.get("n")), level, owner)
            for step, level, owner in ew.walk_steps(steps)
        ]

        assert seen == [
            ("RetryStep", 1, "a"),
            ("AutoStep", 2, "a"),
            ("ParallelStep", 1, "b"),
            (3, 2, "b"),
            (4, 2, "b"),
        ]


# ---------------------------------------------------------------- identity


def _envelope(**over: Any) -> ex.WorkflowEnvelope:
    envelope = ex.parse_executable(_doc(**over))
    assert isinstance(envelope, ex.WorkflowEnvelope)
    return envelope


def _body_dict(envelope: ex.WorkflowEnvelope) -> dict[str, Any]:
    return {
        "initial_step_id": envelope.body.initial_step_id,
        "steps": envelope.body.steps,
    }


class TestIdentity:
    def test_template_identity_ignores_the_key_order_of_the_steps_and_sees_a_field(
        self,
    ):
        steps = {"a": _auto("a", "b"), "b": _auto("b")}
        flipped = {"b": _auto("b"), "a": _auto("a", "b")}
        reordered = {
            "a": dict(reversed(list(_auto("a", "b").items()))),
            "b": _auto("b"),
        }

        base = ei.identity_of(_envelope(steps=steps)).template_identity
        assert ei.identity_of(_envelope(steps=flipped)).template_identity == base
        assert ei.identity_of(_envelope(steps=reordered)).template_identity == base
        changed = {"a": _auto("a", "b", name="other"), "b": _auto("b")}
        assert ei.identity_of(_envelope(steps=changed)).template_identity != base

    def test_the_instance_identity_is_the_digest_of_the_body_tree(self):
        envelope = _envelope(steps={"a": _auto("a", "b"), "b": _auto("b")})
        body = _body_dict(envelope)

        identity = ei.identity_of(envelope, body)

        assert identity.instance_identity == ei.canonical_digest(body)
        assert identity.instance_identity != identity.template_identity
        assert ei.identity_of(envelope).instance_identity is None

    def test_the_instance_identity_does_not_depend_on_the_header(self):
        one = _envelope(name="One", description="first")
        two = _envelope(name="Two", description="second", version="9", id="other")

        a = ei.identity_of(one, _body_dict(one))
        b = ei.identity_of(two, _body_dict(two))

        assert a.instance_identity == b.instance_identity
        assert a.template_identity != b.template_identity

    def test_an_instance_is_hashed_as_written_even_when_it_loads_as_an_fsm(self):
        envelope = _envelope()
        fsm = kc._fsm(
            "Mini", "a", kc._state("a", transitions=[kc._go("b")]), kc._state("b")
        )
        fsm["handler_only_keys"] = []  # a written default: the FSM digest drops it
        assert definition_digest(FSMDefinition(**fsm)) != ei.canonical_digest(fsm)

        identity = ei.identity_of(envelope, fsm)

        assert identity.instance_identity == ei.canonical_digest(fsm)

    def test_a_body_whose_steps_are_named_like_an_fsm_is_a_workflow(self):
        steps = {
            "initial_state": _auto("initial_state", "states"),
            "states": _auto("states"),
            "name": _auto("name"),
        }
        envelope = _envelope(steps=steps, initial="initial_state")
        body = _body_dict(envelope)

        assert ei.identity_of(envelope, body).instance_identity == ei.canonical_digest(
            body
        )
        assert ei.identity_of(envelope, steps).instance_identity == ei.canonical_digest(
            steps
        )

    def test_the_steps_are_hashed_through_the_tree_walker(self, monkeypatch):
        seen: list[Any] = []
        original = ei._json_tree

        def spy(value: Any) -> Any:
            seen.append(value)
            return original(value)

        monkeypatch.setattr(ei, "_json_tree", spy)
        envelope = _envelope()

        ei.identity_of(envelope)

        assert envelope.body.steps in seen

    def test_a_deep_but_legal_step_tree_hashes_the_same_twice(self):
        nest: Any = "leaf"
        for _ in range(150):
            nest = {"k": nest}
        steps = {"a": _auto("a", note=nest)}

        first = ei.identity_of(_envelope(steps=copy.deepcopy(steps)))
        second = ei.identity_of(_envelope(steps=copy.deepcopy(steps)))

        assert first == second


class TestNoFsmToInstantiate:
    LINE = "a workflow is instantiated by its adapter"

    def test_instantiate_envelope_refuses_one_line(self):
        with pytest.raises(ex.InstantiationError) as info:
            ei.instantiate_envelope(_envelope())

        assert str(info.value) == self.LINE

    @pytest.mark.parametrize(
        "record", [{}, {"options": {}, "variables": {}}, "garbage", None]
    )
    def test_rebuild_instance_refuses_before_it_reads_the_record(self, record):
        with pytest.raises(ex.InstantiationError) as info:
            ei.rebuild_instance(_envelope(), record, "0" * 64)

        assert str(info.value) == self.LINE

    def test_options_variants_and_a_generator_do_not_open_another_door(self):
        for kwargs in ({"options": {"x": 1}}, {"variants": ("v",)}, {"variables": {}}):
            with pytest.raises(
                ex.InstantiationError, match="instantiated by its adapter"
            ):
                ei.instantiate_envelope(_envelope(), **kwargs)
        with pytest.raises(ex.InstantiationError):
            ei.instantiate_body(_envelope())


# ---------------------------------------------------------------- V1: shape and size


def _one_line(error: ex.ExecutableDocumentError) -> str:
    assert len(error.problems) == 1, error.problems
    line = error.problems[0]
    assert len(line) <= ex.PROBLEM_LINE_CHARS
    assert _SECRET not in line
    assert "\x1b" not in line and "\n" not in line
    return line


class TestV1RefusesMalformedBodies:
    @pytest.mark.parametrize(
        "step",
        ["text", 5, None, [], ["AutoStep"], True],
        ids=["text", "int", "none", "list", "list-of-text", "bool"],
    )
    def test_a_step_that_is_no_mapping_is_refused(self, step):
        line = _one_line(_refused(_doc(steps={"a": step})))

        assert line.startswith("body.steps.a")

    @pytest.mark.parametrize(
        "tag",
        [
            None,
            5,
            ["AutoStep"],
            "",
            "bad type!",
            "../x",
            "A" * 65,
            f"x\n{_SECRET}",
            "é",
        ],
        ids=[
            "none",
            "int",
            "list",
            "empty",
            "space",
            "path",
            "long",
            "newline",
            "unicode",
        ],
    )
    def test_a_step_with_no_or_garbage_type_is_refused(self, tag):
        step = {"step_id": "a", "note": _SECRET}
        if tag is not None:
            step["type"] = tag

        _one_line(_refused(_doc(steps={"a": step})))

    def test_a_step_with_no_type_key_at_all_is_refused(self):
        _one_line(_refused(_doc(steps={"a": {"step_id": "a", "note": _SECRET}})))

    def test_a_key_that_is_not_the_step_id_is_refused_and_no_step_id_is_allowed(self):
        _one_line(_refused(_doc(steps={"a": _auto("other")})))
        _one_line(_refused(_doc(steps={"a": {**_auto("a"), "step_id": 5}})))
        without = {
            "type": "AutoStep",
            "note": "x",
        }  # the loader's pydantic reports a missing id
        assert ex.parse_executable(_doc(steps={"a": without}))

    @pytest.mark.parametrize("initial", ["zzz", "A", "", "x" * 300, "a\x1b[31m"])
    def test_an_initial_step_that_is_not_a_key_is_refused(self, initial):
        _one_line(_refused(_doc(initial=initial)))

    def test_a_missing_or_non_text_initial_step_is_refused(self):
        document = _doc()
        del document["body"]["initial_step_id"]
        _refused(document)
        document["body"]["initial_step_id"] = ["a"]
        _refused(document)

    @pytest.mark.parametrize(
        "steps", [{}, [], "steps", None, {"": _auto("")}, {"k" * 200: {}}]
    )
    def test_empty_or_malformed_step_tables_are_refused(self, steps):
        document = _doc()
        document["body"]["steps"] = steps
        _refused(document)

    def test_a_nested_step_or_steps_of_the_wrong_shape_is_refused(self):
        for bad in (
            {"type": "RetryStep", "step": "text"},
            {"type": "RetryStep", "step": ["x"]},
            {"type": "ParallelStep", "steps": {"x": {}}},
            {"type": "ParallelStep", "steps": ["text"]},
            {"type": "ParallelStep", "steps": [{"type": "AutoStep"}, 5]},
        ):
            _one_line(_refused(_doc(steps={"a": {**bad, "step_id": "a"}})))

    def test_the_nesting_boundary_is_exact(self):
        ok = _doc(steps={"a": {**_chain(ew.MAX_STEP_NESTING), "step_id": "a"}})
        too_deep = _doc(
            steps={"a": {**_chain(ew.MAX_STEP_NESTING + 1), "step_id": "a"}}
        )

        assert ex.parse_executable(ok)
        assert "nests deeper" in _one_line(_refused(too_deep))

    @pytest.mark.parametrize("depth", [100, 5_000, 100_000])
    def test_deep_nesting_is_one_line_with_no_recursion_error(
        self, depth, timing_factor
    ):
        document = _doc(steps={"a": {**_chain(depth), "step_id": "a"}})

        with cpu_timer() as timer:
            error = _refused(document)

        _one_line(error)
        assert timer.elapsed < 3.0 * timing_factor

    def test_a_ten_thousand_step_table_is_one_bounded_line(self, timing_factor):
        steps = {f"s{i}": _auto(f"s{i}", f"s{i + 1}") for i in range(10_000)}

        with cpu_timer() as timer:
            error = _refused(_doc(steps=steps, initial="s0"))

        _one_line(error)
        assert timer.elapsed < 3.0 * timing_factor

    def test_ten_thousand_nested_children_are_one_bounded_line(self, timing_factor):
        children = [{"type": "AutoStep", "step_id": f"c{i}"} for i in range(10_000)]
        steps = {"a": {"type": "ParallelStep", "step_id": "a", "steps": children}}

        with cpu_timer() as timer:
            error = _refused(_doc(steps=steps))

        assert "more than" in _one_line(error)
        assert timer.elapsed < 3.0 * timing_factor

    def test_the_count_boundary_counts_nested_steps(self):
        fits = {"type": "ParallelStep", "step_id": "a", "steps": [_auto("c")] * 499}
        spills = {**fits, "steps": [_auto("c")] * 500}

        assert ex.parse_executable(_doc(steps={"a": fits}))
        _one_line(_refused(_doc(steps={"a": spills})))

    def test_a_cyclic_step_dict_is_refused_not_walked_forever(self):
        loop: dict[str, Any] = {"type": "RetryStep", "step_id": "a"}
        loop["step"] = loop

        with pytest.raises(ValueError, match="nests deeper"):
            ew.WorkflowBody(initial_step_id="a", steps={"a": loop})

    def test_no_line_carries_a_value_of_a_step(self):
        hostile = {
            "a": {
                "type": "AutoStep",
                "step_id": "a",
                "name": _SECRET,
                "action": _SECRET,
                "next_step_id": _SECRET,
            },
            "b": {"type": f"bad {_SECRET}", "step_id": "b"},
        }
        error = _refused(_doc(steps=hostile))

        assert _SECRET not in error.args[0]
        assert all(_SECRET not in line for line in error.problems)

    def test_a_key_with_control_characters_is_shown_escaped_and_capped(self):
        key = "a\x1b[31m‮" + "z" * 300
        error = _refused(_doc(steps={key: _auto(key)}, initial="nope"))

        for line in error.problems:
            assert "\x1b" not in line and "‮" not in line
            assert len(line) <= ex.PROBLEM_LINE_CHARS


# ---------------------------------------------------------------- the schema, the guard, laziness


class TestTheSchemaAndTheVerifierGuard:
    def test_the_shipped_schema_has_the_workflow_branch(self):
        schema = ex.executable_json_schema()
        body = schema["$defs"]["WorkflowBody"]

        assert "workflow" in schema["discriminator"]["mapping"]
        assert set(body["properties"]) == {"initial_step_id", "steps"}
        assert set(body["required"]) == {"initial_step_id", "steps"}
        shipped = json.loads(ex.SCHEMA_PATH.read_text(encoding="utf-8"))
        assert shipped == schema

    @pytest.mark.parametrize("origin", ["operator", "model"])
    @pytest.mark.parametrize("name", sorted(corpus.workflow_documents()))
    def test_load_executable_runs_the_workflow_verifier_and_never_reads_an_fsm_body(
        self, origin, name
    ):
        # a workflow has no `body.fsm`, so the load sequence takes its own branch and returns
        # the envelope as parsed (invariant 2).
        doc = corpus.workflow_documents()[name]

        envelope, result = load_executable(
            doc, origin=origin, allow={"callables": doc["requires"]["callables"]}
        )

        assert isinstance(envelope, ex.WorkflowEnvelope)
        assert result.is_valid and result.errors == []

    def test_a_hostile_model_origin_workflow_never_loads(self):
        hostile = _doc(
            steps={"a": {"type": "LLMStep", "step_id": "a", "llm_interface": "x"}},
            requires={"tools": "none"},
        )

        envelope, result = load_executable(hostile, origin="model", allow={})

        assert envelope is None and not result.is_valid


class TestLaziness:
    _CODE = (
        "import sys; sys.path.insert(0, {src!r}); import {module}; "
        "import fsm_llm; assert fsm_llm.__file__.startswith({src!r}); "
        "print(sorted(m for m in sys.modules if m.startswith("
        "'fsm_llm.core.definitions.executable')))"
    )

    def _loaded(self, module: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-c", self._CODE.format(src=str(_SRC), module=module)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
        return json.loads(out.replace("'", '"'))

    def test_import_fsm_llm_loads_no_executable_module(self):
        assert self._loaded("fsm_llm") == []

    def test_import_of_the_workflows_package_loads_only_the_leaf_with_the_names(self):
        assert self._loaded("fsm_llm.workflows") == [
            "fsm_llm.core.definitions.executable_workflow"
        ]
