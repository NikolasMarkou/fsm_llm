"""``load_workflow``: a workflow document back into a ``WorkflowDefinition`` (plan ace7a8c8 step 25).

The document is data: it never executes or imports. A step ``type`` resolves only through the
step-class table (the built-ins plus the classes the caller supplies), a callable field only
through a ``callables`` ``NameRegistry`` the caller supplies, and every failure of a document
is collected into ONE ``WorkflowValidationError``.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.names import NameRegistry
from fsm_llm.workflows import (
    AutoStep,
    WorkflowDefinition,
    WorkflowDefinitionError,
    WorkflowStep,
    WorkflowValidationError,
    create_workflow,
    load_workflow,
)
from tests.test_fsm_llm_workflows.support import (
    callables_registry,
    every_step_objects,
    every_step_workflow,
)


class NoteStep(WorkflowStep):
    """A custom step class: round-trips through ``step_classes``; ``action`` is a shared name."""

    next_step_id: str = ""
    text: str = ""
    action: Callable[..., Any] | None = None

    async def execute(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError


def _custom_act(context: dict[str, Any]) -> dict[str, Any]:
    return {}


def _all_steps_world() -> tuple[WorkflowDefinition, NameRegistry[Any], dict[str, Any]]:
    objects = every_step_objects()
    objects["custom_act"] = _custom_act
    custom = NoteStep(
        step_id="s_custom",
        name="Note",
        next_step_id="s_cond",
        text="hello",
        action=_custom_act,
    )
    workflow = every_step_workflow(objects, custom=custom)
    return workflow, callables_registry(objects), objects


def _small_doc() -> dict[str, Any]:
    workflow = create_workflow("wf", "W")
    workflow.with_initial_step(AutoStep(step_id="a", name="A", next_step_id="b"))
    workflow.with_step(AutoStep(step_id="b", name="B", next_step_id=""))
    return workflow.serialize()


def _errors(document: Any, **kwargs: Any) -> list[str]:
    kwargs.setdefault("objects", NameRegistry("callables"))
    with pytest.raises(WorkflowValidationError) as info:
        load_workflow(document, **kwargs)
    return [str(e) for e in info.value.validation_errors]


def _has(errors: list[str], *needles: str) -> bool:
    return any(all(n in e for n in needles) for e in errors)


class TestRoundTrip:
    def test_a_definition_holding_every_step_class_and_a_custom_one_round_trips(self):
        workflow, registry, _ = _all_steps_world()
        document = workflow.serialize(callables=registry)
        loaded = load_workflow(document, objects=registry, step_classes=[NoteStep])
        assert loaded == workflow

    def test_the_loaded_callables_are_the_same_registered_objects(self):
        workflow, registry, objects = _all_steps_world()
        loaded = load_workflow(
            workflow.serialize(callables=registry),
            objects=registry,
            step_classes=[NoteStep],
        )
        steps = loaded.steps
        assert steps["s_auto"].action is objects["act"]
        assert steps["s_cond"].condition is objects["cond"]
        assert steps["s_api"].api_function is objects["api"]
        assert steps["s_llm"].llm_interface is objects["llm"]
        assert steps["s_agent"].agent is objects["agent"]
        assert steps["s_par"].aggregation_function is objects["agg"]
        assert steps["s_par"].steps[0].action is objects["child_act"]
        assert steps["s_retry"].step.action is objects["inner_act"]
        assert steps["s_custom"].action is objects["custom_act"]
        assert type(steps["s_custom"]) is NoteStep

    def test_the_round_trip_holds_through_json_text(self):
        workflow, registry, _ = _all_steps_world()
        text = json.dumps(workflow.serialize(callables=registry))
        loaded = load_workflow(
            json.loads(text), objects=registry, step_classes=[NoteStep]
        )
        assert loaded == workflow

    def test_a_loaded_workflow_serializes_to_the_document_it_came_from(self):
        workflow, registry, _ = _all_steps_world()
        document = workflow.serialize(callables=registry)
        loaded = load_workflow(document, objects=registry, step_classes=[NoteStep])
        assert loaded.serialize(callables=registry) == document

    def test_the_loader_never_changes_or_aliases_the_document(self):
        workflow, registry, _ = _all_steps_world()
        document = workflow.serialize(callables=registry)
        before = copy.deepcopy(document)
        loaded = load_workflow(document, objects=registry, step_classes=[NoteStep])
        assert document == before
        loaded.metadata["owner"]["tags"].append("z")
        assert document == before

    def test_the_loaded_definition_is_validated(self):
        document = _small_doc()
        document["steps"]["orphan"] = {
            "type": "AutoStep",
            "step_id": "orphan",
            "name": "Orphan",
            "next_step_id": "",
        }
        errors = _errors(document)
        assert _has(errors, "orphan", "not reachable")

    def test_the_definition_model_config_is_unchanged(self):
        # the loader refuses an unknown top-level key itself; the model does not
        assert WorkflowDefinition.model_config == {}
        built = WorkflowDefinition(workflow_id="w", name="W", foo=1)  # type: ignore[call-arg]
        assert not hasattr(built, "foo")


class TestObjectFormFsmDefinition:
    @staticmethod
    def _fsm() -> FSMDefinition:
        return FSMDefinition.model_validate(
            {
                "name": "f",
                "description": "d",
                "initial_state": "s",
                "persona": "p",
                "states": {"s": {"id": "s", "description": "d", "purpose": "p"}},
            }
        )

    def test_the_weaker_form_holds_serialize_of_load_equals_the_document(self):
        objects = every_step_objects()
        registry = callables_registry(objects)
        workflow = every_step_workflow(objects, object_form=self._fsm())
        document = workflow.serialize(callables=registry)
        loaded = load_workflow(document, objects=registry)
        assert loaded.serialize(callables=registry) == document

    def test_known_limit_equality_does_not_hold_for_the_object_form(self):
        objects = every_step_objects()
        registry = callables_registry(objects)
        workflow = every_step_workflow(objects, object_form=self._fsm())
        loaded = load_workflow(workflow.serialize(callables=registry), objects=registry)
        assert isinstance(workflow.steps["s_conv"].fsm_definition, FSMDefinition)
        assert isinstance(loaded.steps["s_conv"].fsm_definition, dict)
        assert loaded != workflow

    def test_an_object_in_a_document_is_normalised_to_a_dict(self):
        document = _small_doc()
        document["steps"]["b"] = {
            "type": "ConversationStep",
            "step_id": "b",
            "name": "B",
            "fsm_definition": self._fsm(),
        }
        loaded = load_workflow(document, objects=NameRegistry("callables"))
        assert loaded.steps["b"].fsm_definition == self._fsm().model_dump()


class TestOneErrorWithEveryFailure:
    def test_a_bad_document_reports_all_its_failures_in_one_error(self):
        document = _small_doc()
        document["bogus_key"] = 1
        steps = document["steps"]
        steps["t1"] = {"type": "NopeStep", "step_id": "t1", "name": "T1"}
        steps["t2"] = {"step_id": "t2", "name": "T2", "next_step_id": ""}
        steps["t3"] = {
            "type": "AutoStep",
            "step_id": "t3",
            "name": "T3",
            "next_step_id": "",
            "action": "ghost",
        }
        steps["t4"] = {
            "type": "AutoStep",
            "step_id": "t4",
            "name": "T4",
            "next_step_id": "",
            "colour": "red",
        }
        steps["t5"] = {"type": "AutoStep", "step_id": "t5", "name": "T5"}

        class AutoStep2(WorkflowStep):  # shadows nothing: renamed below
            async def execute(self, context: dict[str, Any]) -> Any:
                raise NotImplementedError

        AutoStep2.__name__ = "AutoStep"  # a custom class named as a built-in
        errors = _errors(document, step_classes=[AutoStep2, NoteStep, NoteStep])
        assert _has(errors, "bogus_key")
        assert _has(errors, "t1", "NopeStep")
        assert _has(errors, "t2", "type")
        assert _has(errors, "t3", "ghost", "action")
        assert _has(errors, "t4", "colour")
        assert _has(errors, "t5", "next_step_id")
        assert _has(errors, "AutoStep", "built-in")
        assert _has(errors, "NoteStep", "twice")

    def test_a_structure_error_and_a_step_error_each_use_the_one_error_type(self):
        document = _small_doc()
        document["steps"]["b"]["next_step_id"] = "nowhere"
        assert _has(_errors(document), "nowhere")

    def test_a_bad_nested_step_is_reported_with_its_path(self):
        document = _small_doc()
        document["steps"]["p"] = {
            "type": "ParallelStep",
            "step_id": "p",
            "name": "P",
            "next_step_id": "",
            "steps": [{"type": "AutoStep", "step_id": "c", "name": "C"}, {"name": "x"}],
        }
        document["steps"]["r"] = {
            "type": "RetryStep",
            "step_id": "r",
            "name": "R",
            "step": {"type": "Missing", "step_id": "i", "name": "I"},
        }
        errors = _errors(document)
        assert _has(errors, "'p'", "steps[0]", "next_step_id")
        assert _has(errors, "'p'", "steps[1]", "type")
        assert _has(errors, "'r'", "Missing")

    def test_a_document_that_is_not_a_mapping_is_refused(self):
        assert _has(_errors(["not", "a", "mapping"]), "mapping")

    def test_a_steps_value_that_is_not_a_mapping_is_refused(self):
        document = _small_doc()
        document["steps"] = ["a"]
        assert _has(_errors(document), "steps", "mapping")

    def test_a_pydantic_error_names_the_step_and_the_field(self):
        document = _small_doc()
        document["steps"]["b"]["timeout_seconds"] = "soon"
        assert _has(_errors(document), "'b'", "timeout_seconds")


class TestTheTypeTagIsDataNotCode:
    def test_the_tag_is_matched_exactly_not_case_insensitively(self):
        document = _small_doc()
        document["steps"]["b"]["type"] = "autostep"
        assert _has(_errors(document), "autostep")

    def test_a_custom_class_is_unknown_unless_the_caller_supplies_it(self):
        document = _small_doc()
        document["steps"]["b"] = {"type": "NoteStep", "step_id": "b", "name": "B"}
        assert _has(_errors(document), "NoteStep")
        loaded = load_workflow(
            document, objects=NameRegistry("callables"), step_classes=[NoteStep]
        )
        assert type(loaded.steps["b"]) is NoteStep

    @pytest.mark.parametrize(
        "tag",
        [
            "fsm_llm.workflows.steps.AutoStep",
            "fsm_llm.workflows.steps:AutoStep",
            "antigravity",
            "os.system",
            ["AutoStep"],
            5,
        ],
    )
    def test_a_tag_that_looks_like_an_import_path_is_unknown_and_imports_nothing(
        self, tag: Any
    ):
        sys.modules.pop("antigravity", None)
        document = _small_doc()
        document["steps"]["b"]["type"] = tag
        errors = _errors(document)
        assert _has(errors, "'b'", "type")
        assert "antigravity" not in sys.modules

    def test_the_loader_module_imports_nothing_by_a_string(self):
        import fsm_llm.workflows.loader as loader

        source = Path(loader.__file__).read_text(encoding="utf-8")
        for banned in ("importlib", "__import__", "import_module", "exec_module"):
            assert banned not in source
        assert "eval(" not in source


class TestNamedObjects:
    def test_a_name_missing_from_the_registry_is_an_error_never_a_silent_none(self):
        document = _small_doc()
        document["steps"]["a"]["action"] = "ghost"
        errors = _errors(document)
        assert _has(errors, "'a'", "action", "ghost")

    def test_a_name_that_is_not_a_string_is_refused(self):
        document = _small_doc()
        document["steps"]["a"]["action"] = {"name": "act"}
        assert _has(_errors(document), "'a'", "action", "name")

    def test_a_null_object_field_stays_none(self):
        document = _small_doc()
        document["steps"]["a"]["action"] = None
        loaded = load_workflow(document, objects=NameRegistry("callables"))
        assert loaded.steps["a"].action is None

    def test_a_registry_that_is_not_a_callables_registry_is_refused(self):
        document = _small_doc()
        with pytest.raises(WorkflowDefinitionError, match="callables"):
            load_workflow(document, objects={"act": print})  # type: ignore[arg-type]
        with pytest.raises(WorkflowDefinitionError, match="'tools'"):
            load_workflow(document, objects=NameRegistry("tools"))
