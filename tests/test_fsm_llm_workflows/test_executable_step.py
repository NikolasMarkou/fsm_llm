"""``ExecutableStep``: a workflow step that runs an executable by id (plan ace7a8c8 step 26, X5.2).

The step reads ``registry.get(target)``, maps workflow keys to the executable's inputs, awaits
``arun`` under a ``RunContext`` and maps the ``RunResult`` back. It owns no conversation lifecycle (the
executable does: the adapter ends its conversation in ``finally``) and imports no ``API``. A ``failed`` or
``aborted`` result is a failure routed to ``error_step_id``, never to the success route (5d1e7a3b/D-003); a
``returns`` entry that names an internal-prefixed workflow key is refused (876e7164/D-003). Rows run the
REAL core with a scripted model, or a small code-backed executable; cancel rows use events and bounded
waits (D7).
"""

from __future__ import annotations

import ast
import asyncio
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

import fsm_llm
from fsm_llm import API
from fsm_llm.core.codes import Reason, Status
from fsm_llm.core.definitions.executable import Contract, Requires
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.results import RunResult
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.executables import (
    CodeBackedExecutable,
    ExecutableNotFoundError,
    ExecutableRegistry,
)
from fsm_llm.engine.executables import adapters as adapters_module
from fsm_llm.testing import ScriptedLLM
from fsm_llm.workflows import (
    WorkflowEngine,
    WorkflowStatus,
    auto_step,
    create_workflow,
    executable_step,
    load_workflow,
)
from fsm_llm.workflows.exceptions import WorkflowValidationError
from fsm_llm.workflows.loader import builtin_step_classes
from fsm_llm.workflows.steps import ExecutableStep, ParallelStep
from tests.test_fsm_llm import kind_corpus as kc

_BOUND = 10  # seconds: a wait that would hang fails instead


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


# ---- executables under test -------------------------------------------------------------------


class _Code(CodeBackedExecutable):
    """A code-backed executable whose behaviour a row sets: it records every run."""

    def __init__(
        self,
        ident: str = "code",
        *,
        status: Status = Status.OK,
        reason: Reason | None = None,
        outputs: Mapping[str, Any] | None = None,
        answer: str | None = "done",
        inputs: tuple[str, ...] = ("topic",),
        optional_inputs: tuple[str, ...] = (),
        callables: tuple[str, ...] = (),
        park: bool = False,
        barrier: Any = None,
    ) -> None:
        super().__init__(
            id=ident,
            kind="strategy",
            impl="tests",
            contract=Contract(inputs=list(inputs), outputs=list(outputs or ())),
            requires=Requires(callables=list(callables)),
            optional_inputs=optional_inputs,
        )
        self.status, self.reason = status, reason
        self.result_outputs = dict(outputs or {})
        self.answer = answer
        self.park, self.barrier = park, barrier
        self.runs: list[dict[str, Any]] = []
        self.threads: list[int] = []
        self.cancelled = 0
        self.entered = threading.Event()

    async def execute(self, inputs, objects, context, identity) -> RunResult:
        self.runs.append(
            {
                "inputs": dict(inputs),
                "objects": dict(objects),
                "callables": context.callables,
                "max_steps": context.max_steps,
                "max_seconds": context.max_seconds,
            }
        )
        self.threads.append(threading.get_ident())
        try:
            if self.park:
                self.entered.set()
                await asyncio.Event().wait()
            if self.barrier is not None:
                await asyncio.wait_for(self.barrier.wait(), _BOUND)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.status is not Status.OK:
            return self.stopped(self.status, self.reason, identity)
        return RunResult(
            status=Status.OK,
            outputs=self.result_outputs,
            answer=self.answer,
            identity=identity,
        )


def _registry(*executables: Any) -> ExecutableRegistry:
    registry = ExecutableRegistry()
    for executable in executables:
        registry.register(executable)
    return registry


def _fsm_doc(ident: str = "demo") -> dict[str, Any]:
    """a -> b: state ``a`` extracts ``result`` and speaks, ``b`` is terminal and speaks."""
    return {
        "schema": "fsm-llm/executable",
        "kind": "fsm",
        "id": ident,
        "contract": {"inputs": ["topic"], "outputs": ["result"]},
        "body": {
            "fsm": kc._fsm(
                "Demo",
                "a",
                kc._state(
                    "a",
                    extraction_instructions="Extract the result.",
                    required_context_keys=["result"],
                    response_instructions="A.",
                    transitions=[kc._go("b")],
                ),
                kc._state("b", response_instructions="B."),
            )
        },
    }


class _InjectingAPI(API):
    """The ``API`` of the FSM adapter, with a scripted model and a record of its conversations."""

    llm: ClassVar[Any] = None
    made: ClassVar[list[_InjectingAPI]] = []

    @classmethod
    def from_definition(cls, definition, **kwargs):  # type: ignore[override]
        if kwargs.get("llm_interface") is None:  # the adapter passes an explicit None
            kwargs["llm_interface"] = cls.llm
        api = super().from_definition(definition, **kwargs)
        cls.made.append(api)
        return api

    def end_conversation(self, conversation_id: str) -> None:
        self.__dict__.setdefault("ended", []).append(
            {
                "id": conversation_id,
                "gate_free": not self.fsm_manager._active_turns,
            }
        )
        super().end_conversation(conversation_id)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Make the FSM adapter's ``API`` run on a scripted model; returns that model's installer."""
    _InjectingAPI.made = []

    def install(llm: ScriptedLLM) -> ScriptedLLM:
        _InjectingAPI.llm = llm
        monkeypatch.setattr(adapters_module, "API", _InjectingAPI)
        return llm

    return install


def _llm(**kwargs: Any) -> ScriptedLLM:
    kwargs.setdefault("extraction_data", {"result": "42"})
    kwargs.setdefault("default_response", "r")
    return ScriptedLLM(**kwargs)


def _step(registry: Any, target: str = "demo", **kwargs: Any) -> ExecutableStep:
    kwargs.setdefault("inputs", {"topic": "subject"})
    kwargs.setdefault("returns", {"final": "result"})
    kwargs.setdefault("success_step_id", "next")
    kwargs.setdefault("error_step_id", "err")
    return ExecutableStep(
        step_id="x", name="X", target=target, registry=registry, **kwargs
    )


# ---- parity with the adapter result -----------------------------------------------------------


class TestParity:
    async def test_the_step_maps_the_result_the_adapter_reports(self, scripted) -> None:
        """One real FSM executable, run directly and through the step: the same status, both
        identities and outputs. ``returns`` copies only non-``None`` outputs."""
        scripted(_llm())
        registry = ExecutableRegistry()
        executable = registry.add_document(_fsm_doc())
        direct = await executable.arun({"topic": "t"}, context=RunContext())
        step = _step(registry, returns={"final": "result", "absent": "nothing"})
        result = await step.execute({"subject": "t"})
        assert (result.success, result.next_step_id) == (True, "next")
        identity = {
            "template_identity": direct.identity.template_identity,
            "instance_identity": direct.identity.instance_identity,
        }
        assert identity["template_identity"] and identity["instance_identity"]
        assert result.data == {
            "executable_status": "ok",
            "executable_identity": identity,
            "executable_x_status": "ok",
            "executable_x_identity": identity,
            "executable_answer": direct.answer,
            "executable_x_answer": direct.answer,
            "final": direct.outputs[
                "result"
            ],  # "absent" is not in the outputs: not copied
        }

    async def test_a_none_output_is_not_copied(self) -> None:
        code = _Code(outputs={"kept": "v", "gone": None})
        step = _step(_registry(code), "code", returns={"a": "kept", "b": "gone"})
        result = await step.execute({"subject": "t"})
        assert result.data["a"] == "v" and "b" not in result.data

    async def test_only_mapped_workflow_keys_reach_the_executable(self) -> None:
        code = _Code(optional_inputs=("depth",))
        step = _step(
            _registry(code), "code", inputs={"topic": "subject", "depth": "missing"}
        )
        await step.execute({"subject": "t", "other": "x"})
        # a workflow key the context lacks is omitted (the executable then judges its own contract)
        assert code.runs[0]["inputs"] == {"topic": "t"}

    async def test_a_workflow_runs_the_step_end_to_end(self, scripted) -> None:
        scripted(_llm())
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        workflow = (
            create_workflow("wf", "W")
            .with_initial_step(
                _step(registry, success_step_id="done", error_step_id="bad")
            )
            .with_step(auto_step("done", "Done", next_step_id=""))
            .with_step(auto_step("bad", "Bad", next_step_id=""))
        )
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        iid = await engine.start("wf", initial_context={"subject": "t"})
        assert engine.get_status(iid) == WorkflowStatus.COMPLETED
        context = engine.get_context(iid)
        assert context["final"] == "42" and context["executable_x_status"] == "ok"
        await engine.close()


# ---- failure routing --------------------------------------------------------------------------


class TestFailureRouting:
    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (Status.FAILED, Reason.PROVIDER_ERROR),
            (Status.FAILED, Reason.INPUTS_MISSING),
            (Status.ABORTED, Reason.BUDGET_SPENT),
            (Status.ABORTED, Reason.CANCELLED),
            (Status.RUNNING, None),
        ],
    )
    async def test_a_result_that_is_not_ok_goes_to_the_error_route(
        self, status: Status, reason: Reason | None
    ) -> None:
        """Never the success route: the data still reports the status and the identity."""
        code = _Code(status=status, reason=reason)
        result = await _step(_registry(code), "code").execute({"subject": "t"})
        assert result.success is False
        assert result.next_step_id == "err"
        assert result.next_step_id != "next"
        assert result.data["executable_status"] == status.value
        assert "final" not in result.data

    async def test_with_no_error_route_the_instance_fails(self) -> None:
        """5d1e7a3b/D-003: a failure with no ``error_step_id`` has no route, the engine FAILS the
        instance and never continues down the success route."""
        code = _Code(status=Status.ABORTED, reason=Reason.BUDGET_SPENT)
        step = _step(_registry(code), "code", error_step_id=None)
        assert (await step.execute({"subject": "t"})).next_step_id is None
        workflow = (
            create_workflow("wf", "W")
            .with_initial_step(
                _step(
                    _registry(code), "code", success_step_id="done", error_step_id=None
                )
            )
            .with_step(auto_step("done", "Done", next_step_id=""))
        )
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        iid = await engine.start("wf", initial_context={"subject": "t"})
        assert engine.get_status(iid) == WorkflowStatus.FAILED
        await engine.close()

    async def test_an_unknown_target_is_the_same_failure(self) -> None:
        result = await _step(ExecutableRegistry(), "nope").execute({"subject": "t"})
        assert (result.success, result.next_step_id) == (False, "err")
        assert "nope" in str(result.error)
        with pytest.raises(ExecutableNotFoundError):
            ExecutableRegistry().get("nope")

    async def test_a_missing_declared_input_fails_with_no_model_call(
        self, scripted
    ) -> None:
        llm = scripted(_llm())
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        result = await _step(registry).execute({})  # no "subject" in the context
        assert (result.success, result.next_step_id) == (False, "err")
        assert result.data["executable_status"] == "failed"
        assert llm.call_history == []

    async def test_an_undeclared_input_is_a_failure_with_the_reason_in_the_text(
        self,
    ) -> None:
        code = _Code()
        step = _step(_registry(code), "code", inputs={"unknown": "subject"})
        result = await step.execute({"subject": "t"})
        assert (result.success, result.next_step_id) == (False, "err")
        assert "declares no input" in str(result.error)
        assert code.runs == []


# ---- callables --------------------------------------------------------------------------------


class TestCallables:
    async def test_a_target_that_needs_callables_is_refused_without_them(self) -> None:
        code = _Code(callables=("tools",))
        result = await _step(_registry(code), "code").execute({"subject": "t"})
        assert (result.success, result.next_step_id) == (False, "err")
        # the STEP refuses (the adapter would refuse too, with its own text)
        assert "names no callables registry" in str(result.error)
        assert code.runs == []  # refused before the run

    async def test_with_a_callables_registry_the_run_gets_it(self) -> None:
        objects: NameRegistry[Any] = NameRegistry("callables")
        marker = object()
        objects.register("tools", marker)
        code = _Code(callables=("tools",))
        step = _step(_registry(code), "code", callables=objects)
        result = await step.execute({"subject": "t"})
        assert result.success is True
        assert code.runs[0]["callables"] is objects
        assert code.runs[0]["objects"] == {"tools": marker}

    async def test_a_callables_registry_of_another_namespace_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="callables"):
            _step(ExecutableRegistry(), callables=NameRegistry("tools"))
        with pytest.raises(ValidationError, match="callables"):
            _step(ExecutableRegistry(), callables={"tools": 1})

    async def test_the_budgets_reach_the_run_context(self) -> None:
        code = _Code()
        await _step(_registry(code), "code", max_steps=7, max_seconds=3.0).execute(
            {"subject": "t"}
        )
        await _step(_registry(code), "code").execute({"subject": "t"})
        assert [(r["max_steps"], r["max_seconds"]) for r in code.runs] == [
            (7, 3.0),
            (None, None),
        ]

    def test_a_budget_that_is_not_positive_is_refused(self) -> None:
        for bad in ({"max_steps": 0}, {"max_seconds": 0}, {"max_seconds": -1.0}):
            with pytest.raises(ValidationError):
                _step(ExecutableRegistry(), **bad)


# ---- the internal-key rule --------------------------------------------------------------------


class TestReturnsNeverNameAnInternalKey:
    @pytest.mark.parametrize(
        "key", ["_waiting_info", "_timer_info", "_x", "system_x", "internal_x", "__x"]
    )
    def test_a_return_naming_an_internal_prefixed_key_is_refused(
        self, key: str
    ) -> None:
        """876e7164/D-003: the engine drops internal-prefixed result keys except a whitelist
        (``_waiting_info``, ``_timer_info``): a step must not be able to name one."""
        with pytest.raises(ValidationError, match="internal"):
            _step(ExecutableRegistry(), returns={key: "result"})

    async def test_a_return_set_after_construction_is_refused_at_run_time(self) -> None:
        code = _Code(outputs={"v": "1"})
        step = _step(_registry(code), "code", returns={"ok_key": "v"})
        step.returns = {
            "_waiting_info": "v"
        }  # a post-construction edit bypasses the validator
        result = await step.execute({"subject": "t"})
        assert result.success is False and "internal" in str(result.error)
        assert "_waiting_info" not in result.data

    async def test_the_step_writes_no_internal_key_of_its_own(self) -> None:
        result = await _step(_registry(_Code(outputs={"v": "1"})), "code").execute(
            {"subject": "t"}
        )
        assert all(not k.startswith("_") for k in result.data)


# ---- cancel and overlap -----------------------------------------------------------------------


class TestCancelAndOverlap:
    async def test_a_time_limit_cancels_a_code_backed_run(self, timing_factor) -> None:
        code = _Code(park=True)
        step = _step(_registry(code), "code", timeout_seconds=0.2 * timing_factor)
        result = await asyncio.wait_for(step.execute({"subject": "t"}), _BOUND)
        assert (result.success, result.next_step_id) == (False, "err")
        assert "timed out" in str(result.error)
        assert code.cancelled == 1

    async def test_a_time_limit_cancels_an_fsm_run_and_the_adapter_ends_its_conversation(
        self, scripted, timing_factor
    ) -> None:
        """Park the model in the first step of the run (the 2nd Pass-2 await; the 1st is the
        greeting). The limit cancels it, the core rolls the turn back, the adapter ends the
        conversation exactly once and the gate is free; the step owns no lifecycle of its own."""
        from tests.test_fsm_llm_workflows.test_native_steps import _Probe

        probe = scripted(_Probe(park_at=2, extraction_data={"result": "42"}))
        registry = ExecutableRegistry()
        registry.add_document(_fsm_doc())
        step = _step(registry, timeout_seconds=0.3 * timing_factor)
        result = await asyncio.wait_for(step.execute({"subject": "t"}), _BOUND)
        assert (result.success, result.next_step_id) == (False, "err")
        assert "timed out" in str(result.error)
        assert probe.entered.is_set() and probe.cancelled == 1  # type: ignore[attr-defined]
        (api,) = _InjectingAPI.made
        assert len(api.ended) == 1 and api.ended[0]["gate_free"] is True  # type: ignore[attr-defined]
        assert api.fsm_manager.instances == {}

    async def test_two_steps_of_a_parallel_step_overlap_on_one_loop(self) -> None:
        barrier = asyncio.Barrier(2)
        codes = [_Code("c1", barrier=barrier), _Code("c2", barrier=barrier)]
        registry = _registry(*codes)
        children = [
            ExecutableStep(
                step_id=f"x{i}",
                name="X",
                target=code.id,
                registry=registry,
                inputs={"topic": "subject"},
            )
            for i, code in enumerate(codes)
        ]
        parallel = ParallelStep(step_id="p", name="P", steps=children, next_step_id="n")
        result = await asyncio.wait_for(parallel.execute({"subject": "t"}), _BOUND)
        assert result.success is True
        assert {t for c in codes for t in c.threads} == {threading.get_ident()}


# ---- the type, the factory, the loader --------------------------------------------------------


class TestTheStepClass:
    def test_the_serialize_tag_is_the_class_name_and_no_object_is_written(self) -> None:
        registry = _registry(_Code())
        workflow = create_workflow("wf", "W").with_initial_step(_step(registry, "code"))
        document = workflow.serialize()["steps"]["x"]
        assert document["type"] == "ExecutableStep"
        assert "registry" not in document and "callables" not in document
        assert document["target"] == "code"
        assert document["inputs"] == {"topic": "subject"}

    def test_both_routing_fields_are_referenced_with_no_override(self) -> None:
        step = _step(ExecutableRegistry())
        assert step.referenced_step_ids() == {"next", "err"}
        assert "referenced_step_ids" not in vars(ExecutableStep)

    def test_an_unknown_keyword_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _step(ExecutableRegistry(), llm_interface=object())

    def test_a_target_and_a_registry_are_required(self) -> None:
        with pytest.raises(ValidationError):
            ExecutableStep(step_id="x", name="X", registry=ExecutableRegistry())
        with pytest.raises(ValidationError, match="registry"):
            ExecutableStep(step_id="x", name="X", target="t", registry=object())
        with pytest.raises(ValidationError):
            _step(ExecutableRegistry(), "")

    def test_the_factory_builds_the_same_step(self) -> None:
        registry = ExecutableRegistry()
        built = executable_step(
            "x",
            "X",
            "demo",
            registry,
            inputs={"topic": "subject"},
            returns={"final": "result"},
            success_step_id="next",
            error_step_id="err",
            max_steps=5,
            max_seconds=2.0,
            timeout_seconds=9.0,
        )
        assert isinstance(built, ExecutableStep)
        assert built == _step(
            registry, max_steps=5, max_seconds=2.0, timeout_seconds=9.0
        )

    def test_the_module_imports_no_api_and_no_agents_or_reasoning(self) -> None:
        import fsm_llm.workflows.steps.executable as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
        assert "API" not in names
        for forbidden in ("fsm_llm.agents", "fsm_llm.reasoning", "fsm_llm.engine.api"):
            assert not any(
                n == forbidden or n.startswith(forbidden + ".") for n in names
            )
        # the executable side comes only from the executables package and the context
        executable_side = [n for n in names if n.startswith("fsm_llm.engine")]
        assert set(executable_side) <= {
            "fsm_llm.engine.context",
            "fsm_llm.engine.executables",
        }

    def test_the_loader_knows_the_class_through_its_one_table(self) -> None:
        assert builtin_step_classes()["ExecutableStep"] is ExecutableStep


class TestRoundTripThroughTheLoader:
    def _objects(self, registry: Any) -> NameRegistry[Any]:
        objects: NameRegistry[Any] = NameRegistry("callables")
        objects.register("exec_registry", registry)
        objects.register("run_objects", objects)
        return objects

    def test_load_of_serialize_equals_the_definition(self) -> None:
        registry = _registry(_Code())
        objects = self._objects(registry)
        workflow = (
            create_workflow("wf", "W")
            .with_initial_step(
                _step(
                    registry,
                    "code",
                    callables=objects,
                    max_steps=4,
                    max_seconds=2.5,
                    timeout_seconds=7.0,
                )
            )
            .with_step(auto_step("next", "N", next_step_id=""))
            .with_step(auto_step("err", "E", next_step_id=""))
        )
        document = workflow.serialize(callables=objects)
        assert document["steps"]["x"]["registry"] == "exec_registry"
        assert document["steps"]["x"]["callables"] == "run_objects"
        assert document["steps"]["x"]["type"] == "ExecutableStep"
        assert load_workflow(document, objects=objects) == workflow

    def test_a_registry_the_document_cannot_name_is_refused_at_serialize(self) -> None:
        from fsm_llm.workflows.exceptions import WorkflowDefinitionError

        objects: NameRegistry[Any] = NameRegistry("callables")
        workflow = create_workflow("wf", "W").with_initial_step(
            _step(_registry(_Code()), "code", success_step_id="", error_step_id=None)
        )
        with pytest.raises(WorkflowDefinitionError, match="registry"):
            workflow.serialize(callables=objects)

    def test_a_document_naming_an_unregistered_registry_is_one_error(self) -> None:
        registry = _registry(_Code())
        objects = self._objects(registry)
        workflow = create_workflow("wf", "W").with_initial_step(
            _step(registry, "code", success_step_id="", error_step_id=None)
        )
        document = workflow.serialize(callables=objects)
        document["steps"]["x"]["registry"] = "nowhere"
        with pytest.raises(WorkflowValidationError, match="invalid") as info:
            load_workflow(document, objects=objects)
        assert any("nowhere" in e for e in info.value.validation_errors)
