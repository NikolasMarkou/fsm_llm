"""The policy layer (`fsm_llm.policy`) and its seam with `fsm_llm.agents`.

Step 86 of plan d454fd62 moved the approval types and the human-in-the-loop manager
(`HumanInTheLoop`, `ApprovalRequest`, `ApprovalPolicy`, `ApprovalCallback`,
`EscalationCallback`, `ApprovalDeniedError`) from `fsm_llm.agents` down into the layer
package `fsm_llm.policy`, and left the gate handler `make_hitl_checker` in
`fsm_llm.agents.tools` (D-037). These tests pin what that move promised:

- the agents constants classes keep every attribute they had, and each policy-layer
  attribute is the SAME object (one source of value, `vars()` still lists it);
- `fsm_llm.agents` re-exports the policy-layer objects, and the old module paths no
  longer define them (no shim);
- the layer is below the agents package: importing it loads no agents module and its
  source names no module above it;
- the log records of the manager carry the new module name;
- `make_hitl_checker`, which no other test named, behaves as before.

The behaviour of the manager itself (timeouts, redaction, the driver-only grant) is
pinned by `test_hitl.py`, `test_hitl_security.py`, `test_trust_boundary.py` and
`test_secret_hygiene.py`, which were only retargeted.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from loguru import logger

import fsm_llm.agents as agents
import fsm_llm.agents.constants as agents_constants
import fsm_llm.agents.tools as agents_tools
import fsm_llm.policy as policy
import fsm_llm.policy.constants as policy_constants
from fsm_llm.core.definitions import (
    AgentError,
    FieldExtractionRequest,
    FieldExtractionResponse,
    ResponseGenerationRequest,
    ResponseGenerationResponse,
)
from fsm_llm.llm import LLMInterface
from fsm_llm.tools import ToolCall, ToolRegistry
from tests.test_fsm_llm_agents.support import _layer_probe_output

SRC = Path(policy.__file__).resolve().parents[1]
_ATTR_CLASSES = ("Defaults", "LogMessages")


def _public_attributes(cls: type) -> dict[str, object]:
    return {k: v for k, v in vars(cls).items() if not k.startswith("__")}


class TestConstantsKeepOneSource:
    @pytest.mark.parametrize("name", _ATTR_CLASSES)
    def test_every_policy_attribute_is_the_same_object_in_the_agents_class(
        self, name: str
    ) -> None:
        policy_cls = getattr(policy_constants, name)
        agents_cls = getattr(agents_constants, name)
        attributes = _public_attributes(policy_cls)
        assert attributes, f"{name} of the policy layer is empty"
        for attr, value in attributes.items():
            assert attr in vars(agents_cls), (
                f"{name}.{attr} left vars() of the agents class"
            )
            assert vars(agents_cls)[attr] is value, (
                f"{name}.{attr} is a copy, not the policy value"
            )

    @pytest.mark.parametrize("name", _ATTR_CLASSES)
    def test_the_agents_class_is_its_own_class_not_a_subclass(self, name: str) -> None:
        agents_cls = getattr(agents_constants, name)
        assert agents_cls is not getattr(policy_constants, name)
        assert agents_cls.__bases__ == (object,)

    def test_the_policy_values_are_the_documented_ones(self) -> None:
        # a golden copy of what the manager depends on: a change here is a
        # behaviour change (a default threshold, a log line), not a refactor
        assert policy_constants.Defaults.CONFIDENCE_THRESHOLD == 0.3
        assert policy_constants.LogMessages.APPROVAL_REQUESTED == (
            "Requesting approval for: {action}"
        )
        assert policy_constants.LogMessages.APPROVAL_RESULT == (
            "Approval {result} for: {action}"
        )
        assert policy_constants.LogMessages.ESCALATION == (
            "Escalating to human: {reason}"
        )

    def test_the_default_threshold_is_what_a_manager_uses(self) -> None:
        assert policy.HumanInTheLoop().confidence_threshold == 0.3
        assert agents_constants.Defaults.CONFIDENCE_THRESHOLD == 0.3


class TestTheAgentsFacadeReexports:
    @pytest.mark.parametrize("name", sorted(policy.__all__))
    def test_the_public_name_is_the_policy_layer_object(self, name: str) -> None:
        assert getattr(agents, name) is getattr(policy, name)
        assert name in agents.__all__

    @pytest.mark.parametrize(
        "name, home",
        [
            ("HumanInTheLoop", "fsm_llm.policy.hitl"),
            ("ApprovalRequest", "fsm_llm.policy.approval"),
            ("ApprovalDeniedError", "fsm_llm.policy.approval"),
        ],
    )
    def test_each_class_is_defined_in_its_module(self, name: str, home: str) -> None:
        assert getattr(policy, name).__module__ == home

    def test_approval_denied_error_is_an_agent_error_of_core(self) -> None:
        assert issubclass(policy.ApprovalDeniedError, AgentError)
        assert agents.AgentError is AgentError
        with pytest.raises(AgentError, match="Approval denied for action: x"):
            raise policy.ApprovalDeniedError("x")
        assert policy.ApprovalDeniedError("y").action_description == "y"

    def test_the_synthesizer_reads_the_same_manager(self) -> None:
        from fsm_llm.synthesis import synthesizer

        assert synthesizer.HumanInTheLoop is policy.HumanInTheLoop


class TestOldPathsAreGone:
    """No shim: a moved name is not DEFINED at its old module."""

    def test_the_hitl_module_is_gone_from_agents(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("fsm_llm.agents.hitl")

    @pytest.mark.parametrize(
        "module, names",
        [
            ("fsm_llm.agents.definitions", ["ApprovalRequest"]),
            ("fsm_llm.agents.exceptions", ["ApprovalDeniedError"]),
        ],
    )
    def test_the_module_defines_none_of_them(
        self, module: str, names: list[str]
    ) -> None:
        path = Path(importlib.import_module(module).__file__ or "")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        assert not set(names) & defined
        for name in names:
            assert not hasattr(importlib.import_module(module), name)

    def test_make_hitl_checker_stays_in_the_agents_tools_module(self) -> None:
        assert agents_tools.make_hitl_checker.__module__ == "fsm_llm.agents.tools"
        assert not hasattr(policy, "make_hitl_checker")
        assert "make_hitl_checker" not in policy.__all__


class TestTheLayerIsBelowTheAgents:
    _PROBE = (
        "import sys\n"
        "import fsm_llm.policy\n"
        "import fsm_llm.policy.hitl, fsm_llm.policy.approval, fsm_llm.policy.constants\n"
        "above = ('agents', 'reasoning', 'workflows', 'monitor', 'eval', 'synthesis',\n"
        "         'okf', 'skills', 'testing', 'runner')\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if any(m == f'fsm_llm.{a}' or m.startswith(f'fsm_llm.{a}.') for a in above))\n"
        "print('LOADED', bad)\n"
    )

    def test_importing_it_loads_no_subpackage_and_no_peer_layer(self) -> None:
        out = _layer_probe_output(self._PROBE)
        assert out.strip() == "LOADED []", out

    def test_the_modules_of_the_layer_are_exactly_these(self) -> None:
        assert sorted(p.name for p in (SRC / "policy").glob("*.py")) == [
            "__init__.py",
            "approval.py",
            "constants.py",
            "hitl.py",
        ]

    def test_no_module_of_the_layer_names_a_higher_module_in_its_source(self) -> None:
        forbidden = (
            "fsm_llm.agents",
            "fsm_llm.context",
            "fsm_llm.engine",
            "fsm_llm.handlers",
            "fsm_llm.llm",
            "fsm_llm.okf",
            "fsm_llm.skills",
            "fsm_llm.synthesis",
        )
        for path in sorted((SRC / "policy").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for name in names:
                    assert not name.startswith(forbidden), f"{path.name}: {name}"

    def test_the_tools_layer_does_not_know_the_policy_layer(self) -> None:
        for path in sorted((SRC / "tools").glob("*.py")):
            assert "fsm_llm.policy" not in path.read_text(encoding="utf-8"), path.name


class TestTheFacade:
    def test_every_name_of_all_resolves_and_nothing_else_is_public(self) -> None:
        for name in policy.__all__:
            assert hasattr(policy, name)
        public = {
            n for n in vars(policy) if not n.startswith("_") and n != "annotations"
        }
        assert public - {"approval", "constants", "hitl"} == set(policy.__all__)
        assert sorted(policy.__all__) == [
            "ApprovalCallback",
            "ApprovalDeniedError",
            "ApprovalPolicy",
            "ApprovalRequest",
            "EscalationCallback",
            "HumanInTheLoop",
        ]


class TestLogRecordsCarryTheNewModuleName:
    def test_the_approval_records_come_from_the_policy_module(self) -> None:
        records: list[tuple[str, str]] = []
        sink = logger.add(
            lambda m: records.append((m.record["name"], m.record["message"])),
            level="INFO",
        )
        logger.enable("fsm_llm")
        try:
            hitl = policy.HumanInTheLoop(approval_callback=lambda request: True)
            hitl.request_approval(ToolCall(tool_name="send", parameters={"a": 1}), {})
            hitl.escalate("low", {})
        finally:
            logger.disable("fsm_llm")
            logger.remove(sink)
        assert records == [
            ("fsm_llm.policy.hitl", "Requesting approval for: send({'a': 1})"),
            ("fsm_llm.policy.hitl", "Approval APPROVED for: send"),
            ("fsm_llm.policy.hitl", "Escalating to human: low"),
        ]


class TestTheApprovalRequestModel:
    def test_the_fields_and_their_defaults(self) -> None:
        request = policy.ApprovalRequest(tool_name="send")
        assert list(policy.ApprovalRequest.model_fields) == [
            "tool_name",
            "parameters",
            "reasoning",
            "context_summary",
        ]
        assert request.model_dump() == {
            "tool_name": "send",
            "parameters": {},
            "reasoning": "",
            "context_summary": {},
        }

    def test_the_defaults_are_not_shared_between_requests(self) -> None:
        first = policy.ApprovalRequest(tool_name="a")
        first.parameters["x"] = 1
        first.context_summary["y"] = 2
        second = policy.ApprovalRequest(tool_name="b")
        assert second.parameters == {}
        assert second.context_summary == {}

    def test_a_request_needs_a_tool_name(self) -> None:
        with pytest.raises(ValueError):
            policy.ApprovalRequest()  # type: ignore[call-arg]


class TestWhatTheApproverIsShown:
    """The context summary handed to the callback is a human-facing surface: internal
    and secret-looking entries are kept out of it (the old suite pinned only the secret
    ones), while the call's `parameters` stay raw because the approver must see the
    exact call it authorises (D-016 of plan 06a5ec0a)."""

    def _summary_and_request(self, context: dict, parameters: dict | None = None):
        seen: list[policy.ApprovalRequest] = []

        def callback(request: policy.ApprovalRequest) -> bool:
            seen.append(request)
            return True

        hitl = policy.HumanInTheLoop(approval_callback=callback)
        call = ToolCall(tool_name="send", parameters=parameters or {})
        assert hitl.request_approval(call, context) is True
        return seen[0]

    @pytest.mark.parametrize(
        "key",
        [
            "_internal",
            "_Mixed",
            "system_note",
            "SYSTEM_A",
            "internal_y",
            "Internal_Z",
            "__dunder",
        ],
    )
    def test_internal_prefix_keys_never_reach_the_approver(self, key: str) -> None:
        request = self._summary_and_request({key: "hidden", "task": "visible"})
        assert request.context_summary == {"task": "visible"}

    def test_the_observations_never_reach_the_approver(self) -> None:
        request = self._summary_and_request(
            {"observations": ["a long tool output"], "task": "t"}
        )
        assert request.context_summary == {"task": "t"}

    def test_a_secret_looking_entry_is_dropped_and_a_nested_one_redacted(self) -> None:
        request = self._summary_and_request(
            {
                "api_key": "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4",
                "nested": {"password": "hunter2", "n": 1},
                "task": "t",
            }
        )
        assert "api_key" not in request.context_summary
        assert request.context_summary["task"] == "t"
        assert request.context_summary["nested"] == {
            "password": "<redacted>",
            "n": 1,
        }

    def test_the_parameters_stay_raw_and_the_log_line_redacts_them(self) -> None:
        records: list[str] = []
        sink = logger.add(lambda m: records.append(m.record["message"]), level="INFO")
        logger.enable("fsm_llm")
        try:
            request = self._summary_and_request(
                {}, {"api_key": "sk-raw-value", "to": "a"}
            )
        finally:
            logger.disable("fsm_llm")
            logger.remove(sink)
        assert request.parameters == {"api_key": "sk-raw-value", "to": "a"}
        assert "sk-raw-value" not in records[0]
        assert "<redacted>" in records[0]


class _SelectsSend(LLMInterface):
    """Selects the gated tool `send` on the first think turn, then concludes."""

    def __init__(self) -> None:
        self.model = "mock-model"
        self.selected = 0

    def extract_field(self, request: FieldExtractionRequest) -> FieldExtractionResponse:
        name = request.field_name
        if name == "tool_name":
            self.selected += 1
            value: object = "send" if self.selected == 1 else "none"
        else:
            value = {
                "tool_input": {"x": "1"},
                "reasoning": "send it",
                "should_terminate": self.selected > 1,
                "final_answer": "done",
            }.get(name, "text")
        return FieldExtractionResponse(
            field_name=name, value=value, confidence=0.9, reasoning="m", is_valid=True
        )

    def generate_response(
        self, request: ResponseGenerationRequest
    ) -> ResponseGenerationResponse:
        return ResponseGenerationResponse(
            message="ok", message_type="response", reasoning="mock"
        )


class TestOnlyTheValueTrueApproves:
    """The driver reads the callback's answer with `is True`: a truthy non-bool answer
    (`"yes"`, `1`) is a denial, so a callback that returns a string cannot approve by
    accident (the old suite pinned forged context values, not the callback's reply)."""

    def _run(self, answer: object) -> list[str]:
        executions: list[str] = []
        registry = ToolRegistry()

        def send(params: dict) -> str:
            executions.append("send")
            return "sent"

        registry.register_function(
            send,
            name="send",
            description="Send it",
            parameter_schema={"properties": {"x": {"type": "string"}}},
            requires_approval=True,
        )
        hitl = policy.HumanInTheLoop(approval_callback=lambda request: answer)  # type: ignore[arg-type, return-value]
        agent = agents.ReactAgent(
            tools=registry,
            config=agents.AgentConfig(max_iterations=4, model="mock/model"),
            llm_interface=_SelectsSend(),
            hitl=hitl,
        )
        agent.run("send the report")
        return executions

    def test_true_runs_the_gated_tool(self) -> None:
        assert self._run(True) == ["send"]

    @pytest.mark.parametrize("answer", ["yes", 1, [1], False, None, 0])
    def test_anything_else_does_not(self, answer: object) -> None:
        assert self._run(answer) == []


class TestTheApprovalGateHandler:
    """`make_hitl_checker` (agents/tools.py) had no direct test before the move."""

    def _checker(self, **kwargs):
        hitl = policy.HumanInTheLoop(
            approval_policy=lambda call, ctx: call.tool_name == "send"
        )
        return agents_tools.make_hitl_checker(hitl, **kwargs)

    def test_a_gated_tool_sets_approval_required(self) -> None:
        check = self._checker()
        assert check({"tool_name": "send", "tool_input": {"to": "a"}}) == {
            "approval_required": True
        }

    def test_an_ungated_tool_clears_approval_required(self) -> None:
        check = self._checker()
        assert check({"tool_name": "search"}) == {"approval_required": False}

    @pytest.mark.parametrize("context", [{}, {"tool_name": ""}, {"tool_name": "none"}])
    def test_no_tool_changes_nothing(self, context: dict) -> None:
        assert self._checker()(context) == {}

    def test_the_policy_argument_replaces_the_manager_policy(self) -> None:
        seen: list[tuple[str, dict, dict]] = []

        def decide(call, ctx):
            seen.append((call.tool_name, call.parameters, ctx))
            return True

        check = self._checker(policy=decide)
        context = {"tool_name": "search", "tool_input": '{"q": "x"}', "reasoning": "r"}
        assert check(context) == {"approval_required": True}
        assert seen == [("search", {"q": "x"}, context)]

    def test_a_manager_without_a_policy_gates_nothing(self) -> None:
        check = agents_tools.make_hitl_checker(policy.HumanInTheLoop())
        assert check({"tool_name": "send"}) == {"approval_required": False}
