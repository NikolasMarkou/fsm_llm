"""Contract of the promoted doubles `MockLLM2Interface` and `PromptGroundedLLM`.

Step 22 of plan d454fd62 moved both classes out of `tests/conftest.py` into
`fsm_llm.testing` (step 26 then moved every importer to the public module). The move is byte-equivalent apart from one dead parameter
(`MockLLM2Interface(transition_target=)`, stored and never read), so these tests
pin what the repo's own tests rely on: mutable attributes, subclassing,
`call_history.clear()`, the cut-off flags, evidence grounding, bulk quoting and
`complete` raising. `TestPromptGroundedLLM` in
`tests/test_fsm_llm_agents/test_grounded_patterns.py` stays the second oracle.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

import fsm_llm.testing as public
import tests.conftest as conftest
from fsm_llm import API
from fsm_llm.definitions import (
    BulkExtractionRequest,
    CompletionRequest,
    FieldExtractionRequest,
    FSMDefinition,
    LLMResponseError,
    ResponseGenerationRequest,
)
from fsm_llm.llm import LLMInterface, implements_complete
from fsm_llm.testing import MockLLM2Interface, PromptGroundedLLM

_REPO = Path(__file__).resolve().parents[2]


def _field(name: str = "city", **overrides: object) -> FieldExtractionRequest:
    fields: dict[str, object] = {
        "system_prompt": "extract",
        "user_message": "hello",
        "field_name": name,
    }
    fields.update(overrides)
    return FieldExtractionRequest.model_validate(fields)


def _reply(system_prompt: str = "reply") -> ResponseGenerationRequest:
    return ResponseGenerationRequest(system_prompt=system_prompt, user_message="hi")


def _bulk(system_prompt: str, user_message: str = "hi") -> BulkExtractionRequest:
    return BulkExtractionRequest(system_prompt=system_prompt, user_message=user_message)


def _two_state_fsm() -> dict:
    return {
        "name": "doubles_probe",
        "description": "start asks for a name, done ends",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "ask",
                "purpose": "collect the name",
                "required_context_keys": ["user_name"],
                "response_instructions": "Ask for the name",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "name known",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "name present",
                                "requires_context_keys": ["user_name"],
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "end",
                "purpose": "wrap up",
                "response_instructions": "Say goodbye",
            },
        },
    }


class TestExports:
    def test_both_names_are_public_and_static(self) -> None:
        assert {"MockLLM2Interface", "PromptGroundedLLM"} <= set(public.__all__)
        tree = ast.parse(inspect.getsource(public))
        (assign,) = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and [t.id for t in n.targets if isinstance(t, ast.Name)] == ["__all__"]
        ]
        assert isinstance(assign.value, (ast.List, ast.Tuple))

    def test_no_test_module_imports_the_doubles_from_conftest(self) -> None:
        # Step 26 of plan d454fd62 moved every importer to `fsm_llm.testing` and
        # removed the re-exports: a test imports the doubles from the public module.
        moved = {"MockLLM2Interface", "PromptGroundedLLM", "_CURRENT_STATE_TAG"}
        offenders = []
        for path in sorted((_REPO / "tests").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (
                    (node.module or "").rsplit(".", 1)[-1] == "conftest"
                ):
                    hit = moved & {alias.name for alias in node.names}
                elif isinstance(node, ast.Attribute) and node.attr in moved:
                    hit = (
                        {node.attr}
                        if isinstance(node.value, ast.Name)
                        and node.value.id == "conftest"
                        else set()
                    )
                else:
                    continue
                if hit:
                    where = f"{path.relative_to(_REPO).as_posix()}:{node.lineno}"
                    offenders.append(f"{where} {sorted(hit)}")
        assert offenders == []
        assert not hasattr(conftest, "PromptGroundedLLM")
        assert not hasattr(conftest, "_CURRENT_STATE_TAG")

    def test_conftest_defines_no_class_of_its_own_double(self) -> None:
        tree = ast.parse(inspect.getsource(conftest))
        classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        assert classes == set()

    @pytest.mark.parametrize("cls", [MockLLM2Interface, PromptGroundedLLM])
    def test_class_carries_an_interface_contract(self, cls: type) -> None:
        assert cls.__module__ == "fsm_llm.testing.doubles"
        assert "Interface contract" in (inspect.getdoc(cls) or "")
        assert issubclass(cls, LLMInterface)

    def test_the_dead_transition_target_is_gone(self) -> None:
        assert (
            "transition_target" not in inspect.signature(MockLLM2Interface).parameters
        )
        with pytest.raises(TypeError):
            MockLLM2Interface(transition_target="x")  # type: ignore[call-arg]
        assert not hasattr(MockLLM2Interface(), "transition_target")

    def test_no_call_site_passes_transition_target(self) -> None:
        offenders = []
        for root in ("tests", "scripts", "src", "examples", "evaluation"):
            for path in sorted((_REPO / root).rglob("*.py")):
                if "__pycache__" in path.parts or path == Path(__file__).resolve():
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    named_keyword = (
                        isinstance(node, ast.keyword)
                        and node.arg == "transition_target"
                    )
                    named_attribute = (
                        isinstance(node, ast.Attribute)
                        and node.attr == "transition_target"
                    )
                    hit = named_keyword or named_attribute
                    if hit:
                        offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
        assert offenders == []


class TestMockLLM2Interface:
    def test_defaults(self) -> None:
        llm = MockLLM2Interface()
        assert llm.extraction_data == {}
        assert llm.response_text == "Hello! How can I help you?"
        assert llm.call_history == []
        assert llm.cut_reply is False
        assert llm.cut_fields == set()

    def test_none_and_empty_extraction_data_are_both_an_empty_dict(self) -> None:
        assert MockLLM2Interface(extraction_data=None).extraction_data == {}
        assert MockLLM2Interface(extraction_data={}).extraction_data == {}

    def test_a_known_field_comes_back_valid(self) -> None:
        llm = MockLLM2Interface(extraction_data={"city": "Paris"})

        response = llm.extract_field(_field("city"))

        assert (response.value, response.confidence, response.is_valid) == (
            "Paris",
            1.0,
            True,
        )
        assert response.truncated is False

    def test_an_unknown_field_is_null_invalid_and_not_cut(self) -> None:
        response = MockLLM2Interface(extraction_data={"a": 1}).extract_field(
            _field("b")
        )

        assert (response.value, response.confidence, response.is_valid) == (
            None,
            0.0,
            False,
        )
        assert response.truncated is False

    def test_a_cut_field_is_null_and_truncated_even_when_known(self) -> None:
        llm = MockLLM2Interface(extraction_data={"city": "Paris"}, cut_fields=["city"])

        response = llm.extract_field(_field("city"))

        assert (response.value, response.is_valid, response.truncated) == (
            None,
            False,
            True,
        )

    def test_cut_fields_is_copied_into_a_set(self) -> None:
        names = ["a", "a", "b"]
        assert MockLLM2Interface(cut_fields=names).cut_fields == {"a", "b"}

    def test_the_reply_is_the_fixed_text_and_carries_the_cut_flag(self) -> None:
        plain = MockLLM2Interface(response_text="fixed").generate_response(_reply())
        cut = MockLLM2Interface(response_text="fixed", cut_reply=True)

        assert (plain.message, plain.message_type, plain.truncated) == (
            "fixed",
            "response",
            False,
        )
        assert cut.generate_response(_reply()).truncated is True

    def test_call_history_records_both_calls_in_order_and_can_be_cleared(
        self,
    ) -> None:
        llm = MockLLM2Interface()
        field_request, reply_request = _field(), _reply()

        llm.extract_field(field_request)
        llm.generate_response(reply_request)

        assert llm.call_history == [
            ("extract_field", field_request),
            ("generate_response", reply_request),
        ]
        history = llm.call_history
        llm.call_history.clear()
        assert llm.call_history == [] and llm.call_history is history

    def test_attributes_changed_between_turns_take_effect(self) -> None:
        llm = MockLLM2Interface(extraction_data={"city": "Paris"})

        assert llm.extract_field(_field("city")).value == "Paris"
        llm.extraction_data = {"city": "Rome"}
        assert llm.extract_field(_field("city")).value == "Rome"
        llm.cut_fields.add("city")
        assert llm.extract_field(_field("city")).truncated is True
        llm.cut_reply = True
        llm.response_text = "changed"
        reply = llm.generate_response(_reply())
        assert (reply.message, reply.truncated) == ("changed", True)

    def test_a_subclass_can_wrap_the_calls_and_forward_args_and_kwargs(self) -> None:
        class Counting(MockLLM2Interface):
            def __init__(self, *args: object, step: int = 1, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)  # type: ignore[arg-type]
                self.step = step
                self.seen = 0

            def extract_field(self, request: FieldExtractionRequest):  # type: ignore[no-untyped-def]
                self.seen += self.step
                return super().extract_field(request)

        llm = Counting({"city": "Paris"}, "text", step=3)

        assert llm.extract_field(_field("city")).value == "Paris"
        assert llm.seen == 3
        assert llm.response_text == "text"
        assert [name for name, _ in llm.call_history] == ["extract_field"]

    def test_it_implements_neither_bulk_nor_complete_nor_a_meter(self) -> None:
        llm = MockLLM2Interface()

        assert implements_complete(llm) is False
        with pytest.raises(NotImplementedError):
            llm.extract_bulk_data(_bulk("p"))
        with pytest.raises(NotImplementedError):
            llm.complete(CompletionRequest(messages=[{"role": "user", "content": "x"}]))
        assert not hasattr(llm, "usage")
        assert not hasattr(llm, "model")

    def test_it_drives_a_real_conversation_and_records_the_turns(self) -> None:
        llm = MockLLM2Interface(
            extraction_data={"user_name": "Ada"}, response_text="hi"
        )
        api = API(fsm_definition=_two_state_fsm(), llm_interface=llm)
        conversation_id, _greeting = api.start_conversation()

        api.converse("I am Ada", conversation_id)

        assert api.get_current_state(conversation_id) == "done"
        assert api.get_data(conversation_id)["user_name"] == "Ada"
        kinds = {name for name, _ in llm.call_history}
        assert kinds == {"extract_field", "generate_response"}


class TestPromptGroundedLLM:
    FACTS: ClassVar[dict[str, tuple[object, str]]] = {
        "city": ("Paris", "capital of France")
    }

    def test_defaults_and_the_model_attribute(self) -> None:
        llm = PromptGroundedLLM()
        assert llm.model == "prompt-grounded-fake"
        assert (llm.facts, llm.responses, llm.default_response) == ({}, {}, "ok")
        assert llm.requests == []

    def test_arguments_are_copied(self) -> None:
        facts, responses = dict(self.FACTS), {"s": "r"}
        llm = PromptGroundedLLM(facts=facts, responses=responses)
        facts["x"] = ("y", "z")
        responses["t"] = "u"
        assert set(llm.facts) == {"city"} and set(llm.responses) == {"s"}

    def test_evidence_in_prompt_message_or_context_grounds_a_field(self) -> None:
        llm = PromptGroundedLLM(facts=self.FACTS)
        seen = [
            llm.extract_field(_field(system_prompt="the capital of France")).value,
            llm.extract_field(_field(user_message="the capital of France?")).value,
            llm.extract_field(_field(context={"k": "capital of France"})).value,
        ]
        assert seen == ["Paris"] * 3

    def test_no_evidence_or_empty_evidence_or_unknown_field_is_null(self) -> None:
        llm = PromptGroundedLLM(facts={**self.FACTS, "blank": ("v", "")})

        for name in ("city", "blank", "other"):
            response = llm.extract_field(_field(name))
            assert (response.value, response.is_valid, response.confidence) == (
                None,
                False,
                0.0,
            )

    def test_bulk_returns_only_quoted_names_with_evidence_in_prompt_or_message(
        self,
    ) -> None:
        llm = PromptGroundedLLM(
            facts={
                "city": ("Paris", "capital of France"),
                "pet": ("cat", "my cat"),
                "hidden": ("x", "secret"),
            }
        )

        out = llm.extract_bulk_data(
            _bulk('Extract "city" and "pet" and "hidden"', "the capital of France")
        )
        assert out.extracted_data == {"city": "Paris"}
        assert llm.extract_bulk_data(_bulk("Extract things")).extracted_data == {}
        # the field-extraction context is not searched in bulk mode
        quoted_only = llm.extract_bulk_data(_bulk('"hidden"', "secret"))
        assert quoted_only.extracted_data == {"hidden": "x"}

    def test_the_reply_follows_the_current_state_tag(self) -> None:
        llm = PromptGroundedLLM(responses={"a": "reply a"}, default_response="dflt")

        tagged = llm.generate_response(_reply("<current_state>a</current_state> x"))
        other = llm.generate_response(_reply("<current_state>b</current_state>"))
        none = llm.generate_response(_reply("no tag"))

        assert (tagged.message, other.message, none.message) == (
            "reply a",
            "dflt",
            "dflt",
        )

    def test_complete_raises_and_is_not_recorded(self) -> None:
        llm = PromptGroundedLLM()
        request = CompletionRequest(messages=[{"role": "user", "content": "x"}])

        with pytest.raises(LLMResponseError, match="no completion replies"):
            llm.complete(request)
        assert llm.requests == []
        # an override exists, so core accepts it for a classifying FSM
        assert implements_complete(llm) is True

    def test_requests_record_every_other_call_and_calls_filters_by_kind(self) -> None:
        llm = PromptGroundedLLM(facts=self.FACTS)
        field_request, bulk_request, reply_request = _field(), _bulk("p"), _reply()

        llm.extract_field(field_request)
        llm.extract_bulk_data(bulk_request)
        llm.generate_response(reply_request)

        assert [kind for kind, _ in llm.requests] == [
            "extract_field",
            "extract_bulk_data",
            "generate_response",
        ]
        assert llm.calls("extract_field") == [field_request]
        assert llm.calls("generate_response") == [reply_request]
        assert llm.calls("nothing") == []

    def test_attributes_changed_between_turns_take_effect(self) -> None:
        llm = PromptGroundedLLM()
        assert (
            llm.extract_field(_field(system_prompt="capital of France")).value is None
        )
        llm.facts["city"] = ("Paris", "capital of France")
        assert (
            llm.extract_field(_field(system_prompt="capital of France")).value
            == "Paris"
        )

    def test_a_subclass_overriding_one_method_inherits_the_rest(self) -> None:
        class Broken(PromptGroundedLLM):
            def generate_response(self, request: ResponseGenerationRequest):  # type: ignore[no-untyped-def]
                raise LLMResponseError("down")

        llm = Broken(facts=self.FACTS)

        with pytest.raises(LLMResponseError, match="down"):
            llm.generate_response(_reply())
        assert llm.extract_field(_field(system_prompt="capital of France")).value == (
            "Paris"
        )

    def test_it_drives_a_real_conversation_through_grounded_bulk_extraction(
        self,
    ) -> None:
        definition = FSMDefinition.model_validate(_two_state_fsm())
        llm = PromptGroundedLLM(
            responses={"start": "asking", "done": "bye"}, default_response="ok"
        )
        api = API(fsm_definition=definition, llm_interface=llm)
        conversation_id, greeting = api.start_conversation()

        reply = api.converse("hello", conversation_id)

        assert greeting == "asking" and reply == "asking"
        assert api.get_current_state(conversation_id) == "start"
        assert llm.calls("generate_response")


class TestNoNetworkNoPytest:
    def test_both_doubles_run_with_every_socket_refused_and_without_pytest(
        self,
    ) -> None:
        code = (
            "import socket, sys\n"
            "def refuse(*a, **k):\n"
            "    raise AssertionError('socket used')\n"
            "socket.socket.connect = refuse\n"
            "socket.create_connection = refuse\n"
            "from fsm_llm.testing import MockLLM2Interface, PromptGroundedLLM\n"
            "from fsm_llm.definitions import ResponseGenerationRequest\n"
            "r = ResponseGenerationRequest(system_prompt='s', user_message='u')\n"
            "assert MockLLM2Interface().generate_response(r).message\n"
            "assert PromptGroundedLLM().generate_response(r).message == 'ok'\n"
            "assert 'pytest' not in sys.modules\n"
            "print('ok')\n"
        )

        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_REPO,
            env={
                "PATH": "",
                "OLLAMA_API_BASE": "http://127.0.0.1:9",
                "OLLAMA_HOST": "http://127.0.0.1:9",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
