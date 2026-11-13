"""Contract of the reply builders, `state_in_prompt` and `hash_embed`.

The builders make the three `CompletionResponse` shapes a scripted double
returns; `state_in_prompt` reads the one tag the response prompt carries, with a
drift test against the real prompt builder; `hash_embed` is a pure embedder.
Expected `kind` values come from `fsm_llm.core.codes`, never from literals.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

import fsm_llm.testing as public
from fsm_llm import API
from fsm_llm.context.layout import ResponseGenerationPromptBuilder
from fsm_llm.core.codes import CompletionKind
from fsm_llm.core.definitions import CompletionResponse, ModelToolCall
from fsm_llm.testing import (
    ScriptedLLM,
    final_reply,
    hash_embed,
    malformed_reply,
    state_in_prompt,
    tool_call_reply,
)
from fsm_llm.testing import doubles as doubles_module


def _fsm() -> dict:
    return {
        "name": "tag_probe",
        "description": "two states",
        "initial_state": "intake",
        "states": {
            "intake": {
                "id": "intake",
                "description": "ask",
                "purpose": "collect the name",
                "response_instructions": "Ask for the name",
                "transitions": [
                    {
                        "target_state": "wrap_up",
                        "description": "always",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "always",
                                "logic": {"==": [1, 1]},
                            }
                        ],
                    }
                ],
            },
            "wrap_up": {
                "id": "wrap_up",
                "description": "end",
                "purpose": "wrap up",
                "response_instructions": "Say goodbye",
            },
        },
    }


class TestFinalReply:
    def test_is_a_final_completion_response(self) -> None:
        reply = final_reply("hello")
        assert isinstance(reply, CompletionResponse)
        assert reply.kind == CompletionKind.FINAL
        assert reply.text == "hello"
        assert reply.calls == ()
        assert reply.finish_reason is None
        assert not reply.truncated

    def test_finish_reason_length_is_a_cut_reply(self) -> None:
        assert final_reply("half a sent", finish_reason="length").truncated

    def test_none_text_is_a_reply_with_no_text(self) -> None:
        assert final_reply(None).text is None


class TestToolCallReply:
    def test_one_call(self) -> None:
        reply = tool_call_reply(("lookup", {"city": "Rome"}))
        assert reply.kind == CompletionKind.CALLS
        assert reply.calls == (
            ModelToolCall(id="call_1", name="lookup", arguments={"city": "Rome"}),
        )

    def test_several_calls_get_distinct_ids_in_order(self) -> None:
        reply = tool_call_reply(("a", {}), ("b", {"x": 1}), ("c", {}))
        assert [c.id for c in reply.calls] == ["call_1", "call_2", "call_3"]
        assert [c.name for c in reply.calls] == ["a", "b", "c"]

    def test_text_beside_the_calls_is_kept(self) -> None:
        assert tool_call_reply(("a", {}), text="checking").text == "checking"

    def test_no_call_is_a_value_error_naming_the_fix(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            tool_call_reply()

    def test_arguments_are_copied(self) -> None:
        args = {"k": 1}
        reply = tool_call_reply(("a", args))
        args["k"] = 2
        assert reply.calls[0].arguments == {"k": 1}


class TestMalformedReply:
    def test_is_a_malformed_completion_with_no_calls(self) -> None:
        reply = malformed_reply()
        assert reply.kind == CompletionKind.MALFORMED
        assert reply.calls == ()
        assert reply.text is None

    def test_text_is_kept(self) -> None:
        assert malformed_reply("not json").text == "not json"


class TestStateInPrompt:
    def test_reads_the_tag(self) -> None:
        assert state_in_prompt("a <current_state>intake</current_state> b") == "intake"

    def test_no_tag_is_none(self) -> None:
        assert state_in_prompt("no tag here") is None

    def test_the_first_tag_wins(self) -> None:
        text = "<current_state>a</current_state><current_state>b</current_state>"
        assert state_in_prompt(text) == "a"

    def test_there_is_one_tag_regex_in_the_package(self) -> None:
        sources = [
            path.read_text(encoding="utf-8")
            for path in sorted(Path(public.__file__).parent.glob("*.py"))
        ]
        assert sum(s.count('re.compile(r"<current_state>') for s in sources) == 1

    def test_prompt_grounded_llm_reads_the_same_regex(self) -> None:
        tag = r"<current_state>([^<]+)</current_state>"
        assert doubles_module._CURRENT_STATE_TAG.pattern == tag

    @pytest.mark.parametrize("state_id", ["intake", "wrap_up"])
    def test_every_state_of_a_real_definition_round_trips(self, state_id: str) -> None:
        llm = ScriptedLLM(responses={"intake": "hi", "wrap_up": "bye"})
        api = API(fsm_definition=_fsm(), llm_interface=llm)
        conversation_id, _ = api.start_conversation()
        instance = api.fsm_manager.instances[conversation_id]
        definition = api.fsm_manager.get_fsm_definition(instance.fsm_id)
        prompt = ResponseGenerationPromptBuilder().build_response_prompt(
            instance, definition.states[state_id], definition
        )
        assert state_in_prompt(prompt) == state_id

    def test_a_scripted_reply_is_chosen_by_the_state_the_pipeline_wrote(self) -> None:
        llm = ScriptedLLM(responses={"intake": "hi", "wrap_up": "bye"})
        api = API(fsm_definition=_fsm(), llm_interface=llm)
        conversation_id, greeting = api.start_conversation()
        assert greeting == "hi"
        assert api.converse("go", conversation_id) == "bye"


class TestHashEmbed:
    def test_is_a_pure_function_of_the_text(self) -> None:
        embed = hash_embed()
        assert embed("the quick fox") == embed("the quick fox")
        assert hash_embed()("the quick fox") == embed("the quick fox")

    def test_dimension_defaults_to_eight_and_is_chosen(self) -> None:
        assert len(hash_embed()("x")) == 8
        assert len(hash_embed(dim=3)("x")) == 3
        assert len(hash_embed(64)("x")) == 64

    def test_unit_length_for_every_text_including_empty(self) -> None:
        embed = hash_embed()
        for text in ("", "a", "many words in one sentence here", "!!!"):
            vector = embed(text)
            assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)
            assert all(isinstance(v, float) for v in vector)

    def test_similar_text_is_closer_than_unrelated_text(self) -> None:
        embed = hash_embed(dim=64)

        def cosine(a: str, b: str) -> float:
            va, vb = embed(a), embed(b)
            return sum(x * y for x, y in zip(va, vb, strict=True))

        close = cosine("book a table for two", "book a table for four")
        far = cosine("book a table for two", "quarterly tax filing deadline")
        assert close > far

    def test_case_and_punctuation_do_not_matter(self) -> None:
        embed = hash_embed()
        assert embed("Hello, World!") == embed("hello world")

    @pytest.mark.parametrize("dim", [0, -1])
    def test_a_non_positive_dimension_is_refused(self, dim: int) -> None:
        with pytest.raises(ValueError, match="dim"):
            hash_embed(dim)

    def test_a_non_int_dimension_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            hash_embed(2.5)  # type: ignore[arg-type]


class TestExports:
    @pytest.mark.parametrize(
        "name",
        [
            "ScriptedLLM",
            "UnscriptedCallError",
            "final_reply",
            "tool_call_reply",
            "malformed_reply",
            "state_in_prompt",
            "hash_embed",
        ],
    )
    def test_public_with_an_interface_contract(self, name: str) -> None:
        assert name in public.__all__
        assert "Interface contract" in (inspect.getdoc(getattr(public, name)) or "")
