"""A pushed child owns its synthesis result (step 55 of plan d454fd62, D-022).

A child pushed with ``inherit_context=True`` used to inherit the parent's ``synthesis_result``: its
synthesize state's edges read the key, so they routed on the stale result before any run wrote one.
The drop is on the child's initial context only (``KindSpec.inherits_result_on_push``, read by
``API._build_push_context``). These rows lived in ``test_runner.py`` and moved here when
``ProcedureRunner`` was deleted (plan baa37f40 step 22, D-041): they never needed the runner, only
a host with a synthesize state, which now takes an executor. A PASSIVE executor (the conversation
body under the kind's name) stands in: nothing here runs a procedure, the rule is the engine's.
"""

import json
from typing import Any

from fsm_llm.engine.api.conversation import API
from fsm_llm.engine.kinds import ConversationExecutor
from tests.test_fsm_llm_synthesis.conftest import ScriptedRunnerLLM
from tests.test_fsm_llm_synthesis.rig import CONTEXT, REPLIES, make_host

_STALE = {"status": "ok", "procedure_id": "stale", "version": "1.0", "reason": None}


def _api(context: dict[str, Any]) -> tuple[API, str]:
    api = API.from_definition(
        definition=make_host(),
        llm_interface=ScriptedRunnerLLM({"answer": "42"}, dict(REPLIES)),
        executors={"synthesize": ConversationExecutor()},
    )
    conversation_id, _ = api.start_conversation(context)
    return api, conversation_id


def _pushed_child() -> tuple[API, str]:
    api, conversation_id = _api({**CONTEXT, "synthesis_result": _STALE})
    api.push_fsm(conversation_id, make_host())  # inherit_context defaults to True
    return api, conversation_id


class TestPushedChildOwnsItsSynthesisResult:
    def test_the_child_waits_for_its_own_result_instead_of_routing_on_the_stale_one(
        self,
    ):
        api, conversation_id = _pushed_child()
        assert api.get_stack_depth(conversation_id) == 2
        api.advance(conversation_id)  # the child's start -> solve
        assert api.get_current_state(conversation_id) == "solve"

        api.advance(conversation_id)

        # no edge passes before a run wrote a result: the stale `ok` would have
        # taken the child on to `booked` without any procedure ever running
        assert api.get_current_state(conversation_id) == "solve"

    def test_the_child_does_not_see_the_inherited_result(self):
        api, conversation_id = _pushed_child()
        data = api.get_data(conversation_id)
        assert "synthesis_result" not in data
        assert data["topic"] == "tables"  # the rest is inherited as before

    def test_an_explicitly_passed_result_is_kept(self):
        api, conversation_id = _api({**CONTEXT, "synthesis_result": _STALE})
        handed = {**_STALE, "procedure_id": "handed"}
        api.push_fsm(
            conversation_id, make_host(), context_to_pass={"synthesis_result": handed}
        )
        assert api.get_data(conversation_id)["synthesis_result"] == handed

    def test_a_result_key_the_child_does_not_own_is_still_inherited(self):
        api, conversation_id = _api({**CONTEXT, "synthesis_result": _STALE})
        other = json.loads(
            json.dumps(make_host(result_key="other_result")).replace(
                "synthesis_result", "other_result"
            )
        )
        api.push_fsm(conversation_id, other)
        assert api.get_data(conversation_id)["synthesis_result"] == _STALE

    def test_the_parent_keeps_its_own_result_through_a_push_and_a_pop(self):
        api, conversation_id = _pushed_child()
        api.pop_fsm(conversation_id, context_to_return={"from_child": 1})
        assert api.get_stack_depth(conversation_id) == 1
        data = api.get_data(conversation_id)
        assert data["synthesis_result"] == _STALE
        assert data["from_child"] == 1
