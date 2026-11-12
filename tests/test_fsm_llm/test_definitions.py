"""Definition-model behaviour: history summary, not-found error, entity coercer, version and session docs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fsm_llm.definitions import FSMError
from tests.test_fsm_llm.support import _a2026_09_22_api


class TestStep7HistorySizeZeroSummary:
    """P1-2: ``max_history_size=0`` digests exchanges before clearing them."""

    def test_p1_2_size_zero_summarises_before_clearing(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=0)
        conv.add_user_message("hello")
        conv.add_system_message("hi there")
        conv.add_user_message("how are you")
        conv.add_system_message("good")
        assert conv.exchanges == []
        assert conv.summary is not None
        assert "user: hello" in conv.summary
        assert "system: good" in conv.summary

    def test_p1_2_size_zero_keeps_the_summary_cap(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=0, summary="x" * 1990)
        conv.add_user_message("hello")
        conv.add_system_message("hi there")
        assert conv.summary is not None
        assert len(conv.summary) == 2000
        assert conv.summary.startswith("x" * 1990 + " | user: ")

    def test_p1_2_summary_already_at_cap_stays_at_cap(self):
        from fsm_llm.definitions import Conversation

        conv = Conversation(max_history_size=0, summary="y" * 2000)
        conv.add_user_message("hello")
        conv.add_system_message("hi")
        assert conv.summary == "y" * 2000
        assert conv.exchanges == []


class TestStep7FSMDefinitionNotFoundError:
    """Typed error for the registry-less id branch of ``load_fsm_definition``."""

    def test_id_branch_raises_typed_error_that_is_fsm_and_value_error(self):
        from fsm_llm import FSMDefinitionNotFoundError
        from fsm_llm.utilities import load_fsm_definition

        with pytest.raises(FSMDefinitionNotFoundError) as exc:
            load_fsm_definition("no-such-id")
        err = exc.value
        assert isinstance(err, FSMError)
        assert isinstance(err, ValueError)
        assert err.fsm_id == "no-such-id"
        msg = str(err)
        assert "Unknown FSM ID" in msg
        assert "no FSM registry" in msg
        assert "'no-such-id'" in msg

    def test_error_pickles_with_its_attributes(self):
        import pickle

        from fsm_llm.definitions import FSMDefinitionNotFoundError

        err = FSMDefinitionNotFoundError("abc123")
        clone = pickle.loads(pickle.dumps(err))
        assert type(clone) is FSMDefinitionNotFoundError
        assert str(clone) == str(err)
        assert clone.fsm_id == "abc123"
        assert clone.details == {"fsm_id": "abc123"}

    def test_api_loader_fallback_raises_the_typed_error(self):
        from fsm_llm.definitions import FSMDefinitionNotFoundError

        api = _a2026_09_22_api()
        with pytest.raises(FSMDefinitionNotFoundError):
            api.fsm_manager.fsm_loader("evicted-content-hash")

    def test_exported_in_all(self):
        import fsm_llm

        assert "FSMDefinitionNotFoundError" in fsm_llm.__all__


class TestStep7SharedEntityCoercer:
    """Both entity validators delegate to one module-level coercer."""

    def test_both_validators_delegate_to_the_shared_function(self):
        import fsm_llm.definitions as d

        real = d._coerce_entity_values
        with patch.object(d, "_coerce_entity_values", wraps=real) as spy:
            d.IntentScore(intent="a", confidence=0.5, entities={"k": 1})
            d.ClassificationResult(
                reasoning="r", intent="a", confidence=0.5, entities={"k": [1, 2]}
            )
        assert [c.args for c in spy.call_args_list] == [({"k": 1},), ({"k": [1, 2]},)]

    def test_shared_function_behaviour(self):
        from fsm_llm.definitions import _coerce_entity_values

        assert _coerce_entity_values({"a": None, "b": 3, "c": ["x", 1]}) == {
            "a": None,
            "b": "3",
            "c": "x, 1",
        }
        assert _coerce_entity_values("junk") == {}


class TestStep7VersionAndSessionDocs:
    def test_architecture_is_2_pass(self):
        from fsm_llm import get_version_info

        assert get_version_info()["architecture"] == "2-pass"

    def test_p1_6_stack_depth_documented_as_advisory(self):
        from fsm_llm.session import SessionState

        description = SessionState.model_fields["stack_depth"].description or ""
        assert "advisory" in description
        assert "never read by restore" in description
