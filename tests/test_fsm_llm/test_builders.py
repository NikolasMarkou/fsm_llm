"""Tests for the core ``BuildError`` shared by every ``build()``."""

import pickle

import pytest

import fsm_llm
from fsm_llm import BuildError, FSMError


class TestBuildError:
    def test_is_fsm_error_and_value_error(self):
        err = BuildError("bad")
        assert isinstance(err, FSMError)
        assert isinstance(err, ValueError)

    def test_caught_as_value_error(self):
        with pytest.raises(ValueError, match="bad"):
            raise BuildError("bad")

    def test_errors_default_is_message(self):
        assert BuildError("bad").errors == ["bad"]

    def test_errors_explicit(self):
        err = BuildError("2 problems", errors=["a", "b"])
        assert err.errors == ["a", "b"]
        assert str(err) == "2 problems"

    def test_details_passthrough(self):
        err = BuildError("bad", details={"k": 1})
        assert err.details == {"k": 1}
        assert BuildError("bad").details == {}

    def test_pickle_round_trip(self):
        err = BuildError("2 problems", errors=["a", "b"], details={"k": 1})
        back = pickle.loads(pickle.dumps(err))
        assert isinstance(back, BuildError)
        assert str(back) == "2 problems"
        assert back.errors == ["a", "b"]
        assert back.details == {"k": 1}

    def test_exported_from_package(self):
        assert "BuildError" in fsm_llm.__all__
        assert fsm_llm.BuildError is BuildError


class TestHandlerBuilderBuildError:
    def test_missing_lambda_is_build_error_and_value_error(self):
        builder = fsm_llm.create_handler("h")
        with pytest.raises(BuildError, match="Execution lambda is required") as exc:
            builder.build()
        assert isinstance(exc.value, ValueError)

    def test_non_callable_do_raises_build_error(self):
        builder = fsm_llm.create_handler("h")
        with pytest.raises(BuildError, match="must be callable") as exc:
            builder.do("not callable")  # type: ignore[arg-type]
        assert isinstance(exc.value, ValueError)

    def test_do_records_and_build_refuses_non_callable(self):
        builder = fsm_llm.HandlerBuilder("h")
        builder.execution_lambda = 42  # type: ignore[assignment]
        with pytest.raises(BuildError, match="must be callable"):
            builder.build()

    def test_callable_still_builds(self):
        handler = fsm_llm.create_handler("h").do(lambda ctx: {})
        assert handler.name == "h"


# --------------------------------------------------------------
# APIBuilder / FSMManagerBuilder
# --------------------------------------------------------------


def _definition_dict():
    return {
        "name": "B",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "p",
                "response_instructions": "r",
                "transitions": [
                    {"target_state": "b", "description": "go", "priority": 1}
                ],
            },
            "b": {
                "id": "b",
                "description": "b",
                "purpose": "p",
                "response_instructions": "r",
            },
        },
    }


def _mock_llm():
    from tests.conftest import MockLLM2Interface

    return MockLLM2Interface()


class TestAPIBuilder:
    def test_mutators_return_builder(self):
        from fsm_llm import APIBuilder

        b = APIBuilder()
        d = _definition_dict()
        assert b.set_definition(d) is b
        assert b.set_llm_interface(_mock_llm()) is b
        assert b.set_model("m") is b
        assert b.set_api_key("k") is b
        assert b.set_temperature(0.1) is b
        assert b.set_max_tokens(5) is b
        assert b.set_llm_option("seed", 3) is b
        assert b.add_handler(fsm_llm.create_handler("h").do(lambda c: {})) is b
        assert b.set_handler_error_mode("raise") is b
        assert b.set_transition_config(fsm_llm.TransitionEvaluatorConfig()) is b
        assert b.set_session_store(None) is b
        assert b.set_handler_timeout(1.0) is b
        assert b.set_max_history_size(3) is b
        assert b.set_max_message_length(10) is b
        assert b.set_max_fsm_cache_size(2) is b

    def test_missing_definition_raises_build_error(self):
        from fsm_llm import APIBuilder

        with pytest.raises(BuildError, match="definition"):
            APIBuilder().build()

    def test_refusal_parity(self):
        from fsm_llm import API, APIBuilder

        d = _definition_dict()
        with pytest.raises(ValueError) as direct:
            API(d, llm_interface=_mock_llm(), temperature=0.2)
        b = APIBuilder().set_definition(d).set_llm_interface(_mock_llm())
        b.set_temperature(0.2)
        with pytest.raises(BuildError) as built:
            b.build()
        assert str(direct.value) in str(built.value)
        assert isinstance(built.value.__cause__, ValueError)

    def test_open_option_reaches_interface(self):
        from fsm_llm import APIBuilder

        api = (
            APIBuilder()
            .set_definition(_definition_dict())
            .set_model("some/model")
            .set_llm_option("seed", 3)
            .build()
        )
        assert api.llm_interface.kwargs["seed"] == 3

    def test_unset_values_equal_api_defaults(self):
        from fsm_llm import API, APIBuilder

        d = _definition_dict()
        built = APIBuilder().set_definition(d).build()
        direct = API(d)
        assert built.llm_interface.model == direct.llm_interface.model
        assert built.llm_interface.temperature == direct.llm_interface.temperature
        assert built.llm_interface.max_tokens == direct.llm_interface.max_tokens
        assert built.fsm_manager.max_history_size == direct.fsm_manager.max_history_size
        assert (
            built.fsm_manager.max_message_length
            == direct.fsm_manager.max_message_length
        )
        assert (
            built.fsm_manager._max_fsm_cache_size
            == direct.fsm_manager._max_fsm_cache_size
        )
        assert built.fsm_id == direct.fsm_id

    def test_set_values_reach_api(self):
        from fsm_llm import APIBuilder

        api = (
            APIBuilder()
            .set_definition(_definition_dict())
            .set_llm_interface(_mock_llm())
            .set_max_history_size(2)
            .set_max_message_length(7)
            .set_max_fsm_cache_size(4)
            .build()
        )
        assert api.fsm_manager.max_history_size == 2
        assert api.fsm_manager.max_message_length == 7
        assert api.fsm_manager._max_fsm_cache_size == 4

    def test_cache_size_error_is_wrapped(self):
        from fsm_llm import APIBuilder

        b = (
            APIBuilder()
            .set_definition(_definition_dict())
            .set_llm_interface(_mock_llm())
            .set_max_fsm_cache_size(0)
        )
        with pytest.raises(BuildError, match="max_fsm_cache_size must be >= 1") as e:
            b.build()
        assert isinstance(e.value.__cause__, ValueError)

    def test_bad_definition_is_wrapped(self):
        from fsm_llm import APIBuilder

        b = APIBuilder().set_definition({"name": "x"}).set_llm_interface(_mock_llm())
        with pytest.raises(BuildError, match="Invalid FSM definition"):
            b.build()

    def test_isolation_and_independent_second_build(self):
        from fsm_llm import APIBuilder

        d = _definition_dict()
        h = fsm_llm.create_handler("h").do(lambda c: {})
        b = APIBuilder().set_definition(d).set_llm_interface(_mock_llm())
        b.add_handler(h)
        first = b.build()
        d["states"]["a"]["purpose"] = "changed"
        d["name"] = "changed"
        b.add_handler(fsm_llm.create_handler("h2").do(lambda c: {}))
        assert first.fsm_definition.name == "B"
        assert first.fsm_definition.states["a"].purpose == "p"
        assert len(first.handler_system.handlers) == 1
        second = b.build()
        assert second.fsm_definition.name == "changed"
        assert second is not first
        assert second.fsm_definition is not first.fsm_definition
        assert len(second.handler_system.handlers) == 2
        assert len(first.handler_system.handlers) == 1

    def test_definition_object_is_copied(self):
        from fsm_llm import API, APIBuilder

        definition = API(_definition_dict(), llm_interface=_mock_llm()).fsm_definition
        api = (
            APIBuilder()
            .set_definition(definition)
            .set_llm_interface(_mock_llm())
            .build()
        )
        assert api.fsm_definition is not definition
        assert api.fsm_definition == definition

    def test_builders_module_does_not_import_litellm(self):
        import fsm_llm.builders as mod

        import re

        with open(mod.__file__) as fh:
            assert not re.search(r"import litellm|from litellm", fh.read())


class TestFSMManagerBuilder:
    def test_mutators_return_builder(self):
        from fsm_llm import FSMManagerBuilder

        b = FSMManagerBuilder()
        assert b.set_fsm_loader(lambda i: None) is b
        assert b.set_llm_interface(_mock_llm()) is b
        assert b.set_max_history_size(2) is b
        assert b.set_max_message_length(2) is b
        assert b.set_handler_error_mode("raise") is b
        assert b.set_max_fsm_cache_size(2) is b
        assert b.set_handler_system(None) is b
        assert b.set_transition_evaluator(None) is b
        assert b.set_data_extraction_prompt_builder(None) is b
        assert b.set_response_generation_prompt_builder(None) is b
        assert b.set_field_extraction_prompt_builder(None) is b

    def test_missing_llm_interface_raises_build_error(self):
        from fsm_llm import FSMManagerBuilder

        with pytest.raises(BuildError, match="llm_interface is required") as e:
            FSMManagerBuilder().build()
        assert isinstance(e.value.__cause__, ValueError)

    def test_cache_size_parity(self):
        from fsm_llm import FSMManager, FSMManagerBuilder

        with pytest.raises(ValueError) as direct:
            FSMManager(llm_interface=_mock_llm(), max_fsm_cache_size=0)
        b = FSMManagerBuilder().set_llm_interface(_mock_llm())
        b.set_max_fsm_cache_size(0)
        with pytest.raises(BuildError) as built:
            b.build()
        assert str(direct.value) in str(built.value)
        assert built.value.__cause__ is not None

    def test_defaults_equal_constructor(self):
        from fsm_llm import FSMManager, FSMManagerBuilder

        llm = _mock_llm()
        built = FSMManagerBuilder().set_llm_interface(llm).build()
        direct = FSMManager(llm_interface=llm)
        assert built.max_history_size == direct.max_history_size
        assert built.max_message_length == direct.max_message_length
        assert built._max_fsm_cache_size == direct._max_fsm_cache_size
        assert built.fsm_loader is direct.fsm_loader

    def test_set_values_reach_manager(self):
        from fsm_llm import FSMManagerBuilder

        loader = lambda i: None  # noqa: E731
        m = (
            FSMManagerBuilder()
            .set_llm_interface(_mock_llm())
            .set_fsm_loader(loader)
            .set_max_history_size(2)
            .set_max_fsm_cache_size(9)
            .build()
        )
        assert m.fsm_loader is loader
        assert m.max_history_size == 2
        assert m._max_fsm_cache_size == 9

    def test_second_build_independent(self):
        from fsm_llm import FSMManagerBuilder

        b = FSMManagerBuilder().set_llm_interface(_mock_llm())
        first = b.build()
        b.set_max_history_size(1)
        second = b.build()
        assert first.max_history_size != 1
        assert second is not first
        assert second.fsm_cache is not first.fsm_cache
