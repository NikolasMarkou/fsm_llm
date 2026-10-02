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
