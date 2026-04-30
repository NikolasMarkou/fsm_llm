"""Tests for the unified ``fsm-llm`` CLI (R7).

Covers subcommand routing, target detection, alias delegation,
``explain`` output shape (AST + leaf schemas + plans), and value
coercion helpers in :mod:`fsm_llm.cli.run`.

These tests do NOT exercise the live LLM — ``run`` factory-mode tests
build terms whose evaluation does not require an oracle (e.g. plain
``var(...)`` terms with env-bound values), and FSM-mode invocations
either route through ``--help`` or use the ``--turns 0`` non-
interactive smoke shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from fsm_llm.cli.main import (
    SUBCOMMANDS,
    main_cli,
    meta_alias,
    monitor_alias,
    validate_alias,
    visualize_alias,
)
from fsm_llm.cli.run import (
    _coerce,
    _is_factory_string,
    _parse_kv_list,
    _resolve_factory,
)

# -----------------------------------------------------------------------
# Fixture helpers
# -----------------------------------------------------------------------


@pytest.fixture
def fsm_json_path(tmp_path: Path, minimal_fsm_dict: dict) -> str:
    """Write the minimal FSM dict to a tmp JSON file and return the path."""
    p = tmp_path / "minimal.json"
    p.write_text(json.dumps(minimal_fsm_dict))
    return str(p)


# -----------------------------------------------------------------------
# SUBCOMMANDS registry shape (closed set)
# -----------------------------------------------------------------------


class TestSubcommandsRegistry:
    def test_six_subcommands(self) -> None:
        assert len(SUBCOMMANDS) == 6

    def test_subcommand_names(self) -> None:
        assert SUBCOMMANDS == (
            "run",
            "explain",
            "validate",
            "visualize",
            "meta",
            "monitor",
        )


# -----------------------------------------------------------------------
# Top-level --help / --version routing
# -----------------------------------------------------------------------


class TestTopLevel:
    def test_top_level_help_returns_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main_cli(["--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        # All 6 subcommands appear in the help text.
        for sub in SUBCOMMANDS:
            assert sub in captured.out

    def test_top_level_version_returns_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main_cli(["--version"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "fsm_llm" in captured.out

    def test_missing_subcommand_errors(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main_cli([])
        # argparse uses 2 for usage errors.
        assert exc.value.code == 2


# -----------------------------------------------------------------------
# Per-subcommand --help (smoke — must dispatch without errors)
# -----------------------------------------------------------------------


class TestSubcommandHelp:
    @pytest.mark.parametrize("sub", ["run", "explain", "validate", "visualize"])
    def test_subcommand_help_exits_zero(self, sub: str) -> None:
        with pytest.raises(SystemExit) as exc:
            main_cli([sub, "--help"])
        assert exc.value.code == 0


# -----------------------------------------------------------------------
# Target detection (cli.run helpers)
# -----------------------------------------------------------------------


class TestIsFactoryString:
    def test_factory_string_simple(self) -> None:
        assert _is_factory_string("pkg.mod:factory") is True

    def test_factory_string_nested_pkg(self) -> None:
        assert _is_factory_string("fsm_llm.stdlib.long_context:niah_term") is True

    def test_not_factory_when_existing_path(self, fsm_json_path: str) -> None:
        # Existing file path should never be classified as factory.
        assert _is_factory_string(fsm_json_path) is False

    def test_not_factory_when_no_colon(self) -> None:
        assert _is_factory_string("examples/foo/bar.json") is False

    def test_not_factory_with_two_colons(self) -> None:
        assert _is_factory_string("a:b:c") is False

    def test_not_factory_when_empty_half(self) -> None:
        assert _is_factory_string(":foo") is False
        assert _is_factory_string("foo:") is False

    def test_not_factory_with_invalid_identifier(self) -> None:
        assert _is_factory_string("9pkg:factory") is False
        assert _is_factory_string("pkg:9factory") is False


class TestResolveFactory:
    def test_resolve_existing(self) -> None:
        # niah is a stable stdlib factory.
        factory = _resolve_factory("fsm_llm.stdlib.long_context:niah_term")
        assert callable(factory)

    def test_resolve_unknown_module(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _resolve_factory("nonexistent_pkg_xyz:factory")
        assert "cannot import" in str(exc.value)

    def test_resolve_unknown_attr(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _resolve_factory("fsm_llm:nonexistent_attribute_xyz")
        assert "no attribute" in str(exc.value)

    def test_resolve_non_callable(self) -> None:
        # __version__ is a string, not callable.
        with pytest.raises(SystemExit) as exc:
            _resolve_factory("fsm_llm.__version__:__version__")
        assert "not callable" in str(exc.value)


# -----------------------------------------------------------------------
# Value coercion + KV parsing
# -----------------------------------------------------------------------


class TestCoerce:
    def test_bool_true(self) -> None:
        assert _coerce("true") is True

    def test_bool_false(self) -> None:
        assert _coerce("false") is False

    def test_bool_case_insensitive(self) -> None:
        assert _coerce("TRUE") is True
        assert _coerce("False") is False

    def test_int(self) -> None:
        assert _coerce("42") == 42

    def test_negative_int(self) -> None:
        assert _coerce("-7") == -7

    def test_float(self) -> None:
        assert _coerce("3.14") == 3.14

    def test_string_fallback(self) -> None:
        assert _coerce("hello") == "hello"

    def test_string_with_digits(self) -> None:
        assert _coerce("abc123") == "abc123"


class TestParseKvList:
    def test_empty(self) -> None:
        assert _parse_kv_list([]) == {}

    def test_single(self) -> None:
        assert _parse_kv_list(["foo=bar"]) == {"foo": "bar"}

    def test_with_coercion(self) -> None:
        result = _parse_kv_list(["n=42", "flag=true", "name=alice"])
        assert result == {"n": 42, "flag": True, "name": "alice"}

    def test_value_with_equals(self) -> None:
        # Only the FIRST '=' splits.
        assert _parse_kv_list(["url=http://x?a=1"]) == {"url": "http://x?a=1"}

    def test_missing_equals_errors(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _parse_kv_list(["bare_key"])
        assert "KEY=VALUE" in str(exc.value)


# -----------------------------------------------------------------------
# explain subcommand — both modes
# -----------------------------------------------------------------------


class TestExplainSubcommand:
    def test_explain_factory_mode_outputs_ast_shape(self, capsys) -> None:
        rc = main_cli(
            [
                "explain",
                "fsm_llm.stdlib.long_context:niah_term",
                "--factory-arg",
                "question=What is X?",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "AST Shape" in captured.out
        assert "Fix" in captured.out  # niah always has a Fix subtree
        assert "Leaf" in captured.out

    def test_explain_factory_mode_with_n_K_populates_plans(self, capsys) -> None:
        rc = main_cli(
            [
                "explain",
                "fsm_llm.stdlib.long_context:niah_term",
                "--factory-arg",
                "question=What is X?",
                "--n",
                "1024",
                "--K",
                "8192",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        # Plans section should have content (no longer just "(empty …)").
        assert "Plans" in captured.out
        assert "plan[0]" in captured.out
        assert "predicted_calls" in captured.out

    def test_explain_factory_mode_json_output(self, capsys) -> None:
        rc = main_cli(
            [
                "explain",
                "fsm_llm.stdlib.long_context:niah_term",
                "--factory-arg",
                "question=What is X?",
                "--json",
                "--n",
                "1024",
                "--K",
                "8192",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        # Output must be valid JSON.
        data = json.loads(captured.out)
        assert "ast_shape" in data
        assert "leaf_schemas" in data
        assert "plans" in data
        assert isinstance(data["plans"], list)
        assert len(data["plans"]) >= 1
        assert "predicted_calls" in data["plans"][0]

    def test_explain_fsm_mode_outputs_case(self, capsys, fsm_json_path: str) -> None:
        rc = main_cli(["explain", fsm_json_path])
        captured = capsys.readouterr()
        assert rc == 0
        assert "AST Shape" in captured.out
        # Compiled FSM always has an outer Case discriminant.
        assert "Case" in captured.out

    def test_explain_fsm_mode_plans_empty_with_n_K(
        self, capsys, fsm_json_path: str
    ) -> None:
        # FSM-mode programs have no Fix subtree today (R6 deferred). Plans
        # stay empty even when (n, K) supplied — falsification trigger from
        # D-STEP-08-RESOLUTION; documented expected behaviour.
        rc = main_cli(
            ["explain", fsm_json_path, "--n", "1024", "--K", "8192", "--json"]
        )
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["plans"] == []


# -----------------------------------------------------------------------
# Alias functions
# -----------------------------------------------------------------------


class TestAliases:
    def test_validate_alias_routes_through_cli(
        self, fsm_json_path: str, monkeypatch
    ) -> None:
        # Simulate: fsm-llm-validate --fsm <path>
        monkeypatch.setattr(sys, "argv", ["fsm-llm-validate", "--fsm", fsm_json_path])
        # Should run validator and return cleanly. Exit code 0 means valid.
        rc = validate_alias()
        assert rc == 0

    def test_visualize_alias_routes_through_cli(
        self, fsm_json_path: str, monkeypatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["fsm-llm-visualize", "--fsm", fsm_json_path])
        rc = visualize_alias()
        assert rc == 0

    def test_meta_alias_passes_through(self, monkeypatch, capsys) -> None:
        # meta_alias is a direct passthrough — patch the underlying main_cli
        # to verify it's called (and avoid invoking the real interactive
        # builder).
        called = {"n": 0}

        def _fake_meta_main():
            called["n"] += 1

        with patch(
            "fsm_llm.stdlib.agents.meta_cli.main_cli",
            new=_fake_meta_main,
        ):
            monkeypatch.setattr(sys, "argv", ["fsm-llm-meta", "--help"])
            rc = meta_alias()
        assert rc == 0
        assert called["n"] == 1

    def test_monitor_alias_passes_through(self, monkeypatch) -> None:
        called = {"n": 0}

        def _fake_monitor_main():
            called["n"] += 1

        # Patch via the import path the alias uses (fsm_llm_monitor.__main__).
        with patch(
            "fsm_llm_monitor.__main__.main_cli",
            new=_fake_monitor_main,
        ):
            monkeypatch.setattr(sys, "argv", ["fsm-llm-monitor", "--version"])
            rc = monitor_alias()
        assert rc == 0
        assert called["n"] == 1


# -----------------------------------------------------------------------
# validate / visualize subcommand routes via main_cli
# -----------------------------------------------------------------------


class TestValidateAndVisualizeSubcommands:
    def test_validate_subcommand_on_minimal_fsm(self, fsm_json_path: str) -> None:
        rc = main_cli(["validate", "--fsm", fsm_json_path])
        assert rc == 0

    def test_visualize_subcommand_on_minimal_fsm(self, fsm_json_path: str) -> None:
        rc = main_cli(["visualize", "--fsm", fsm_json_path, "--style", "minimal"])
        assert rc == 0


# -----------------------------------------------------------------------
# run subcommand: factory mode without LLM
# -----------------------------------------------------------------------


class TestRunFactoryMode:
    def test_run_factory_no_llm(self, capsys, monkeypatch) -> None:
        """Build a Program from a trivial factory whose Term needs no oracle.

        We register a no-LLM factory at module scope and reference it by
        ``module:attr``.
        """
        # Minimal factory: returns a Var('x') term. .run(x="hello") returns
        # the env binding directly (no Leaf, no oracle call).
        from fsm_llm.runtime import var

        def _trivial_factory():
            return var("x")

        # Inject into a real importable module so cli.run can resolve it.
        import fsm_llm.cli.run as run_mod

        run_mod._test_trivial_factory = _trivial_factory  # type: ignore[attr-defined]

        rc = main_cli(
            [
                "run",
                "fsm_llm.cli.run:_test_trivial_factory",
                "--env",
                "x=hello",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "hello" in captured.out
