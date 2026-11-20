"""The generic JSON patch applier of core (plan ace7a8c8, roadmap E1.2).

``apply_json_patch`` is the engine of ``synthesis.apply_patch`` with no scope:
the scope rule enters only through the optional ``check_path`` hook. These rows
call it with no scope, and pin what a generic caller can rely on: the hook, the
order of refusals, the code set, and that core imports nothing of synthesis.
``tests/test_fsm_llm_synthesis/test_patch.py`` is the scoped oracle and stays as is.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

from fsm_llm.core.definitions import (
    MAX_INDEX_DIGITS,
    MAX_PATCH_DEPTH,
    MAX_POINTER_TOKEN_CHARS,
    MAX_POINTER_TOKENS,
    PATCH_OPS,
    FSMError,
    JsonPatchError,
    apply_json_patch,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "fsm_llm"

#: Every code core raises: the eleven of the engine and ``out-of-scope`` of the hook.
CORE_CODES = frozenset(
    {
        "too-many-ops",
        "too-large",
        "too-deep",
        "bad-value",
        "bad-document",
        "op-shape",
        "op-not-allowed",
        "bad-pointer",
        "bad-index",
        "out-of-scope",
        "no-target",
        "test-failed",
    }
)


def _doc() -> dict[str, Any]:
    return {"a": {"b": [1, 2, 3], "c": "x"}, "k/e": 1, "m~n": 2}


def _apply(doc, ops, *, max_ops=50, max_chars=10_000, check_path=None):
    return apply_json_patch(
        doc, ops, max_ops=max_ops, max_chars=max_chars, check_path=check_path
    )


def _refusal(doc, ops, **kwargs) -> JsonPatchError:
    before = copy.deepcopy(doc)
    with pytest.raises(JsonPatchError) as info:
        _apply(doc, ops, **kwargs)
    assert doc == before  # atomic: the caller's document is untouched
    return info.value


class TestGenericCall:
    def test_the_four_ops_apply_with_no_scope_and_the_document_is_untouched(self):
        doc = _doc()
        before = copy.deepcopy(doc)
        out = _apply(
            doc,
            [
                {"op": "add", "path": "/a/b/-", "value": 4},
                {"op": "replace", "path": "/a/c", "value": "y"},
                {"op": "remove", "path": "/a/b/0"},
                {"op": "test", "path": "/a/c", "value": "y"},
                {"op": "add", "path": "/top", "value": {"n": [1]}},
            ],
        )
        assert out == {
            "a": {"b": [2, 3, 4], "c": "y"},
            "k/e": 1,
            "m~n": 2,
            "top": {"n": [1]},
        }
        assert doc == before and out is not doc

    def test_pointer_tokens_unescape_tilde_one_before_tilde_zero(self):
        out = _apply(
            _doc(),
            [
                {"op": "replace", "path": "/k~1e", "value": 9},
                {"op": "replace", "path": "/m~0n", "value": 8},
            ],
        )
        assert out["k/e"] == 9 and out["m~n"] == 8
        assert _refusal(_doc(), [{"op": "remove", "path": "/k~01e"}]).code == (
            "no-target"
        )

    def test_max_ops_and_max_chars_are_required_keywords(self):
        with pytest.raises(TypeError):
            apply_json_patch(_doc(), [], 5, 100)  # type: ignore[misc]
        with pytest.raises(TypeError):
            apply_json_patch(_doc(), [])  # type: ignore[call-arg]

    def test_the_limits_are_the_documented_values(self):
        assert PATCH_OPS == ("add", "remove", "replace", "test")
        assert (
            MAX_PATCH_DEPTH,
            MAX_POINTER_TOKENS,
            MAX_POINTER_TOKEN_CHARS,
            MAX_INDEX_DIGITS,
        ) == (16, 12, 100, 9)


class TestCheckPathHook:
    def test_a_false_return_refuses_the_whole_patch_at_that_op(self):
        doc = _doc()
        ops = [
            {"op": "replace", "path": "/a/c", "value": "y"},
            {"op": "replace", "path": "/a/c", "value": "z"},
            {"op": "remove", "path": "/k~1e"},
        ]
        seen: list[list[str]] = []

        def only_under_a(tokens: list[str]) -> bool:
            seen.append(tokens)
            return tokens[0] == "a"

        err = _refusal(doc, ops, check_path=only_under_a)
        assert (err.code, err.op) == ("out-of-scope", 2)
        assert seen == [["a", "c"], ["a", "c"], ["k/e"]]  # unescaped tokens, in order

    def test_a_true_return_for_every_path_changes_nothing(self):
        ops = [{"op": "replace", "path": "/a/c", "value": "y"}]
        assert _apply(_doc(), ops, check_path=lambda _t: True) == _apply(_doc(), ops)

    def test_a_malformed_pointer_outranks_the_hook_of_the_same_op(self):
        def never(_t: list[str]) -> bool:
            raise AssertionError("the hook sees only a well-formed pointer")

        err = _refusal(_doc(), [{"op": "remove", "path": "a/b"}], check_path=never)
        assert (err.code, err.op) == ("bad-pointer", 0)

    def test_a_hook_refusal_is_reported_before_any_target_is_looked_up(self):
        ops = [
            {"op": "remove", "path": "/a/c/zz"},
            {"op": "remove", "path": "/k~1e"},
        ]
        err = _refusal(_doc(), ops, check_path=lambda t: t[0] == "a")
        assert (err.code, err.op) == ("out-of-scope", 1)  # no-target of op 0 is later


class TestRefusalOrder:
    @pytest.mark.parametrize(
        ("label", "doc", "ops", "limits", "expected"),
        [
            ("document first", [], "not a list", {}, "bad-document"),
            ("then the ops type", {}, {"op": "add"}, {}, "op-shape"),
            ("then the count", {}, [{}, {}, {}], {"max_ops": 2}, "too-many-ops"),
            (
                "then the size, before any op is read",
                {},
                ["x" * 50, 3],
                {"max_chars": 20},
                "too-large",
            ),
            ("then op shape", {}, [3], {}, "op-shape"),
            ("then op name", {}, [{"op": "move", "path": "/a"}], {}, "op-not-allowed"),
            ("then pointer", {}, [{"op": "remove", "path": "a"}], {}, "bad-pointer"),
        ],
    )
    def test_each_check_runs_in_its_place(self, label, doc, ops, limits, expected):
        with pytest.raises(JsonPatchError) as info:
            _apply(doc, ops, **limits)
        assert info.value.code == expected, label

    def test_depth_is_judged_below_the_op_object_before_any_op_is_read(self):
        value: Any = 1
        for _ in range(MAX_PATCH_DEPTH + 1):
            value = [value]
        ops = [{"op": "add", "path": "/x", "value": value}, 3]
        assert _refusal({}, ops).code == "too-deep"  # op 1 is not even read
        ok: Any = 1
        for _ in range(MAX_PATCH_DEPTH):
            ok = [ok]
        assert _apply({}, [{"op": "add", "path": "/x", "value": ok}]) == {"x": ok}

    def test_the_value_and_extra_key_checks_follow_the_op_name(self):
        assert _refusal({}, [{"op": "add", "path": "/x"}]).code == "op-shape"
        assert (
            _refusal({}, [{"op": "remove", "path": "/x", "value": 1}]).code
            == "op-shape"
        )

    def test_bad_values_are_refused_before_the_walk(self):
        for bad in (float("nan"), 2**70, object(), {1: 2}):
            err = _refusal({}, [{"op": "add", "path": "/x", "value": bad}])
            assert err.code == "bad-value"


class TestCodes:
    def test_every_code_core_raises_is_reachable_and_in_the_set(self):
        deep: Any = 1
        for _ in range(MAX_PATCH_DEPTH + 1):
            deep = [deep]
        cases: list[tuple[Any, Any, dict[str, Any]]] = [
            ([], [], {}),
            ({}, "x", {}),
            ({}, [{}, {}], {"max_ops": 1}),
            ({}, [{"op": "add", "path": "/x", "value": "y" * 99}], {"max_chars": 50}),
            ({}, [{"op": "add", "path": "/x", "value": deep}], {}),
            ({}, [{"op": "add", "path": "/x", "value": {1: 1}}], {}),
            ({}, [3], {}),
            ({}, [{"op": "move", "path": "/x"}], {}),
            ({}, [{"op": "remove", "path": "x"}], {}),
            ({"l": [1]}, [{"op": "remove", "path": "/l/01"}], {}),
            ({}, [{"op": "remove", "path": "/x"}], {}),
            ({"x": 1}, [{"op": "test", "path": "/x", "value": 2}], {}),
            (
                {"x": 1},
                [{"op": "remove", "path": "/x"}],
                {"check_path": lambda _t: False},
            ),
        ]
        seen = {_refusal(doc, ops, **kw).code for doc, ops, kw in cases}
        assert seen == CORE_CODES

    def test_the_synthesis_code_set_is_the_core_set_plus_bad_scope(self):
        from fsm_llm.synthesis.constants import PATCH_CODES

        assert set(PATCH_CODES) == CORE_CODES | {"bad-scope"}

    def test_the_error_is_an_fsm_error_with_a_code_and_an_op_and_no_patch_text(self):
        err = _refusal({}, [{"op": "remove", "path": "/secret-token"}])
        assert isinstance(err, FSMError) and not isinstance(err, ValueError)
        assert (err.code, err.op) == ("no-target", 0)
        assert str(err) == "patch refused: no-target (op 0)"
        assert "secret-token" not in str(err)
        whole = _refusal([], [])
        assert (whole.code, whole.op) == ("bad-document", None)
        assert str(whole) == "patch refused: bad-document"

    def test_a_document_too_deep_to_copy_is_bad_document(self):
        doc: dict[str, Any] = {}
        node = doc
        for _ in range(5000):
            node["n"] = {}
            node = node["n"]
        with pytest.raises(JsonPatchError) as info:
            _apply(doc, [{"op": "add", "path": "/x", "value": 1}])
        assert info.value.code == "bad-document"


class TestPlacement:
    def test_core_imports_nothing_of_synthesis_or_the_layers_above(self):
        tree = ast.parse((_SRC / "core" / "definitions" / "patch.py").read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(("." * node.level) + (node.module or ""))
            elif isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
        assert not [m for m in imported if "synthesis" in m], imported
        assert not [
            m for m in imported if m.startswith(("fsm_llm.engine", "fsm_llm.llm"))
        ]
        assert "logger" not in (_SRC / "core" / "definitions" / "patch.py").read_text()

    def test_synthesis_reads_the_depth_limit_from_core(self):
        from fsm_llm.core.definitions import patch as core_patch
        from fsm_llm.synthesis import constants as synthesis_constants

        assert synthesis_constants.MAX_PATCH_DEPTH is core_patch.MAX_PATCH_DEPTH
        for name in ("PATCH_OPS", "MAX_POINTER_TOKENS", "MAX_INDEX_DIGITS"):
            assert not hasattr(synthesis_constants, name)  # one home, no second copy
