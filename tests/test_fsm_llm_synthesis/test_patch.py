"""
Tests for ``fsm_llm.synthesis.patch``: the in-house RFC 6902 subset.

The RFC 6902 Appendix A examples that fall inside the four ops are the oracle
(run under ``/states/s`` so they pass the scope rule); the rest is hostile
input: every refusal is one ``PatchError`` with a code, the caller's document
is never touched, and nothing recurses or converts a huge number on a value
that has not been measured.
"""

import copy
import json
import sys
from typing import Any

import pytest

from fsm_llm.definitions import FSMError
from fsm_llm.synthesis import (
    PatchError,
    SynthesisError,
    SynthesisPolicy,
    apply_patch,
)
from fsm_llm.synthesis.constants import MAX_PATCH_DEPTH, PATCH_CODES

POLICY = SynthesisPolicy()


def _doc() -> dict[str, Any]:
    return {
        "name": "P",
        "persona": "x",
        "initial_state": "a",
        "handler_only_keys": ["k"],
        "states": {
            "a": {
                "id": "a",
                "purpose": "ask",
                "field_extractions": [{"field_name": "n"}],
                "transitions": [{"target_state": "b", "priority": 1}],
            },
            "b": {"id": "b", "purpose": "end", "transitions": []},
            "ab": {"id": "ab", "purpose": "other"},
            "a/b": {"id": "a/b", "purpose": "slash"},
            "abort": {"id": "abort", "purpose": "silent"},
        },
    }


def _nest(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


def _run(doc, ops, scope=("a",), policy=POLICY):
    return apply_patch(doc, ops, scope_states=scope, policy=policy)


def _code(doc, ops, scope=("a",), policy=POLICY) -> str:
    before = copy.deepcopy(doc)
    with pytest.raises(PatchError) as info:
        _run(doc, ops, scope, policy)
    assert doc == before  # atomic: the caller's dict is untouched
    return info.value.code


def _rfc(doc, ops):
    """Run an Appendix A example under /states/s and give back that member."""
    wrapped = {"states": {"s": doc}}
    fixed = [{**op, "path": "/states/s" + op["path"]} for op in ops]
    return apply_patch(wrapped, fixed, scope_states=("s",), policy=POLICY)["states"][
        "s"
    ]


class TestRfc6902AppendixA:
    def test_a1_add_an_object_member(self):
        out = _rfc({"foo": "bar"}, [{"op": "add", "path": "/baz", "value": "qux"}])
        assert out == {"baz": "qux", "foo": "bar"}

    def test_a2_add_an_array_element(self):
        out = _rfc(
            {"foo": ["bar", "baz"]},
            [{"op": "add", "path": "/foo/1", "value": "qux"}],
        )
        assert out == {"foo": ["bar", "qux", "baz"]}

    def test_a3_remove_an_object_member(self):
        out = _rfc({"baz": "qux", "foo": "bar"}, [{"op": "remove", "path": "/baz"}])
        assert out == {"foo": "bar"}

    def test_a4_remove_an_array_element(self):
        out = _rfc({"foo": ["bar", "qux", "baz"]}, [{"op": "remove", "path": "/foo/1"}])
        assert out == {"foo": ["bar", "baz"]}

    def test_a5_replace_a_value(self):
        out = _rfc(
            {"baz": "qux", "foo": "bar"},
            [{"op": "replace", "path": "/baz", "value": "boo"}],
        )
        assert out == {"baz": "boo", "foo": "bar"}

    def test_a8_a_successful_test(self):
        doc = {"baz": "qux", "foo": ["a", 2, "c"]}
        ops = [
            {"op": "test", "path": "/baz", "value": "qux"},
            {"op": "test", "path": "/foo/1", "value": 2},
        ]
        assert _rfc(doc, ops) == doc

    def test_a9_a_failing_test_is_an_error(self):
        ops = [{"op": "test", "path": "/baz", "value": "bar"}]
        with pytest.raises(PatchError) as info:
            _rfc({"baz": "qux"}, ops)
        assert info.value.code == "test-failed"

    def test_a10_add_a_nested_object(self):
        out = _rfc(
            {"foo": "bar"},
            [{"op": "add", "path": "/child", "value": {"grandchild": {}}}],
        )
        assert out == {"foo": "bar", "child": {"grandchild": {}}}

    def test_a11_unrecognised_elements_are_refused_not_ignored(self):
        op = {"op": "add", "path": "/states/a/x", "value": 1, "xyz": 123}
        assert _code(_doc(), [op]) == "op-shape"

    def test_a12_add_under_a_missing_parent_is_an_error(self):
        with pytest.raises(PatchError) as info:
            _rfc({"foo": "bar"}, [{"op": "add", "path": "/baz/bat", "value": "q"}])
        assert info.value.code == "no-target"

    def test_a14_tilde_escapes_unescape_slash_first(self):
        doc = {"/": 9, "~1": 10}
        assert _rfc(doc, [{"op": "test", "path": "/~01", "value": 10}]) == doc
        assert _rfc(doc, [{"op": "test", "path": "/~1", "value": 9}]) == doc

    def test_a15_a_string_is_not_a_number(self):
        with pytest.raises(PatchError) as info:
            _rfc({"/": 9, "~1": 10}, [{"op": "test", "path": "/~01", "value": "10"}])
        assert info.value.code == "test-failed"

    def test_a16_add_an_array_value_with_the_dash_index(self):
        out = _rfc(
            {"foo": ["bar"]},
            [{"op": "add", "path": "/foo/-", "value": ["abc", "def"]}],
        )
        assert out == {"foo": ["bar", ["abc", "def"]]}


class TestSemantics:
    def test_the_result_is_a_new_dict_and_the_input_is_not_aliased(self):
        doc, value = _doc(), {"field_name": "m"}
        out = _run(
            doc,
            [{"op": "add", "path": "/states/a/field_extractions/-", "value": value}],
        )
        assert out is not doc and doc == _doc()
        value["field_name"] = "changed"
        assert out["states"]["a"]["field_extractions"][1] == {"field_name": "m"}

    def test_the_four_ops_in_sequence(self):
        ops = [
            {"op": "test", "path": "/states/a/purpose", "value": "ask"},
            {"op": "replace", "path": "/states/a/purpose", "value": "ask more"},
            {"op": "add", "path": "/states/a/description", "value": "d"},
            {"op": "remove", "path": "/states/a/transitions/0/priority"},
        ]
        out = _run(_doc(), ops)["states"]["a"]
        assert out["purpose"] == "ask more" and out["description"] == "d"
        assert out["transitions"] == [{"target_state": "b"}]

    def test_add_at_the_length_appends_and_past_it_is_refused(self):
        add = lambda i: [  # noqa: E731
            {"op": "add", "path": f"/states/a/transitions/{i}", "value": {}}
        ]
        assert len(_run(_doc(), add(1))["states"]["a"]["transitions"]) == 2
        assert _code(_doc(), add(2)) == "no-target"

    def test_remove_replace_and_test_need_an_existing_target(self):
        for op, extra in (
            ("remove", {}),
            ("replace", {"value": 1}),
            ("test", {"value": 1}),
        ):
            for path in ("/states/a/nope", "/states/a/transitions/1"):
                assert _code(_doc(), [{"op": op, "path": path, **extra}]) == "no-target"

    def test_the_dash_index_is_for_add_only(self):
        for op in ("remove", "replace", "test"):
            extra = {} if op == "remove" else {"value": 1}
            path = "/states/a/transitions/-"
            assert _code(_doc(), [{"op": op, "path": path, **extra}]) == "bad-index"

    def test_a_failing_last_op_leaves_nothing(self):
        ops = [
            {"op": "replace", "path": "/states/a/purpose", "value": "changed"},
            {"op": "remove", "path": "/states/a/nope"},
        ]
        assert _code(_doc(), ops) == "no-target"

    def test_an_empty_patch_is_an_equal_copy(self):
        doc = _doc()
        out = _run(doc, [])
        assert out == doc and out is not doc

    def test_walking_through_a_scalar_is_no_target(self):
        op = {"op": "add", "path": "/states/a/purpose/x", "value": 1}
        assert _code(_doc(), [op]) == "no-target"


class TestStrictEquality:
    @pytest.mark.parametrize(
        ("have", "want", "passes"),
        [
            (True, 1, False),
            (1, True, False),
            (0, False, False),
            (1, 1.0, False),
            (1.0, 1, False),
            ("1", 1, False),
            (1, 1, True),
            (True, True, True),
            (None, None, True),
            (None, False, False),
            ([1, [True]], [1, [1]], False),
            ([1, [True]], [1, [True]], True),
            ({"a": 1, "b": [1.5]}, {"b": [1.5], "a": 1}, True),
            ({"a": 1}, {"a": 1, "b": 2}, False),
            ({"a": 1}, {"a": True}, False),
            ([1, 2], [1, 2, 3], False),
            ([], {}, False),
        ],
    )
    def test_test_is_json_equality_not_python_equality(self, have, want, passes):
        doc = _doc()
        doc["states"]["a"]["probe"] = have
        ops = [{"op": "test", "path": "/states/a/probe", "value": want}]
        if passes:
            assert _run(doc, ops) == doc
        else:
            assert _code(doc, ops) == "test-failed"

    def test_a_deep_target_is_compared_without_recursion(self):
        deep = _nest(MAX_PATCH_DEPTH)
        doc = _doc()
        doc["states"]["a"]["probe"] = deep
        assert _run(doc, [{"op": "test", "path": "/states/a/probe", "value": deep}])


class TestScope:
    @pytest.mark.parametrize(
        "path",
        [
            "/initial_state",
            "/handler_only_keys",
            "/handler_only_keys/0",
            "/persona",
            "/name",
            "/states",
            "/states/a",
            "/states/b/purpose",  # a state outside the scope
            "/states/ab/purpose",  # a longer id that starts with the scope id
            "/states/abort/purpose",
            "/states/a~1b/purpose",  # the id `a/b` is not `a`
            "/states/A/purpose",
        ],
    )
    @pytest.mark.parametrize("op", ["add", "remove", "replace", "test"])
    def test_every_out_of_scope_path_is_refused(self, path, op):
        extra = {} if op == "remove" else {"value": "x"}
        assert _code(_doc(), [{"op": op, "path": path, **extra}]) == "out-of-scope"

    def test_abort_is_refused_even_when_a_caller_lists_it(self):
        op = {"op": "replace", "path": "/states/abort/purpose", "value": "x"}
        assert _code(_doc(), [op], scope=("a", "abort")) == "out-of-scope"

    def test_a_scope_state_id_with_a_slash_matches_by_token_not_prefix(self):
        op = {"op": "replace", "path": "/states/a~1b/purpose", "value": "x"}
        out = _run(_doc(), [op], scope=("a/b",))
        assert out["states"]["a/b"]["purpose"] == "x"
        # `/states/a/b/purpose` is state `a` key `b`, not state `a/b`
        raw = {"op": "replace", "path": "/states/a/b/purpose", "value": "x"}
        assert _code(_doc(), [raw], scope=("a/b",)) == "out-of-scope"

    def test_dotdot_is_a_literal_key_and_escapes_nothing(self):
        op = {"op": "add", "path": "/states/a/..", "value": 1}
        out = _run(_doc(), [op])
        assert out["states"]["a"][".."] == 1
        assert {k: v for k, v in out.items() if k != "states"} == {
            k: v for k, v in _doc().items() if k != "states"
        }
        assert {k: v for k, v in out["states"].items() if k != "a"} == {
            k: v for k, v in _doc()["states"].items() if k != "a"
        }

    def test_a_new_state_cannot_be_created(self):
        op = {"op": "add", "path": "/states/new/x", "value": 1}
        assert _code(_doc(), [op], scope=("new",)) == "no-target"

    def test_scope_is_checked_for_every_op_before_any_is_applied(self):
        ops = [
            {"op": "replace", "path": "/states/a/purpose", "value": "changed"},
            {"op": "replace", "path": "/persona", "value": "evil"},
        ]
        assert _code(_doc(), ops) == "out-of-scope"

    def test_a_bare_string_scope_is_refused(self):
        op = {"op": "replace", "path": "/states/a/purpose", "value": "x"}
        assert _code(_doc(), [op], scope="a") == "bad-scope"


class TestShapeAndPointers:
    @pytest.mark.parametrize("name", ["move", "copy", "MOVE", "", None, 1, ["add"], {}])
    def test_only_the_four_ops_are_allowed(self, name):
        op = {"op": name, "path": "/states/a/x", "value": 1}
        assert _code(_doc(), [op]) == "op-not-allowed"

    @pytest.mark.parametrize(
        "op",
        [
            {"op": "move", "from": "/states/a/y", "path": "/states/a/x"},
            {"op": "copy", "from": "/states/a/y", "path": "/states/a/x"},
        ],
    )
    def test_move_and_copy_are_refused(self, op):
        assert _code(_doc(), [op]) == "op-not-allowed"

    @pytest.mark.parametrize(
        "op",
        [
            {"op": "add", "path": "/states/a/x"},  # value missing
            {"op": "replace", "path": "/states/a/x"},
            {"op": "test", "path": "/states/a/x"},
            {"op": "remove", "path": "/states/a/x", "value": 1},
            {"op": "add", "path": "/states/a/x", "value": 1, "from": "/p"},
            {"op": "add", "value": 1},
            {"path": "/states/a/x", "value": 1},
            "add",
            ["add", "/states/a/x", 1],
            None,
        ],
    )
    def test_an_op_has_exactly_op_path_and_value_where_needed(self, op):
        assert _code(_doc(), [op]) in ("op-shape", "op-not-allowed", "bad-pointer")

    def test_ops_must_be_a_list(self):
        for ops in (None, {}, "x", ({"op": "remove", "path": "/states/a/x"},)):
            assert _code(_doc(), ops) == "op-shape"

    @pytest.mark.parametrize(
        "path",
        [
            "",  # the whole document
            "states/a/purpose",  # no leading slash
            "/states/a//purpose",
            "/states//purpose",
            "/states/a/purpose/",
            "/states/a/~2",
            "/states/a/~",
            "/states/a/a~",
            "/states/a/~x",
            "/" + "/".join(["a"] * 13),  # 13 tokens
            "/states/a/" + "x" * 101,
            7,
            None,
            ["states"],
        ],
    )
    def test_a_bad_pointer_is_refused(self, path):
        op = {"op": "add", "path": path, "value": 1}
        assert _code(_doc(), [op]) == "bad-pointer"

    def test_twelve_tokens_are_the_most(self):
        path = "/states/a/" + "/".join(["k"] * 9)  # 12 tokens
        assert _code(_doc(), [{"op": "add", "path": path, "value": 1}]) == "no-target"

    @pytest.mark.parametrize(
        "token", ["01", "-0", "-1", "+1", "1.0", "1e2", "٣", "²", "x"]
    )
    def test_a_bad_array_index_is_refused(self, token):
        path = f"/states/a/transitions/{token}"
        assert _code(_doc(), [{"op": "add", "path": path, "value": {}}]) == "bad-index"

    def test_index_digits_are_capped_before_any_conversion(self):
        big = "1" * 10_000
        path = f"/states/a/transitions/{big}"
        wide = SynthesisPolicy(max_patch_chars=50_000)
        op = {"op": "test", "path": path, "value": 1}
        assert _code(_doc(), [op], policy=wide) == "bad-pointer"  # token too long
        path = f"/states/a/transitions/{'9' * 10}"
        assert _code(_doc(), [{"op": "test", "path": path, "value": 1}]) == "bad-index"
        path = f"/states/a/transitions/{'0' * 9}"
        assert _code(_doc(), [{"op": "test", "path": path, "value": 1}]) == "bad-index"
        path = f"/states/a/transitions/{'9' * 9}"
        assert _code(_doc(), [{"op": "test", "path": path, "value": 1}]) == "no-target"

    def test_a_numeric_token_in_an_object_is_just_a_key(self):
        op = {"op": "add", "path": "/states/a/01", "value": 1}
        assert _run(_doc(), [op])["states"]["a"]["01"] == 1


class TestHostileValues:
    def test_a_5000_level_value_is_refused_without_recursion(self):
        value = _nest(5000)
        op = {"op": "add", "path": "/states/a/x", "value": value}
        wide = SynthesisPolicy(max_patch_chars=10**6)
        before = sys.getrecursionlimit()
        assert _code(_doc(), [op], policy=wide) == "too-deep"
        assert sys.getrecursionlimit() == before

    def test_a_self_referencing_value_is_too_deep_not_a_hang(self):
        value: list[Any] = []
        value.append(value)
        op = {"op": "add", "path": "/states/a/x", "value": value}
        assert _code(_doc(), [op]) == "too-deep"

    def test_depth_16_passes_and_17_is_refused(self):
        nest = _nest
        wide = SynthesisPolicy(max_patch_chars=10**6)
        ok = {"op": "add", "path": "/states/a/x", "value": nest(MAX_PATCH_DEPTH)}
        assert _run(_doc(), [ok], policy=wide)["states"]["a"]["x"] == nest(16)
        bad = {"op": "add", "path": "/states/a/x", "value": nest(MAX_PATCH_DEPTH + 1)}
        assert _code(_doc(), [bad], policy=wide) == "too-deep"

    def test_dict_nesting_counts_too(self):
        value: Any = 1
        for _ in range(MAX_PATCH_DEPTH + 1):
            value = {"k": value}
        op = {"op": "add", "path": "/states/a/x", "value": value}
        assert _code(_doc(), [op], policy=SynthesisPolicy(max_patch_chars=10**6)) == (
            "too-deep"
        )

    def test_a_huge_int_is_refused_never_converted(self):
        for n in (10**5000, -(10**5000), 2**64, -(2**64)):
            op = {"op": "add", "path": "/states/a/x", "value": n}
            assert _code(_doc(), [op]) == "bad-value"
        ok = {"op": "add", "path": "/states/a/x", "value": 2**63}
        assert _run(_doc(), [ok])["states"]["a"]["x"] == 2**63

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), -float("inf"), (1, 2), {1, 2}, b"x", object(), 1j],
    )
    def test_a_non_json_value_is_refused(self, value):
        op = {"op": "add", "path": "/states/a/x", "value": value}
        assert _code(_doc(), [op]) == "bad-value"

    def test_a_non_string_key_is_refused(self):
        op = {"op": "add", "path": "/states/a/x", "value": {1: "a"}}
        assert _code(_doc(), [op]) == "bad-value"

    def test_the_size_cap_counts_every_string_and_key(self):
        small = SynthesisPolicy(max_patch_chars=200)
        for value in ("x" * 300, ["y" * 40] * 10, {"k" * 50: "v" * 160}, [0] * 100):
            op = {"op": "add", "path": "/states/a/x", "value": value}
            assert _code(_doc(), [op], policy=small) == "too-large"

    def test_a_million_key_dict_is_refused_unwalked(self):
        big = dict.fromkeys(map(str, range(300_000)), 0)
        op = {"op": "add", "path": "/states/a/x", "value": big}
        assert _code(_doc(), [op]) == "too-large"

    def test_a_million_element_list_is_refused_unwalked(self):
        op = {"op": "add", "path": "/states/a/x", "value": [0] * 1_000_000}
        assert _code(_doc(), [op]) == "too-large"

    def test_the_default_8000_characters_hold_a_real_edit(self):
        text = "word " * 100  # 500 characters
        op = {"op": "add", "path": "/states/a/response_instructions", "value": text}
        assert _run(_doc(), [op])["states"]["a"]["response_instructions"] == text
        assert len(json.dumps(op)) < POLICY.max_patch_chars

    def test_too_many_ops_is_refused_before_the_walk(self):
        op = {"op": "test", "path": "/states/a/purpose", "value": "ask"}
        assert _run(_doc(), [op] * 12) == _doc()
        assert _code(_doc(), [op] * 13) == "too-many-ops"
        tight = SynthesisPolicy(max_patch_ops=2)
        assert _code(_doc(), [op] * 3, policy=tight) == "too-many-ops"

    def test_a_non_dict_document_is_refused(self):
        for doc in (None, [], "x", 3):
            with pytest.raises(PatchError) as info:
                _run(doc, [])
            assert info.value.code == "bad-document"


class TestErrorSurface:
    def test_patch_error_is_a_synthesis_error_with_a_code_and_no_patch_text(self):
        secret = "SECRET-TEXT-" + "z" * 50
        op = {"op": "add", "path": f"/states/a/{secret}/x", "value": secret}
        with pytest.raises(PatchError) as info:
            _run(_doc(), [op])
        err = info.value
        assert isinstance(err, SynthesisError) and isinstance(err, FSMError)
        assert err.code == "no-target" and err.op == 0
        assert secret not in str(err) and "z" * 20 not in str(err)

    def test_the_refusal_table_reaches_every_code_and_no_other(self):
        seen = set()
        for ops, scope in [
            ([{"op": "test", "path": "/states/a/purpose", "value": "x"}] * 13, ("a",)),
            ([{"op": "test", "path": "/states/a/p", "value": "x" * 9000}], ("a",)),
            ([{"op": "add", "path": "/states/a/p", "value": _nest(17)}], ("a",)),
            ([{"op": "add", "path": "/states/a/p", "value": float("nan")}], ("a",)),
            ([{"op": "add", "path": "/states/a/p"}], ("a",)),
            ([{"op": "move", "path": "/states/a/p"}], ("a",)),
            ([{"op": "add", "path": "", "value": 1}], ("a",)),
            ([{"op": "add", "path": "/states/a/transitions/01", "value": 1}], ("a",)),
            ([{"op": "add", "path": "/persona", "value": 1}], ("a",)),
            ([{"op": "remove", "path": "/states/a/zzz"}], ("a",)),
            ([{"op": "test", "path": "/states/a/purpose", "value": "no"}], ("a",)),
            ([{"op": "test", "path": "/states/a/purpose", "value": "ask"}], "a"),
        ]:
            seen.add(_code(_doc(), ops, scope=scope))
        with pytest.raises(PatchError) as info:
            _run([], [])
        seen.add(info.value.code)
        assert seen == set(PATCH_CODES)
