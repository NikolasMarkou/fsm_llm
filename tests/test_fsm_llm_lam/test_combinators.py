from __future__ import annotations

"""Tests for fsm_llm.lam.combinators — the 7 pure ℒ∖{𝓜} impls."""

import pytest

from fsm_llm.runtime.combinators import (
    BUILTIN_OPS,
    ReduceOp,
    concat_impl,
    cross_impl,
    filter_impl,
    map_impl,
    peek_impl,
    reduce_impl,
    split_impl,
)
from fsm_llm.runtime.errors import TerminationError


class TestSplit:
    def test_split_string_rank_reducing(self) -> None:
        pieces = split_impl("abcdefgh", 2)
        assert len(pieces) <= 2
        assert all(len(q) < len("abcdefgh") for q in pieces)
        assert "".join(pieces) == "abcdefgh"

    def test_split_list_rank_reducing(self) -> None:
        pieces = split_impl([1, 2, 3, 4, 5], 3)
        assert len(pieces) <= 3
        assert all(len(q) < 5 for q in pieces)
        flat = [x for piece in pieces for x in piece]
        assert flat == [1, 2, 3, 4, 5]

    def test_split_identity_on_small(self) -> None:
        # E1: rank-1 input → identity
        assert split_impl("a", 2) == ["a"]
        assert split_impl([], 2) == [[]]

    def test_split_identity_on_k_equals_1(self) -> None:
        assert split_impl("abc", 1) == ["abc"]

    def test_split_rejects_bad_k(self) -> None:
        with pytest.raises(TerminationError):
            split_impl("abc", 0)
        with pytest.raises(TerminationError):
            split_impl("abc", -1)
        with pytest.raises(TerminationError):
            split_impl("abc", "two")  # type: ignore[arg-type]

    def test_split_non_sized_identity(self) -> None:
        obj = object()
        assert split_impl(obj, 2) == [obj]


class TestPeek:
    def test_peek_string(self) -> None:
        assert peek_impl("hello world", 5) == "hello"

    def test_peek_list(self) -> None:
        assert peek_impl([1, 2, 3, 4], 2) == [1, 2]

    def test_peek_beyond_length(self) -> None:
        assert peek_impl("ab", 100) == "ab"

    def test_peek_zero(self) -> None:
        assert peek_impl("abc", 0) == ""

    def test_peek_rejects_negative(self) -> None:
        with pytest.raises(TerminationError):
            peek_impl("abc", -1)


class TestMap:
    def test_map_basic(self) -> None:
        assert map_impl(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    def test_map_empty(self) -> None:
        # E2: empty → empty
        assert map_impl(lambda x: x, []) == []


class TestFilter:
    def test_filter_basic(self) -> None:
        assert filter_impl(lambda x: x > 1, [0, 1, 2, 3]) == [2, 3]

    def test_filter_empty(self) -> None:
        assert filter_impl(lambda x: True, []) == []


class TestReduce:
    def test_reduce_with_reduceop_nonempty(self) -> None:
        assert reduce_impl(BUILTIN_OPS["sum"], [1, 2, 3, 4]) == 10

    def test_reduce_empty_with_unit(self) -> None:
        # E2 happy path: unit declared
        assert reduce_impl(BUILTIN_OPS["sum"], []) == 0
        assert reduce_impl(BUILTIN_OPS["concat_str"], []) == ""

    def test_reduce_empty_no_unit_raises(self) -> None:
        # E2 unhappy path: no unit, empty input → TerminationError
        with pytest.raises(TerminationError):
            reduce_impl(BUILTIN_OPS["max"], [])

    def test_reduce_bare_callable(self) -> None:
        assert reduce_impl(lambda a, b: a + b, [1, 2, 3]) == 6

    def test_reduce_bare_callable_empty_raises(self) -> None:
        with pytest.raises(TerminationError):
            reduce_impl(lambda a, b: a + b, [])

    def test_reduce_associative_flag_respected(self) -> None:
        # All built-ins are declared associative.
        for op in BUILTIN_OPS.values():
            assert op.associative is True


class TestConcat:
    def test_concat_lists(self) -> None:
        assert concat_impl([1, 2], [3, 4], [5]) == [1, 2, 3, 4, 5]

    def test_concat_mixed_iterables(self) -> None:
        assert concat_impl("ab", "cd") == ["a", "b", "c", "d"]

    def test_concat_empty(self) -> None:
        assert concat_impl() == []


class TestCross:
    def test_cross_basic(self) -> None:
        result = cross_impl([1, 2], ["a", "b"])
        assert result == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]

    def test_cross_one_empty(self) -> None:
        assert cross_impl([], [1, 2]) == []
        assert cross_impl([1, 2], []) == []


class TestReduceOpRegistry:
    def test_custom_reduce_op(self) -> None:
        xor_op = ReduceOp(name="xor", fn=lambda a, b: a ^ b, associative=True, unit=0)
        assert reduce_impl(xor_op, [1, 2, 3]) == (1 ^ 2 ^ 3)
        assert reduce_impl(xor_op, []) == 0
