from __future__ import annotations

"""Tests for fsm_llm.lam.cost — accumulator record/reset semantics."""

import pytest

from fsm_llm.runtime.cost import CostAccumulator, LeafCall


class TestRecord:
    def test_record_single(self) -> None:
        acc = CostAccumulator()
        acc.record("leaf_A", tokens_in=10, tokens_out=20, cost=0.001)
        assert acc.total_calls == 1
        assert acc.total_tokens_in == 10
        assert acc.total_tokens_out == 20
        assert acc.total_cost == pytest.approx(0.001)

    def test_record_many(self) -> None:
        acc = CostAccumulator()
        for _ in range(5):
            acc.record("leaf_B", 3, 4, 0.5)
        assert acc.total_calls == 5
        assert acc.total_tokens_in == 15
        assert acc.total_tokens_out == 20
        assert acc.total_cost == pytest.approx(2.5)

    def test_record_no_cost(self) -> None:
        acc = CostAccumulator()
        acc.record("leaf", 1, 1)
        assert acc.total_cost == 0.0

    def test_record_negative_raises(self) -> None:
        acc = CostAccumulator()
        with pytest.raises(ValueError):
            acc.record("leaf", -1, 0)
        with pytest.raises(ValueError):
            acc.record("leaf", 0, -1)
        with pytest.raises(ValueError):
            acc.record("leaf", 0, 0, -0.1)


class TestByLeaf:
    def test_groups(self) -> None:
        acc = CostAccumulator()
        acc.record("A", 1, 1)
        acc.record("A", 1, 1)
        acc.record("B", 1, 1)
        assert acc.by_leaf() == {"A": 2, "B": 1}


class TestReset:
    def test_reset_clears(self) -> None:
        acc = CostAccumulator()
        acc.record("leaf", 1, 2, 0.1)
        acc.reset()
        assert acc.total_calls == 0
        assert acc.total_tokens_in == 0
        assert acc.total_tokens_out == 0
        assert acc.total_cost == 0.0


class TestLeafCallImmutable:
    def test_frozen(self) -> None:
        call = LeafCall(leaf_id="x", tokens_in=1, tokens_out=2, cost=0.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            call.cost = 1.0  # type: ignore[misc]
