from __future__ import annotations

"""
Cost accumulator for Leaf-level oracle usage.

The executor calls ``record()`` after each successful ``Oracle.invoke``.
The accumulator keeps per-leaf entries (indexed by a caller-supplied
``leaf_id`` — typically the Leaf node's ``template`` or a hash) and
aggregates totals. In M1 this is in-memory only; M3 (Monitor integration)
will add a sink.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeafCall:
    """Record of a single Leaf oracle invocation."""

    leaf_id: str
    tokens_in: int
    tokens_out: int
    cost: float


@dataclass
class CostAccumulator:
    """Append-only accumulator of Leaf calls + aggregate totals."""

    calls: list[LeafCall] = field(default_factory=list)

    def record(
        self,
        leaf_id: str,
        tokens_in: int,
        tokens_out: int,
        cost: float = 0.0,
    ) -> None:
        """Append a Leaf call record. Costs may be 0.0 when the backend
        doesn't expose per-call pricing; token counts still accumulate."""
        if tokens_in < 0 or tokens_out < 0:
            raise ValueError(
                f"tokens_in/tokens_out must be non-negative, got "
                f"in={tokens_in}, out={tokens_out}"
            )
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}")
        self.calls.append(
            LeafCall(
                leaf_id=leaf_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
            )
        )

    # ----- aggregates -----

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.calls)

    @property
    def total_tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

    def by_leaf(self) -> dict[str, int]:
        """Call count grouped by ``leaf_id``."""
        out: dict[str, int] = {}
        for c in self.calls:
            out[c.leaf_id] = out.get(c.leaf_id, 0) + 1
        return out

    def reset(self) -> None:
        """Clear all recorded calls. Useful between Executor runs that
        share an accumulator."""
        self.calls.clear()


__all__ = ["LeafCall", "CostAccumulator"]
