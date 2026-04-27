from __future__ import annotations

"""
fsm_llm.stdlib.long_context — Category-C long-context λ-term factories.

This sub-package provides named λ-term factories for the patterns described
in the paper that motivates ``docs/lambda.md`` (Roy et al., 2026): recursive
SPLIT → FMAP(self) → REDUCE decomposition over inputs that exceed the
oracle's context window. Each factory returns a ``Term`` ready to be passed
to ``fsm_llm.runtime.Executor.run``.

M5 slice 1 ships ``niah`` (needle-in-haystack QA). Slice 2 adds
``aggregate`` (synthesise across all chunks). Future slices add
``pairwise``, ``multi_hop``, and OOLONG/OOL-Pairs equivalents.

Purity invariant: this package MUST NOT import ``fsm_llm.llm``,
``fsm_llm.fsm``, or ``fsm_llm.pipeline``. It is a pure term-builder layer
on top of ``fsm_llm.runtime``.
"""

from .aggregate import aggregate, aggregate_op
from .multi_hop import (
    make_dynamic_hop_runner,
    multi_hop,
    multi_hop_dynamic,
    not_found_gate,
)
from .niah import best_answer_op, make_size_bucket, niah
from .niah_padded import (
    aligned_size,
    make_pad_callable,
    niah_padded,
    pad_to_aligned,
)
from .pairwise import compare_op, oracle_compare_op, pairwise

__all__ = [
    "niah",
    "aggregate",
    "pairwise",
    "multi_hop",
    "multi_hop_dynamic",
    "make_dynamic_hop_runner",
    "not_found_gate",
    "niah_padded",
    "make_size_bucket",
    "best_answer_op",
    "aggregate_op",
    "compare_op",
    "oracle_compare_op",
    "aligned_size",
    "pad_to_aligned",
    "make_pad_callable",
]
