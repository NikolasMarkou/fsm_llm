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

from .aggregate import aggregate as aggregate_term
from .aggregate import aggregate_op
from .multi_hop import (
    make_dynamic_hop_runner,
    not_found_gate,
)
from .multi_hop import multi_hop as multi_hop_term
from .multi_hop import multi_hop_dynamic as multi_hop_dynamic_term
from .niah import best_answer_op, make_size_bucket
from .niah import niah as niah_term
from .niah_padded import (
    aligned_size,
    make_pad_callable,
    pad_to_aligned,
)
from .niah_padded import niah_padded as niah_padded_term
from .pairwise import compare_op, oracle_compare_op
from .pairwise import pairwise as pairwise_term

# ---------------------------------------------------------------------------
# Bare-name back-compat aliases — emit DeprecationWarning at access time.
#
# Per the 0.6.0 cleanup (CHANGELOG): long-context factories were renamed to
# the ``*_term`` convention used by every other stdlib slice (``react_term``,
# ``analytical_term``, ``linear_term``, …). The bare names (``niah``,
# ``aggregate``, …) remain reachable via module-level ``__getattr__`` and
# warn at access time. Removal: 0.7.0.
# ---------------------------------------------------------------------------

_BARE_ALIASES = {
    "niah": "niah_term",
    "aggregate": "aggregate_term",
    "pairwise": "pairwise_term",
    "multi_hop": "multi_hop_term",
    "multi_hop_dynamic": "multi_hop_dynamic_term",
    "niah_padded": "niah_padded_term",
}


def __getattr__(name):  # PEP 562
    if name in _BARE_ALIASES:
        from fsm_llm._api.deprecation import warn_deprecated

        canonical = _BARE_ALIASES[name]
        warn_deprecated(
            f"fsm_llm.stdlib.long_context.{name}",
            since="0.6.0",
            removal="0.7.0",
            replacement=f"fsm_llm.stdlib.long_context.{canonical}",
        )
        # Resolve the canonical *_term symbol from this module's globals.
        return globals()[canonical]
    raise AttributeError(
        f"module 'fsm_llm.stdlib.long_context' has no attribute {name!r}"
    )


__all__ = [
    # Canonical *_term factory names (0.6.0+)
    "niah_term",
    "aggregate_term",
    "pairwise_term",
    "multi_hop_term",
    "multi_hop_dynamic_term",
    "niah_padded_term",
    # Helpers
    "make_dynamic_hop_runner",
    "not_found_gate",
    "make_size_bucket",
    "best_answer_op",
    "aggregate_op",
    "compare_op",
    "oracle_compare_op",
    "aligned_size",
    "pad_to_aligned",
    "make_pad_callable",
]
