from __future__ import annotations

"""
Typed λ-AST nodes for the M1 kernel.

All nodes are frozen Pydantic v2 ``BaseModel`` instances (D-003): immutable,
serialisable via ``model_dump`` / ``model_validate``, and hashable by
structure. The recursive types are resolved with ``model_rebuild()`` at the
end of the module, mirroring the pattern used in ``fsm_llm.definitions``
for the ``FSMDefinition → State → Transition`` cycle.

Node taxonomy (closed set):

- ``Var(name)``              — variable reference, resolved in the executor
                               environment.
- ``Abs(param, body)``       — λ-abstraction.
- ``App(fn, arg)``            — application.
- ``Let(name, value, body)``  — sugar for ``App(Abs(name, body), value)``,
                               kept as its own node so serialised terms
                               remain readable.
- ``Case(scrutinee, branches, default=None)``
                              — finite discrimination; scrutinee must
                               evaluate to a value in ``branches`` or
                               ``default`` must be set.
- ``Combinator(op, args)``    — one of the 7 library combinators. ``op``
                               is a closed ``CombinatorOp`` enum (I2).
- ``Fix(body)``               — bounded recursion; ``body`` is an ``Abs``
                               whose single parameter receives the
                               recursive binding.
- ``Leaf(template, input_vars, schema=None)``
                              — the ONLY node that invokes 𝓜 (I1).
                               Carries a prompt template, the env var
                               names to substitute, and an optional
                               structured-output schema (referenced by
                               fully-qualified name so the AST remains
                               picklable — ``BaseModel`` classes themselves
                               are not round-trip-serialisable through
                               ``model_dump``).

The ``schema`` field on ``Leaf`` is ``str | None`` — a dotted path to a
Pydantic model. The executor resolves it at oracle-dispatch time. Carrying
a class object here would break JSON round-trip (SC3).
"""

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CombinatorOp(str, Enum):
    """Closed set of combinator operators (I2).

    Extending this enum requires edits in three files: this enum, the
    executor dispatch table (``executor.py``), and the impl in
    ``combinators.py``. The three-way coupling is intentional — it's what
    makes the stdlib "pre-verified".
    """

    SPLIT = "SPLIT"
    PEEK = "PEEK"
    MAP = "MAP"
    FILTER = "FILTER"
    REDUCE = "REDUCE"
    CONCAT = "CONCAT"
    CROSS = "CROSS"
    # R5 — host-callable bridge. A ``Combinator(op=HOST_CALL, args=(Var(name),
    # *call_args))`` resolves ``Var(name)`` against the executor env (must be
    # a Python callable), evaluates the remaining args, and invokes it. NOT
    # an oracle call (no Leaf): used to splice handler hooks and the streaming
    # response bypass into the compiled term. See plan_43d56276 step 1 + R5.
    HOST_CALL = "HOST_CALL"


# --------------------------------------------------------------
# Base node
# --------------------------------------------------------------


class _FrozenNode(BaseModel):
    """Base class for all AST nodes. Frozen, hashable, extra=forbid."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# Forward-declared in the sum type; concrete classes below all inherit
# from _FrozenNode and carry a ``kind`` discriminator so ``Term`` can be
# validated as a tagged union.


class Var(_FrozenNode):
    kind: Literal["Var"] = "Var"
    name: str = Field(..., min_length=1, max_length=200)


class Abs(_FrozenNode):
    kind: Literal["Abs"] = "Abs"
    param: str = Field(..., min_length=1, max_length=200)
    body: Term


class App(_FrozenNode):
    kind: Literal["App"] = "App"
    fn: Term
    arg: Term


class Let(_FrozenNode):
    kind: Literal["Let"] = "Let"
    name: str = Field(..., min_length=1, max_length=200)
    value: Term
    body: Term


class Case(_FrozenNode):
    kind: Literal["Case"] = "Case"
    scrutinee: Term
    # ``branches`` keys are string-valued discriminants. Non-string cases
    # should be encoded by the caller (e.g., ``str(int_value)``). This
    # keeps the AST JSON-roundtrippable.
    branches: dict[str, Term]
    default: Term | None = None


class Combinator(_FrozenNode):
    kind: Literal["Combinator"] = "Combinator"
    op: CombinatorOp
    args: tuple[Term, ...] = Field(default_factory=tuple)


class Fix(_FrozenNode):
    kind: Literal["Fix"] = "Fix"
    body: Term  # expected to be an Abs; enforced by validator


class Leaf(_FrozenNode):
    kind: Literal["Leaf"] = "Leaf"
    template: str = Field(..., min_length=1)
    input_vars: tuple[str, ...] = Field(default_factory=tuple)
    # Dotted path to a Pydantic model class used for structured output.
    # None → unstructured ``generate_response`` path.
    schema_ref: str | None = None
    # Optional model override (e.g., to route this leaf through a smaller
    # or larger model than the oracle's default). None → oracle default.
    model_override: str | None = None


# Tagged-union alias — all nodes tagged by ``kind`` so a dict validates
# unambiguously to the right subclass.
Term = Annotated[
    Var | Abs | App | Let | Case | Combinator | Fix | Leaf,
    Field(discriminator="kind"),
]


# Resolve forward references for all recursive fields. See
# ``definitions.py:626`` for the same pattern applied to FSMDefinition.
for _cls in (Abs, App, Let, Case, Combinator, Fix):
    _cls.model_rebuild()


def is_term(obj: Any) -> bool:
    """True iff ``obj`` is an instance of one of the concrete term types."""
    return isinstance(obj, Var | Abs | App | Let | Case | Combinator | Fix | Leaf)


__all__ = [
    "CombinatorOp",
    "Var",
    "Abs",
    "App",
    "Let",
    "Case",
    "Combinator",
    "Fix",
    "Leaf",
    "Term",
    "is_term",
]
