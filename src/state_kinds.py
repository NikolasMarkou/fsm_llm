"""The state kinds: which kinds of state exist and the rules each kind owns.

A state is one of three kinds today: ``conversation`` (the default: field,
bulk and classification extraction, then a reply), ``completion`` (Pass 1 is
one model completion over a consumer-owned transcript) and ``synthesize`` (a
runner outside core pushes a compiled procedure). Until this module the rules
of the two non-default kinds were spelled out in the State validator, the
definition loader, the data-flow and liveness analyses, the dry-run gate, the
pipeline and ``API.push_fsm``. Row 0.14 of the roadmap gathers them here, one
spec per kind, and each later step of plan d454fd62 moves one rule family to
read its spec instead of testing ``state.completion`` or ``state.synthesize``.

A leaf core module, like ``schema_registry`` and ``codes``: it imports only the
standard library, never logs and never raises on a loaded state. A state is read
through ``getattr`` (its kind config by field name), so this module needs no
``State`` import and ``definitions`` can import it. ``fsm_llm/__init__.py`` does
not import it; the registry is internal until the kind extension point is
public (roadmap C10).

Interface contract:

- ``KINDS``: the ordered tuple of specs, the default kind first. The order is
  the detection order: ``kind_of`` returns the first config-bearing spec whose
  config field is set, else the default. A state holding two configs is refused
  by the State validator; ``model_construct`` and ``model_copy(update=)`` can
  bypass that, which is why the order is fixed.
- ``kind_of(state)``: the ``KindSpec`` of a state or any object with the same
  attributes. Never raises; an object without a config attribute is the default
  kind.
- ``get_kind(name)``: the spec with that name, ``KeyError`` for an unknown one.
- ``KindSpec`` fields (all declarative; the engine's executors are a separate
  module, the verifier hooks that need the evaluator are added with the steps
  that move them):

  - ``name``: the kind name, the value of ``State.kind``.
  - ``config_field``: the ``State`` field holding the kind's config model, or
    ``None`` for the default kind (declared by the absence of any config).
  - ``result_key(state)``: the context key the kind writes its result to, or
    ``None``. Such a key is handler-only: no extraction may write it.
  - ``inherits_result_on_push``: ``False`` when ``API.push_fsm`` must drop the
    parent's value of the result key from the child's initial context.
  - ``forbidden_declarations``: the ``State`` fields the kind cannot also
    declare (they would be dead config); the loader refuses them.
  - ``dryrun_judged``: ``False`` when gate G6a does not judge the edges that
    leave a state of the kind.

``State.kind`` is a plain property over ``kind_of`` (never a field, a computed
field or a private attribute: any of them would reach ``model_dump`` and move
``fsm_id`` and the library digest).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

__all__ = [
    "COMPLETION",
    "CONVERSATION",
    "KINDS",
    "SYNTHESIZE",
    "CompletionStateKind",
    "KindSpec",
    "get_kind",
    "kind_of",
]

_E = TypeVar("_E", bound=Enum)


class CompletionStateKind(str, Enum):
    """The names of the three built-in state kinds (the values of ``State.kind``).

    Named for the state kinds, not ``codes.CompletionKind`` (the ``kind`` of a
    model reply: calls, final, malformed). Use ``.value`` in messages.
    """

    CONVERSATION = "conversation"
    COMPLETION = "completion"
    SYNTHESIZE = "synthesize"

    @classmethod
    def from_string(cls: type[_E], value: str) -> _E:
        """The member for *value*; ``ValueError`` for an unknown string."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown state kind {value!r}; expected one of {[m.value for m in cls]}"
        )


def _no_result_key(state: Any) -> str | None:
    return None


def _result_key_of(config_field: str) -> Callable[[Any], str | None]:
    """The hook reading ``<config_field>.result_key`` (``None`` when unset)."""

    def result_key(state: Any) -> str | None:
        config = getattr(state, config_field, None)
        return None if config is None else config.result_key

    return result_key


@dataclass(frozen=True)
class KindSpec:
    """One state kind: its name, config field and declarative rules.

    See the module docstring for the field contract. Frozen: a spec is shared
    by every state of its kind.
    """

    name: str
    config_field: str | None = None
    result_key: Callable[[Any], str | None] = _no_result_key
    inherits_result_on_push: bool = True
    forbidden_declarations: tuple[str, ...] = ()
    dryrun_judged: bool = True


# The extraction channels a kind that replaces Pass 1 cannot also declare: they
# would never run (the loader refuses them, in this order).
_EXTRACTION_DECLARATIONS = (
    "field_extractions",
    "classification_extractions",
    "required_context_keys",
    "extraction_instructions",
)

CONVERSATION = KindSpec(name=CompletionStateKind.CONVERSATION.value)

COMPLETION = KindSpec(
    name=CompletionStateKind.COMPLETION.value,
    config_field="completion",
    result_key=_result_key_of("completion"),
    # An inherited completion result made the child's completion state skip its
    # call (skip-if-set) and answer with the parent's reply, so the key is
    # dropped on push (the rule stays anchored in `API.push_fsm`).
    inherits_result_on_push=False,
    forbidden_declarations=_EXTRACTION_DECLARATIONS,
    dryrun_judged=False,
)

# The synthesize result key is NOT dropped on push today (an inherited
# `synthesis_result` makes a nested synthesize state skip itself): the flag
# keeps that behaviour until its own fix (plan d454fd62 step 55).
SYNTHESIZE = KindSpec(
    name=CompletionStateKind.SYNTHESIZE.value,
    config_field="synthesize",
    result_key=_result_key_of("synthesize"),
    inherits_result_on_push=True,
    forbidden_declarations=("completion", *_EXTRACTION_DECLARATIONS),
    dryrun_judged=False,
)

KINDS: tuple[KindSpec, ...] = (CONVERSATION, COMPLETION, SYNTHESIZE)


def get_kind(name: str) -> KindSpec:
    """The spec named *name*; ``KeyError`` for an unknown kind."""
    for spec in KINDS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown state kind {name!r}")


def kind_of(state: Any) -> KindSpec:
    """The kind of *state*: the first spec in ``KINDS`` whose config is set,
    else the default kind. Reads ``state`` by ``getattr`` only; never raises."""
    for spec in KINDS:
        if spec.config_field and getattr(state, spec.config_field, None) is not None:
            return spec
    return CONVERSATION
