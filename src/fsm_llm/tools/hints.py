"""Signature and type-hint readers shared by the argument binder and the schema inference."""

from __future__ import annotations

import collections.abc
import functools
import inspect
import re
import types
import typing
from typing import Any

# A string (``from __future__``) dict annotation: dict, dict[...], (typing.)Dict[...]
_DICT_ANNOTATION_RE = re.compile(r"\s*(dict|(typing\.)?Dict)(\[.*\])?\s*")


def _is_dict_annotation(ann: Any) -> bool:
    """True for ``dict``, a ``dict[...]`` generic, or a string annotation
    (``from __future__``) spelling one of those. Never raises."""
    return (
        ann is dict
        or typing.get_origin(ann) is dict
        or (isinstance(ann, str) and _DICT_ANNOTATION_RE.fullmatch(ann) is not None)
    )


# Mapping types a tool parameter can be advertised as ``object``: ``dict``, the
# abstract ``Mapping``/``MutableMapping`` and a ``TypedDict``. The lone-positional
# dict-style CALL (``_is_dict_annotation``) is deliberately narrower: dict only.
_MAPPING_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)


def _is_mapping_hint(hint: Any) -> bool:
    """True for ``dict``/``dict[...]``/``typing.Dict``, ``Mapping``,
    ``MutableMapping`` (bare or parametrized) and a ``TypedDict`` class.
    Never raises."""
    origin = typing.get_origin(hint) or hint
    return origin in _MAPPING_ORIGINS or (
        isinstance(hint, type) and typing.is_typeddict(hint)
    )


def _without_none(hint: Any) -> Any:
    """The one non-``None`` member of ``Optional[X]`` (``X | None``), else *hint*."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return hint


def _takes_mapping(annotation: Any) -> bool:
    """True when a parameter's raw annotation is a mapping type, bare or in
    ``Optional`` (a string annotation counts only when it spells ``dict``).
    Never raises."""
    return _is_dict_annotation(annotation) or _is_mapping_hint(
        _without_none(annotation)
    )


_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _accepts_extra_keywords(sig: inspect.Signature) -> bool:
    """True when the function names a parameter AND takes ``**kw``.

    Such a function receives every key the model sends (the schema lists only
    the named ones). A lone ``**kw`` is excluded on purpose: it declares nothing,
    so an explicit schema (an MCP tool) stays its contract and filters.
    """
    params = list(sig.parameters.values())
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params) and any(
        p.kind not in _VAR_KINDS for p in params
    )


def callable_name(fn: Any) -> str:
    """A display name for any callable: ``__name__``, else its type's name.

    ``functools.partial`` objects and callable instances have no ``__name__``
    (review item 3: ``register_function`` raised AttributeError for them).
    """
    return getattr(fn, "__name__", None) or type(fn).__name__


def _introspection_target(fn: Any) -> Any:
    """The function whose hints and coroutine flag describe *fn*.

    Unwraps ``functools.partial`` (nested too) to its ``func``; a callable
    instance maps to its class's ``__call__``. Functions, methods and builtins
    are returned as is. Never raises.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    if inspect.isroutine(fn):
        return fn
    return type(fn).__call__ if callable(fn) else fn
