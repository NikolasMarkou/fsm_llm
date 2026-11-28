"""Weighted aliases (roadmap E2.2): one name served by several versions, deterministically.

An alias maps a name to routes ``{version label or template identity: weight}``. Which route a
conversation takes is a pure function of the experiment name and the conversation id, so an
assignment is reproducible and a log can name the identity that was served. A resumed
conversation passes the identity it was served and never rolls again (the registry does that
check before it asks this module).

Interface contract:

- ``make_alias(name, routes, target=None) -> AliasSpec``: ``routes`` is a non-empty mapping of a
  non-blank ``str`` key to a finite number that is not a ``bool`` and is at least 0, with a
  positive sum; ``target`` (default ``name``) is the executable id a version label names. A
  violation is a ``ValueError``. The spec keeps the routes sorted by key.
- ``is_template_identity(key)``: true for a 64-character lowercase hex string (a template
  identity); any other key is a version label.
- ``pick_route(spec, experiment, conversation_id) -> str``: a one-route alias returns its key
  with no seed; otherwise both seeds are required (``ValueError``: an unseeded roll would not be
  reproducible). The draw is the first 64 bits of ``sha256(f"{experiment}:{conversation_id}")``
  as a fraction of 2**64 scaled by the total weight over the cumulative weights of the sorted
  routes. A zero-weight route is never picked.

Plan ace7a8c8 D-059. Pure functions: no lock, no clock, no randomness.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["AliasSpec", "is_template_identity", "make_alias", "pick_route"]

_IDENTITY = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AliasSpec:
    """A validated alias: the id its version labels name and the routes sorted by key."""

    name: str
    target: str
    routes: tuple[tuple[str, float], ...]


def is_template_identity(key: str) -> bool:
    """True when *key* is a 64-hex template identity rather than a version label."""
    return _IDENTITY.fullmatch(key) is not None


def make_alias(
    name: str, routes: Mapping[str, float], target: str | None = None
) -> AliasSpec:
    """Validate and freeze an alias (see the module docstring)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("an alias name must be a non-blank str")
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError(f"alias {name!r} needs at least one route")
    checked: list[tuple[str, float]] = []
    for key, weight in routes.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"alias {name!r}: a route key must be a non-blank str")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                f"alias {name!r}: the weight of {key!r} must be a finite number >= 0, "
                f"got {weight!r}"
            )
        checked.append((key, float(weight)))
    if not sum(w for _, w in checked) > 0:
        raise ValueError(f"alias {name!r}: the weights must sum to more than 0")
    return AliasSpec(name, name if target is None else target, tuple(sorted(checked)))


def pick_route(
    spec: AliasSpec, experiment: str | None, conversation_id: str | None
) -> str:
    """The route key of one conversation (see the module docstring)."""
    live = [(key, weight) for key, weight in spec.routes if weight > 0]
    if len(live) == 1:
        return live[0][0]
    if not experiment or not conversation_id:
        raise ValueError(
            f"alias {spec.name!r} has several routes: an experiment name and a "
            "conversation id are both required to pick one"
        )
    digest = hashlib.sha256(f"{experiment}:{conversation_id}".encode()).hexdigest()
    point = int(digest[:16], 16) / 2**64 * sum(weight for _, weight in live)
    total = 0.0
    for key, weight in live:
        total += weight
        if point < total:
            return key
    return live[-1][0]
