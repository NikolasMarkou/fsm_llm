"""
The in-memory OKF bundle: concepts by id, the link graph and lexical search.

``OKFBundle`` is built by ``load_bundle`` or by hand; it never touches the file
system and never raises on bundle content.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from fsm_llm.constants import OKF_INCOMPLETE_NOTE, OKF_INDEX_MAX_CHARS
from fsm_llm.text import strip_unsafe_text

from .links import _body_links, _Ids, _link_candidates
from .models import INCOMPLETE_CODES, OKFConcept, OKFIssue

_Concepts = list[OKFConcept]
_Hits = list[tuple[OKFConcept, float]]
_HAY_FIELDS = ((3, 0), (2, 1), (1, 2))  # (weight, index into the haystack)
_HAY_WEIGHT = sum(weight for weight, _ in _HAY_FIELDS)  # best score per token
_CHECK_DEADLINE = OKFIssue(  # frozen, so one instance serves every bundle
    severity="error",
    code="check-deadline",
    message="check time limit (max_seconds) reached; the rest was not checked",
)


class OKFBundle:
    """Concepts by id, plus the link graph and lexical search over them.

    Interface contract:
        - Build with ``OKFBundle(concepts, root=, issues=, okf_version=,
          index_text=, max_seconds=)`` or ``load_bundle``; ``root`` is ``None``
          for an in-memory bundle. ``index_text`` is control-stripped and
          capped. ``max_seconds`` (``None``: unbounded) bounds ``check_bundle``
          and the first ``backlinks`` call, each on its own clock.
        - ``get(id)`` -> concept or ``None``; ``len``, ``in`` and iteration
          (id order) work as for a mapping of ids; ``list(type=, prefix=,
          include_deprecated=False)`` -> concepts in id order (``type`` is
          compared case-insensitively).
        - ``search(query, limit=5, include_deprecated=False)`` -> ``(concept,
          score)`` best first, score in (0, 1], ties by id; lexical, tokens of
          3+ characters; ``[]`` for a non-string or token-less query.
        - ``resolve_link(target, from_id)`` -> the id of an EXISTING concept
          or ``None`` (external, escaping, broken and non-concept targets all
          give ``None``); it also serves path-valued keys such as ``resource``.
        - ``outgoing(id)`` / ``backlinks(id)`` -> ids in order, no self-links;
          outgoing covers body links, ``resource`` and ``sources[].resource``.
          A ``backlinks`` build past ``max_seconds`` adds one ``check-deadline``
          issue and answers from the part built; a finished build removes it.
        - ``incomplete`` -> the ``INCOMPLETE_CODES`` in ``issues`` (a load
          stop); ``put(concept)`` adds or replaces and invalidates every cache.
        Nothing here raises on bundle content or touches the filesystem.
    """

    def __init__(
        self,
        concepts: Iterable[OKFConcept] = (),
        *,
        root: Path | None = None,
        issues: Iterable[OKFIssue] = (),
        okf_version: str | None = None,
        index_text: str = "",
        max_seconds: float | None = None,
    ) -> None:
        self.root = root
        self.max_seconds = max_seconds
        self.issues: list[OKFIssue] = list(issues)
        self.okf_version = okf_version
        self.index_text = strip_unsafe_text(index_text, multiline=True)[
            :OKF_INDEX_MAX_CHARS
        ]
        self._concepts: dict[str, OKFConcept] = {}
        self._reset()
        for concept in concepts:
            self.put(concept)

    @property
    def incomplete(self) -> tuple[str, ...]:
        seen = {i.code for i in self.issues}
        return tuple(c for c in INCOMPLETE_CODES if c in seen)

    def _deadline(self) -> float:
        return time.monotonic() + (self.max_seconds or math.inf)

    def _reset(self) -> None:
        self._ids: _Ids | None = None
        self._out: dict[str, _Ids] = {}
        self._back: dict[str, _Ids] | None = None
        self._hay: dict[str, tuple[str, str, str]] = {}

    def put(self, concept: OKFConcept) -> None:
        self._concepts[concept.id] = concept
        self._reset()

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: object) -> bool:
        return concept_id in self._concepts

    def _sorted_ids(self) -> _Ids:
        if self._ids is None:
            self._ids = sorted(self._concepts)
        return self._ids

    def __iter__(self) -> Iterator[OKFConcept]:
        return (self._concepts[cid] for cid in self._sorted_ids())

    def get(self, concept_id: str) -> OKFConcept | None:
        return self._concepts.get(concept_id)

    def list(
        self,
        type: str | None = None,
        prefix: str | None = None,
        include_deprecated: bool = False,
    ) -> _Concepts:
        wanted = type.strip().casefold() if type else None
        return [
            c
            for c in self
            if (include_deprecated or not c.is_deprecated)
            and (wanted is None or c.type.casefold() == wanted)
            and (not prefix or c.id.startswith(prefix))
        ]

    def search(
        self, query: str, limit: int = 5, include_deprecated: bool = False
    ) -> _Hits:
        if not isinstance(query, str) or limit < 1:
            return []
        found = re.findall(r"\w{3,}", query[:1000].casefold())
        tokens = list(dict.fromkeys(found))[:20]
        hits: _Hits = []
        for concept in self:
            if tokens and (include_deprecated or not concept.is_deprecated):
                hay = self._haystack(concept)
                score = sum(
                    weight
                    for token in tokens
                    for weight, i in _HAY_FIELDS
                    if token in hay[i]
                )
                if score:
                    hits.append((concept, score / (_HAY_WEIGHT * len(tokens))))
        hits.sort(key=lambda hit: (-hit[1], hit[0].id))
        return hits[:limit]

    def _haystack(self, c: OKFConcept) -> tuple[str, str, str]:
        if c.id not in self._hay:
            self._hay[c.id] = (
                f"{c.title or ''}\n{' '.join(c.tags)}".casefold(),
                f"{c.description or ''}\n{c.id}".casefold(),
                c.body.casefold(),
            )
        return self._hay[c.id]

    def resolve_link(self, target: object, from_id: str) -> str | None:
        return next(
            (c for c in _link_candidates(target, from_id) if c in self._concepts),
            None,
        )

    def outgoing(self, concept_id: str) -> _Ids:
        concept = self._concepts.get(concept_id)
        if concept is None:
            return []
        if concept_id not in self._out:
            refs = [
                *_body_links(concept.body),
                concept.resource,
                *(s.get("resource") for s in concept.sources),
            ]
            found = dict.fromkeys(
                t for r in refs if (t := self.resolve_link(r, concept_id))
            )
            found.pop(concept_id, None)
            self._out[concept_id] = list(found)
        return list(self._out[concept_id])

    def backlinks(self, concept_id: str) -> _Ids:
        if self._back is None:
            back: dict[str, _Ids] = {}
            deadline = self._deadline()
            for cid in self._sorted_ids():
                if time.monotonic() > deadline:
                    if not any(i.code == "check-deadline" for i in self.issues):
                        self.issues.append(_CHECK_DEADLINE)
                    return list(back.get(concept_id, []))
                for target in self.outgoing(cid):
                    back.setdefault(target, []).append(cid)
            self._back = back
            self.issues[:] = [i for i in self.issues if i.code != "check-deadline"]
        return list(self._back.get(concept_id, []))


def incomplete_note(bundle: OKFBundle) -> str:
    """The fixed note for a partial bundle (codes only, no issue text); else ``""``."""
    codes = bundle.incomplete
    return OKF_INCOMPLETE_NOTE.format(codes=", ".join(codes)) if codes else ""
