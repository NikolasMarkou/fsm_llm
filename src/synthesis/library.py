"""
The procedure library: a content-addressed, file-backed store of compiled FSMs.

Layout under the library root::

    definitions/<sha256>.json   the definition, immutable, named by its digest
    records/<sha256>.json       its ``LibraryRecord`` (lifecycle, stats, provenance)

The digest is the full sha256 of the same canonical dump ``API`` hashes
(``model_dump(exclude_defaults=True)``, sorted keys); its first 8 hex digits
are the suffix of ``API.process_fsm_definition``'s id (pinned by a test).

Trust data fails loud: a corrupt or unparseable file, a definition whose bytes
do not hash to its name, a record that names another digest and an unreadable
directory all raise ``LibraryError``, and a corrupt file is never overwritten.
Stored goal text is data, never an instruction; it is capped on store.

Concurrency: one lock per ``FSMLibrary`` makes each read-modify-write of a
record atomic between threads. Two processes on one directory are last-writer
wins on the stats (an accepted residual: counters may undercount; lifecycle
edges and definitions stay valid).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable, Collection, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, model_validator

from fsm_llm.agents.exceptions import require_int_limit
from fsm_llm.agents.semantic_tools import EmbedTextsFn, cosine_similarity
from fsm_llm.definitions import FSMDefinition
from fsm_llm.logging import logger
from fsm_llm.okf import capped_line, iso_utc
from fsm_llm.security import resolve_within

from .constants import (
    LIFECYCLE_DEPRECATED,
    LIFECYCLE_DRAFT,
    LIFECYCLE_TRUSTED,
    LIFECYCLE_VERIFIED,
    LIFECYCLES,
)
from .exceptions import LibraryError, SynthesisError

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")
_GOAL_CHARS = 500  # stored goal text; longer goals are cut
_LABEL_CHARS = 120  # stored model name, run id and approver
_PROVENANCE_KEYS = frozenset(
    {
        "synthesizing_model",
        "goal",
        "parent_run",
        "gates_passed",
        "approved_by",
        "parent",
        "created_at",
    }
)
_SIGNATURE_KEYS = frozenset({"inputs_required", "return_keys", "tools_used"})
# The explicit lifecycle edge table: nothing leaves deprecated, and a draft
# never skips verification.
_EDGES = {
    LIFECYCLE_DRAFT: {LIFECYCLE_VERIFIED, LIFECYCLE_DEPRECATED},
    LIFECYCLE_VERIFIED: {LIFECYCLE_TRUSTED, LIFECYCLE_DEPRECATED},
    LIFECYCLE_TRUSTED: {LIFECYCLE_DEPRECATED},
    LIFECYCLE_DEPRECATED: set(),
}
_RANK = {LIFECYCLE_DRAFT: 0, LIFECYCLE_VERIFIED: 1, LIFECYCLE_TRUSTED: 2}


class LibraryRecord(BaseModel):
    """What the library knows about one stored definition.

    ``signature`` is ``{inputs_required, return_keys, tools_used}`` (lists of
    names); ``stats`` is ``{runs, successes}``; ``provenance`` holds only the
    keys in ``_PROVENANCE_KEYS``; ``version`` is ``major.minor`` per lineage.
    A file with any other shape fails validation (``extra="forbid"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    digest: str
    name: str
    version: str
    lifecycle: str
    signature: dict[str, list[str]]
    provenance: dict[str, str | list[str] | None]
    stats: dict[str, int]

    @model_validator(mode="after")
    def _check_shape(self) -> LibraryRecord:
        problems = [
            "digest" if not _DIGEST_RE.fullmatch(self.digest) else "",
            "version" if not _VERSION_RE.fullmatch(self.version) else "",
            "lifecycle" if self.lifecycle not in LIFECYCLES else "",
            "signature" if set(self.signature) != _SIGNATURE_KEYS else "",
            "provenance" if not set(self.provenance) <= _PROVENANCE_KEYS else "",
            "stats"
            if set(self.stats) != {"runs", "successes"} or min(self.stats.values()) < 0
            else "",
        ]
        if bad := [p for p in problems if p]:
            raise ValueError(f"invalid record field(s): {', '.join(bad)}")
        return self


class LibraryMatch(NamedTuple):
    """One search hit: the record, its score and the scale of that score.

    ``embedded`` is True for a cosine from the embedder, a similarity claim a
    threshold may judge. False is the word fallback: 1.0 only for the same
    goal (case, punctuation and spacing aside), at most 0.5 otherwise, so it
    ranks near-misses and never passes a reuse threshold above 0.5.
    """

    record: LibraryRecord
    score: float
    embedded: bool = False


def _canonical(definition: FSMDefinition) -> tuple[str, str]:
    """``(digest, canonical JSON text)``: the recipe of ``API.process_fsm_definition``."""
    text = json.dumps(definition.model_dump(exclude_defaults=True), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest(), text


def digest_of(definition: FSMDefinition) -> str:
    """The 64-hex content digest of a typed definition (the library's id)."""
    return _canonical(definition)[0]


def _words(text: str) -> list[str]:
    """The lowercase alphanumeric tokens of *text*, in order."""
    return re.findall(r"[^\W_]+", text.lower())


def _lexical(want: list[str], have: list[str]) -> float:
    """1.0 for the same token sequence, else half the Jaccard word overlap.
    Set overlap ignores negation and word order, so it is never a reuse score."""
    if want == have:
        return 1.0
    return len(set(want) & set(have)) / len(set(want) | set(have)) / 2


def _scores(
    goal: str, texts: list[str], embed: EmbedTextsFn | None
) -> tuple[list[float], bool]:
    """``(scores, embedded)`` of *texts* against *goal*: cosine over one batch
    ``embed`` call (``embedded`` True), else (no embedder, or one that raises
    or returns a wrong count, uneven or non-finite vectors) ``_lexical``, with
    one WARNING for a failed embedder."""
    if embed is not None:
        try:
            rows = [[float(x) for x in v] for v in embed([goal, *texts])]
            if len(rows) != len(texts) + 1:
                raise ValueError("wrong vector count")
            if len({len(r) for r in rows}) != 1 or not rows[0]:
                raise ValueError("uneven or empty vectors")
            if not all(math.isfinite(x) for r in rows for x in r):
                raise ValueError("non-finite vector")
            return [cosine_similarity(rows[0], r) for r in rows[1:]], True
        except Exception as exc:  # the embedder is caller code: any failure
            logger.warning(
                f"library search: embedder unusable ({type(exc).__name__}: "
                f"{capped_line(str(exc), 100)}); ranking by word overlap"
            )
    want = _words(goal)
    return [_lexical(want, _words(t)) for t in texts], False


class FSMLibrary:
    """A directory of compiled procedures keyed by content digest.

    Interface contract:
        ``FSMLibrary(path)`` expands ``~`` once and resolves the root once;
        nothing is created until the first write. Every digest argument must
        be 64 lowercase hex characters (else ``LibraryError``); every path is
        built through ``security.resolve_within`` and a symlinked file is
        refused. Methods raise ``LibraryError`` for an unknown digest, a
        corrupt entry, an illegal lifecycle edge and any I/O failure, and
        ``SynthesisError`` for a bad argument.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.root = Path(os.path.expanduser(os.fspath(path))).resolve()
        if self.root.exists() and not self.root.is_dir():
            raise LibraryError(f"library path is not a directory: {self.root}")
        self._lock = threading.Lock()

    # -- storage ----------------------------------------------------------

    def _confined(self, rel: str) -> Path:
        try:
            if (self.root / rel).is_symlink():
                raise LibraryError(f"library refuses a symlink: {rel}")
            found = resolve_within(self.root, rel)
        except (OSError, RuntimeError, ValueError) as exc:
            raise LibraryError(f"cannot resolve {rel}: {type(exc).__name__}") from exc
        if found is None:
            raise LibraryError(f"library path escapes the root: {rel}")
        return found

    def _path(self, kind: str, digest: str) -> Path:
        if not (isinstance(digest, str) and _DIGEST_RE.fullmatch(digest)):
            raise LibraryError(f"invalid digest: {capped_line(repr(digest), 70)}")
        return self._confined(f"{kind}/{digest}.json")

    @staticmethod
    def _read(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise LibraryError(f"cannot read {path.name}: {exc}") from exc

    # DECISION plan-2026-11-10T194139-2cefeb61/D-008
    # This atomic write is a copy of `FileSessionStore.save` on purpose: the
    # D-011 (a02151fe) and D-004 (876e7164) anchors forbid sharing one helper
    # without a maintainer call. Keep the `finally` (not `except OSError`: that
    # leaks the temp file on every other exit) and do NOT write a record or
    # definition in place: a crash would leave a torn trust file, which is
    # refused as corrupt forever.
    def _write(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, str(path))
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
        except OSError as exc:
            raise LibraryError(f"cannot write {path.name}: {exc}") from exc

    def _read_record(self, digest: str) -> LibraryRecord | None:
        raw = self._read(self._path("records", digest))
        if raw is None:
            return None
        try:
            record = LibraryRecord.model_validate_json(raw)
        except ValueError as exc:
            raise LibraryError(f"corrupt record {digest[:12]}: {exc}") from exc
        if record.digest != digest:
            raise LibraryError(f"record {digest[:12]} names another digest")
        return record

    def _read_definition(self, digest: str) -> FSMDefinition | None:
        raw = self._read(self._path("definitions", digest))
        if raw is None:
            return None
        if hashlib.sha256(raw).hexdigest() != digest:
            raise LibraryError(f"definition {digest[:12]} does not match its digest")
        try:
            return FSMDefinition.model_validate_json(raw)
        except ValueError as exc:
            raise LibraryError(f"corrupt definition {digest[:12]}: {exc}") from exc

    def _save(self, record: LibraryRecord) -> LibraryRecord:
        self._write(
            self._path("records", record.digest), record.model_dump_json(indent=2)
        )
        return record

    def _update(
        self, digest: str, change: Callable[[LibraryRecord], LibraryRecord]
    ) -> LibraryRecord:
        with self._lock:
            return self._save(change(self.get_record(digest)))

    # -- reads ------------------------------------------------------------

    def get_record(self, digest: str) -> LibraryRecord:
        """The record of *digest*; ``LibraryError`` when absent or corrupt."""
        record = self._read_record(digest)
        if record is None:
            raise LibraryError(f"no record for digest {digest[:12]}")
        return record

    def get_definition(self, digest: str) -> FSMDefinition:
        """The stored definition, re-hashed against its name before it is
        returned; ``LibraryError`` when absent, corrupt or mismatched."""
        definition = self._read_definition(digest)
        if definition is None:
            raise LibraryError(f"no definition for digest {digest[:12]}")
        return definition

    # -- writes -----------------------------------------------------------

    def put(
        self,
        definition: FSMDefinition,
        *,
        goal: str,
        inputs_required: Sequence[str],
        return_keys: Sequence[str],
        synthesizing_model: str = "",
        gates_passed: Sequence[str] = (),
        approved_by: str | None = None,
        lifecycle: str = LIFECYCLE_DRAFT,
        parent: str | None = None,
        parent_run: str | None = None,
    ) -> LibraryRecord:
        """Store *definition* and return its record.

        Idempotent: the same digest returns the existing record unchanged
        (``approved_by`` is only added when the record has no approver). A
        different digest is a new entry; with ``parent`` (the digest of the
        definition it repairs, which must exist) the minor version is parent's
        plus one, else ``1.0``. ``lifecycle`` is ``draft`` or ``verified``.
        ``tools_used`` is read from the definition, not from the caller.
        """
        if lifecycle not in (LIFECYCLE_DRAFT, LIFECYCLE_VERIFIED):
            raise SynthesisError(f"a new entry is draft or verified, not {lifecycle!r}")
        digest, text = _canonical(definition)
        tools = {
            t["function"]["name"]
            for s in definition.states.values()
            if s.completion and s.completion.tools
            for t in s.completion.tools
            if isinstance(t.get("function"), dict)
            and isinstance(t["function"].get("name"), str)
        }
        with self._lock:
            old = self._read_record(digest)
            if self._read_definition(digest) is None:
                self._write(self._path("definitions", digest), text)
            who = approved_by and capped_line(approved_by, _LABEL_CHARS)
            if old is not None:
                if who and not old.provenance["approved_by"]:
                    return self._save(
                        old.model_copy(
                            update={
                                "provenance": {**old.provenance, "approved_by": who}
                            }
                        )
                    )
                return old
            version = "1.0"
            if parent is not None:
                major, minor = _VERSION_RE.fullmatch(
                    self.get_record(parent).version
                ).groups()  # type: ignore[union-attr]  # validated on read
                version = f"{major}.{int(minor) + 1}"
            return self._save(
                LibraryRecord(
                    digest=digest,
                    name=definition.name,
                    version=version,
                    lifecycle=lifecycle,
                    signature={
                        "inputs_required": sorted(set(inputs_required)),
                        "return_keys": sorted(set(return_keys)),
                        "tools_used": sorted(tools),
                    },
                    provenance={
                        "synthesizing_model": capped_line(
                            synthesizing_model, _LABEL_CHARS
                        ),
                        "goal": capped_line(goal, _GOAL_CHARS),
                        "parent_run": parent_run
                        and capped_line(parent_run, _LABEL_CHARS),
                        "gates_passed": list(gates_passed),
                        "approved_by": who or None,
                        "parent": parent,
                        "created_at": iso_utc(datetime.now(timezone.utc)),
                    },
                    stats={"runs": 0, "successes": 0},
                )
            )

    @staticmethod
    def _moved(record: LibraryRecord, target: str, **provenance: str) -> LibraryRecord:
        if target not in _EDGES[record.lifecycle]:
            raise LibraryError(
                f"illegal lifecycle edge {record.lifecycle} -> {target} "
                f"for {record.digest[:12]}"
            )
        return record.model_copy(
            update={
                "lifecycle": target,
                "provenance": {**record.provenance, **provenance},
            }
        )

    def verify(self, digest: str) -> LibraryRecord:
        """``draft -> verified``."""
        return self._update(digest, lambda r: self._moved(r, LIFECYCLE_VERIFIED))

    def approve(self, digest: str, by: str) -> LibraryRecord:
        """``verified -> trusted``, recording who approved (a non-blank name)."""
        who = capped_line(by, _LABEL_CHARS)
        if not who:
            raise SynthesisError("approve needs a non-blank approver name")
        return self._update(
            digest, lambda r: self._moved(r, LIFECYCLE_TRUSTED, approved_by=who)
        )

    def deprecate(self, digest: str) -> LibraryRecord:
        """Any non-deprecated lifecycle to ``deprecated``; nothing leaves it."""
        return self._update(digest, lambda r: self._moved(r, LIFECYCLE_DEPRECATED))

    def record_run(
        self,
        digest: str,
        success: bool,
        *,
        trust_after_successes: int | None = None,
    ) -> LibraryRecord:
        """Count one run. A ``verified`` entry becomes ``trusted`` once its
        successes reach ``trust_after_successes`` (``None``: never by count)."""
        if not isinstance(success, bool):
            raise SynthesisError("success must be a bool")
        if trust_after_successes is not None:
            require_int_limit(
                "trust_after_successes",
                trust_after_successes,
                minimum=1,
                error=SynthesisError,
            )

        def count(record: LibraryRecord) -> LibraryRecord:
            stats = {
                "runs": record.stats["runs"] + 1,
                "successes": record.stats["successes"] + success,
            }
            record = record.model_copy(update={"stats": stats})
            if (
                trust_after_successes is not None
                and record.lifecycle == LIFECYCLE_VERIFIED
                and stats["successes"] >= trust_after_successes
            ):
                record = self._moved(record, LIFECYCLE_TRUSTED)
            return record

        return self._update(digest, count)

    # -- retrieval --------------------------------------------------------

    def search(
        self,
        goal: str,
        *,
        inputs_available: Collection[str],
        return_keys: Collection[str],
        tools_allowed: Collection[str],
        min_lifecycle: str,
        embed: EmbedTextsFn | None = None,
        top_k: int = 3,
    ) -> list[LibraryMatch]:
        """Hits for *goal*, best first (ties by digest).

        Scores come in two scales, told apart by ``LibraryMatch.embedded``.
        An embedder score is a SIMILARITY claim: a cosine does not understand
        negation, so "do not X" can sit next to "X". Without an embedder the
        word fallback scores 1.0 only for the same goal and at most 0.5 for
        anything else (see ``LibraryMatch``). For exact reuse pass
        ``embed=None``.

        The signature filter runs FIRST and is exact: every input the
        procedure requires is in ``inputs_available``, its return keys cover
        ``return_keys``, its tools are within ``tools_allowed``, it is not
        deprecated and is at least ``min_lifecycle`` (``draft``, ``verified``
        or ``trusted``). Only then are the survivors scored against the stored
        goal text, so a high score can never admit a procedure the signature
        refuses. A corrupt record raises ``LibraryError`` for the whole search.
        """
        if min_lifecycle not in _RANK:
            raise SynthesisError(f"min_lifecycle must be one of {sorted(_RANK)}")
        require_int_limit("top_k", top_k, minimum=1, error=SynthesisError)
        have, want, tools = set(inputs_available), set(return_keys), set(tools_allowed)
        try:
            names = os.listdir(self._confined("records"))
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise LibraryError(f"cannot list the library: {exc}") from exc
        kept = []
        for name in sorted(n for n in names if n.endswith(".json")):
            record = self.get_record(name[: -len(".json")])
            sig = record.signature
            if (
                record.lifecycle in _RANK
                and _RANK[record.lifecycle] >= _RANK[min_lifecycle]
                and set(sig["inputs_required"]) <= have
                and want <= set(sig["return_keys"])
                and set(sig["tools_used"]) <= tools
            ):
                kept.append(record)
        if not kept:
            return []
        scores, embedded = _scores(
            goal, [str(r.provenance.get("goal") or "") for r in kept], embed
        )
        ranked = sorted(
            zip(kept, scores, strict=True), key=lambda p: (-p[1], p[0].digest)
        )
        return [LibraryMatch(r, s, embedded) for r, s in ranked[:top_k]]
