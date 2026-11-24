"""``ExecutableRegistry``: serve executables by id, reload atomically, retain templates only.

The registry holds the DOCUMENTS and code-backed executables of a process and hands out
executables by ``(id, version)``, by template identity and through weighted aliases (roadmap
E2.1, E2.2; plan ace7a8c8 D-059). It never holds an instance.

Interface contract:

- ``ExecutableRegistry(paths=(), *, kinds=None, sources=None,
  max_retained_templates=MAX_RETAINED_TEMPLATES)``; ``from_paths(paths, **kw)`` also reloads
  once. ``kinds`` and ``sources`` default to the ``EXECUTABLE_KINDS`` and ``EXECUTABLE_SOURCES``
  namespaces.
- ``add_document(data, *, variables=None, resolver=None) -> Executable``: core
  ``load_executable`` (a ``pattern`` document with no ``resolver`` resolves its handler and
  generator names through the ``HANDLERS`` and ``GENERATORS`` namespaces, D-013 of plan a555dcfe);
  an invalid document raises ``ExecutableError`` carrying every error,
  a duplicate ``(id, version)`` raises ``ExecutableCollisionError``. ``register(executable)``
  adds a code-backed entry under the same collision rule.
- ``load_sources() -> ReloadReport``: calls each factory of the sources namespace that this
  registry has not called yet, once; a factory returns an iterable of executables or envelope
  documents; a failing factory or item is reported, never raised.
- ``reload() -> ReloadReport``: re-reads the path sources. A file that fails keeps its previous
  entries (reported), a file that is gone drops out, memory entries stay, the first source in
  registration order wins a duplicate ``(id, version)`` (reported).
- ``get(id, version=None)`` (greatest version by integer tuple; an unparseable label only as the
  sole version of its id), ``get_by_template(identity)``, ``list(kind=None, capability=None,
  launchable=None)`` (sorted by id and version; ``launchable`` is false when ``requires.callables``
  is non-empty): ``ExecutableNotFoundError`` / ``ExecutableEvictedError`` on a miss.
- ``resolve_target(target_id, *, conversation_id=None) -> ResolvedTarget | None``: the
  ``TargetResolver`` of the ``call`` kind (``engine.kinds.targets``, plan baa37f40 D-021): the
  greatest version of *target_id* instantiated with no variables, answered as its
  ``FSMDefinition`` and both identities when the instance is an FSM body (kinds ``fsm``,
  ``strategy``, ``procedure``) and, for a ``strategy``, its declared ``results`` as the target's own
  output map (``ResolvedTarget.returns``, D-030 of plan baa37f40); ``None`` for a miss, a body
  that does not instantiate, a code-backed kind (``pattern``) and, whatever its kind, an executable
  whose ``requires.handlers`` is non-empty (a frame runs no handler; plan a555dcfe D-010). Never raises for a miss, never loads a module
  by the id and keeps no instance.
- ``resolve_delegated(target_id, *, launcher, context=None, input_keys=(), max_steps=None) ->
  DelegatedTarget | None`` (plan baa37f40 phase K7, D-049): the delegated twin, for the kinds
  ``pattern`` only (a ``workflow`` kind is no target: D-053, D-055). ``None`` (the one answer a caller maps to
  ``failed/unknown_target``) for: a *launcher* that is not a ``RunLauncher`` (``is_run_launcher``), an unknown or
  deprecated id, a kind that is not code-backed (a body kind stays a frame target), the
  capability ``conversational`` (a delegated run has no user to talk to), a name of
  ``requires.callables`` that ``context.callables`` lacks (judged against the HOST context, not
  the global ``launchable`` flag), an input key outside ``contract.inputs`` plus the executable's
  ``optional_inputs``, and ``max_steps`` when the executable's ``refused_budgets`` names it. Never
  raises for a miss, never loads a module by the id and keeps no instance.
- ``set_alias(name, routes, *, target=None)`` and ``resolve(name, *, experiment=None,
  conversation_id=None, pinned_template_identity=None) -> Resolution`` (see ``aliases``).
- ``pin(executable)`` / ``unpin(executable)`` (counted, in memory): a pinned template survives
  eviction. ``rebuild(template_identity, record, expected_instance_identity)`` returns the
  instance rebuilt from a logged record (``InstanceIdentityMismatchError`` on a tampered one).

Concurrency: readers take no lock but one attribute read of an immutable snapshot, then one leaf
lock to touch the retention LRU; ``reload`` swaps the snapshot under that one lock. Plan
ace7a8c8 D-059.
"""

from __future__ import annotations

import builtins
import json
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from fsm_llm.core.codes import Lifecycle
from fsm_llm.core.definitions import FSMDefinition
from fsm_llm.core.definitions.executable import Capability, InstantiationError
from fsm_llm.core.names import NameRegistry
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.engine.context import RunContext
from fsm_llm.engine.kinds.targets import (
    DelegatedTarget,
    ResolvedTarget,
    is_run_launcher,
)
from fsm_llm.logging import logger
from fsm_llm.security import resolve_within

from .aliases import AliasSpec, is_template_identity, make_alias, pick_route
from .namespaces import (
    EXECUTABLE_KINDS,
    EXECUTABLE_SOURCES,
    adapter_for,
    namespace_resolver,
)
from .protocol import (
    Executable,
    ExecutableCollisionError,
    ExecutableError,
    ExecutableEvictedError,
    ExecutableNotFoundError,
    Instance,
)

__all__ = [
    "MAX_DOCUMENT_BYTES",
    "MAX_RETAINED_TEMPLATES",
    "ExecutableRegistry",
    "ReloadReport",
    "Resolution",
]

#: The bound of the template LRU (superseded and in-flight templates beside the current ones).
MAX_RETAINED_TEMPLATES = 256
#: The largest document file the registry reads.
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024

_MEMORY = "memory"
#: The kinds that run as code on a host thread (delegated mode); a body kind is a frame and a
#: ``workflow`` is no target until E5.2 and E5.3 (D-053, D-055).
_DELEGATED_KINDS = ("pattern",)
#: The kinds that are never a frame: their handlers, hooks and per-run objects would not run.
_NEVER_A_FRAME_KINDS = ("pattern",)
_VERSION = re.compile(r"^[0-9]+(\.[0-9]+)*$")


def _version_key(label: str) -> tuple[int, ...] | None:
    return tuple(int(p) for p in label.split(".")) if _VERSION.match(label) else None


class _Entry(NamedTuple):
    executable: Any
    source: str


class _Snapshot:
    """An immutable view: ``(id, version)`` to entry, plus the lookups derived from it."""

    __slots__ = ("by_template", "entries", "latest")

    def __init__(self, entries: dict[tuple[str, str], _Entry]) -> None:
        self.entries = entries
        best: dict[str, tuple[tuple[int, ...], Any]] = {}
        for (ident, version), entry in entries.items():
            key = _version_key(version) or ()
            if ident not in best or key > best[ident][0]:
                best[ident] = (key, entry.executable)
        self.latest = {ident: exe for ident, (_, exe) in best.items()}
        self.by_template = {
            e.executable.template_identity: e.executable for e in entries.values()
        }


def _assemble(
    candidates: Iterable[tuple[Any, str]],
) -> tuple[dict[tuple[str, str], _Entry], list[str]]:
    """First-wins entries over *candidates* in order, and the refusals (pure, no I/O)."""
    entries: dict[tuple[str, str], _Entry] = {}
    versions: dict[str, list[str]] = {}
    refused: list[str] = []
    for executable, source in candidates:
        key = (executable.id, executable.version)
        if key in entries:
            refused.append(
                f"{key[0]!r} version {key[1]!r} from {source} is already served from "
                f"{entries[key].source}"
            )
            continue
        known = versions.get(key[0], [])
        numeric = _version_key(key[1])
        if numeric is not None and any(_version_key(v) == numeric for v in known):
            refused.append(
                f"{key[0]!r} version {key[1]!r} from {source} equals a version already "
                "served as an integer dotted tuple"
            )
            continue
        if known and any(_version_key(v) is None for v in [*known, key[1]]):
            refused.append(
                f"{key[0]!r} version {key[1]!r} from {source}: a version label that is "
                "not an integer dotted tuple is allowed only as the sole version"
            )
            continue
        entries[key] = _Entry(executable, source)
        versions.setdefault(key[0], []).append(key[1])
    return entries, refused


@dataclass(frozen=True)
class ReloadReport:
    """What a reload or a source load did: the keys served, per-source failures, refusals."""

    ids: tuple[tuple[str, str], ...]
    failed: Mapping[str, tuple[str, ...]]
    collisions: tuple[str, ...]


class Resolution(NamedTuple):
    """What ``resolve`` served: the executable, its template identity and the route taken."""

    executable: Any
    template_identity: str
    route: str


class ExecutableRegistry:
    """Executables by id. See the module docstring."""

    def __init__(
        self,
        paths: Iterable[str | os.PathLike[str]] = (),
        *,
        kinds: NameRegistry[Callable[..., Any]] | None = None,
        sources: NameRegistry[Callable[[], Iterable[Any]]] | None = None,
        max_retained_templates: int = MAX_RETAINED_TEMPLATES,
    ) -> None:
        if (
            isinstance(max_retained_templates, bool)
            or not isinstance(max_retained_templates, int)
            or max_retained_templates < 1
        ):
            raise ValueError("max_retained_templates must be an integer >= 1")
        self._paths = tuple(Path(p) for p in paths)
        self._kinds = EXECUTABLE_KINDS if kinds is None else kinds
        self._sources = EXECUTABLE_SOURCES if sources is None else sources
        self._bound = max_retained_templates
        # DECISION plan-2026-11-20T055240-ace7a8c8/D-059
        # `_lock` is a LEAF lock: it guards dict and counter operations only. No file read, no
        # loader, no validation, no logger and no callback runs under it (copy of the
        # `_fsm_cache_lock` rule); `reload` reads and validates every source first and swaps the
        # new immutable snapshot in one step, so a reader sees a wholly old or a wholly new view.
        # The registry retains TEMPLATES (the snapshot plus a bounded LRU, pinned ones exempt) and
        # NEVER an instance: instances are rebuilt from the template and the logged record. Do NOT
        # keep every identity served (a server running millions of distinct tasks would grow
        # without bound: the iter-2 rule is void), do NOT call a loader or log under `_lock`, do
        # NOT cache an `Instance` in any field.
        self._lock = threading.Lock()
        self._snapshot = _Snapshot({})
        self._retained: OrderedDict[str, Any] = OrderedDict()
        self._pins: dict[str, int] = {}
        self._aliases: dict[str, AliasSpec] = {}
        self._called: set[str] = set()

    @classmethod
    def from_paths(
        cls, paths: Iterable[str | os.PathLike[str]], **options: Any
    ) -> ExecutableRegistry:
        """A registry over *paths* (files or directories of ``*.json``), loaded once."""
        registry = cls(paths, **options)
        registry.reload()
        return registry

    # ---- adding -------------------------------------------------------------------------

    def add_document(
        self,
        data: object,
        *,
        variables: Mapping[str, str] | None = None,
        resolver: Callable[[str, str], bool] | None = None,
    ) -> Executable:
        """Load and add one envelope document (see the module docstring)."""
        return self.register(self._executable_of(data, variables, resolver))

    def register(self, executable: Executable) -> Executable:
        """Add a built executable under the collision rule; returns it."""
        key = (executable.id, executable.version)
        with self._lock:
            candidates = [
                (e.executable, e.source) for e in self._snapshot.entries.values()
            ]
            entries, refused = _assemble([*candidates, (executable, _MEMORY)])
            accepted = key in entries and entries[key].executable is executable
            if accepted:
                self._snapshot = _Snapshot(entries)
                self._retain_locked(executable)
        if not accepted:
            raise ExecutableCollisionError(
                f"cannot add executable: {refused[-1] if refused else 'refused'}"
            )
        return executable

    def _executable_of(
        self,
        data: object,
        variables: Mapping[str, str] | None = None,
        resolver: Callable[[str, str], bool] | None = None,
    ) -> Executable:
        if (
            resolver is None
            and isinstance(data, Mapping)
            and data.get("kind") == "pattern"
        ):
            # D-013: a pattern names handlers and generators, so its names resolve through the two
            # namespaces by default; any other kind keeps "names are not resolved"
            resolver = namespace_resolver()
        envelope, report = load_executable(data, variables=variables, resolver=resolver)
        if envelope is None or not report.is_valid:
            raise ExecutableError(
                "invalid executable document: " + "; ".join(report.errors[:10])
            )
        factory = adapter_for(envelope.kind, self._kinds)
        built: Executable = factory(envelope, pins=self)
        return built

    def load_sources(self) -> ReloadReport:
        """Call each not-yet-called source factory once and add what it returns."""
        self._sources.load_entry_points()
        failed: dict[str, list[str]] = {}
        refused: list[str] = []
        for name, factory in self._sources.items():
            with self._lock:
                if name in self._called:
                    continue
                self._called.add(name)
            try:
                produced = list(factory())
            except Exception as error:
                failed[name] = [f"{type(error).__name__}: {error}"]
                continue
            for item in produced:
                try:
                    if isinstance(item, Mapping):
                        self.add_document(item)
                    else:
                        self.register(item)
                except ExecutableCollisionError as error:
                    refused.append(f"source {name!r}: {error}")
                except ExecutableError as error:
                    failed.setdefault(name, []).append(str(error))
        for name, messages in failed.items():
            logger.warning(f"executable source {name!r}: {'; '.join(messages)}")
        return self._report(failed, refused)

    # ---- reading ------------------------------------------------------------------------

    def get(self, id: str, version: str | None = None) -> Executable:
        """The executable *id* (its greatest version, or the named one)."""
        snapshot = self._snapshot
        found = (
            snapshot.latest.get(id)
            if version is None
            else getattr(snapshot.entries.get((id, version)), "executable", None)
        )
        if found is None:
            raise ExecutableNotFoundError(
                f"no executable {id!r}"
                + ("" if version is None else f" version {version!r}")
            )
        with self._lock:
            self._retain_locked(found)
        return found  # type: ignore[no-any-return]

    def get_by_template(self, template_identity: str) -> Executable:
        """The executable whose template identity is *template_identity* (current or retained)."""
        found = self._snapshot.by_template.get(template_identity)
        with self._lock:
            if found is None:
                found = self._retained.get(template_identity)
            if found is not None:
                self._retain_locked(found)
        if found is None:
            raise ExecutableEvictedError(
                f"template identity {template_identity!r} is neither current nor "
                "retained (never loaded, or evicted and unpinned)"
            )
        return found  # type: ignore[no-any-return]

    def list(
        self,
        kind: str | None = None,
        capability: Capability | str | None = None,
        launchable: bool | None = None,
    ) -> builtins.list[Executable]:
        """The current executables, sorted by id and version, optionally filtered."""
        wanted = None if capability is None else Capability.from_string(capability)
        chosen = [
            e.executable
            for e in self._snapshot.entries.values()
            if (kind is None or e.executable.kind == kind)
            and (wanted is None or wanted in e.executable.capabilities)
            and (
                launchable is None
                or launchable == (not e.executable.requires.callables)
            )
        ]
        return sorted(
            chosen, key=lambda e: (e.id, _version_key(e.version) or (), e.version)
        )

    # ---- reload -------------------------------------------------------------------------

    def reload(self) -> ReloadReport:
        """Re-read the path sources; validate everything first, then swap one snapshot."""
        loaded, order, keep, failed = self._read_paths()  # no lock held
        with self._lock:
            current = builtins.list(self._snapshot.entries.values())
            candidates = [
                (e.executable, e.source) for e in current if e.source == _MEMORY
            ]
            for label in order:
                if label in loaded:
                    candidates.append((loaded[label], label))
                elif label in keep:
                    # a file label equals its entries' source; a directory that could not be
                    # listed is its resolved root, and its entries sit under it
                    under = label + os.sep
                    candidates += [
                        (e.executable, e.source)
                        for e in current
                        if e.source == label or e.source.startswith(under)
                    ]
            entries, refused = _assemble(candidates)
            self._snapshot = _Snapshot(entries)
            for entry in entries.values():
                self._retain_locked(entry.executable)
        for label, messages in failed.items():
            logger.warning(f"executable source {label}: {'; '.join(messages)}")
        for message in refused:
            logger.warning(f"executable refused: {message}")
        return self._report(failed, refused)

    def _read_paths(
        self,
    ) -> tuple[
        dict[str, Any], builtins.list[str], set[str], dict[str, builtins.list[str]]
    ]:
        """Read and validate every path source: ``(loaded, order, keep, failed)``."""
        loaded: dict[str, Any] = {}
        order: list[str] = []
        keep: set[str] = set()
        failed: dict[str, list[str]] = {}
        for path in self._paths:
            for label, target, problem in _members(path):
                order.append(label)
                if problem is None and target is not None:
                    problem = self._load_file(target, loaded, label)
                if problem is not None:
                    failed[label] = [problem]
                    keep.add(label)
        return loaded, order, keep, failed

    def _load_file(
        self, target: Path, loaded: dict[str, Any], label: str
    ) -> str | None:
        try:
            if target.stat().st_size > MAX_DOCUMENT_BYTES:
                return f"larger than {MAX_DOCUMENT_BYTES} bytes"
            with target.open("rb") as handle:
                text = handle.read(MAX_DOCUMENT_BYTES + 1).decode("utf-8-sig")
            data = json.loads(text)
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as error:
            return f"unreadable: {type(error).__name__}"
        if not isinstance(data, dict):
            return "a document is a JSON object"
        try:
            loaded[label] = self._executable_of(data)
        except ExecutableError as error:
            return str(error)[:600]
        return None

    def _report(
        self, failed: Mapping[str, Sequence[str]], refused: Sequence[str]
    ) -> ReloadReport:
        return ReloadReport(
            ids=tuple(sorted(self._snapshot.entries)),
            failed={k: tuple(v) for k, v in failed.items()},
            collisions=tuple(refused),
        )

    # ---- retention ----------------------------------------------------------------------

    def pin(self, executable: Executable) -> None:
        """Keep *executable*'s template retained until a matching ``unpin`` (counted)."""
        identity = executable.template_identity
        with self._lock:
            self._pins[identity] = self._pins.get(identity, 0) + 1
            self._retain_locked(executable)

    def unpin(self, executable: Executable) -> None:
        """Release one pin; an unpinned template is evictable again. A stray unpin is a no-op."""
        identity = executable.template_identity
        with self._lock:
            count = self._pins.get(identity, 0)
            if count > 1:
                self._pins[identity] = count - 1
            else:
                self._pins.pop(identity, None)
            self._evict_locked()

    def _retain_locked(self, executable: Any) -> None:
        identity = executable.template_identity
        self._retained.setdefault(identity, executable)
        self._retained.move_to_end(identity)
        self._evict_locked()

    def _evict_locked(self) -> None:
        # Drop the least recently used UNPINNED templates beyond the bound (pure dict work).
        if len(self._retained) <= self._bound:
            return
        for identity in [i for i in self._retained if i not in self._pins]:
            if len(self._retained) <= self._bound:
                break
            del self._retained[identity]

    def rebuild(
        self,
        template_identity: str,
        record: Mapping[str, Any],
        expected_instance_identity: str,
    ) -> Instance:
        """The instance of a logged run, rebuilt from its template and instantiation record."""
        executable = self.get_by_template(template_identity)
        rebuild = getattr(executable, "rebuild", None)
        if rebuild is None:
            raise ExecutableError(
                f"executable {executable.id!r} has no static instance to rebuild"
            )
        rebuilt: Instance = rebuild(record, expected_instance_identity)
        return rebuilt

    # ---- target resolution --------------------------------------------------------------

    def resolve_target(
        self, target_id: str, *, conversation_id: str | None = None
    ) -> ResolvedTarget | None:
        """The definition a ``call`` pushes for *target_id*, or ``None`` (see the module docstring).

        ``conversation_id`` is accepted for the ``TargetResolver`` Protocol; aliases and
        experiments are not routed here (carried).
        """
        try:
            executable = self.get(target_id)
        except ExecutableNotFoundError:
            return None
        if getattr(executable, "lifecycle", None) is Lifecycle.DEPRECATED:
            return None  # selection refuses a deprecated strategy: so does a call (D-044 (2))
        # DECISION plan-2026-11-24T043150-a555dcfe/D-010
        # A frame is a plain child conversation: the engine runs no handler of it. A pattern and
        # any executable whose `requires.handlers` is non-empty is never pushed as a frame (the
        # call ends `failed/unknown_target`); a pattern runs delegated. Do NOT relax this for
        # `requires.handlers` alone, do NOT move it into the loader (an operator document that
        # lists handlers still runs standalone), do NOT judge it from a record: the executable's
        # own claims decide.
        requires = getattr(executable, "requires", None)
        if getattr(executable, "kind", None) in _NEVER_A_FRAME_KINDS or getattr(
            requires, "handlers", None
        ):
            return None
        try:
            instance = executable.instantiate()
        except InstantiationError as error:
            logger.warning(
                f"call target {target_id!r} does not instantiate: {type(error).__name__}"
            )
            return None
        definition = getattr(instance, "definition", None)
        if not isinstance(
            definition, FSMDefinition
        ):  # a code-backed kind has no body to push
            return None
        envelope = getattr(executable, "envelope", None)
        decl = getattr(getattr(envelope, "body", None), "strategy", None)
        return ResolvedTarget(
            definition,
            executable.template_identity,
            instance.instance_identity,
            returns=None if decl is None else dict(decl.results),
        )

    def resolve_delegated(
        self,
        target_id: str,
        *,
        launcher: object | None,
        context: RunContext | None = None,
        input_keys: Iterable[str] = (),
        max_steps: int | None = None,
    ) -> DelegatedTarget | None:
        """The code-backed executable a delegated ``call`` runs, or ``None`` (see the docstring)."""
        if not is_run_launcher(launcher):
            return None  # no launcher injected: a code-backed kind is never a target
        try:
            executable = self.get(target_id)
        except ExecutableNotFoundError:
            return None
        kind = getattr(executable, "kind", None)
        if kind not in _DELEGATED_KINDS:
            return None
        if getattr(executable, "lifecycle", None) is Lifecycle.DEPRECATED:
            return None
        if Capability.CONVERSATIONAL in executable.capabilities:
            return None
        callables = None if context is None else context.callables
        if any(
            callables is None or name not in callables
            for name in executable.requires.callables
        ):
            return None
        allowed = {
            *executable.contract.inputs,
            *getattr(executable, "optional_inputs", ()),
        }
        if any(key not in allowed for key in input_keys):
            return None
        if max_steps is not None and "max_steps" in getattr(
            executable, "refused_budgets", ()
        ):
            return None
        return DelegatedTarget(
            target_id,
            kind,
            executable,
            executable.template_identity,
            None,
            tuple(executable.contract.outputs),
        )

    # ---- aliases ------------------------------------------------------------------------

    def set_alias(
        self, name: str, routes: Mapping[str, float], *, target: str | None = None
    ) -> None:
        """Define or replace an alias (see ``fsm_llm.engine.executables.aliases``)."""
        spec = make_alias(name, routes, target)
        with self._lock:
            self._aliases = {**self._aliases, name: spec}

    def resolve(
        self,
        name: str,
        *,
        experiment: str | None = None,
        conversation_id: str | None = None,
        pinned_template_identity: str | None = None,
    ) -> Resolution:
        """The executable an alias (or a plain id) serves for one conversation.

        A pinned identity is returned as is and never rolls: a resumed conversation keeps what
        it was served. No ``executable/resolve`` event is written (phase S).
        """
        if pinned_template_identity is not None:
            return Resolution(
                self.get_by_template(pinned_template_identity),
                pinned_template_identity,
                "pinned",
            )
        spec = self._aliases.get(name)
        if spec is None:
            plain = self.get(name)
            return Resolution(plain, plain.template_identity, plain.version)
        key = pick_route(spec, experiment, conversation_id)
        chosen = (
            self.get_by_template(key)
            if is_template_identity(key)
            else self.get(spec.target, key)
        )
        return Resolution(chosen, chosen.template_identity, key)


def _members(path: Path) -> list[tuple[str, Path | None, str | None]]:
    """The files one path source contributes: ``(label, target, problem)``, no lock involved.

    A file is one member; a directory contributes its ``*.json`` files in sorted order, each
    confined by ``resolve_within``; a symlink is reported and skipped; a path that does not exist
    contributes nothing (its entries drop out).
    """
    if path.is_symlink():
        return [(str(path), None, "a symbolic link is skipped")]
    if path.is_file():
        return [(str(path), path, None)]
    if not path.is_dir():
        return []
    root = path.resolve()
    members: list[tuple[str, Path | None, str | None]] = []
    try:
        names = sorted(n for n in os.listdir(root) if n.lower().endswith(".json"))
    except OSError as error:
        # reported under the resolved root: the label `reload` matches its entries against
        return [(str(root), None, f"unreadable directory: {type(error).__name__}")]
    for name in names:
        label = str(root / name)
        if (root / name).is_symlink():
            members.append((label, None, "a symbolic link is skipped"))
        elif (target := resolve_within(root, name)) is None:
            members.append((label, None, "outside the directory"))
        else:
            members.append((label, target, None))
    return members
