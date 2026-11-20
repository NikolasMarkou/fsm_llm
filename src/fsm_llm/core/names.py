"""One registry primitive for names (roadmap 0.17, contract C10): ``NameRegistry``.

A ``NameRegistry[T]`` maps names to objects inside one namespace (``state_kinds``,
later ``executable_kinds``, ``executable_sources``, ``callables`` and the rest of the
C10 list). Definitions refer to objects by name only; the registry is where a name
becomes code, which keeps "no code in a definition" true.

Interface contract:

- ``NameRegistry(namespace, *, entry_point_group=None)``. A registry is a plain
  instance: a global one is a named module constant, a per-run one is a local. There is
  no process-wide registry of registries.
- ``register(name, obj, *, override=False)``: a taken name is refused with
  ``NameCollisionError`` unless ``override=True``, which replaces it. One OBJECT under a
  second name is refused whatever ``override`` says, so ``name_of`` is unambiguous
  (identity, ``is``). After ``freeze()`` it raises ``RegistryFrozenError``. A name must
  be a non-blank ``str`` (``TypeError`` and ``ValueError`` otherwise).
- ``get(name, default=None)``, ``resolve(name)`` (``NameNotFoundError`` on a miss),
  ``name in registry``, ``list()`` (sorted names; a pure read), ``items()`` (``(name,
  object)`` pairs in registration order, a snapshot; a replaced name keeps its first
  position), ``name_of(obj)`` (the name or ``None``), ``frozen``, ``namespace``.
- Entry points: when ``entry_point_group`` names a group, the group is read ONCE, on the
  first miss of ``get``, ``resolve`` or ``in``, or by ``load_entry_points()`` or
  ``freeze()`` (a frozen registry is complete). Each entry point ``name = module:attr``
  registers the loaded object under its name, never overriding: an existing name wins and
  the entry is warned and skipped; an entry whose load fails is warned once and skipped. A
  later miss never reloads. ``list()`` shows only what is registered or already loaded.
  ``entry_point_group=None`` (the default) loads nothing.
- Locking: one leaf lock guards the two dicts; no loader, logger or other callback runs
  under it. A separate re-entrant lock serialises the one entry-point load, so a thread
  that misses while another loads waits for it instead of reporting a false miss, and a
  loaded module may read the same registry without deadlock.

Imported by name (``from fsm_llm.core.names import NameRegistry``);
``fsm_llm/__init__.py`` does not import it. ``importlib.metadata`` is imported inside
``load_entry_points``, so importing this module costs nothing.
"""

from __future__ import annotations

import builtins
import threading
from typing import Any, Generic, TypeVar

from fsm_llm.core.definitions.errors import FSMError
from fsm_llm.logging import logger

__all__ = [
    "NameCollisionError",
    "NameNotFoundError",
    "NameRegistry",
    "NameRegistryError",
    "RegistryFrozenError",
]

T = TypeVar("T")


class NameRegistryError(FSMError):
    """Root of the name registry errors: ``namespace`` and ``name`` say where."""

    def __init__(
        self,
        message: str,
        *,
        namespace: str | None = None,
        name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.namespace = namespace
        self.name = name


class NameCollisionError(NameRegistryError):
    """A name is taken, or the object is already registered under another name."""


class NameNotFoundError(NameRegistryError):
    """``resolve`` found no object for the name."""


class RegistryFrozenError(NameRegistryError):
    """``register`` was called on a frozen registry."""


class NameRegistry(Generic[T]):
    """A namespaced mapping of names to objects. See the module docstring."""

    def __init__(self, namespace: str, *, entry_point_group: str | None = None) -> None:
        self._namespace = namespace
        self._group = entry_point_group
        self._lock = threading.Lock()  # leaf: guards _objects, _names, _frozen
        self._load_lock = threading.RLock()  # serialises the one entry-point load
        self._objects: dict[str, T] = {}
        self._names: dict[int, str] = {}  # id(object) -> name, objects are held above
        self._frozen = False
        self._loaded = entry_point_group is None

    @property
    def namespace(self) -> str:
        """The namespace label, for messages."""
        return self._namespace

    @property
    def frozen(self) -> bool:
        """True after ``freeze()``."""
        return self._frozen

    def register(self, name: str, obj: T, *, override: bool = False) -> None:
        """Register ``obj`` under ``name``; see the module docstring for the refusals."""
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("name must not be blank")
        with self._lock:
            if self._frozen:
                raise RegistryFrozenError(
                    f"namespace {self._namespace!r} is frozen: cannot register {name!r}",
                    namespace=self._namespace,
                    name=name,
                )
            if name in self._objects and not override:
                raise NameCollisionError(
                    f"{name!r} is already registered in namespace {self._namespace!r}",
                    namespace=self._namespace,
                    name=name,
                )
            holder = self._names.get(id(obj))
            if holder is not None and holder != name:
                raise NameCollisionError(
                    f"that object is already registered as {holder!r} in namespace "
                    f"{self._namespace!r}: cannot also register it as {name!r}",
                    namespace=self._namespace,
                    name=name,
                )
            old = self._objects.get(name)
            if old is not None:
                del self._names[id(old)]
            self._objects[name] = obj
            self._names[id(obj)] = name

    def get(self, name: str, default: T | None = None) -> T | None:
        """The object for ``name``, or ``default``. A miss loads the entry points once."""
        found, obj = self._lookup(name)
        return obj if found else default

    def resolve(self, name: str) -> T:
        """The object for ``name``; ``NameNotFoundError`` when there is none."""
        found, obj = self._lookup(name)
        if not found:
            raise NameNotFoundError(
                f"{name!r} is not registered in namespace {self._namespace!r}",
                namespace=self._namespace,
                name=name,
            )
        return obj  # type: ignore[return-value]

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self._lookup(name)[0]

    def list(self) -> builtins.list[str]:
        """The registered names, sorted. Loads nothing."""
        with self._lock:
            return sorted(self._objects)

    def items(self) -> builtins.list[tuple[str, T]]:
        """``(name, object)`` pairs in registration order, a snapshot. Loads nothing.

        A replaced name keeps the position of its first registration. The returned list is
        the caller's: the lock is released before it returns.
        """
        with self._lock:
            return list(self._objects.items())

    def name_of(self, obj: object) -> str | None:
        """The name ``obj`` (this very object, by identity) is registered under, or ``None``."""
        with self._lock:
            return self._names.get(id(obj))

    def freeze(self) -> None:
        """Seal the registry: pending entry points load first, then ``register`` raises."""
        self.load_entry_points()
        with self._lock:
            self._frozen = True

    def load_entry_points(self) -> None:
        """Read the entry-point group now (once; later calls and misses do nothing)."""
        with self._load_lock:
            if self._loaded:
                return
            self._loaded = (
                True  # set first: a loader that reads this registry reloads nothing
            )
            self._load()

    def _lookup(self, name: str) -> tuple[bool, T | None]:
        with self._lock:
            if name in self._objects:
                return True, self._objects[name]
        self.load_entry_points()
        with self._lock:
            if name in self._objects:
                return True, self._objects[name]
        return False, None

    def _load(self) -> None:
        # No lock of the data is held here: a loader may import code and log.
        from importlib.metadata import entry_points

        group = self._group
        if group is None:
            return
        try:
            entries = sorted(entry_points(group=group), key=lambda e: e.name)
        except Exception as error:
            logger.warning(
                f"namespace {self._namespace!r}: cannot read entry points of {group!r}: "
                f"{type(error).__name__}: {error}"
            )
            return
        for entry in entries:
            try:
                obj = entry.load()
            except Exception as error:
                logger.warning(
                    f"namespace {self._namespace!r}: entry point {entry.name!r} of "
                    f"{group!r} failed to load and is skipped: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            try:
                self.register(entry.name, obj)
            except NameRegistryError as error:
                logger.warning(
                    f"namespace {self._namespace!r}: entry point {entry.name!r} of "
                    f"{group!r} is skipped: {error}"
                )
