#!/usr/bin/env python
"""Print the public API of `fsm_llm`, one deterministic text, for `docs/api.txt`.

Run `.venv/bin/python scripts/api_snapshot.py` (stdout) or
`.venv/bin/python scripts/api_snapshot.py --write docs/api.txt` (`make
api-snapshot`). `tests/test_fsm_llm_regression/test_api_snapshot.py` compares
the stdout with the committed file, so a removed or renamed name in the
listed `__all__` lists, or a changed signature, model field, default or enum
member, fails the build until the file is regenerated and reviewed in the same
commit. Not covered: module functions outside `__all__`, `model_config`, the
values of plain constant classes, the contents of factory defaults and
property types.

Public means the names in the static `__all__` of the modules in `MODULES`,
plus the names listed in `EXPLICIT` for a module that has no `__all__`
(`flow_analysis`, public by the AGENTS.md key-files row). `__version__` is left
out: the release version is pinned elsewhere and would force a regeneration
every release. The text is stable across Python 3.10 to 3.12, the environment,
the working directory and the hash seed: types are rendered by one formatter,
set defaults are sorted, classes are named by `__qualname__` only (so a later
module split does not change the file), and nothing prints an address.

Interface: `snapshot() -> str` (the whole text, ends with one newline) and
`main(argv) -> int` (0, or 2 on a usage error). An unimportable module raises.
Run it through the installed `fsm_llm`, never with `src/` on `sys.path`.
"""

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import re
import sys
import typing

MODULES = (
    "fsm_llm",
    "fsm_llm.agent_skills",
    "fsm_llm.codes",
    "fsm_llm.flow_analysis",
    "fsm_llm.frontmatter",
    "fsm_llm.okf",
    "fsm_llm.okf_bridge",
    "fsm_llm.schema_registry",
    "fsm_llm.agents",
    "fsm_llm.eval",
    "fsm_llm.monitor",
    "fsm_llm.reasoning",
    "fsm_llm.synthesis",
    "fsm_llm.workflows",
)

# Modules with no `__all__`: the public names are listed here.
EXPLICIT = {
    "fsm_llm.flow_analysis": (
        "FlowFinding",
        "analyze_data_flow",
        "analyze_liveness",
        "declared_available",
        "strongly_connected_components",
    ),
}

_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")
_TYPING = re.compile(r"\btyping\.")


def _t(a) -> str:
    """Render a runtime type object the same way on 3.10, 3.11 and 3.12."""
    import collections.abc as cabc
    import types as _types

    if a is None or a is type(None):
        return "None"
    if a is Ellipsis:
        return "..."
    if isinstance(a, str):
        return a
    if isinstance(a, typing.ForwardRef):
        return a.__forward_arg__
    origin, args = typing.get_origin(a), typing.get_args(a)
    if origin is typing.Union or origin is getattr(_types, "UnionType", None):
        return " | ".join(_t(x) for x in args)
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(x) for x in args) + "]"
    if origin is not None:
        name = getattr(origin, "__qualname__", repr(origin))
        if origin is cabc.Callable and args and isinstance(args[0], list):
            return f"Callable[[{', '.join(_t(x) for x in args[0])}], {_t(args[1])}]"
        return f"{name}[{', '.join(_t(x) for x in args)}]" if args else name
    if isinstance(a, type):
        return a.__qualname__
    return _ADDR.sub("", _TYPING.sub("", repr(a)))


def ann(a) -> str:
    if a is inspect.Parameter.empty:
        return ""
    return _t(a)


def default(v) -> str:
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(sorted(repr(x) for x in v)) + "}"
    if isinstance(v, enum.Enum):
        return f"{type(v).__name__}.{v.name}"
    if callable(v) and hasattr(v, "__qualname__"):
        return v.__qualname__
    return _ADDR.sub("", repr(v))


def sig(obj) -> str:
    try:
        s = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    parts, star, prev = [], False, None
    for p in s.parameters.values():
        if (
            prev is not None
            and prev.kind is p.POSITIONAL_ONLY
            and p.kind is not p.POSITIONAL_ONLY
        ):
            parts.append("/")
        if (
            p.kind is p.KEYWORD_ONLY
            and not star
            and not any(q.kind is q.VAR_POSITIONAL for q in s.parameters.values())
        ):
            parts.append("*")
            star = True
        prev = p
        t = p.name
        if p.kind is p.VAR_POSITIONAL:
            t = "*" + t
        elif p.kind is p.VAR_KEYWORD:
            t = "**" + t
        if p.annotation is not p.empty:
            t += f": {ann(p.annotation)}"
        if p.default is not p.empty:
            t += (" = " if p.annotation is not p.empty else "=") + default(p.default)
        parts.append(t)
    r = f" -> {ann(s.return_annotation)}" if s.return_annotation is not s.empty else ""
    return "(" + ", ".join(parts) + ")" + r


def is_pydantic(c) -> bool:
    return any(
        b.__module__.startswith("pydantic") and b.__name__ == "BaseModel"
        for b in c.__mro__
    )


def cls_lines(prefix: str, name: str, c: type) -> list[str]:
    bases = [b.__qualname__ for b in c.__bases__ if b is not object]
    out = [
        f"{prefix}.{name} class({', '.join(bases)})"
        if bases
        else f"{prefix}.{name} class"
    ]
    if issubclass(c, enum.Enum):
        for m in c:
            out.append(f"  {name}.{m.name} = {m.value!r}")
        return out
    if is_pydantic(c):
        for fname, f in c.model_fields.items():  # declaration order
            req = (
                "required"
                if f.is_required()
                else f"= {default(f.get_default(call_default_factory=False)) if f.default_factory is None else 'factory'}"
            )
            out.append(f"  field {fname}: {ann(f.annotation)} {req}")
    for mname, raw in sorted(vars(c).items()):
        if mname.startswith("_") and mname != "__init__":
            continue
        if mname == "__init__" and is_pydantic(c):
            continue
        fn = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        if isinstance(raw, property):
            out.append(f"  property {mname}")
        elif inspect.isfunction(fn):
            kind = (
                "static "
                if isinstance(raw, staticmethod)
                else "classmethod "
                if isinstance(raw, classmethod)
                else ""
            )
            out.append(f"  {kind}def {mname}{sig(fn)}")
    return out


def module_lines(modname: str) -> list[str]:
    m = importlib.import_module(modname)
    out = [f"# {modname}"]
    for name in sorted(EXPLICIT.get(modname) or m.__all__):
        if name == "__version__":
            continue
        obj = getattr(m, name)
        if typing.get_origin(obj) is not None:
            out.append(f"{modname}.{name} alias {_t(obj)}")
        elif inspect.isclass(obj):
            out += cls_lines(modname, name, obj)
        elif inspect.isroutine(obj):
            out.append(f"{modname}.{name} def{sig(obj)}")
        elif inspect.ismodule(obj):
            out.append(f"{modname}.{name} module")
        elif isinstance(obj, (str, int, float, bool, tuple, frozenset, type(None))):
            out.append(f"{modname}.{name} = {default(obj)}")
        else:
            out.append(f"{modname}.{name} {type(obj).__qualname__}")
    return out


def snapshot() -> str:
    lines: list[str] = []
    for mod in MODULES:
        lines += module_lines(mod)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write", metavar="PATH", help="write the text to PATH instead of stdout"
    )
    args = parser.parse_args(argv)
    text = snapshot()
    if args.write:
        with open(args.write, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
