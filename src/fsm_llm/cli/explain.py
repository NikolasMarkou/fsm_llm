"""``fsm-llm explain <target>`` implementation.

Prints :meth:`fsm_llm.program.Program.explain` output:

* ``ast_shape`` — indented multi-line rendering of the term skeleton.
* ``leaf_schemas`` — map of synthesised leaf-id → schema class name.
* ``plans`` — currently empty for both FSM-mode and term-mode programs
  per :class:`fsm_llm.program.ExplainOutput` (R1 deferral; will be
  wired in step 17 of the active plan).

Target detection mirrors :mod:`fsm_llm.cli.run`:

* path or ``*.json`` → FSM JSON. Built via
  :meth:`fsm_llm.program.Program.from_fsm`.
* ``pkg.mod:attr`` → factory. Built via
  :meth:`fsm_llm.program.Program.from_factory` with kwargs from
  ``--factory-arg``.

JSON output mode (``--json``) emits a single JSON object with keys
``ast_shape``, ``leaf_schemas`` (schema class names as strings), and
``plans`` (list of dicts) — for machine consumers (test harnesses,
docs generators).
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from typing import Any

from .run import _is_factory_string, _parse_kv_list, _resolve_factory


def _build_program(args: argparse.Namespace):
    from ..dialog.api import API
    from ..program import Program

    target: str = args.target
    if _is_factory_string(target):
        factory = _resolve_factory(target)
        factory_kwargs = _parse_kv_list(args.factory_arg)
        return Program.from_factory(factory, factory_kwargs=factory_kwargs)
    # FSM JSON path. Use API.from_file → from_fsm to honour the same
    # construction path the canonical FSM CLI uses.
    api = API.from_file(target)
    # Program.from_fsm takes an FSMDefinition. Reach into the API's
    # cached definition rather than re-parsing the file.
    fsm_def = api.fsm_definition
    api.close()  # release any session resources from the temporary API
    return Program.from_fsm(fsm_def)


def _render_human(target: str, output: Any) -> str:
    parts: list[str] = []
    parts.append(f"# Program.explain() for {target!r}")
    parts.append("")
    parts.append("## AST Shape")
    parts.append(output.ast_shape or "(empty)")
    parts.append("")
    parts.append("## Leaf Schemas")
    if not output.leaf_schemas:
        parts.append("(no Leaves)")
    else:
        for leaf_id, schema in output.leaf_schemas.items():
            schema_name = (
                getattr(schema, "__name__", repr(schema)) if schema is not None else "—"
            )
            parts.append(f"- {leaf_id} → {schema_name}")
    parts.append("")
    parts.append("## Plans")
    if not output.plans:
        parts.append("(empty — Plan list not yet wired; see plan v1 step 17)")
    else:
        for i, plan in enumerate(output.plans):
            parts.append(f"- plan[{i}]: {plan!r}")
    return "\n".join(parts)


def _render_json(output: Any) -> str:
    schemas_serialisable = {
        leaf_id: (
            getattr(schema, "__name__", str(schema)) if schema is not None else None
        )
        for leaf_id, schema in (output.leaf_schemas or {}).items()
    }
    plans_serialisable: list[dict[str, Any]] = []
    for plan in output.plans or []:
        # Plan is a frozen dataclass / pydantic model; try the common shapes.
        if hasattr(plan, "model_dump"):
            plans_serialisable.append(plan.model_dump())
        elif hasattr(plan, "__dict__"):
            plans_serialisable.append(
                {k: v for k, v in vars(plan).items() if not k.startswith("_")}
            )
        else:
            plans_serialisable.append({"repr": repr(plan)})
    return _json.dumps(
        {
            "ast_shape": output.ast_shape,
            "leaf_schemas": schemas_serialisable,
            "plans": plans_serialisable,
        },
        indent=2,
        default=str,
    )


def explain(args: argparse.Namespace) -> int:
    try:
        program = _build_program(args)
    except SystemExit:
        raise
    except Exception as e:
        print(f"fsm-llm explain: failed to build program: {e}", file=sys.stderr)
        return 1
    plan_kwargs: dict[str, Any] = {}
    tau = getattr(args, "tau", None)
    if tau is not None:
        plan_kwargs["tau"] = tau
    output = program.explain(
        n=getattr(args, "n", None),
        K=getattr(args, "K", None),
        plan_kwargs=plan_kwargs or None,
    )
    if getattr(args, "json", False):
        print(_render_json(output))
    else:
        print(_render_human(args.target, output))
    return 0
