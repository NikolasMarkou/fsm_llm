"""The lifetime turn budget on the ``API`` side: the defaults of an ``API`` and what each new
conversation or frame is given (plan 5a6a1f80, step 7, roadmap row 1.7).

Layer ``engine``. The record, the charge and the validator live in ``engine/fsm/budget.py`` (a
layer below: ``engine/fsm`` imports no ``API``); this module only decides WHICH limits a
conversation starts with. These are MODULE-LEVEL functions taking the API, not methods called
through ``self``, for the reason the module docstring of ``events.py`` gives (a ``self._helper()``
call resolves to a ``Mock`` under the ``Mock(spec=API)`` unbound-self tests, so the default is read
with ``getattr`` and a double without it counts as "no default").

Interface contract:
    - ``budget_defaults(max_steps, max_seconds) -> (max_steps, max_seconds)``: validates the
      ``API`` constructor's limits (``ValueError``, as ``FSMManager.start_conversation``) and
      returns them; ``API.__init__`` keeps the pair as ``self._turn_budget``.
    - ``budget_kwargs(api, max_steps=None, max_seconds=None) -> dict``: the keyword arguments for
      ``FSMManager.start_conversation`` of a new conversation or pushed frame: an explicit
      limit wins, otherwise the ``API`` default; a limit that is ``None`` is left OUT, so with no
      default and no argument the manager is called exactly as before. Validates the explicit
      values (``ValueError``) so a refusal happens before anything is registered. Fork, resume and
      a snapshot restore call it with no argument: they start a FRESH budget from the API defaults
      (a spent count is never carried, D-011 of the plan).
"""

from __future__ import annotations

from typing import Any

from fsm_llm.engine.fsm.budget import check_turn_limits

from .surface import _ApiSurface


def budget_defaults(
    max_steps: int | None, max_seconds: float | None
) -> tuple[int | None, float | None]:
    """Validate the ``API`` constructor's lifetime limits and return them (see the module docstring)."""
    check_turn_limits(max_steps, max_seconds)
    return max_steps, max_seconds


def budget_kwargs(
    api: _ApiSurface,
    max_steps: int | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Return the budget keywords for one new conversation or frame (see the module docstring)."""
    check_turn_limits(max_steps, max_seconds)
    defaults = getattr(api, "_turn_budget", None)
    default_steps, default_seconds = (
        defaults if isinstance(defaults, tuple) else (None, None)
    )
    chosen = {
        "max_steps": default_steps if max_steps is None else max_steps,
        "max_seconds": default_seconds if max_seconds is None else max_seconds,
    }
    return {name: value for name, value in chosen.items() if value is not None}
