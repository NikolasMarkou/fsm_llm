"""Unified CLI surface for FSM-LLM (R7).

Exposes a single ``fsm-llm`` binary with subcommands ``run / explain /
validate / visualize / meta / monitor``. Old console scripts
(``fsm-llm-validate``, ``fsm-llm-visualize``, ``fsm-llm-monitor``,
``fsm-llm-meta``) survive as thin aliases that delegate into
:func:`fsm_llm.cli.main.main_cli` with their subcommand prepended to
``argv`` — see ``# DECISION D-PLAN-04`` in
``plans/plan_2026-04-27_43d56276/decisions.md``.

This package was created in plan v1 R7 step 16 after R6 was deferred to
its own fresh plan (see ``D-STEP-08-RESOLUTION``). The dispatcher does
not depend on any R6 artefact.
"""

from __future__ import annotations

# NOTE: ``main_cli`` is intentionally NOT re-exported eagerly here —
# importing it would fire ``from .main import …`` at package-load time
# and trigger a ``runpy`` warning when running ``python -m
# fsm_llm.cli.main``. Consumers should import directly from
# ``fsm_llm.cli.main``.

__all__: list[str] = []
