"""An agent run logs no internal-prefix WARNING (plan 0068b055, step 12.29).

Agents seed their own driver keys (``ContextKeys.ITERATION_BUDGET`` =
``_max_iterations`` and others) through ``initial_context``. Core used to warn
``Context update contains internal-prefix key`` for each, on every run
(findings/live-runs-3.md Defect 4). The core pins are in
``tests/test_fsm_llm/test_internal_key_seeding_log.py``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from fsm_llm.agents import ReactAgent
from fsm_llm.agents.constants import ContextKeys
from tests.conftest import PromptGroundedLLM
from tests.test_fsm_llm_agents.test_advance_driver import (
    _ANSWER,
    _FACTS,
    _TASK,
    _registry,
)


def _run_logged(agent: Any) -> tuple[Any, list[str]]:
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m.record["message"]), level="WARNING")
    logger.enable("fsm_llm")
    try:
        result = agent.run(_TASK)
    finally:
        logger.disable("fsm_llm")
        logger.remove(sink)
    return result, records


def test_a_react_run_logs_no_internal_prefix_warning():
    runs: list[str] = []
    llm = PromptGroundedLLM(facts=_FACTS, default_response=_ANSWER)
    agent = ReactAgent(tools=_registry(runs), llm_interface=llm)

    result, records = _run_logged(agent)

    assert runs == ["capital of France"]  # the run really used its seeded budget
    assert result.success is True
    assert ContextKeys.ITERATION_BUDGET == "_max_iterations"
    assert [r for r in records if "internal-prefix" in r] == []
