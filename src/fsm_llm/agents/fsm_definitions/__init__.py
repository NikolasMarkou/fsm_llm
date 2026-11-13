"""Pre-built FSM definitions for agent patterns: one module per pattern.

Every public name of the old single module is re-exported here (P1), so
``from fsm_llm.agents.fsm_definitions import build_react_fsm`` keeps working
(``examples/agents/workflow_agent/run.py`` imports ``build_self_consistency_fsm``
this way). The shared helpers live in ``_common`` (state, edge and condition
constructors) and ``_loop`` (the think, act and conclude parts of the loop
patterns); import a private helper from its defining module, never from here.
"""

from ._loop import conclude_on_evidence_logic, loop_field_context_keys
from .adapt import build_adapt_fsm
from .debate import build_debate_fsm
from .evalopt import build_evalopt_fsm
from .maker_checker import build_maker_checker_fsm
from .meta_builder import build_meta_builder_fsm
from .native_fc import build_native_fc_fsm
from .orchestrator import build_orchestrator_fsm
from .parallel_react import build_parallel_react_fsm
from .plan_execute import build_plan_execute_fsm
from .prompt_chain import build_prompt_chain_fsm
from .react import build_react_fsm
from .reflexion import build_reflexion_fsm
from .rewoo import build_rewoo_fsm
from .self_consistency import build_self_consistency_fsm

__all__ = [
    "build_adapt_fsm",
    "build_debate_fsm",
    "build_evalopt_fsm",
    "build_maker_checker_fsm",
    "build_meta_builder_fsm",
    "build_native_fc_fsm",
    "build_orchestrator_fsm",
    "build_parallel_react_fsm",
    "build_plan_execute_fsm",
    "build_prompt_chain_fsm",
    "build_react_fsm",
    "build_reflexion_fsm",
    "build_rewoo_fsm",
    "build_self_consistency_fsm",
    "conclude_on_evidence_logic",
    "loop_field_context_keys",
]
