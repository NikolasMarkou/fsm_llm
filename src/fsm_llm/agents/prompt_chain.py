"""
PromptChainAgent -- Fixed sequential pipeline with validation gates.

Chains a user-defined list of LLM steps, each with optional validation
gates that can short-circuit the pipeline on failure.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fsm_llm.agents.constants import ContextKeys, ErrorMessages
from fsm_llm.agents.fsm_definitions.prompt_chain import chain_options
from fsm_llm.core.definitions import AgentError

from .base import artifact_text
from .definitions import AgentConfig, ChainStep
from .pattern_agent import PatternAgent


class PromptChainAgent(PatternAgent):
    """
    Sequential prompt chaining agent.

    Executes a fixed pipeline of LLM steps in order, with optional
    validation gates between steps that can terminate early on failure.
    The FSM is written by the generator ``agents.prompt_chain`` from the chain (the file
    ``patterns/prompt_chain.json``, plan a555dcfe step 11); the gates are the library handler
    ``agents.chain_gate_checker``.

    Usage::

        from fsm_llm.agents import PromptChainAgent, ChainStep

        chain = [
            ChainStep(
                step_id="outline",
                name="Generate outline",
                extraction_instructions="Extract a structured outline as JSON.",
                response_instructions="Present the outline clearly.",
            ),
            ChainStep(
                step_id="draft",
                name="Write draft",
                extraction_instructions="Extract the full draft text.",
                response_instructions="Write a complete draft from the outline.",
                validation_fn=lambda ctx: len(ctx.get("chain_step_result", "")) > 50,
            ),
        ]

        agent = PromptChainAgent(chain=chain)
        result = agent.run("Write an essay about climate change.")
        print(result.answer)
    """

    # No `tools` argument: `create_agent` raises for tools given to a chain (`accepts_tools`).
    takes_tools: ClassVar[bool] = False

    # Run outputs caller context may not seed (D-052 of plan 06a5ec0a).
    _run_output_keys = frozenset(
        {
            ContextKeys.CHAIN_STEP_INDEX,
            ContextKeys.CHAIN_STEP_RESULT,
            ContextKeys.CHAIN_RESULTS,
            ContextKeys.GATE_PASSED,
        }
    )

    def __init__(
        self,
        chain: list[ChainStep],
        config: AgentConfig | None = None,
        **api_kwargs: Any,
    ) -> None:
        """
        Initialize a prompt chain agent.

        :param chain: Ordered list of ChainStep definitions (read once, here)
        :param config: Agent configuration (defaults to AgentConfig())
        :param api_kwargs: Additional kwargs passed to fsm_llm.API
        """
        if not chain:
            raise AgentError(ErrorMessages.EMPTY_CHAIN)
        from .pattern_files import load_envelope  # lazy: reads the file

        chain = list(chain)
        super().__init__(
            load_envelope("prompt_chain"),
            config=config,
            options=chain_options(chain),
            callables={"validation_fns": [step.validation_fn for step in chain]},
            **api_kwargs,
        )
        self.chain = chain

    def _budget_kwargs(self) -> dict[str, Any]:
        """Hard ceiling on iterations based on chain length (the ``max_iterations`` option)."""
        return {"max_iterations": self._options["max_iterations"]}

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """The last chain step result, else the last non-empty reply."""
        # DECISION plan-2026-11-05T134105-5ed71cf2/D-010: any non-empty last
        # result or reply is the answer. Do NOT re-add a minimum length: it
        # dropped "Paris"/"42" and shipped the failure sentence as success.
        chain_results = final_context.get(ContextKeys.CHAIN_RESULTS, [])
        last = artifact_text(chain_results[-1]).strip() if chain_results else ""
        return last or super()._extract_answer(final_context, responses)

    def _answer_source_keys(
        self, final_context: dict[str, Any], extra_answer_keys: list[str] | None
    ) -> list[str] | None:
        """Where ``_extract_answer`` reads the answer (see the base)."""
        chain_results = final_context.get(ContextKeys.CHAIN_RESULTS, [])
        if chain_results and artifact_text(chain_results[-1]).strip():
            return [ContextKeys.CHAIN_STEP_RESULT]
        return None
