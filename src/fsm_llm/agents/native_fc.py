"""
NativeFunctionCallingReactAgent: a ReAct loop on provider-native function
calling, run by core as an FSM.

The stock agents select tools by typed field extraction (the model writes
``tool_name``/``tool_input`` into context). This agent instead declares the
registry's tools to the provider (``tools=registry.get_json_schemas()``) and
reads the structured ``tool_calls`` the model returns, through core's
completion state (``State.completion``, ``LLMInterface.complete``).

``build_native_fc_fsm`` (``fsm_definitions/native_fc.py``) is the FSM: ``call_model``
(one model turn with the tools) -> ``run_tools`` (``NativeFCHandlers`` run
each call through ``ToolRegistry.execute``) -> ``call_model``, then at most
one forced final-tool turn, at most one ``output_schema`` repair turn, and
``conclude``. ``BaseAgent._standard_run`` drives it on core's bounded run;
every model request goes through the run's ``LLMInterface`` (inject one with
``llm_interface=``, the test and metering seam).

Example::

    from fsm_llm.agents import AgentConfig, ToolRegistry
    from fsm_llm.agents.native_fc import NativeFunctionCallingReactAgent

    agent = NativeFunctionCallingReactAgent(
        tools=registry, config=AgentConfig(model="gpt-4o-mini"),
    )
    result = agent.run("What is the weather in Paris?")

``system_policy`` appends caller-owned STANDING instructions (rules, gates,
output shape: anything true for every turn of the dispatch) to the system
message, leaving the user turn to carry only that turn's task. It is read at
``run()`` time, so a caller that obtains the agent from a factory may set the
attribute afterwards. Default ``None`` reproduces the base system message
byte for byte.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fsm_llm.agents.constants import (
    NATIVE_FC_HANDLER_ONLY_KEYS,
    Defaults,
    NativeFCContextKeys,
    NativeFCStates,
)
from fsm_llm.tools import ToolRegistry

from .definitions import AgentConfig
from .fsm_definitions.native_fc import build_native_fc_fsm
from .handlers.native_fc import _FORCE_WRITE_PROMPT as _FORCE_WRITE_PROMPT
from .handlers.native_fc import _REPAIR_PROMPT as _REPAIR_PROMPT
from .handlers.native_fc import NativeFCHandlers as NativeFCHandlers
from .pattern_agent import PatternAgent

_K = NativeFCContextKeys
_SYSTEM_PROMPT = (
    "You are a capable AI agent. Use the provided tools to gather information "
    "and complete the task. Call tools when you need external data; when you "
    "have enough information, reply with the final answer and no further tool "
    "calls."
)


class NativeFunctionCallingReactAgent(PatternAgent):
    """ReAct loop driven by provider-native function calling, run by core.

    Args:
        tools: Tool registry (non-empty, no ``requires_approval`` tool: this
            pattern has no approval gate).
        config: AgentConfig (model, max_iterations = model loop turns,
            timeout, temperature, output_schema, force_final_tool, ...).
        system_policy: Standing instructions appended to the system message.
            ``None`` (the default) falls back to ``config.instructions``; when
            both are unset the system message is the base prompt.
        seed: Optional sampling seed sent with every model request (an
            ``api_kwargs`` entry of the run's ``LiteLLMInterface``). ``None``
            (the default) sends no ``seed`` key at all.
        **api_kwargs: Passed to ``API.from_definition`` like every pattern's
            (``llm_interface=`` injects the interface every model turn uses).
    """

    tools: ToolRegistry  # always set (``PatternAgent.tools`` may be None)

    # Keys the run writes for itself; a caller's ``initial_context`` never
    # seeds them (``strip_caller_context``; the seed handler resets them too).
    _run_output_keys: ClassVar[frozenset[str]] = frozenset(NATIVE_FC_HANDLER_ONLY_KEYS)
    # DECISION plan-2026-10-01T093600-944e2692/D-032: past the model loop the
    # run finishes even when the clock is spent, so a final answer in hand is
    # never lost to the forced or repair turn (e1f63a9 checked the clock only
    # before loop turns). Do NOT add a clock check, a deadline flag or a loop
    # in this agent: core's `seconds_exempt_states` carries it. See D-032.
    _seconds_exempt_states: ClassVar[frozenset[str]] = NativeFCStates.SECONDS_EXEMPT

    def __init__(
        self,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        system_policy: str | None = None,
        *,
        seed: int | None = None,
        **api_kwargs: Any,
    ) -> None:
        from .base import reject_misplaced_kwargs
        from .pattern_files import load_envelope  # lazy: reads the file

        # the base denylist (`hitl=` is a TypeError here): `PatternAgent` names `hitl` itself
        reject_misplaced_kwargs(type(self).__name__, api_kwargs)
        super().__init__(
            load_envelope("native_fc"),
            tools=tools,
            config=config,
            callables={"parse_structured": self._try_parse_structured_output},
            **api_kwargs,
        )
        self.seed = seed
        self.system_policy = (
            system_policy if system_policy is not None else self.config.instructions
        )

    @property
    def system_policy(self) -> str | None:
        """Standing instructions appended to the system message, or ``None``.

        Public and settable after construction on purpose (see
        :meth:`_system_message`); setting anything but a ``str`` or ``None``
        raises ``TypeError``.
        """
        return self._system_policy

    @system_policy.setter
    def system_policy(self, value: str | None) -> None:
        # DECISION plan-2026-10-01T093600-944e2692/D-033: the third
        # positional parameter was `complete_fn` before D-028; a caller of
        # the old API bound a function here and its repr was sent as the
        # system policy. Do NOT accept a non-str value silently.
        if value is not None and not isinstance(value, str):
            raise TypeError(
                "system_policy must be a str or None, got "
                f"{type(value).__name__} (complete_fn was removed: inject "
                "llm_interface= instead)"
            )
        self._system_policy = value

    @property
    def seed(self) -> int | None:
        """The sampling seed of every model request, or ``None`` (no key).

        Stored as the ``seed`` entry of the agent's ``api_kwargs``, which
        ``_create_api`` reads at each ``run()``: setting it after construction
        takes effect on the next run.
        """
        seed = self._api_kwargs.get("seed")
        return seed if isinstance(seed, int) else None

    @seed.setter
    def seed(self, value: int | None) -> None:
        if value is None:
            self._api_kwargs.pop("seed", None)
        else:
            self._api_kwargs["seed"] = value

    def _system_message(self) -> str:
        """Return this run's system message: the base prompt plus any policy.

        Interface contract (1 call site, :meth:`_fsm_for_run`; the attribute it reads is
        public because callers set it post-construction):
            - Reads ``self.system_policy`` at CALL time, not at construction, so
              a caller that receives the agent from a factory can still supply
              standing instructions.
            - Returns ``_SYSTEM_PROMPT`` unchanged when the policy is unset or
              blank -- the no-policy path is byte-identical to before.
            - Never raises.
        """
        # DECISION plan-2026-07-21T191807-bf7ffe24/D-021
        # Standing instructions belong in the SYSTEM message, and this seam
        # exists because that placement was MEASURED to be the difference
        # between a 4B model doing the work and only talking about it. Live
        # `ollama_chat/qwen3.5:4b`, an EXECUTE role, n=5 per arm, bytes
        # stat'd on disk: the whole role prompt in the USER turn wrote 0/5;
        # the SAME text, unchanged and complete, with everything standing moved
        # into the system message wrote 4/5. Content ablations in between
        # (fixing the writes line: 0/5; deleting the rules block: 2/5) did not
        # reproduce it, so this is placement, not wording.
        # Do NOT "simplify" this into a `system_prompt` REPLACEMENT parameter:
        # the measured arm kept `_SYSTEM_PROMPT` and appended to it, and the
        # base prompt is what tells the model it may call tools at all.
        # Do NOT make it constructor-only either -- `roles.py` obtains this
        # agent through an injectable builder that cannot see the dispatch, so
        # a construction-time-only parameter would be unreachable exactly where
        # it was measured to matter. See decisions.md D-021.
        if not self.system_policy:
            return _SYSTEM_PROMPT
        return f"{_SYSTEM_PROMPT}\n\n{self.system_policy}"

    def _fsm_for_run(
        self, task: str, initial_context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The FSM of one run through ``build_native_fc_fsm``: the tools, the system message,
        ``config.force_final_tool`` and ``config.output_schema`` are read now."""
        # DECISION plan-2026-10-01T093600-944e2692/D-028: this agent is an FSM run by core
        # (`_standard_run`, through `PatternAgent.run`); its model turns are completion states. Do NOT
        # bring back a loop over model turns, a provider call, a request builder or an injectable
        # completion function here: every request rule lives once in core's `llm` package (D-027)
        # and the test seam is `llm_interface=`. See decisions.md D-009, D-028.
        return build_native_fc_fsm(
            self.tools.get_json_schemas(),
            instructions=self._system_message(),
            force_final_tool=self.config.force_final_tool,
            repair=self.config.output_schema is not None,
        )

    def _step_ceiling(self, max_iterations: int) -> tuple[int, str]:
        """Two core steps per model loop turn plus the post-loop turns."""
        per_turn = Defaults.NATIVE_FC_STEPS_PER_TURN
        post_loop = Defaults.NATIVE_FC_POST_LOOP_STEPS
        return (
            per_turn * max_iterations + post_loop,
            f"{per_turn} x max_iterations {max_iterations} + {post_loop} "
            "post-loop steps",
        )

    def _extract_answer(
        self,
        final_context: dict[str, Any],
        responses: list[str],
        extra_keys: list[str] | None = None,
    ) -> str:
        """The run's answer, exactly as the handlers recorded it (``""`` if none).

        Every state is silent, so there is no reply to fall back to.
        """
        answer = final_context.get(_K.ANSWER)
        return answer if isinstance(answer, str) else ""

    def _try_parse_structured_output(
        self, answer: str, context: dict[str, Any] | None = None
    ) -> Any:
        """Parse *answer* against ``config.output_schema``; context is ignored.

        The answer is this pattern's only structured source: a context key
        named like a schema field (a caller's ``initial_context`` entry, a
        handler key) must never stand in for what the model answered.
        """
        return super()._try_parse_structured_output(answer)
