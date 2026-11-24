"""``MetaBuilderAgent``, the shell of the meta-builder agent (plan d454fd62 step 90; the old ``agents/meta_builder.py``, same dotted path).

The class is a shell (construction and the session attributes) composed of six mixins:
``HintsMixin`` (``hints.py``), ``HandlersMixin`` (``handlers.py``), ``AssemblyMixin`` (``assembly.py``),
``RunMixin`` (``runs.py``), ``SessionMixin`` (``session.py``) and ``ResultsMixin`` (``results.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fsm_llm import API
from fsm_llm.agents.base import reject_misplaced_kwargs
from fsm_llm.agents.definitions import (
    AgentConfig,
    MetaBuilderConfig,
    MetaBuilderResult,
    default_agent_config,
)

from .assembly import AssemblyMixin
from .handlers import HandlersMixin
from .hints import HintsMixin
from .results import ResultsMixin
from .runs import RunMixin
from .session import SessionMixin


class MetaBuilderAgent(
    HintsMixin, HandlersMixin, AssemblyMixin, RunMixin, SessionMixin, ResultsMixin
):
    """Meta-builder: classify the artifact type, collect, then build.

    A build is one structured completion (the ``build`` state of the meta
    FSM) whose JSON spec is assembled on a fresh builder and validated in
    Python. In turn-by-turn mode, ``send()`` collects requirements until a
    build trigger ("build it"); ``start()`` never builds. A build with
    validation errors keeps the session open for another try. ``is_valid``
    on a workflow or agent result means a structurally complete spec, not a
    loadable object.

    ``api_kwargs`` go to the core ``API`` (``llm_interface=`` injects the
    interface every model call goes through; other names are LLM call
    kwargs). ``model``, ``temperature`` and ``max_tokens`` belong on
    ``MetaBuilderConfig`` and raise ``TypeError`` here.

    Usage (single-shot)::

        agent = MetaBuilderAgent()
        result = agent.run("Build a customer support chatbot with 3 states")
        print(result.artifact_json)

    Usage (turn-by-turn)::

        agent = MetaBuilderAgent()
        response = agent.start()
        while not agent.is_complete():
            response = agent.send(input("> "))
        result = agent.get_result()
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        agent_types: Iterable[str] = (),
        **api_kwargs: Any,
    ) -> None:
        reject_misplaced_kwargs(type(self).__name__, api_kwargs)
        # A plain AgentConfig (create_agent, the builder) becomes a
        # MetaBuilderConfig here; see MetaBuilderConfig.from_agent_config.
        self.meta_config = (
            default_agent_config(api_kwargs.get("llm_interface"), MetaBuilderConfig)
            if config is None
            else MetaBuilderConfig.from_agent_config(config)
        )
        self._api_kwargs = api_kwargs
        # Agent pattern ids beside ``VALID_AGENT_TYPES`` (``fsm-llm-meta --executables DIR``).
        self._agent_types = tuple(sorted(set(agent_types)))

        # The turn-by-turn session (``start``) and the result of ``run``.
        self._api: API | None = None
        self._conversation_id: str | None = None
        self._result: MetaBuilderResult | None = None
        # Sends so far, the ``max_turns`` budget (driver bookkeeping).
        self._turn_count = 0
