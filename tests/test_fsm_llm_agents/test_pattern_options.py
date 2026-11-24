"""The resolved options of a pattern agent (plan a555dcfe, step 28.2, D-035).

The Phase P1 review found that the factories and hooks of the handler library read only the options the
CALLER passed and fell back to Python constants, while the FSM was built from the file's defaults: a
preset that set a default (``max_verify_retries: 3``) got one value in the FSM and another in its
handlers. The agent now resolves its options once at construction (the caller's values checked against
the options schema, the file's default for every other declared option) and hands that one mapping to
the handler context, the hooks and the FSM builder; a bad ``options=`` is refused at construction.
"""

from __future__ import annotations

from typing import Any

import pytest

from fsm_llm.agents import SemanticMemoryStore, create_agent
from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.constants import Defaults
from fsm_llm.agents.definitions import AgentConfig
from fsm_llm.agents.handlers import hooks
from fsm_llm.agents.pattern_agent import PatternAgent
from fsm_llm.core.definitions import AgentError
from fsm_llm.engine.executables import ExecutableRegistry
from tests import pattern_runs as runs


def _lookup(query: str) -> str:
    """Look it up."""
    return "x"


def _registry(*docs: dict[str, Any]) -> ExecutableRegistry:
    import fsm_llm.agents.executables as ae

    registry = ExecutableRegistry()
    for held in (*ae.builtin_executables(), *ae.preset_executables()):
        registry.register(held)
    for doc in docs:
        registry.add_document(doc)
    return registry


def _preset(doc_id: str, base: str, **options: Any) -> dict[str, Any]:
    return {
        "schema": "fsm-llm/executable",
        "kind": "pattern",
        "id": doc_id,
        "name": doc_id,
        "description": "d",
        "body": {"extends": base, "options": options},
    }


def _verified(registry: ExecutableRegistry, doc_id: str, calls: list[str], **kw: Any):
    def never(answer: str, context: dict[str, Any]) -> bool:
        calls.append(answer)
        return False

    return create_agent(
        doc_id,
        [_lookup],
        registry=registry,
        config=AgentConfig(model="m", max_iterations=5, verification_fn=never),
        llm_interface=runs.ScriptedModel("_lookup"),
        **kw,
    )


class TestAFileDefaultReachesTheHandlers:
    """The probe of the review (p4): a preset default and the same value passed by the caller agree."""

    def test_a_preset_default_retries_as_often_as_the_same_caller_option(self):
        registry = _registry(
            _preset("user.strict", "verified_react", max_verify_retries=3)
        )
        by_default: list[str] = []
        by_option: list[str] = []

        _verified(registry, "user.strict", by_default).run("q")
        _verified(
            registry, "user.strict", by_option, options={"max_verify_retries": 3}
        ).run("q")

        assert (
            len(by_default) == len(by_option) == 4
        )  # the first attempt and three retries

    def test_a_caller_option_beats_the_file_default(self):
        registry = _registry(
            _preset("user.strict", "verified_react", max_verify_retries=3)
        )
        calls: list[str] = []

        _verified(
            registry, "user.strict", calls, options={"max_verify_retries": 0}
        ).run("q")

        assert len(calls) == 1

    def test_the_handler_context_holds_the_resolved_options(self):
        registry = _registry(
            _preset("user.strict", "verified_react", max_verify_retries=3)
        )

        agent = _verified(registry, "user.strict", [])

        assert agent._handler_context().options["max_verify_retries"] == 3

    def test_a_preset_default_of_the_recall_reaches_the_before_run_hook(self):
        registry = _registry(_preset("user.recall", "auto_memory", recall_k=7))
        seen: list[int] = []

        class _Store(SemanticMemoryStore):
            def search(self, query: str, k: int = 3, **kw: Any):
                seen.append(k)
                return []

        agent = create_agent(
            "user.recall",
            [_lookup],
            registry=registry,
            config=runs._config(),
            callables={"memory": _Store(embed_fn=lambda t: [1.0, 0.0])},
            llm_interface=runs.ScriptedModel("_lookup"),
        )
        agent.run("q")

        assert seen and seen[0] == 7


class TestTheDefaultsEqualTheOldPythonConstants:
    """A file default that equals the constant a factory used to fall back to changes nothing."""

    @pytest.mark.parametrize(
        ("pattern", "option", "constant"),
        [
            ("orchestrator", "max_workers", Defaults.MAX_WORKERS),
            ("reflexion", "max_reflections", Defaults.MAX_REFLECTIONS),
            ("plan_execute", "max_replans", Defaults.MAX_REPLANS),
            ("maker_checker", "max_revisions", Defaults.MAX_REVISIONS),
            ("maker_checker", "quality_threshold", Defaults.QUALITY_THRESHOLD),
            ("debate", "max_rounds", Defaults.MAX_DEBATE_ROUNDS),
            ("evaluator_optimizer", "max_refinements", Defaults.MAX_REFINEMENTS),
            ("verified_react", "max_verify_retries", hooks.DEFAULT_MAX_VERIFY_RETRIES),
            ("auto_memory", "recall_k", hooks.DEFAULT_RECALL_K),
            ("auto_memory", "recall_min_score", hooks.DEFAULT_RECALL_MIN_SCORE),
        ],
    )
    def test_the_file_default_is_the_constant(
        self, pattern: str, option: str, constant: Any
    ):
        assert pf.load_envelope(pattern).body.options[option] == constant

    @pytest.mark.parametrize(
        ("pattern", "option", "fallback"),
        [
            ("auto_memory", "auto_remember", True),
            ("auto_memory", "remember_only_on_success", False),
        ],
    )
    def test_the_boolean_defaults_are_the_fallbacks_of_the_hook(
        self, pattern: str, option: str, fallback: bool
    ):
        assert pf.load_envelope(pattern).body.options[option] is fallback


class TestOptionsAreCheckedAtConstruction:
    def _doc(self) -> dict[str, Any]:
        doc = pf.load_document("auto_memory")
        doc["id"] = "user.mem"
        return doc

    def test_an_unknown_option_is_refused_before_any_hook_or_model_call(self):
        model = runs.ScriptedModel("_lookup")
        searched: list[str] = []

        class _Store(SemanticMemoryStore):
            def search(self, query: str, k: int = 3, **kw: Any):
                searched.append(query)
                return []

        registry = _registry(self._doc())
        with pytest.raises(AgentError, match="nope"):
            create_agent(
                "user.mem",
                [_lookup],
                registry=registry,
                config=runs._config(),
                options={"nope": 1},
                callables={"memory": _Store(embed_fn=lambda t: [1.0, 0.0])},
                llm_interface=model,
            )
        assert model.log == [] and searched == []

    @pytest.mark.parametrize("value", ["three", -1, 1.5, None, True])
    def test_an_invalid_value_is_refused_at_construction(self, value: Any):
        registry = _registry(self._doc())
        with pytest.raises(AgentError, match="recall_k"):
            create_agent(
                "user.mem",
                [_lookup],
                registry=registry,
                config=runs._config(),
                options={"recall_k": value},
                callables={"memory": SemanticMemoryStore(embed_fn=lambda t: [1.0])},
                llm_interface=runs.ScriptedModel("_lookup"),
            )

    def test_the_message_names_the_option_and_never_its_value(self):
        registry = _registry(self._doc())
        with pytest.raises(AgentError) as caught:
            create_agent(
                "user.mem",
                [_lookup],
                registry=registry,
                config=runs._config(),
                options={"recall_k": "SECRET-VALUE"},
                callables={"memory": SemanticMemoryStore(embed_fn=lambda t: [1.0])},
                llm_interface=runs.ScriptedModel("_lookup"),
            )
        assert "SECRET-VALUE" not in str(caught.value)

    def test_a_valid_option_still_builds_a_pattern_agent(self):
        registry = _registry(self._doc())
        agent = create_agent(
            "user.mem",
            [_lookup],
            registry=registry,
            config=runs._config(),
            options={"recall_k": 2},
            callables={"memory": SemanticMemoryStore(embed_fn=lambda t: [1.0])},
            llm_interface=runs.ScriptedModel("_lookup"),
        )
        assert type(agent) is PatternAgent
        assert agent._options["recall_k"] == 2


class TestTheAgentConfigOptionsKeepTheirPrecedence:
    """``temperature`` and ``max_iterations`` of a preset live in the CONFIG (D-033): the resolved
    options must not shadow the host's explicit config with the file default."""

    def test_the_host_max_iterations_beats_the_preset_default_in_the_limiter(self):
        registry = _registry()
        agent = create_agent(
            "code-review",
            [_lookup],
            registry=registry,
            config=AgentConfig(model="m", max_iterations=7),
            llm_interface=runs.ScriptedModel("_lookup"),
        )

        assert agent.config.max_iterations == 7
        assert "max_iterations" not in agent._handler_context().options

    def test_the_default_applies_to_the_config_when_the_host_set_nothing(self):
        agent = create_agent(
            "code-review",
            [_lookup],
            registry=_registry(),
            config=AgentConfig(model="m"),
            llm_interface=runs.ScriptedModel("_lookup"),
        )

        assert agent.config.max_iterations == 5
        assert agent.config.temperature == pytest.approx(0.3)

    def test_a_caller_option_reaches_the_config_and_the_options(self):
        agent = create_agent(
            "code-review",
            [_lookup],
            registry=_registry(),
            config=AgentConfig(model="m", max_iterations=7),
            options={"max_iterations": 4},
            llm_interface=runs.ScriptedModel("_lookup"),
        )

        assert agent.config.max_iterations == 4
        assert agent._handler_context().options["max_iterations"] == 4


class TestTheResolvedMappingIsTheAgentsOwn:
    def test_a_list_default_is_a_copy_not_the_files_own(self):
        envelope = pf.load_envelope("native_fc")
        tools = runs._tools()
        agent = PatternAgent(
            envelope,
            tools=tools,
            config=runs._config(),
            llm_interface=runs.ScriptedModel(),
        )
        before = list(envelope.body.options["tool_schemas"])

        agent._options["tool_schemas"].append({"name": "x"})

        assert envelope.body.options["tool_schemas"] == before

    def test_a_thin_class_keeps_the_default_of_an_option_named_like_a_config_field(
        self,
    ):
        """``max_iterations`` of debate is a step budget, not the config's: its default stays."""

        class Thin(PatternAgent):
            takes_tools = False

        agent = Thin(
            pf.load_envelope("debate"),
            config=runs._config(),
            llm_interface=runs.ScriptedModel(),
        )

        assert agent._options["max_iterations"] == 18
