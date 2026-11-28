"""Selection of a reasoning strategy over a registry (plan baa37f40, step 17, D-033, roadmap E3.4a).

``strategy_choices(registry)`` is the one function that decides which registered executables a solve
may choose; ``ReasoningEngine(registry=)`` closes the orchestrator's ``choose.targets`` over it, a
user strategy is reached by ``preferred_reasoning_type`` and keeps its answer, an ineligible one is
skipped with a WARNING naming why. Every row is scripted: no network.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

import fsm_llm
from fsm_llm.core.definitions.executable import Capability, Contract
from fsm_llm.engine.executables import CodeBackedExecutable, ExecutableRegistry
from fsm_llm.logging import logger
from fsm_llm.reasoning import (
    ReasoningEngine,
    ReasoningRegistryError,
    get_available_reasoning_types,
)
from fsm_llm.reasoning import selection as selection_module
from fsm_llm.reasoning.constants import ContextKeys, OrchestratorStates
from fsm_llm.reasoning.executables import strategy_registry
from fsm_llm.reasoning.handlers import OutputFormatter
from fsm_llm.reasoning.selection import (
    StrategyChoice,
    builtin_choices,
    choices_for,
    find_choice,
    strategy_choices,
)
from fsm_llm.reasoning.strategy_files import (
    STRATEGY_ORDER,
    load_document,
    load_fsm,
    strategy_id,
)
from fsm_llm.reasoning.utilities import lookup_reasoning_type, map_reasoning_type

from .test_engine_scripted import _PROBLEM, _VALID_SCRIPT, _engine, _ScriptedLLM
from .test_executables import _script

REPO = Path(__file__).resolve().parents[2]
CALL_STATE = OrchestratorStates.EXECUTE_REASONING
_PIRATE_ANSWER = (
    "value of key_insights"  # what the user strategy below returns as its answer
)


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(selection_module.__file__).resolve().is_relative_to(REPO / "src")


def _warnings(run: Any) -> list[str]:
    """The WARNING messages the library logged while ``run()`` ran."""
    records: list[str] = []
    logger.enable("fsm_llm")  # library logging is off by default
    sink = logger.add(
        lambda message: records.append(message.record["message"]), level="WARNING"
    )
    try:
        run()
    finally:
        logger.remove(sink)
        logger.disable("fsm_llm")
    return records


def _document(
    name: str = "pirate",
    *,
    ident: str | None = None,
    description: str = "Answers like a pirate",
    aliases: tuple[str, ...] = ("arr",),
    solution_key: str = "pirate_answer",
    lifecycle: str = "draft",
    inputs: tuple[str, ...] = ("problem_statement",),
    version: str = "1",
) -> dict[str, Any]:
    """A strategy document: the analytical body under another name, its answer under a new key."""
    document = copy.deepcopy(load_document("analytical"))
    document["id"] = ident or f"user.{name}"
    document["name"] = name
    document["version"] = version
    document["description"] = description
    document["lifecycle"] = lifecycle
    document["contract"] = {"inputs": list(inputs), "outputs": []}
    document["body"]["strategy"] = {
        "aliases": list(aliases),
        "results": {solution_key: "key_insights"},
        "solution_key": solution_key,
    }
    return document


def _registry(*documents: dict[str, Any]) -> ExecutableRegistry:
    registry = strategy_registry()
    for document in documents:
        registry.add_document(document)
    return registry


def _script_for_user_strategy() -> dict[str, Any]:
    """Every key the analytical body needs, and NO orchestrator-level answer: the answer of the
    solve can only be the one the strategy returns under its own solution key."""
    script = {**_VALID_SCRIPT, **_script("analytical")}
    for key in (ContextKeys.FINAL_SOLUTION, ContextKeys.PROPOSED_SOLUTION):
        script.pop(key, None)
    return script


# ---- the one function ---------------------------------------------------------------------------


class TestStrategyChoices:
    def test_the_builtin_registry_gives_the_nine_in_order(self) -> None:
        choices = strategy_choices(strategy_registry())
        assert [c.name for c in choices] == list(STRATEGY_ORDER)
        assert [c.target_id for c in choices] == [
            strategy_id(n) for n in STRATEGY_ORDER
        ]
        assert all(c.description and "\n" not in c.description for c in choices)
        assert all(isinstance(c, StrategyChoice) for c in choices)
        assert dataclasses.is_dataclass(choices[0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            choices[0].name = "x"  # type: ignore[misc]

    def test_a_user_strategy_follows_the_builtins_and_user_ones_sort_by_id(
        self,
    ) -> None:
        registry = _registry(_document("zeta"), _document("alpha"))
        names = [c.name for c in strategy_choices(registry)]
        assert names == [*STRATEGY_ORDER, "alpha", "zeta"]

    def test_the_choice_carries_the_declaration_of_the_document(self) -> None:
        (user,) = [
            c
            for c in strategy_choices(_registry(_document(aliases=("Arr ", "matey"))))
            if c.name == "pirate"
        ]
        assert user == StrategyChoice(
            name="pirate",
            target_id="user.pirate",
            description="Answers like a pirate",
            aliases=("arr", "matey"),
            solution_key="pirate_answer",
        )

    def test_a_description_is_one_capped_line_and_falls_back_to_the_name(self) -> None:
        long = "x" * 500
        registry = _registry(
            _document("liner", description="first\nsecond\r\nthird"),
            _document("longer", description=long),
            _document("silent", description=""),
        )
        by_name = {c.name: c for c in strategy_choices(registry)}
        assert by_name["liner"].description == "first second  third"
        assert by_name["longer"].description == "x" * 300
        assert by_name["silent"].description == "silent"

    def test_an_older_version_of_an_id_is_shadowed_silently(self) -> None:
        registry = _registry(
            _document("pirate", version="1", description="old"),
            _document("pirate", version="2", description="new"),
        )
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        (user,) = [c for c in found if c.name == "pirate"]
        assert user.description == "new" and warned == []

    @pytest.mark.parametrize(
        ("document", "why"),
        [
            (_document("old", lifecycle="deprecated"), "it is deprecated"),
            (
                _document("greedy", inputs=("problem_statement", "wallet")),
                "inputs outside the problem keys: ['wallet']",
            ),
            (_document("9lives"), "not an ASCII identifier"),
            (_document("two words", ident="user.spaced"), "not an ASCII identifier"),
            (_document("strat\n", ident="user.strat"), "not an ASCII identifier"),
        ],
        ids=["deprecated", "inputs", "digit-name", "space-name", "newline-name"],
    )
    def test_an_ineligible_strategy_is_skipped_with_one_warning_naming_why(
        self, document: dict[str, Any], why: str
    ) -> None:
        registry = _registry(document)
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        assert [c.name for c in found] == list(STRATEGY_ORDER)
        (message,) = warned
        assert document["id"] in message and why in message

    def test_a_duplicate_name_is_skipped_and_the_first_in_order_wins(self) -> None:
        registry = _registry(
            _document("Pirate", ident="user.a"), _document("pirate", ident="user.b")
        )
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        assert [c.target_id for c in found if c.name.lower() == "pirate"] == ["user.a"]
        (message,) = warned
        assert "'user.b'" in message and "already taken by 'user.a'" in message

    def test_a_user_name_equal_to_a_builtin_name_is_skipped(self) -> None:
        registry = _registry(_document("analytical", ident="user.analytical"))
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        assert [c.target_id for c in found] == [strategy_id(n) for n in STRATEGY_ORDER]
        assert any("already taken by 'reasoning.analytical'" in m for m in warned)

    @pytest.mark.parametrize(
        ("name", "owner"),
        [("math", "simple_calculator"), ("logic", "deductive")],
    )
    def test_a_user_name_equal_to_a_builtin_alias_is_skipped_and_the_builtin_keeps_it(
        self, name: str, owner: str
    ) -> None:
        """D-044 (3), review iter1 note 9: built-ins win, aliases included. Kills: the
        built-in aliases not seeded before the user names are judged (the user strategy took
        the alias and `preferred_reasoning_type=math` ran it)."""
        registry = _registry(_document(name, ident=f"user.{name}"))
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        assert [c.target_id for c in found] == [strategy_id(n) for n in STRATEGY_ORDER]
        assert any(
            f"user.{name}" in m and f"already taken by 'reasoning.{owner}'" in m
            for m in warned
        )
        chosen = find_choice(found, name)
        assert chosen is not None and chosen.name == owner

    def test_an_alias_that_names_something_else_is_dropped_and_a_name_beats_an_alias(
        self,
    ) -> None:
        registry = _registry(
            _document(
                "early", aliases=("late", "math")
            ),  # `late` is a later NAME, `math` a built-in alias
            _document("late", aliases=("late",)),
        )
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        by_name = {c.name: c for c in found}
        assert by_name["early"].aliases == ()
        assert by_name["late"].aliases == ()
        assert any(
            "alias 'math'" in m and "reasoning.simple_calculator" in m for m in warned
        )
        assert any("alias 'late'" in m and "user.late" in m for m in warned)
        assert find_choice(found, "late") is by_name["late"]
        assert find_choice(found, "math") is by_name["simple_calculator"]

    def test_a_code_backed_strategy_has_no_declaration_and_is_skipped(self) -> None:
        class _Code(CodeBackedExecutable):
            async def execute(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

        registry = _registry()
        registry.register(
            _Code(
                id="user.coded",
                kind="strategy",
                impl="x.y",
                contract=Contract(inputs=["problem_statement"]),
                capabilities=[Capability.STRATEGY],
            )
        )
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        assert [c.name for c in found] == list(STRATEGY_ORDER)
        (message,) = warned
        assert "user.coded" in message and "no strategy declaration" in message

    def test_a_procedure_is_never_eligible_whatever_it_claims(self) -> None:
        """STOP IF a model-written ``procedure`` executable becomes selectable: the kind filter
        is the gate, the capability a procedure declares is a claim."""
        document = _document("written")
        document["kind"] = "procedure"
        document["body"] = {"fsm": document["body"]["fsm"], "provenance": {}}
        registry = strategy_registry()
        registry.add_document(document)
        assert registry.get("user.written").kind == "procedure"
        assert Capability.STRATEGY in registry.get("user.written").capabilities
        found: list[StrategyChoice] = []
        warned = _warnings(lambda: found.extend(strategy_choices(registry)))
        assert [c.name for c in found] == list(STRATEGY_ORDER) and warned == []

    def test_a_kind_fsm_with_the_capability_is_not_a_strategy(self) -> None:
        document = _document("plain")
        document["kind"] = "fsm"
        document["body"] = {"fsm": document["body"]["fsm"]}
        registry = strategy_registry()
        registry.add_document(document)
        assert "plain" not in [c.name for c in strategy_choices(registry)]


class TestFindChoice:
    def test_a_name_or_an_alias_in_any_case_and_padding(self) -> None:
        choices = builtin_choices()
        assert find_choice(choices, "  Math ").name == "simple_calculator"
        assert find_choice(choices, "DEDUCTIVE").name == "deductive"
        assert find_choice(choices, "no such type") is None

    def test_a_fresh_list_each_call_never_a_module_table(self) -> None:
        first = builtin_choices()
        first.clear()
        assert len(builtin_choices()) == 9
        assert builtin_choices() is not builtin_choices()
        assert choices_for(None) == builtin_choices()


# ---- the utilities read the registry ------------------------------------------------------------


class TestTheUtilitiesReadTheRegistry:
    def test_map_reasoning_type_knows_a_user_strategy_by_name_and_alias(self) -> None:
        registry = _registry(_document())
        assert map_reasoning_type("pirate", registry=registry) == "pirate"
        assert map_reasoning_type(" ARR ", registry=registry) == "pirate"
        assert map_reasoning_type("math", registry=registry) == "simple_calculator"

    def test_without_a_registry_it_is_the_builtin_nine(self) -> None:
        warned = _warnings(lambda: map_reasoning_type("pirate"))
        assert map_reasoning_type("pirate") == "analytical"
        assert warned and "Unknown reasoning type 'pirate'" in warned[0]

    def test_lookup_returns_none_for_an_unknown_spelling_and_logs_nothing(self) -> None:
        registry = _registry(_document())
        assert lookup_reasoning_type("arr", registry=registry) == "pirate"
        assert lookup_reasoning_type("arr") is None
        assert _warnings(lambda: lookup_reasoning_type("nothing")) == []

    def test_the_description_table_lists_the_user_strategy_after_the_nine(self) -> None:
        listed = get_available_reasoning_types(registry=_registry(_document()))
        assert list(listed) == [*STRATEGY_ORDER, "pirate"]
        assert listed["pirate"] == "Answers like a pirate"
        assert list(get_available_reasoning_types()) == list(STRATEGY_ORDER)


# ---- the engine ---------------------------------------------------------------------------------


class TestTheEngineTakesARegistry:
    def test_with_the_builtin_registry_the_orchestrator_file_is_unchanged(self) -> None:
        engine = _engine(_ScriptedLLM(_VALID_SCRIPT))
        assert engine.main_fsm == load_fsm("orchestrator")
        choose = engine.main_fsm["states"][CALL_STATE]["call"]["choose"]
        assert choose["fallback"] == "reasoning.analytical"
        assert engine.classifier_fsm == load_fsm("classifier")

    def test_the_closed_list_is_every_eligible_target_and_the_fallback_stays(
        self,
    ) -> None:
        registry = _registry(_document(), _document("old", lifecycle="deprecated"))
        engine = ReasoningEngine(
            model="mock", llm_interface=_ScriptedLLM(_VALID_SCRIPT), registry=registry
        )
        choose = engine.main_fsm["states"][CALL_STATE]["call"]["choose"]
        assert choose["targets"] == [
            *(strategy_id(n) for n in STRATEGY_ORDER),
            "user.pirate",
        ]
        assert choose["fallback"] == "reasoning.analytical"
        assert choose["targets_from"] == {"capability": "strategy"}
        assert engine.registry is registry

    def test_registry_is_an_explicit_parameter_never_swallowed_into_the_api(
        self,
    ) -> None:
        engine = ReasoningEngine(
            model="mock",
            llm_interface=_ScriptedLLM(_VALID_SCRIPT),
            registry=_registry(_document()),
        )
        assert "registry" not in engine.api_kwargs
        assert "targets" not in engine.api_kwargs
        assert not any(
            "registry" in p for p in _signature_names(fsm_llm.API.from_definition)
        )

    def test_registry_and_targets_are_refused_when_they_are_not_what_they_claim(
        self,
    ) -> None:
        llm = _ScriptedLLM(_VALID_SCRIPT)
        with pytest.raises(TypeError, match="registry"):
            ReasoningEngine(model="mock", llm_interface=llm, registry=["not", "one"])
        with pytest.raises(TypeError, match="registry="):
            ReasoningEngine(
                model="mock", llm_interface=llm, targets=strategy_registry()
            )

    def test_a_registry_without_analytical_fails_at_construction_named(self) -> None:
        registry = ExecutableRegistry()
        for name in STRATEGY_ORDER:
            if name != "analytical":
                registry.register(_builtin(name))
        with pytest.raises(ReasoningRegistryError, match="analytical"):
            ReasoningEngine(
                model="mock",
                llm_interface=_ScriptedLLM(_VALID_SCRIPT),
                registry=registry,
            )

    def test_a_registry_with_a_single_strategy_fails_at_construction_named(
        self,
    ) -> None:
        registry = ExecutableRegistry()
        registry.register(_builtin("analytical"))
        with pytest.raises(ReasoningRegistryError, match="two"):
            ReasoningEngine(
                model="mock",
                llm_interface=_ScriptedLLM(_VALID_SCRIPT),
                registry=registry,
            )

    def test_the_named_error_is_a_value_error_and_a_reasoning_error(self) -> None:
        from fsm_llm.reasoning import ReasoningError

        assert issubclass(ReasoningRegistryError, (ReasoningError, ValueError))


def _builtin(name: str) -> Any:
    from fsm_llm.reasoning.executables import builtin_executables

    return next(e for e in builtin_executables() if e.id == strategy_id(name))


def _signature_names(function: Any) -> list[str]:
    import inspect

    return list(inspect.signature(function).parameters)


class TestAUserStrategyIsChosenAndKeepsItsAnswer:
    def _solve(self, registry: ExecutableRegistry, context: dict[str, Any]) -> Any:
        llm = _ScriptedLLM(_script_for_user_strategy())
        engine = ReasoningEngine(model="mock", llm_interface=llm, registry=registry)
        return engine, llm, engine.solve_problem(_PROBLEM, context)

    @pytest.mark.parametrize("spelling", ["pirate", "Arr"])
    def test_preferred_reasoning_type_reaches_it_and_its_answer_is_returned(
        self, spelling: str
    ) -> None:
        _e, _llm, (solution, trace) = self._solve(
            _registry(_document()), {"preferred_reasoning_type": spelling}
        )
        context = trace["final_context"]
        assert context[ContextKeys.CALL_RESULT]["target"] == "user.pirate"
        assert context[ContextKeys.CALL_RESULT]["status"] == "ok"
        assert context[ContextKeys.REASONING_TYPE_SELECTED] == "pirate"
        assert trace["reasoning_trace"]["reasoning_types_used"] == ["pirate"]
        assert context["pirate_answer"] == _PIRATE_ANSWER
        # no orchestrator level answer exists: the one returned is the strategy's own
        assert solution == _PIRATE_ANSWER

    def test_the_cached_builtin_keys_would_lose_the_answer(self) -> None:
        """Mutant row: ``extract_final_solution`` with no keys given reads the nine built-in
        solution keys only (the defect of the cached table, D-033 (5)), so it misses the answer."""
        _e, _l, (_solution, trace) = self._solve(
            _registry(_document()), {"preferred_reasoning_type": "pirate"}
        )
        final = trace["final_context"]
        assert OutputFormatter.extract_final_solution(final) != _PIRATE_ANSWER
        keys = (
            ContextKeys.FINAL_SOLUTION,
            ContextKeys.PROPOSED_SOLUTION,
            "pirate_answer",
        )
        assert OutputFormatter.extract_final_solution(final, keys) == _PIRATE_ANSWER

    def test_an_ineligible_strategy_is_never_called_even_when_named(self) -> None:
        registry = _registry(_document("old", lifecycle="deprecated"))
        warned: list[str] = []

        def run() -> None:
            _e, _l, (_solution, trace) = self._solve(
                registry, {"preferred_reasoning_type": "old"}
            )
            call = trace["final_context"][ContextKeys.CALL_RESULT]
            assert call["target"] != "user.old"

        warned = _warnings(run)
        assert any("user.old" in m and "deprecated" in m for m in warned)
        assert any("names no known strategy" in m for m in warned)

    def test_a_forged_choice_key_outside_the_closed_list_takes_the_fallback(
        self,
    ) -> None:
        """D-054: the engine's own list is closed; a value outside it never reaches the registry."""
        engine = ReasoningEngine(
            model="mock",
            llm_interface=_ScriptedLLM(_script_for_user_strategy()),
            registry=_registry(_document()),
        )
        conversation_id, _ = engine.orchestrator.start_conversation(
            {
                ContextKeys.PROBLEM_STATEMENT: _PROBLEM,
                ContextKeys.PROBLEM_TYPE: "logic puzzle",
                ContextKeys.PROBLEM_COMPONENTS: ["a", "b"],
                ContextKeys.CLASSIFIED_PROBLEM_TYPE: "user.not_listed",
                ContextKeys.REASONING_TRACE: [],
                ContextKeys.RETRY_COUNT: 0,
            }
        )
        engine.orchestrator.run_until_terminal(
            conversation_id, max_steps=engine._solve_steps
        )
        record = engine.orchestrator.get_data(conversation_id)[ContextKeys.CALL_RESULT]
        assert record["target"] == "reasoning.analytical"


class TestAStrategyMapCannotWriteAnOrchestratorKey:
    """D-044 (1), review iter1 concern 3: the output map of a user strategy is merged by the
    pop, and the orchestrator's handler-only keys are never a merge target (the loader refuses
    them for a declared `returns`)."""

    @pytest.mark.parametrize(
        "key", ["hybrid_loop_count", "validation_result", "max_retries_reached"]
    )
    def test_a_map_naming_a_handler_only_key_never_writes_it(self, key: str) -> None:
        """Kills: `_mapped_returns` with no handler-only rule (the child's model value landed
        on the orchestrator's handler-only key)."""
        document = _document()
        document["body"]["strategy"]["results"][key] = "key_insights"
        engine = ReasoningEngine(
            model="mock",
            llm_interface=_ScriptedLLM(_script_for_user_strategy()),
            registry=_registry(document),
        )
        result: list[Any] = []
        warned = _warnings(
            lambda: result.append(
                engine.solve_problem(_PROBLEM, {"preferred_reasoning_type": "pirate"})
            )
        )
        context = result[0][1]["final_context"]

        assert context[ContextKeys.CALL_RESULT]["status"] == "ok"
        assert (
            context["pirate_answer"] == _PIRATE_ANSWER
        )  # the rest of the map still merges
        assert context.get(key) != _PIRATE_ANSWER
        assert any(f"'{key}'" in m and "never merged" in m for m in warned)


class TestNoTargetIsResolvedByAStringImport:
    def test_the_selection_module_names_no_string_import(self) -> None:
        tree = ast.parse(Path(selection_module.__file__).read_text(encoding="utf-8"))
        names = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert not names & {"import_module", "__import__", "find_spec", "exec_module"}
        # the engine module too: ids are registry keys only
        engine_source = (REPO / "src/fsm_llm/reasoning/engine.py").read_text(
            encoding="utf-8"
        )
        assert "importlib" not in engine_source and "__import__" not in engine_source
