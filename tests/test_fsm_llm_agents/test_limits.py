from __future__ import annotations

"""F-AR-13b: every agent ``max_*`` / ``num_*`` limit goes through one helper.

``require_int_limit`` refuses a bool, a float (also 1.0, NaN, inf), a str,
``None``, a value below the pattern's minimum and an int past
``Defaults.MAX_INT_LIMIT``, and its message never holds the value in full (an
int past 4300 digits makes an f-string raise ``ValueError``).
"""

import pytest

from fsm_llm.agents.adapt import ADaPTAgent
from fsm_llm.agents.constants import Defaults
from fsm_llm.agents.debate import DebateAgent
from fsm_llm.agents.evaluator_optimizer import EvaluatorOptimizerAgent
from fsm_llm.agents.exceptions import (
    AgentError,
    bounded_repr,
    require_int_limit,
)
from fsm_llm.agents.maker_checker import MakerCheckerAgent
from fsm_llm.agents.orchestrator import OrchestratorAgent
from fsm_llm.agents.parallel_react import ParallelReactAgent
from fsm_llm.agents.plan_execute import PlanExecuteAgent
from fsm_llm.agents.reflexion import ReflexionAgent
from fsm_llm.agents.self_consistency import SelfConsistencyAgent
from fsm_llm.agents.semantic_memory import SemanticMemoryStore
from fsm_llm.agents.swarm import SwarmAgent
from fsm_llm.agents.tool_registries import CachingToolRegistry, RetryingToolRegistry
from fsm_llm.agents.tools import ToolRegistry, check_timeout
from fsm_llm.agents.verified_react import VerifiedReactAgent
from tests.conftest import MockLLM2Interface

HUGE = 10**5000

_BAD = [
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
    pytest.param(1.5, id="1.5"),
    pytest.param(1.0, id="1.0"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param("3", id="str"),
    pytest.param(None, id="None"),
    pytest.param([1], id="list"),
    pytest.param(10**400, id="10e400"),
    pytest.param(HUGE, id="5000-digits"),
    pytest.param(-HUGE, id="neg-5000-digits"),
]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_function(lambda q: q, name="echo", description="echo")
    return reg


class _Member:
    def run(self, *args, **kwargs):  # pragma: no cover - never run
        raise AssertionError


def _llm() -> MockLLM2Interface:
    return MockLLM2Interface()


def _memory(**kw):
    return SemanticMemoryStore(embed_fn=lambda texts: [[1.0]] * len(texts), **kw)


# (id, constructor taking the limit as a keyword, name, minimum, error type)
_SIBLINGS = [
    (
        "reflexion",
        lambda **kw: ReflexionAgent(tools=_registry(), llm_interface=_llm(), **kw),
        "max_reflections",
        1,
        AgentError,
    ),
    (
        "plan_execute",
        lambda **kw: PlanExecuteAgent(tools=_registry(), llm_interface=_llm(), **kw),
        "max_replans",
        0,
        AgentError,
    ),
    (
        "parallel_react",
        lambda **kw: ParallelReactAgent(tools=_registry(), llm_interface=_llm(), **kw),
        "max_parallel",
        1,
        AgentError,
    ),
    (
        "verified_react",
        lambda **kw: VerifiedReactAgent(tools=_registry(), llm_interface=_llm(), **kw),
        "max_verify_retries",
        0,
        AgentError,
    ),
    (
        "evaluator_optimizer",
        lambda **kw: EvaluatorOptimizerAgent(
            evaluation_fn=lambda output, context: None, llm_interface=_llm(), **kw
        ),
        "max_refinements",
        0,
        AgentError,
    ),
    (
        "maker_checker",
        lambda **kw: MakerCheckerAgent("make", "check", llm_interface=_llm(), **kw),
        "max_revisions",
        0,
        AgentError,
    ),
    (
        "debate",
        lambda **kw: DebateAgent(llm_interface=_llm(), **kw),
        "num_rounds",
        1,
        AgentError,
    ),
    (
        "adapt",
        lambda **kw: ADaPTAgent(llm_interface=_llm(), **kw),
        "max_depth",
        0,
        AgentError,
    ),
    (
        "orchestrator",
        lambda **kw: OrchestratorAgent(llm_interface=_llm(), **kw),
        "max_workers",
        1,
        AgentError,
    ),
    (
        "self_consistency_samples",
        lambda **kw: SelfConsistencyAgent(llm_interface=_llm(), **kw),
        "num_samples",
        1,
        AgentError,
    ),
    (
        "self_consistency_workers",
        lambda **kw: SelfConsistencyAgent(llm_interface=_llm(), **kw),
        "max_workers",
        1,
        AgentError,
    ),
    (
        "swarm",
        lambda **kw: SwarmAgent(agents={"a": _Member()}, entry_agent="a", **kw),
        "max_handoffs",
        0,
        AgentError,
    ),
    (
        "retrying_registry",
        lambda **kw: RetryingToolRegistry(**kw),
        "max_retries",
        0,
        ValueError,
    ),
    (
        "caching_registry",
        lambda **kw: CachingToolRegistry(**kw),
        "max_entries",
        1,
        ValueError,
    ),
    (
        "semantic_memory",
        lambda **kw: _memory(**kw),
        "max_entries",
        0,
        ValueError,
    ),
]
_IDS = [row[0] for row in _SIBLINGS]


# ``None`` means unbounded for these two, so it is a valid value, not a bad one.
_NONE_IS_VALID = {"caching_registry", "semantic_memory"}
_REFUSAL_CASES = [
    pytest.param(row, bad.values[0], id=f"{row[0]}-{bad.id}")
    for row in _SIBLINGS
    for bad in _BAD
    if not (bad.values[0] is None and row[0] in _NONE_IS_VALID)
]


class TestEverySiblingRefusesABadLimit:
    @pytest.mark.parametrize("row,bad", _REFUSAL_CASES)
    def test_refused_with_a_bounded_message(self, row, bad):
        _, make, name, _, error = row
        with pytest.raises(error) as info:
            make(**{name: bad})
        assert name in str(info.value)
        assert len(str(info.value)) < 300

    @pytest.mark.parametrize("row", [r for r in _SIBLINGS if r[0] in _NONE_IS_VALID])
    def test_none_stays_unbounded(self, row):
        _, make, name, _, _ = row
        assert make(**{name: None}) is not None


@pytest.mark.parametrize("row", _SIBLINGS, ids=_IDS)
class TestEverySiblingKeepsItsMinimum:
    def test_below_the_minimum_is_refused_not_clamped(self, row):
        _, make, name, minimum, error = row
        with pytest.raises(error, match=f"{name} must be an int from {minimum} to"):
            make(**{name: minimum - 1})

    def test_the_minimum_is_accepted(self, row):
        _, make, name, minimum, _ = row
        assert make(**{name: minimum}) is not None

    def test_the_largest_limit_is_accepted_and_one_more_refused(self, row):
        _, make, name, _, error = row
        assert make(**{name: Defaults.MAX_INT_LIMIT}) is not None
        with pytest.raises(error):
            make(**{name: Defaults.MAX_INT_LIMIT + 1})


class TestStoredValues:
    def test_an_integral_numpy_value_is_stored_as_a_plain_int(self):
        np = pytest.importorskip("numpy")
        agent = ReflexionAgent(
            tools=_registry(), llm_interface=_llm(), max_reflections=np.int64(2)
        )
        assert agent.max_reflections == 2
        assert type(agent.max_reflections) is int

    @pytest.mark.parametrize("cls_kwargs", [{"num_rounds": 3}, {"num_rounds": 1}])
    def test_debate_keeps_a_valid_value(self, cls_kwargs):
        assert (
            DebateAgent(llm_interface=_llm(), **cls_kwargs).num_rounds
            == (cls_kwargs["num_rounds"])
        )

    def test_zero_is_a_real_setting_where_it_means_none(self):
        assert (
            SwarmAgent(
                agents={"a": _Member()}, entry_agent="a", max_handoffs=0
            )._max_handoffs
            == 0
        )
        assert ADaPTAgent(llm_interface=_llm(), max_depth=0).max_depth == 0
        assert (
            MakerCheckerAgent(
                "m", "c", llm_interface=_llm(), max_revisions=0
            ).max_revisions
            == 0
        )
        assert (
            EvaluatorOptimizerAgent(
                evaluation_fn=lambda o, c: None, llm_interface=_llm(), max_refinements=0
            ).max_refinements
            == 0
        )


class TestRequireIntLimit:
    @pytest.mark.parametrize("bad", _BAD)
    def test_bad_values_raise_the_requested_error(self, bad):
        with pytest.raises(AgentError, match="max_x must be an int from 1 to"):
            require_int_limit("max_x", bad, minimum=1)
        with pytest.raises(ValueError, match="max_x"):
            require_int_limit("max_x", bad, minimum=1, error=ValueError)

    def test_the_message_names_the_type_and_a_short_value(self):
        with pytest.raises(AgentError) as info:
            require_int_limit("max_x", HUGE, minimum=1)
        assert "<int, 16610 bits>" in str(info.value)
        with pytest.raises(AgentError) as info:
            require_int_limit("max_x", -HUGE, minimum=1)
        assert len(str(info.value)) < 200

    def test_a_valid_value_comes_back_as_an_int(self):
        assert require_int_limit("n", 3, minimum=1) == 3
        assert require_int_limit("n", 0, minimum=0) == 0
        assert require_int_limit("n", Defaults.MAX_INT_LIMIT, minimum=0) == (
            Defaults.MAX_INT_LIMIT
        )


class TestBoundedRepr:
    def test_never_raises_and_stays_short(self):
        class _Broken:
            def __repr__(self):
                raise RuntimeError("no")

        assert bounded_repr(_Broken()) == "<_Broken>"
        assert bounded_repr("x" * 500).endswith("...")
        assert len(bounded_repr("x" * 500)) <= 43
        assert bounded_repr(HUGE) == "<int, 16610 bits>"
        assert bounded_repr(-(10**400)).startswith("<int,")
        assert bounded_repr(5) == "5"
        assert bounded_repr(True) == "True"


class TestCheckTimeoutMessageIsBounded:
    """Sibling of the limits: ``check_timeout`` formatted the value in full, so
    a 5000-digit ``timeout_s`` made the error path raise ``ValueError``."""

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(HUGE, id="5000-digits"),
            pytest.param(-HUGE, id="neg-5000-digits"),
            pytest.param(10**400, id="10e400"),
        ],
    )
    def test_a_huge_timeout_is_a_clean_tool_error(self, bad):
        from fsm_llm.agents.exceptions import ToolExecutionError

        with pytest.raises(ToolExecutionError) as info:
            check_timeout("slow", bad)
        assert len(str(info.value)) < 300
