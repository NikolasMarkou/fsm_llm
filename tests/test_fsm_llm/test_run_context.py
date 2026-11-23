"""The run context (plan ace7a8c8 step 21, roadmap 1.14, contract C12): ``RunContext``.

One frozen value given to a run: the interface or a factory, the log, the budgets, the cancel
token, the host callables and the parent run reference. ``derive`` narrows and never widens.
The module is lazy: ``import fsm_llm`` does not load it (the laziness table pins that).
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm.core.definitions import FSMError
from fsm_llm.core.names import NameRegistry
from fsm_llm.engine.context import CancelToken, RunCancelledError, RunContext
from fsm_llm.testing import ScriptedLLM


def test_the_module_under_test_is_this_checkout() -> None:
    root = Path(__file__).resolve().parents[2]
    assert Path(fsm_llm.__file__).resolve().is_relative_to(root / "src")


class TestRunContextFields:
    def test_a_bare_context_holds_nothing_and_a_fresh_token(self) -> None:
        context = RunContext()
        assert context.llm_interface is None
        assert context.llm_factory is None
        assert context.event_log is None
        assert context.max_steps is None
        assert context.max_seconds is None
        assert context.callables is None
        assert context.parent is None
        assert isinstance(context.cancel, CancelToken)
        assert not context.cancel.cancelled
        assert RunContext().cancel is not context.cancel

    def test_it_is_frozen(self) -> None:
        context = RunContext(max_steps=3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.max_steps = 9  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.parent = "x"  # type: ignore[misc]

    def test_it_holds_no_field_beyond_the_ones_with_a_consumer(self) -> None:
        names = {f.name for f in dataclasses.fields(RunContext)}
        assert names == {
            "llm_interface",
            "llm_factory",
            "event_log",
            "max_steps",
            "max_seconds",
            "cancel",
            "callables",
            "parent",
            "registry",
        }

    def test_unknown_keywords_are_refused(self) -> None:
        with pytest.raises(TypeError):
            RunContext(policy=object())  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            RunContext(handlers=object())  # type: ignore[call-arg]


class TestRunContextRefusals:
    def test_an_interface_and_a_factory_together_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"llm_interface.*llm_factory"):
            RunContext(llm_interface=ScriptedLLM(), llm_factory=ScriptedLLM)

    def test_the_interface_must_be_an_llm_interface(self) -> None:
        with pytest.raises(ValueError, match="LLMInterface"):
            RunContext(llm_interface=object())  # type: ignore[arg-type]

    def test_the_factory_must_be_callable(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            RunContext(llm_factory="a model")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
    def test_a_bad_step_budget_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            RunContext(max_steps=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -0.5, True, float("nan"), "3"])
    def test_a_bad_time_budget_is_refused(self, bad: object) -> None:
        with pytest.raises(ValueError, match="max_seconds"):
            RunContext(max_seconds=bad)  # type: ignore[arg-type]

    def test_the_token_and_the_registry_have_a_type(self) -> None:
        with pytest.raises(TypeError, match="CancelToken"):
            RunContext(cancel=threading.Event())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="NameRegistry"):
            RunContext(callables={"f": print})  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_a_blank_parent_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="parent"):
            RunContext(parent=bad)


class TestResolveLlm:
    def test_the_interface_is_returned_as_is(self) -> None:
        llm = ScriptedLLM()
        assert RunContext(llm_interface=llm).resolve_llm() is llm

    def test_a_factory_is_called_once_per_resolve_and_its_result_is_returned(
        self,
    ) -> None:
        made: list[ScriptedLLM] = []

        def factory() -> ScriptedLLM:
            made.append(ScriptedLLM())
            return made[-1]

        context = RunContext(llm_factory=factory)
        assert made == []  # building a context calls nothing
        first = context.resolve_llm()
        assert first is made[0] and len(made) == 1
        second = context.resolve_llm()
        assert second is made[1] and len(made) == 2  # one call per run, no caching

    def test_neither_gives_none(self) -> None:
        assert RunContext().resolve_llm() is None


class TestDerive:
    def test_a_child_shares_the_parents_members_and_names_its_parent(self) -> None:
        registry: NameRegistry[object] = NameRegistry("callables")
        llm = ScriptedLLM()
        parent = RunContext(
            llm_interface=llm,
            event_log="logs",
            max_steps=10,
            max_seconds=60,
            callables=registry,
        )
        child = parent.derive(parent="conv-1")
        assert child is not parent
        assert child.parent == "conv-1"
        assert child.llm_interface is llm
        assert child.event_log == "logs"
        assert child.callables is registry
        assert child.cancel is parent.cancel  # one cancel reaches every child
        assert (child.max_steps, child.max_seconds) == (10, 60)
        assert parent.parent is None  # the parent is untouched

    def test_a_limit_is_narrowed(self) -> None:
        parent = RunContext(max_steps=10, max_seconds=60.0)
        child = parent.derive(parent="p", max_steps=4, max_seconds=5.5)
        assert (child.max_steps, child.max_seconds) == (4, 5.5)
        assert (parent.max_steps, parent.max_seconds) == (10, 60.0)

    def test_the_same_limit_is_allowed(self) -> None:
        child = RunContext(max_steps=10, max_seconds=60).derive(
            parent="p", max_steps=10, max_seconds=60
        )
        assert (child.max_steps, child.max_seconds) == (10, 60)

    def test_a_limit_is_never_widened(self) -> None:
        parent = RunContext(max_steps=10, max_seconds=60)
        with pytest.raises(ValueError, match="max_steps"):
            parent.derive(parent="p", max_steps=11)
        with pytest.raises(ValueError, match="max_seconds"):
            parent.derive(parent="p", max_seconds=60.5)

    def test_a_limit_the_parent_lacks_takes_any_valid_value_and_none_keeps(
        self,
    ) -> None:
        parent = RunContext()
        child = parent.derive(parent="p", max_steps=7)
        assert (child.max_steps, child.max_seconds) == (7, None)
        # None means "keep", never "remove": a limit cannot be dropped by deriving.
        kept = RunContext(max_steps=5).derive(parent="p", max_steps=None)
        assert kept.max_steps == 5

    def test_a_bad_value_is_refused_like_a_constructor_value(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            RunContext(max_steps=5).derive(parent="p", max_steps=0)
        with pytest.raises(ValueError, match="max_seconds"):
            RunContext(max_seconds=5).derive(parent="p", max_seconds=float("nan"))

    def test_the_parent_reference_is_required_and_not_blank(self) -> None:
        with pytest.raises(TypeError):
            RunContext().derive()  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="parent"):
            RunContext().derive(parent=" ")

    def test_a_grandchild_narrows_again_and_keeps_the_token(self) -> None:
        root = RunContext(max_steps=9)
        child = root.derive(parent="a", max_steps=5)
        grandchild = child.derive(parent="b", max_steps=2)
        assert (grandchild.max_steps, grandchild.parent) == (2, "b")
        assert grandchild.cancel is root.cancel
        with pytest.raises(ValueError, match="max_steps"):
            child.derive(
                parent="b", max_steps=6
            )  # below the root's 9, above the child's 5


class TestCancelToken:
    def test_it_starts_clear_and_stays_set(self) -> None:
        token = CancelToken()
        assert token.cancelled is False
        token.raise_if_cancelled()  # no raise
        token.cancel()
        assert token.cancelled is True
        token.cancel()  # idempotent
        assert token.cancelled is True

    def test_raise_if_cancelled_raises_a_named_fsm_error(self) -> None:
        token = CancelToken()
        token.cancel()
        with pytest.raises(RunCancelledError) as info:
            token.raise_if_cancelled()
        assert isinstance(info.value, FSMError)
        # it is not the asyncio cancel: a token is a request checked between steps
        import asyncio

        assert not isinstance(info.value, asyncio.CancelledError)

    def test_a_cancel_from_another_thread_is_seen(self) -> None:
        token = CancelToken()
        started = threading.Event()
        done = threading.Event()

        def canceller() -> None:
            started.wait(5)
            token.cancel()
            done.set()

        thread = threading.Thread(target=canceller)
        thread.start()
        started.set()
        assert done.wait(5)
        thread.join(5)
        assert token.cancelled
        with pytest.raises(RunCancelledError):
            token.raise_if_cancelled()

    def test_many_threads_cancel_and_read_at_once(self) -> None:
        token = CancelToken()
        barrier = threading.Barrier(8)
        seen: list[bool] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            barrier.wait(5)
            if index % 2 == 0:
                token.cancel()
            observed = token.cancelled
            with lock:
                seen.append(observed)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert len(seen) == 8
        assert token.cancelled  # a set token is never cleared by a reader
