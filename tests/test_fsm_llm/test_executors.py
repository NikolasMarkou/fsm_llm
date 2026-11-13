"""State executors and kind dispatch in the turn pipeline (plan d454fd62 step 53).

Row 0.14: the turn pipeline has no kind-specific branch. At the one Pass-1 seam
it asks ``fsm_llm.executors.executor_for(state)`` for the executor of the
state's kind. These tests pin the AST ban (no kind config read in the engine
files), the executor table (``synthesize`` is passive: the conversation
executor), the three hooks (``extract``, ``cleans_extracted``,
``after_transition``) and the dispatch through a real turn. The byte equality of
every request and result is held by ``test_kind_equality.py`` and
``test_completion_state.py``; this file only adds what the move is about.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fsm_llm import executors as ex
from fsm_llm import state_kinds as sk
from fsm_llm.definitions import DataExtractionResponse, State
from fsm_llm.testing import final_reply
from tests.test_fsm_llm.test_completion_state import (
    _RESULT,
    _api,
    _context,
    _ScriptedLLM,
    _start,
    _to_call_model,
    _tool_fsm,
)

_SRC = Path(ex.__file__).parent
_ENGINE_FILES = (
    "pipeline.py",
    "api.py",
    "fsm.py",
    "flow_analysis.py",
    "synthesis/dryrun.py",
)
_BANNED_ATTRIBUTES = {"completion", "synthesize"}
_BANNED_NAMES = {"CompletionStateConfig", "SynthesizeConfig"}


def _kind_config_mentions(source: str) -> list[str]:
    """Every read of a kind config in *source*: `.completion`, `.synthesize`, or
    a name or import of `CompletionStateConfig` / `SynthesizeConfig`."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRIBUTES:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            found.append(f"line {node.lineno}: {node.id}")
        elif isinstance(node, ast.alias) and node.name in _BANNED_NAMES:
            found.append(f"import {node.name}")
    return found


def _state(**fields: Any) -> State:
    base: dict[str, Any] = {
        "id": "s",
        "description": "d",
        "purpose": "p",
        "transitions": [{"target_state": "t", "description": "go"}],
    }
    return State(**{**base, **fields})


_COMPLETION = {"tools": [{"type": "function", "function": {"name": "f"}}]}
_SYNTHESIZE = {"goal": "g", "return_keys": ["k"], "on_failure": "t"}


class TestNoKindBranchInTheEngine:
    """The roadmap done-when: the pipeline and the files around it read no kind
    config; only an executor (or a registry hook) does."""

    @pytest.mark.parametrize("name", _ENGINE_FILES)
    def test_the_file_reads_no_kind_config(self, name: str):
        source = (_SRC / name).read_text(encoding="utf-8")
        assert _kind_config_mentions(source) == []

    def test_the_scan_sees_a_violation(self):
        """Guards the guard: each banned shape is found in a snippet."""
        snippet = (
            "from x import CompletionStateConfig\n"
            "a = state.completion\n"
            "b = state.synthesize\n"
            "c = SynthesizeConfig\n"
        )
        assert len(_kind_config_mentions(snippet)) == 4

    def test_the_completion_turn_left_the_pipeline(self):
        from fsm_llm.pipeline import MessagePipeline

        assert not hasattr(MessagePipeline, "_execute_completion_turn")
        assert hasattr(ex.CompletionExecutor, "turn")


# The retained conversation Pass-1 machinery of ``MessagePipeline`` (step 56,
# D-015): not delegates but the real bodies the conversation executor wraps; they
# move into ``engine/kinds/conversation/`` with the layout split (step 80). Until
# then ONE path reaches them: ``ConversationExecutor``. Do NOT call one from the
# kind-agnostic part of the pipeline: that re-introduces a conversation branch at
# the seam the executors exist to own.
_CONVERSATION_PASS_1 = frozenset(
    {
        "_extract_after_transition",
        "_execute_data_extraction",
        "_build_field_configs_from_state",
        "_bulk_extract_from_instructions",
        "_execute_field_extractions",
        "_validate_field_extraction",
        "_execute_classification_extractions",
        "_handler_only_keys",
    }
)


def _readers_of(source: str, names: frozenset[str]) -> dict[str, set[str]]:
    """``{method: names it reads}``: every ``MessagePipeline`` method of *source*
    that reaches one of *names* through ``self.<name>``; a read at module level
    or in another class is keyed ``"<module>"``."""
    found: dict[str, set[str]] = {}
    tree = ast.parse(source)
    for top in tree.body:
        owners = (
            [(f.name, f) for f in top.body if isinstance(f, ast.FunctionDef)]
            if isinstance(top, ast.ClassDef) and top.name == "MessagePipeline"
            else [("<module>", top)]
        )
        for owner, node in owners:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in names:
                    found.setdefault(owner, set()).add(sub.attr)
    return found


class TestConversationMachineryIsReachedThroughTheExecutor:
    def test_the_kind_agnostic_pipeline_never_calls_it(self):
        readers = _readers_of(
            (_SRC / "pipeline.py").read_text(encoding="utf-8"), _CONVERSATION_PASS_1
        )
        outside = {
            owner: sorted(names)
            for owner, names in readers.items()
            if owner not in _CONVERSATION_PASS_1
        }
        assert outside == {}

    def test_no_other_module_reads_it(self):
        readers = {
            str(p.relative_to(_SRC)): _readers_of(
                p.read_text(encoding="utf-8"), _CONVERSATION_PASS_1
            )
            for p in _SRC.rglob("*.py")
            if p.name not in {"pipeline.py", "executors.py"}
        }
        assert {path: found for path, found in readers.items() if found} == {}

    def test_the_conversation_executor_is_the_reader(self):
        readers = _readers_of(
            (_SRC / "executors.py").read_text(encoding="utf-8"), _CONVERSATION_PASS_1
        )
        assert readers == {
            "<module>": {"_execute_data_extraction", "_extract_after_transition"}
        }

    def test_the_scan_sees_a_violation(self):
        snippet = (
            "class MessagePipeline:\n"
            "    def _run_turn(self):\n"
            "        return self._execute_data_extraction()\n"
            "    def _execute_data_extraction(self):\n"
            "        return self._bulk_extract_from_instructions()\n"
        )
        assert _readers_of(snippet, _CONVERSATION_PASS_1) == {
            "_run_turn": {"_execute_data_extraction"},
            "_execute_data_extraction": {"_bulk_extract_from_instructions"},
        }


class TestExecutorModule:
    def _imports(self) -> list[str]:
        tree = ast.parse((_SRC / "executors.py").read_text(encoding="utf-8"))
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                found.append(("." * node.level) + (node.module or ""))
        return found

    def test_it_never_imports_the_pipeline(self):
        """The pipeline imports the executors; an import back is a cycle."""
        for name in self._imports():
            assert name.split(".")[-1] != "pipeline", name

    def test_it_imports_core_modules_only(self):
        stdlib = set(__import__("sys").stdlib_module_names)
        core = {
            "constants",
            "context",
            "definitions",
            "llm",
            "logging",
            "state_kinds",
        }
        for name in self._imports():
            if name.startswith("."):
                assert name.lstrip(".") in core, name
            else:
                assert name.split(".")[0] in stdlib | {"pydantic"}, name

    def test_the_package_init_does_not_load_it(self):
        tree = ast.parse((_SRC / "__init__.py").read_text(encoding="utf-8"))
        names = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("executors" in n for n in names)

    def test_every_name_in_all_exists(self):
        for name in ex.__all__:
            assert hasattr(ex, name), name


class TestExecutorTable:
    def test_each_kind_has_its_executor(self):
        assert type(ex.executor_for(_state())) is ex.ConversationExecutor
        assert (
            type(ex.executor_for(_state(completion=_COMPLETION)))
            is ex.CompletionExecutor
        )

    def test_synthesize_is_passive_the_conversation_executor(self):
        """D-002 of plan 2cefeb61: a bare converse on a synthesize state stays
        BLOCKED, so the kind shares the conversation executor."""
        state = _state(synthesize=_SYNTHESIZE)
        assert sk.kind_of(state).name == "synthesize"
        assert ex.executor_for(state) is ex.executor_for(_state())

    def test_cleans_extracted_is_false_only_for_completion(self):
        assert ex.ConversationExecutor().cleans_extracted is True
        assert ex.CompletionExecutor().cleans_extracted is False
        assert ex.executor_for(_state(synthesize=_SYNTHESIZE)).cleans_extracted is True

    def test_a_kind_without_an_executor_runs_as_conversation(self, monkeypatch):
        marker = sk.KindSpec(name="marker", config_field="marker")
        monkeypatch.setattr(sk, "KINDS", (*sk.KINDS, marker))
        state = SimpleNamespace(completion=None, synthesize=None, marker={"x": 1})
        assert sk.kind_of(state).name == "marker"
        assert type(ex.executor_for(state)) is ex.ConversationExecutor

    def test_register_executor_adds_and_replaces(self, monkeypatch):
        monkeypatch.setattr(ex, "_EXECUTORS", dict(ex._EXECUTORS))
        marker = sk.KindSpec(name="marker", config_field="marker")
        monkeypatch.setattr(sk, "KINDS", (*sk.KINDS, marker))
        state = SimpleNamespace(completion=None, synthesize=None, marker={"x": 1})
        mine = ex.StateExecutor()
        ex.register_executor("marker", mine)
        assert ex.executor_for(state) is mine
        other = ex.StateExecutor()
        ex.register_executor("marker", other)
        assert ex.executor_for(state) is other

    def test_the_base_hooks(self):
        base = ex.StateExecutor()
        assert base.cleans_extracted is True
        with pytest.raises(NotImplementedError):
            base.extract(None, None, "m", "c", _state())  # type: ignore[arg-type]
        response = DataExtractionResponse(extracted_data={}, confidence=1.0)
        # after_transition of the base (and of completion) runs nothing: a
        # pipeline of None would raise on any call.
        for executor in (base, ex.CompletionExecutor()):
            assert (
                executor.after_transition(
                    None,
                    None,
                    "m",
                    "c",
                    response,
                    "a",
                    _state(),  # type: ignore[arg-type]
                )
                is None
            )


class _Recorder(ex.CompletionExecutor):
    """The real completion executor, recording each hook call."""

    def __init__(self, log: list[tuple[str, str]], cleans: bool = False) -> None:
        self.log = log
        self.cleans_extracted = cleans

    def extract(self, pipeline, instance, user_message, conversation_id, state):
        self.log.append(("extract", state.id))
        return super().extract(pipeline, instance, user_message, conversation_id, state)

    def after_transition(self, *args):
        self.log.append(("after_transition", args[-1].id))
        return super().after_transition(*args)


class TestDispatchThroughATurn:
    def _run(self, monkeypatch, executor, replies):
        monkeypatch.setattr(ex, "_EXECUTORS", dict(ex._EXECUTORS))
        ex.register_executor("completion", executor)
        llm = _ScriptedLLM(*replies)
        api = _api(_tool_fsm(), llm)
        conv_id = _start(api)
        _to_call_model(api, conv_id)
        return api, conv_id

    def test_the_pipeline_asks_the_state_kind_executor(self, monkeypatch):
        log: list[tuple[str, str]] = []
        api, conv_id = self._run(monkeypatch, _Recorder(log), [final_reply("5")])
        # arriving in the completion state runs the NEW state's after_transition
        # and no call; the next turn runs its extract.
        assert log == [("after_transition", "call_model")]
        api.advance(conv_id)
        assert log == [
            ("after_transition", "call_model"),
            ("extract", "call_model"),
        ]
        assert _context(api, conv_id)[_RESULT]["text"] == "5"

    def test_the_flag_decides_whether_the_result_is_cleaned(self, monkeypatch):
        """D-021: the completion result commits as is (a None text survives);
        an executor that cleans drops it, which is the only difference."""
        reply = final_reply(None)
        api, conv_id = self._run(monkeypatch, _Recorder([], cleans=False), [reply])
        api.advance(conv_id)
        assert "text" in _context(api, conv_id)[_RESULT]
        assert _context(api, conv_id)[_RESULT]["text"] is None

        api, conv_id = self._run(monkeypatch, _Recorder([], cleans=True), [reply])
        api.advance(conv_id)
        assert "text" not in _context(api, conv_id)[_RESULT]

    def test_a_cut_completion_reply_is_recorded_by_the_executor(self, monkeypatch):
        api, conv_id = self._run(
            monkeypatch,
            ex.CompletionExecutor(),
            [final_reply("part", finish_reason="length")],
        )
        api.advance(conv_id)
        assert _context(api, conv_id)["_cut_off"] == {"fields": [_RESULT]}
