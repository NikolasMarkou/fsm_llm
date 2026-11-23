"""The reasoning FSMs as package data (plan baa37f40 step 1, D-009, roadmap E3.3 job A).

The nine strategies, the classifier and the orchestrator are ``fsm-llm/executable`` envelopes in
``src/fsm_llm/reasoning/strategies/*.json``. The move is behaviour neutral: ``body.fsm`` of each
file is the E0.1 fixture as decoded text (the fixtures are compared as text, never as bytes, so a
Windows checkout reads the same), the two orders the old tables carried are two explicit tuples,
and the deleted tables stay deleted.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import fsm_llm
from fsm_llm.core.definitions import FSMDefinition, definition_digest, fsm_id_of
from fsm_llm.core.verify.executable import load_executable
from fsm_llm.reasoning import constants, strategy_files, utilities
from fsm_llm.reasoning.constants import ReasoningType
from fsm_llm.reasoning.executables import strategy_id
from fsm_llm.reasoning.strategy_files import (
    FSM_NAMES,
    STRATEGY_ORDER,
    load_document,
    load_fsm,
    solution_keys,
    strategy_facts,
)
from tests import executables_corpus as corpus

REPO = Path(__file__).resolve().parents[2]
STRATEGIES_DIR = REPO / "src" / "fsm_llm" / "reasoning" / "strategies"

#: The order of the former ``ALL_REASONING_FSMS`` and of the former ``STRATEGIES``, written out.
_FSM_ORDER = [
    "orchestrator",
    "classifier",
    "simple_calculator",
    "analytical",
    "deductive",
    "inductive",
    "creative",
    "critical",
    "abductive",
    "analogical",
    "hybrid",
]
_STRATEGY_ORDER = [
    "simple_calculator",
    "analytical",
    "deductive",
    "inductive",
    "creative",
    "critical",
    "hybrid",
    "abductive",
    "analogical",
]


def test_the_module_under_test_is_this_checkout() -> None:
    assert Path(fsm_llm.__file__).resolve().is_relative_to(REPO / "src")
    assert Path(strategy_files.__file__).resolve().is_relative_to(REPO / "src")


class TestTheFiles:
    def test_every_name_has_one_file_and_no_file_is_an_orphan(self) -> None:
        on_disk = sorted(p.stem for p in STRATEGIES_DIR.glob("*.json"))
        assert on_disk == sorted(FSM_NAMES)
        assert len(on_disk) == len(set(on_disk))

    @pytest.mark.parametrize("name", _FSM_ORDER)
    def test_a_file_is_canonical_ascii_lf_text(self, name: str) -> None:
        raw = (STRATEGIES_DIR / f"{name}.json").read_bytes()
        assert raw.isascii()
        assert b"\r" not in raw
        text = raw.decode("utf-8")
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=True) + "\n"

    @pytest.mark.parametrize("name", _FSM_ORDER)
    def test_the_body_is_the_e0_1_fixture_as_decoded_text(self, name: str) -> None:
        fixture = corpus.fixture_path(f"reasoning/{name}").read_text(encoding="utf-8")
        assert corpus.render(load_fsm(name)) == fixture
        assert corpus.render(load_document(name)["body"]["fsm"]) == fixture

    @pytest.mark.parametrize("name", _FSM_ORDER)
    def test_the_identity_of_the_body_is_the_indexed_one(self, name: str) -> None:
        row = json.loads(corpus.INDEX_PATH.read_text(encoding="utf-8"))[
            f"reasoning/{name}"
        ]
        raw = load_fsm(name)
        definition = FSMDefinition(**raw)
        assert corpus.canonical_sha256(raw) == row["sha256"]
        assert fsm_id_of(definition) == row["fsm_id"]
        assert definition_digest(definition) == row["digest64"]

    @pytest.mark.parametrize("name", _FSM_ORDER)
    def test_a_file_is_a_valid_envelope_with_its_own_id(self, name: str) -> None:
        envelope, report = load_executable(load_document(name))
        assert envelope is not None and report.is_valid, report.errors
        assert envelope.id == strategy_id(name)
        assert envelope.kind == ("strategy" if name in _STRATEGY_ORDER else "fsm")

    def test_the_four_facts_sit_in_the_nine_strategy_files(self) -> None:
        for name in _STRATEGY_ORDER:
            facts = strategy_facts(name)
            assert set(facts) == {"description", "aliases", "results", "solution_key"}
            assert facts["description"].strip()
            assert all(a == a.lower().strip() and a for a in facts["aliases"])
            assert facts["solution_key"] in facts["results"]


class TestTheOrders:
    def test_the_two_orders_are_the_old_tables_orders(self) -> None:
        assert list(FSM_NAMES) == _FSM_ORDER
        assert list(STRATEGY_ORDER) == _STRATEGY_ORDER

    def test_the_orders_are_not_the_glob_order(self) -> None:
        # a loader that sorted the directory would silently reorder the solution keys
        assert list(STRATEGY_ORDER) != sorted(STRATEGY_ORDER)
        assert list(FSM_NAMES) != sorted(FSM_NAMES)

    def test_the_strategies_are_exactly_the_reasoning_types(self) -> None:
        assert set(STRATEGY_ORDER) == {t.value for t in ReasoningType}
        assert set(STRATEGY_ORDER) < set(FSM_NAMES)

    def test_the_solution_keys_follow_the_strategy_order(self) -> None:
        expected = tuple(strategy_facts(n)["solution_key"] for n in _STRATEGY_ORDER)
        assert solution_keys() == expected
        assert solution_keys()[0] == "calculation_result"


class TestTheLoader:
    def test_a_loaded_fsm_is_a_copy(self) -> None:
        first = load_fsm("orchestrator")
        first["states"].clear()
        first["handler_only_keys"].append("planted")
        assert load_fsm("orchestrator")["states"]
        assert "planted" not in load_fsm("orchestrator")["handler_only_keys"]

    def test_a_loaded_document_and_the_facts_are_copies(self) -> None:
        document = load_document("hybrid")
        document["body"]["strategy"]["aliases"].append("planted")
        facts = strategy_facts("hybrid")
        facts["results"].clear()
        assert "planted" not in strategy_facts("hybrid")["aliases"]
        assert strategy_facts("hybrid")["results"]

    @pytest.mark.parametrize(
        "name", ["nope", "", "../orchestrator", "orchestrator.json"]
    )
    def test_an_unknown_name_is_a_key_error_and_reads_nothing(self, name: str) -> None:
        with pytest.raises(KeyError):
            load_fsm(name)
        with pytest.raises(KeyError):
            load_document(name)

    def test_only_a_strategy_has_facts(self) -> None:
        for name in ("orchestrator", "classifier", "nope"):
            with pytest.raises(KeyError):
                strategy_facts(name)

    def test_get_reasoning_fsm_names_the_missing_one(self) -> None:
        with pytest.raises(KeyError, match="FSM definition not found: nope"):
            utilities.get_reasoning_fsm("nope")

    def test_every_read_names_its_encoding(self) -> None:
        tree = ast.parse(Path(strategy_files.__file__).read_text(encoding="utf-8"))
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"read_text", "open", "read_bytes"}
        ]
        assert reads, "the loader reads no file?"
        for call in reads:
            assert call.func.attr == "read_text"  # type: ignore[union-attr]
            assert [k.arg for k in call.keywords] == ["encoding"]
            assert ast.literal_eval(call.keywords[0].value) == "utf-8"


class TestTheDeletedTables:
    def test_the_python_dict_package_and_both_tables_are_gone(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            __import__("fsm_llm.reasoning.reasoning_modes")
        assert not hasattr(constants, "STRATEGIES")
        assert not hasattr(constants, "StrategySpec")
        assert not hasattr(utilities, "_TYPE_MAP")
        assert not (REPO / "src" / "fsm_llm" / "reasoning" / "reasoning_modes").exists()

    def test_the_spelling_table_is_built_per_call_not_kept(self) -> None:
        before = utilities.map_reasoning_type("math")
        assert before == "simple_calculator"
        assert utilities._spellings() is not utilities._spellings()
        utilities._spellings()["math"] = "planted"
        assert utilities.map_reasoning_type("math") == "simple_calculator"

    def test_the_description_table_is_built_per_call_not_kept(self) -> None:
        first = utilities.get_available_reasoning_types()
        first["analytical"] = "planted"
        assert utilities.get_available_reasoning_types()["analytical"] != "planted"
        assert list(first) == _STRATEGY_ORDER


class TestTheAnchorMoved:
    def test_the_d_054_anchor_lives_with_the_loader_of_the_orchestrator(self) -> None:
        text = Path(strategy_files.__file__).read_text(encoding="utf-8")
        assert "# DECISION plan-2026-10-01T093600-944e2692/D-054" in text

    def test_the_handler_only_keys_it_guards_are_in_the_file(self) -> None:
        keys = load_fsm("orchestrator")["handler_only_keys"]
        assert list(constants.ORCHESTRATOR_HANDLER_ONLY_KEYS) == keys


class TestThePackaging:
    def test_pyproject_ships_the_files_as_package_data(self) -> None:
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        assert '"fsm_llm.reasoning" = ["strategies/*.json"]' in text
