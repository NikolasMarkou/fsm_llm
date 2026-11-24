"""The shipped pattern files and their loader (plan a555dcfe step 9, D-015).

``agents/patterns/<name>.json`` are executable documents of kind ``pattern``; ``pattern_files``
reads them as package data. These rows hold the loader contract (order, canonical text, deep
copies, one parse), what a file must say beside the FSM (names that equal the library and the
constants, no description text that carries a task), that the wheel ships the directory, and that
``import fsm_llm.agents`` does not load the loader or the adapter.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from fsm_llm.agents import pattern_files as pf
from fsm_llm.agents.constants import FRAMEWORK_ONLY_KEYS, ContextKeys, Defaults
from fsm_llm.agents.handlers.library import LIBRARY
from fsm_llm.constants import FIELD_PROMPT_CONTEXT_LABEL
from fsm_llm.core.definitions.executable import PatternEnvelope
from fsm_llm.core.definitions.executable_pattern import (
    APPROVAL_GATE_HANDLER,
    APPROVAL_STATE,
)
from fsm_llm.engine.executables.namespaces import HANDLERS

REPO = Path(__file__).resolve().parents[2]
PATTERNS = REPO / "src" / "fsm_llm" / "agents" / "patterns"


def _doc(name: str = "react") -> dict:
    return pf.load_document(name)


class TestTheLoader:
    def test_the_order_names_exactly_the_files_on_disk(self):
        on_disk = sorted(p.stem for p in PATTERNS.glob("*.json"))

        assert sorted(pf.PATTERN_ORDER) == on_disk
        assert len(set(pf.PATTERN_ORDER)) == len(pf.PATTERN_ORDER)

    @pytest.mark.parametrize("name", pf.PATTERN_ORDER)
    def test_every_file_is_canonical_text(self, name):
        raw = (PATTERNS / f"{name}.json").read_bytes()
        text = raw.decode("utf-8")
        parsed = json.loads(text)

        assert text == json.dumps(parsed, indent=2, ensure_ascii=True) + "\n"
        assert b"\r" not in raw and raw.isascii()

    @pytest.mark.parametrize("name", pf.PATTERN_ORDER)
    def test_every_file_loads_and_is_a_pattern_envelope_with_its_own_id(self, name):
        envelope = pf.load_envelope(name)

        assert isinstance(envelope, PatternEnvelope)
        assert envelope.id == name
        assert pf.load_envelope(name) is envelope  # parsed once per process

    def test_a_document_is_a_deep_copy_each_time(self):
        first = _doc()
        first["body"]["fsm"]["states"]["think"]["purpose"] = "changed"
        first["body"]["options"]["context_keys"].append("x")

        again = _doc()

        assert again["body"]["fsm"]["states"]["think"]["purpose"] != "changed"
        assert "x" not in again["body"]["options"]["context_keys"]

    @pytest.mark.parametrize("name", ["", "nope", "../react", "react.json", "REACT"])
    def test_an_unknown_name_is_a_key_error_and_never_a_path(self, name):
        with pytest.raises(KeyError):
            pf.load_document(name)
        with pytest.raises(KeyError):
            pf.load_envelope(name)


class TestWhatTheReactFileSays:
    def test_it_binds_the_library_by_name_in_the_registration_order(self):
        document = _doc()
        bound = [b["handler"] for b in document["body"]["handlers"]]

        assert bound == [
            "agents.tool_executor",
            "agents.think_turn_limiter",
            "agents.think_fresh_keys",
            "agents.feedback_consumed",
            "agents.approval_fresh_keys",
            "agents.approval_gate",
        ]
        assert set(bound) == set(document["requires"]["handlers"])
        assert set(bound) <= {name for name, _ in LIBRARY}
        assert all(name in HANDLERS for name in bound)

    def test_the_hitl_claim_is_the_gate_binding_and_the_gate_state(self):
        document = _doc()

        assert "hitl" in document["capabilities"]
        assert APPROVAL_GATE_HANDLER in document["requires"]["handlers"]
        assert APPROVAL_STATE not in document["body"]["fsm"]["states"]
        added = [
            op["path"]
            for op in document["body"]["variants"]["approval"]
            if op["op"] == "add"
        ]
        assert f"/states/{APPROVAL_STATE}" in added

    def test_the_description_never_carries_a_task_and_the_task_is_the_only_input(self):
        document = _doc()

        assert document["contract"]["inputs"] == ["task"]
        assert "{{" not in document["description"]
        assert "{{" not in document["body"]["fsm"]["description"]
        assert document["variables"] == ["think_instructions", "context_names"]

    def test_the_literals_equal_the_constants_they_stand_for(self):
        fsm = _doc()["body"]["fsm"]
        classify = next(
            op
            for op in _doc()["body"]["variants"]["classification"]
            if op["op"] == "add"
        )["value"][0]
        should = fsm["states"]["think"]["field_extractions"][2]

        assert fsm["handler_only_keys"] == list(FRAMEWORK_ONLY_KEYS)
        assert classify["fallback_intent"] == ContextKeys.NO_TOOL
        assert classify["confidence_threshold"] == Defaults.TOOL_CONFIDENCE_THRESHOLD
        assert (
            f"'{FIELD_PROMPT_CONTEXT_LABEL}' context"
            in should["extraction_instructions"]
        )

    def test_the_approval_and_classification_options_are_not_variants_by_name(self):
        document = _doc()

        assert "approval" not in document["body"]["options_schema"]["properties"]
        assert list(document["body"]["variants"]) == ["approval", "classification"]
        assert "classification" in document["body"]["options_schema"]["properties"]

    def test_a_json_file_holds_no_decision_anchor(self):
        for path in PATTERNS.glob("*.json"):
            assert "DECISION" not in path.read_text(encoding="utf-8"), path.name


class TestThePackageShipsTheDirectory:
    def test_the_package_data_lists_the_patterns_glob(self):
        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        data = config["tool"]["setuptools"]["package-data"]

        assert data["fsm_llm.agents"] == ["patterns/*.json"]

    def test_the_wheel_expectation_lists_it_too(self):
        text = (REPO / "tests" / "test_packaging.py").read_text(encoding="utf-8")

        assert '"agents/patterns/*.json"' in text

    def test_a_non_python_data_file_has_no_init_module_beside_it(self):
        assert not (PATTERNS / "__init__.py").exists()


class TestImportingTheAgentsPackageIsLazy:
    def test_the_loader_and_the_adapter_are_not_loaded_by_import(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
            "import fsm_llm.agents\n"
            "for name in ('fsm_llm.agents.pattern_files', 'fsm_llm.agents.executables'):\n"
            "    assert name not in sys.modules, name\n"
            "fsm_llm.agents.ReactAgent  # the class is there; the file is read on the first agent\n"
        )
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert done.returncode == 0, done.stderr

    def test_the_first_agent_reads_the_file(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
            "import fsm_llm.agents as a\n"
            "from fsm_llm.tools import ToolRegistry\n"
            "r = ToolRegistry(); r.register_function(lambda p: 'ok', name='t', description='d')\n"
            "a.ReactAgent(r, llm_interface=object())\n"
            "assert 'fsm_llm.agents.pattern_files' in sys.modules\n"
            "assert 'fsm_llm.agents.executables' not in sys.modules\n"
        )
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert done.returncode == 0, done.stderr
