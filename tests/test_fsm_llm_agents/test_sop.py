"""SOPDefinition and SOPRegistry: registration, ``config_overrides`` validation, load errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_llm.agents.sop import SOPRegistry


class TestSopLoadErrors:
    """F-AI-19: a bad SOP file raised three unrelated types."""

    @pytest.mark.parametrize(
        ("name", "text"),
        [
            ("list.yaml", "- a\n- b\n"),
            ("empty.yaml", ""),
            ("noname.json", '{"description": "x"}'),
            ("bad.json", "{not json"),
        ],
    )
    def test_one_error_type_naming_the_file(self, tmp_path: Path, name: str, text: str):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            SOPRegistry().register_from_file(path)


class TestSOPConfigValidation:
    """Verify invalid SOP config_overrides fail at registration."""

    def test_invalid_config_raises_at_register(self):
        """ValueError raised for invalid config_overrides."""
        from fsm_llm.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(
            name="bad-sop",
            config_overrides={"max_iterations": -1},
        )
        with pytest.raises(ValueError, match="invalid config_overrides"):
            registry.register(sop)

    def test_valid_config_registers_ok(self):
        """Valid config_overrides register without error."""
        from fsm_llm.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(
            name="good-sop",
            config_overrides={"temperature": 0.3, "max_iterations": 5},
        )
        registry.register(sop)
        assert registry.has("good-sop")

    def test_empty_config_registers_ok(self):
        """Empty config_overrides register without validation."""
        from fsm_llm.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(name="empty-sop")
        registry.register(sop)
        assert registry.has("empty-sop")


class TestSOPRegistry:
    """Tests for SOPDefinition and SOPRegistry."""

    def test_sop_definition_creation(self):
        """SOPDefinition stores all fields."""
        from fsm_llm.agents.sop import SOPDefinition

        sop = SOPDefinition(
            name="test-sop",
            description="A test SOP",
            agent_pattern="react",
            task_template="Review: {code}",
        )
        assert sop.name == "test-sop"
        assert sop.agent_pattern == "react"

    def test_render_task(self):
        """render_task substitutes template variables."""
        from fsm_llm.agents.sop import SOPDefinition

        sop = SOPDefinition(
            name="review",
            task_template="Review this {language} code: {code}",
        )
        result = sop.render_task(language="Python", code="x=1")
        assert "Python" in result
        assert "x=1" in result

    def test_render_task_missing_var(self):
        """render_task raises on missing variable."""
        from fsm_llm.agents.sop import SOPDefinition

        sop = SOPDefinition(name="test", task_template="{missing}")
        with pytest.raises(ValueError, match="missing"):
            sop.render_task()

    def test_to_agent_config(self):
        """to_agent_config creates AgentConfig from SOP."""
        from fsm_llm.agents.sop import SOPDefinition

        sop = SOPDefinition(
            name="test",
            config_overrides={"temperature": 0.1, "max_iterations": 3},
        )
        config = sop.to_agent_config()
        assert config.temperature == 0.1
        assert config.max_iterations == 3

    def test_sop_serialization(self):
        """SOPDefinition roundtrips through dict."""
        from fsm_llm.agents.sop import SOPDefinition

        sop = SOPDefinition(name="test", description="Test SOP")
        data = sop.to_dict()
        restored = SOPDefinition.from_dict(data)
        assert restored.name == "test"
        assert restored.description == "Test SOP"

    def test_registry_register_and_get(self):
        """Registry stores and retrieves SOPs."""
        from fsm_llm.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        sop = SOPDefinition(name="my-sop", description="Test")
        registry.register(sop)

        assert registry.has("my-sop")
        assert registry.get("my-sop").description == "Test"

    def test_registry_list(self):
        """list_names returns sorted SOP names."""
        from fsm_llm.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        registry.register(SOPDefinition(name="beta"))
        registry.register(SOPDefinition(name="alpha"))
        assert registry.list_names() == ["alpha", "beta"]

    def test_registry_missing_raises(self):
        """get() raises KeyError for missing SOP."""
        from fsm_llm.agents.sop import SOPRegistry

        registry = SOPRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_load_from_json_file(self, tmp_path):
        """register_from_file loads JSON SOP."""
        from fsm_llm.agents.sop import SOPRegistry

        sop_file = tmp_path / "test.json"
        sop_file.write_text(
            json.dumps(
                {
                    "name": "json-sop",
                    "description": "From JSON",
                    "agent_pattern": "react",
                }
            ),
            encoding="utf-8",
        )

        registry = SOPRegistry()
        registry.register_from_file(sop_file)
        assert registry.has("json-sop")

    def test_register_directory_warns_once_per_failed_file_and_registers_the_rest(
        self, tmp_path
    ):
        """I-126: a broken SOP file is a warning to the caller, not only a log
        line; the good files still register and the return value is the count."""
        from fsm_llm.agents.sop import SOPRegistry

        for name in ("alpha", "gamma"):
            (tmp_path / f"{name}.json").write_text(
                json.dumps({"name": name, "agent_pattern": "react"}), encoding="utf-8"
            )
        (tmp_path / "beta.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "delta.yaml").write_text(
            "description: no name here\n", encoding="utf-8"
        )
        (tmp_path / "notes.txt").write_text(
            "ignored, not an SOP file\n", encoding="utf-8"
        )

        registry = SOPRegistry()
        with pytest.warns(UserWarning) as caught:
            count = registry.register_directory(tmp_path)

        assert count == 2
        assert sorted(registry.list_names()) == ["alpha", "gamma"]
        assert len(caught) == 2
        assert {w.filename for w in caught} == {__file__}
        text = "\n".join(str(w.message) for w in caught)
        assert "beta.json" in text and "delta.yaml" in text

    def test_register_directory_of_good_files_warns_nothing(self, tmp_path):
        import warnings

        from fsm_llm.agents.sop import SOPRegistry

        (tmp_path / "alpha.json").write_text(
            json.dumps({"name": "alpha"}), encoding="utf-8"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert SOPRegistry().register_directory(tmp_path) == 1

    def test_load_builtin_sops(self):
        """load_builtin_sops returns registry with 3 SOPs."""
        from fsm_llm.agents.sop import load_builtin_sops

        registry = load_builtin_sops()
        assert len(registry) == 3
        assert registry.has("code-review")
        assert registry.has("summarize")
        assert registry.has("data-extraction")

    def test_registry_remove(self):
        """remove() deletes SOP and returns True."""
        from fsm_llm.agents.sop import SOPDefinition, SOPRegistry

        registry = SOPRegistry()
        registry.register(SOPDefinition(name="temp"))
        assert registry.remove("temp")
        assert not registry.has("temp")
        assert not registry.remove("temp")  # Already removed


class TestSOPsNeedNoModel:
    """Review gate 1, concern 1: a template registry needs no ``LLM_MODEL``.

    The root autouse fixture sets ``LLM_MODEL`` for every test, so each test
    here deletes it. ``tests/test_fsm_llm_regression/test_no_model_paths.py``
    sweeps the other public entry points the same way.
    """

    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_MODEL", raising=False)

    def test_register_a_sop_that_names_no_model(self):
        from fsm_llm.agents.sop import SOPDefinition

        registry = SOPRegistry()
        registry.register(
            SOPDefinition(name="s", config_overrides={"temperature": 0.3})
        )
        assert registry.get("s").config_overrides == {"temperature": 0.3}

    def test_the_placeholder_never_reaches_the_stored_sop(self):
        from fsm_llm.agents.sop import SOPDefinition

        registry = SOPRegistry()
        sop = SOPDefinition(name="s", config_overrides={"temperature": 0.3})
        registry.register(sop)
        assert "model" not in registry.get("s").config_overrides
        assert "model" not in sop.to_dict()["config_overrides"]

    def test_the_builtin_sops_load(self):
        from fsm_llm.agents.sop import load_builtin_sops

        assert set(load_builtin_sops().list_names()) >= {"code-review", "summarize"}

    def test_a_bad_key_is_still_refused_without_a_model(self):
        from fsm_llm.agents.sop import SOPDefinition

        with pytest.raises(
            ValueError, match=r"(?s)invalid config_overrides.*max_iteration"
        ):
            SOPRegistry().register(
                SOPDefinition(name="s", config_overrides={"max_iteration": 3})
            )

    def test_a_bad_value_is_still_refused_without_a_model(self):
        from fsm_llm.agents.sop import SOPDefinition

        with pytest.raises(ValueError, match="invalid config_overrides"):
            SOPRegistry().register(
                SOPDefinition(name="s", config_overrides={"temperature": 9.0})
            )

    def test_to_agent_config_takes_the_injected_interface_as_the_model(self):
        from fsm_llm.agents.sop import SOPDefinition
        from fsm_llm.testing import ScriptedLLM

        double = ScriptedLLM()
        double.model = "provider/own"
        sop = SOPDefinition(name="s", config_overrides={"temperature": 0.3})
        config = sop.to_agent_config(llm_interface=double)
        assert (config.model, config.temperature) == ("provider/own", 0.3)

    def test_to_agent_config_with_a_named_model_needs_nothing_else(self):
        from fsm_llm.agents.sop import SOPDefinition

        sop = SOPDefinition(name="s", config_overrides={"model": "provider/own"})
        assert sop.to_agent_config().model == "provider/own"

    def test_to_agent_config_for_running_still_needs_a_model(self):
        from fsm_llm import ModelNotConfiguredError
        from fsm_llm.agents.sop import SOPDefinition

        with pytest.raises(ModelNotConfiguredError):
            SOPDefinition(name="s").to_agent_config()
