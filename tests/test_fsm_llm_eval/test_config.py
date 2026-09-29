"""Tests for fsm_llm.eval.config: defaults, layering, validation, model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsm_llm.constants import DEFAULT_LLM_MODEL, ENV_LLM_MODEL
from fsm_llm.eval import (
    EvalConfig,
    EvalConfigError,
    load_config,
    merge_config,
    resolve_model,
)


def _write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


class TestDefaults:
    def test_every_field_has_a_default(self):
        config = EvalConfig()
        assert config.model is None
        assert config.workers == 4
        assert config.timeout == 120
        assert config.output_root == "evaluation"
        assert config.output_dir is None
        assert config.fail_under is None
        assert config.examples_dir == "examples"
        assert config.python is None
        assert config.example_timeouts == {}

    def test_merge_of_nothing_is_the_defaults(self):
        assert merge_config() == EvalConfig()
        assert merge_config(None, {}) == EvalConfig()


class TestLayering:
    def test_later_layers_win(self):
        config = merge_config({"workers": 2, "timeout": 9}, {"workers": 8})
        assert config.workers == 8
        assert config.timeout == 9

    def test_file_then_flags_precedence(self, tmp_path: Path):
        layer = load_config(
            _write_json(tmp_path / "c.json", {"workers": 2, "model": "a"})
        )
        config = merge_config(layer, {"model": "b"})
        assert (config.workers, config.model) == (2, "b")

    def test_tables_merge_key_by_key(self):
        config = merge_config(
            {"example_timeouts": {"a/x": 10, "a/y": 20}},
            {"example_timeouts": {"a/y": 30}},
        )
        assert config.example_timeouts == {"a/x": 10, "a/y": 30}


class TestValidation:
    @pytest.mark.parametrize(
        "layer",
        [
            {"wrokers": 2},
            {"workers": 0},
            {"timeout": 0},
            {"fail_under": 101},
            {"fail_under": -1},
            {"example_timeouts": {"a/x": 0}},
            {"workers": "many"},
        ],
    )
    def test_bad_layers_raise_config_error(self, layer):
        with pytest.raises(EvalConfigError):
            merge_config(layer)

    def test_load_rejects_unknown_key_naming_the_file(self, tmp_path: Path):
        path = _write_json(tmp_path / "bad.json", {"modle": "x"})
        with pytest.raises(EvalConfigError, match=r"bad\.json"):
            load_config(path)

    def test_load_rejects_non_object(self, tmp_path: Path):
        with pytest.raises(EvalConfigError, match="JSON object"):
            load_config(_write_json(tmp_path / "list.json", [1, 2]))

    def test_load_rejects_invalid_json(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(EvalConfigError, match="not JSON"):
            load_config(path)

    def test_load_rejects_missing_file(self, tmp_path: Path):
        with pytest.raises(EvalConfigError, match="cannot read"):
            load_config(tmp_path / "missing.json")

    def test_load_returns_only_the_keys_set(self, tmp_path: Path):
        path = _write_json(tmp_path / "c.json", {"timeout": 60})
        assert load_config(path) == {"timeout": 60}


class TestResolveModel:
    def test_config_model_wins(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        assert resolve_model(EvalConfig(model=" cfg-model ")) == "cfg-model"

    def test_env_when_config_unset_or_blank(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "env-model")
        assert resolve_model(EvalConfig()) == "env-model"
        assert resolve_model(EvalConfig(model="  ")) == "env-model"

    def test_default_when_nothing_set(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_MODEL, "")
        assert resolve_model(EvalConfig()) == DEFAULT_LLM_MODEL
        monkeypatch.delenv(ENV_LLM_MODEL)
        assert resolve_model(EvalConfig()) == DEFAULT_LLM_MODEL
