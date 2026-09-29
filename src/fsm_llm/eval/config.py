"""
Evaluation settings: one validated model, JSON config files, layered merging.

Every setting has a default, so ``EvalConfig()`` is a complete configuration.
Settings come in layers, later layers winning: built-in defaults, a config
embedded in a dataset, a ``--config FILE``, then explicit CLI flags. Each layer
is a plain dict of only the keys it sets; ``merge_config`` folds them and
validates once. Unknown keys are rejected, so a typo is an error rather than a
silently ignored setting.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    field_validator,
)

from fsm_llm.constants import DEFAULT_LLM_MODEL, ENV_LLM_MODEL

from .constants import (
    DEFAULT_EXAMPLES_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TIMEOUT,
    DEFAULT_TRIALS,
    DEFAULT_WORKERS,
)
from .exceptions import EvalConfigError

#: Fields whose dict values are merged key by key across layers, not replaced.
_TABLE_FIELDS = (
    "example_inputs",
    "example_timeouts",
    "category_timeouts",
    "llm_kwargs",
)

#: ``API`` arguments that ``llm_kwargs`` may not set: the evaluation owns them
#: (the first six), or they take Python objects a JSON config cannot express
#: (``handlers``, ``transition_config``, ``session_store``).
_RESERVED_LLM_KWARGS = frozenset(
    {
        "fsm_definition",
        "definition",
        "path",
        "llm_interface",
        "model",
        "temperature",
        "max_tokens",
        "handlers",
        "transition_config",
        "session_store",
    }
)


class EvalConfig(BaseModel):
    """Settings shared by the evaluation commands (all optional).

    ``model`` ``None`` means ``$LLM_MODEL``, else the framework default, read
    when the run starts (:func:`resolve_model`). ``python`` ``None`` means the
    running interpreter. The three table fields add to or override the
    built-in repository tables in :mod:`fsm_llm.eval.constants`.

    Read by conversation cases only: ``trials``, and the LLM settings
    ``temperature`` and ``max_tokens`` (``None`` keeps the framework default)
    and ``llm_kwargs``, passed as extra keyword arguments to ``fsm_llm.API``
    (litellm options such as ``api_base`` or ``api_key``, or ``API`` options
    such as ``max_history_size``). ``llm_kwargs`` merges key by key across
    layers and may not set the FSM, model, temperature or max tokens. None of
    the LLM settings apply when an ``llm_interface_factory`` is given.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    workers: int = Field(default=DEFAULT_WORKERS, ge=1)
    timeout: int = Field(default=DEFAULT_TIMEOUT, ge=1)
    output_root: str = DEFAULT_OUTPUT_ROOT
    output_dir: str | None = None
    fail_under: float | None = Field(default=None, ge=0, le=100)
    examples_dir: str = DEFAULT_EXAMPLES_DIR
    python: str | None = None
    category: str | None = None
    name_filter: str | None = None
    example_inputs: dict[str, str] = Field(default_factory=dict)
    example_timeouts: dict[str, PositiveInt] = Field(default_factory=dict)
    category_timeouts: dict[str, PositiveInt] = Field(default_factory=dict)
    trials: int = Field(default=DEFAULT_TRIALS, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("llm_kwargs")
    @classmethod
    def _no_reserved_llm_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(_RESERVED_LLM_KWARGS & set(value))
        if reserved:
            raise ValueError(
                f"llm_kwargs may not set {reserved}; use the dedicated settings"
            )
        return value


def _validated(layer: Mapping[str, Any], source: str) -> EvalConfig:
    """Validate one layer, turning pydantic's error into ``EvalConfigError``."""
    try:
        return EvalConfig.model_validate(dict(layer))
    except ValidationError as exc:
        raise EvalConfigError(
            f"invalid evaluation config ({source}): {exc}",
            details={"source": source},
        ) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config file and return the layer it sets.

    Interface contract (callers: the CLI ``--config`` option and any script):
        - The file must hold one JSON object whose keys are ``EvalConfig``
          fields; it is validated on its own, so errors name the file.
        - Returns only the keys the file sets (a layer for ``merge_config``).
        - Raises ``EvalConfigError`` for a missing or unreadable (or non-UTF-8)
          file, invalid JSON, a non-object, an unknown key, or a bad value.
    """
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalConfigError(f"cannot read config file {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvalConfigError(f"config file {config_path} is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalConfigError(f"config file {config_path} must hold a JSON object")
    _validated(data, str(config_path))
    return data


def merge_config(*layers: Mapping[str, Any] | None) -> EvalConfig:
    """Fold config layers over the defaults, later layers winning.

    Interface contract (callers: both CLI commands and the Python API):
        - Each layer is a mapping of ``EvalConfig`` field names to values;
          ``None`` layers are skipped. Scalar fields are replaced; the table
          fields merge key by key, so a later layer can override one example
          timeout without restating the others.
        - Returns the validated ``EvalConfig``.
        - Raises ``EvalConfigError`` on an unknown key or a bad value.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if key in _TABLE_FIELDS and isinstance(value, Mapping):
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
    return _validated(merged, "merged layers")


def resolve_model(config: EvalConfig) -> str:
    """The model to evaluate: ``config.model``, then ``$LLM_MODEL``, then the default.

    Read at call time, never at import. A blank value at either tier counts as
    absent and falls through.
    """
    for candidate in (config.model, os.environ.get(ENV_LLM_MODEL)):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return DEFAULT_LLM_MODEL
