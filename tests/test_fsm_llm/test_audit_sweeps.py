"""Cross-cutting sweeps for the 2026-09-22 core audit (plan step 15).

Four sweeps, each pinning agreement between pieces that are maintained apart:

1. Operators: ``ALLOWED_JSONLOGIC_OPERATIONS`` equals the union of the three
   dispatch tables in ``expressions`` (eager ``operations``, ``_data_operators``,
   ``_SHORT_CIRCUIT_OPERATORS``), the tables are pairwise disjoint, and every
   entry is actually reached through ``evaluate_logic``.
2. Context filters: one fixed list of names gets the same verdict at all four
   filters, with the documented contract differences written out (the runner
   redacts instead of dropping and ignores internal prefixes;
   ``_strip_internal_mapping`` drops internal prefixes only;
   ``clean_context_keys`` drops forbidden names only when asked to).
3. Config reachability: every ``HandlerSystem.__init__`` and
   ``FSMManager.__init__`` option is reachable through ``API(...)`` or sits on an
   exemption table with a reason.
4. Tag sanitiser: exhaustive enumeration over a small alphabet of the
   invariants ``_sanitize_text_for_prompt`` states in its docstring and its
   DECISION anchors (D-017/D-024/D-026/D-038/D-040/D-047/D-054), plus the
   256/257 boundary cases. The sanitiser does not claim idempotence, so none is
   asserted.
"""

from __future__ import annotations

import html
import inspect
import itertools
import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from fsm_llm import expressions
from fsm_llm.constants import (
    ALLOWED_JSONLOGIC_OPERATIONS,
    has_internal_prefix,
    is_forbidden_context_entry,
)
from fsm_llm.prompts import BasePromptBuilder
from fsm_llm.transition_evaluator import TransitionEvaluatorConfig

# ---------------------------------------------------------------------------
# 1. Operator allowlist vs implemented dispatch tables
# ---------------------------------------------------------------------------


class TestOperatorSweep:
    def test_short_circuit_operators_never_reach_operations(self):
        short_circuit = expressions._SHORT_CIRCUIT_OPERATORS
        assert short_circuit == {"and", "or", "if"}
        assert short_circuit.isdisjoint(expressions.operations)

    def test_every_allowed_operator_has_exactly_one_dispatcher(self):
        tables = [
            set(expressions.operations),
            set(expressions._data_operators),
            set(expressions._SHORT_CIRCUIT_OPERATORS),
        ]
        assert set().union(*tables) == ALLOWED_JSONLOGIC_OPERATIONS
        assert sum(len(t) for t in tables) == len(ALLOWED_JSONLOGIC_OPERATIONS)

    @pytest.mark.parametrize("operator", sorted(expressions.operations))
    def test_every_eager_operator_is_dispatched(self, operator, monkeypatch):
        """Each ``operations`` entry is the one ``evaluate_logic`` calls."""
        sentinel = object()
        monkeypatch.setitem(expressions.operations, operator, lambda *a: sentinel)
        assert expressions.evaluate_logic({operator: [1, 2]}, {}) is sentinel

    @pytest.mark.parametrize("operator", sorted(expressions._data_operators))
    def test_every_data_operator_is_dispatched(self, operator, monkeypatch):
        sentinel = object()
        monkeypatch.setitem(expressions._data_operators, operator, lambda *a: sentinel)
        assert expressions.evaluate_logic({operator: ["x"]}, {}) is sentinel

    @pytest.mark.parametrize("operator", ["and", "or", "if"])
    def test_every_short_circuit_operator_is_dispatched(self, operator, monkeypatch):
        seen: list[str] = []

        def spy(op, values, data, depth):
            seen.append(op)
            return "spied"

        monkeypatch.setattr(expressions, "_evaluate_short_circuit", spy)
        assert expressions.evaluate_logic({operator: [True, 1, 2]}, {}) == "spied"
        assert seen == [operator]


# ---------------------------------------------------------------------------
# 2. Agreement between the four context filters
# ---------------------------------------------------------------------------

# name -> value. Every entry's class is re-checked against the single oracle
# (`is_forbidden_context_entry`, `has_internal_prefix`) so the list cannot drift.
_FORBIDDEN = {
    "password": "hunter2",
    "api_key": "sk-live-4f9a8b7c6d5e4f3a2b1c",
    "session_token": "eyJhbGciOiJIUzI1NiJ9.abc.def",
    "pinCode": "4821",
    "user_password": "x",
    "client_secret": "x",
    "accessToken": "abcdef123456",
    "otp": "123456",
    "cvv": "123",
    "cookie": "a=b",
}
_INTERNAL = {
    "system_prompt": "x",
    "internal_flag": 1,
    "__dunder": 1,
    "SYSTEM_mode": "x",
}
_INTERNAL_AND_FORBIDDEN = {"_secret": "x"}
_ALLOWED: dict[Any, Any] = {
    "username": "alice",
    "pin_attempts": 3,
    "passenger": "Bob",
    "order_key": "ORD-1",
    "token_count": 5,
    "passport_country": "GR",
    "secretary": "Jane",
    "email": "a@b.co",
    1: "non-str key",
}

KEPT, DROPPED, REDACTED = "kept", "dropped", "redacted"

# filter -> verdict per class. The differences are the documented contracts
# (fsm.py D-010, context.py strip_forbidden_keys, runner.py D-014/D-015).
_EXPECTED = {
    "strip_internal": {
        "forbidden": KEPT,
        "internal": DROPPED,
        "internal_and_forbidden": DROPPED,
        "allowed": KEPT,
    },
    "clean_default": {
        "forbidden": KEPT,  # warned, not dropped, unless strip_forbidden_keys
        "internal": DROPPED,
        "internal_and_forbidden": DROPPED,
        "allowed": KEPT,
    },
    "clean_strict": {
        "forbidden": DROPPED,
        "internal": DROPPED,
        "internal_and_forbidden": DROPPED,
        "allowed": KEPT,
    },
    "prompt": {
        "forbidden": DROPPED,
        "internal": DROPPED,
        "internal_and_forbidden": DROPPED,
        "allowed": KEPT,
    },
    "runner": {
        "forbidden": REDACTED,
        "internal": KEPT,  # log redaction matches secrets only, not prefixes
        "internal_and_forbidden": REDACTED,
        "allowed": KEPT,
    },
}


def _run_filter(name: str, ctx: dict) -> dict:
    if name == "strip_internal":
        from fsm_llm.fsm import _strip_internal_mapping

        return _strip_internal_mapping(ctx)
    if name in ("clean_default", "clean_strict"):
        from fsm_llm.context import clean_context_keys

        return clean_context_keys(
            ctx, "sweep", strip_forbidden_keys=name == "clean_strict"
        )
    if name == "prompt":
        return BasePromptBuilder()._filter_context_for_security(ctx)
    from fsm_llm.runner import _redact_context

    return _redact_context(ctx)


# position -> (wrap(entry) -> context, unwrap(filtered) -> entry mapping)
_POSITIONS = {
    "top": (lambda e: dict(e), lambda out: out),
    "nested_dict": (lambda e: {"profile": dict(e)}, lambda out: out["profile"]),
    "list_of_dicts": (lambda e: {"users": [dict(e)]}, lambda out: out["users"][0]),
    "tuple_of_dicts": (
        lambda e: {"items": (dict(e),)},
        lambda out: out["items"][0],
    ),
}


def _cases() -> list[tuple[str, Any, Any]]:
    return [
        (cls, key, value)
        for cls, table in (
            ("forbidden", _FORBIDDEN),
            ("internal", _INTERNAL),
            ("internal_and_forbidden", _INTERNAL_AND_FORBIDDEN),
            ("allowed", _ALLOWED),
        )
        for key, value in table.items()
    ]


def _verdict(out: dict, key: Any, value: Any) -> str:
    if key not in out:
        return DROPPED
    if out[key] == value:
        return KEPT
    if out[key] == "<redacted>":
        return REDACTED
    return f"rewritten to {out[key]!r}"


class TestContextFilterAgreement:
    def test_name_list_matches_the_oracle(self):
        classes = {
            (False, True): "forbidden",
            (True, False): "internal",
            (True, True): "internal_and_forbidden",
            (False, False): "allowed",
        }
        wrong = []
        for cls, key, value in _cases():
            internal = isinstance(key, str) and has_internal_prefix(key)
            forbidden = isinstance(key, str) and is_forbidden_context_entry(key, value)
            if classes[(internal, forbidden)] != cls:
                wrong.append((key, cls, classes[(internal, forbidden)]))
        assert not wrong

    @pytest.mark.parametrize("position", sorted(_POSITIONS))
    @pytest.mark.parametrize("filter_name", sorted(_EXPECTED))
    def test_every_filter_gives_its_contract_verdict(self, filter_name, position):
        wrap, unwrap = _POSITIONS[position]
        disagreements = []
        for cls, key, value in _cases():
            # A benign sibling proves the container itself survives.
            entry = {key: value, "anchor_field": "keep-me"}
            out = unwrap(_run_filter(filter_name, wrap(entry)))
            assert out["anchor_field"] == "keep-me", (key, out)
            verdict = _verdict(out, key, value)
            if verdict != _EXPECTED[filter_name][cls]:
                disagreements.append((key, cls, verdict))
        assert not disagreements


# ---------------------------------------------------------------------------
# 3. Every HandlerSystem / FSMManager option is reachable from API(...)
# ---------------------------------------------------------------------------

# (class, kwarg) -> (API kwarg, non-default value factory, read-back from API)
_REACHABLE = {
    ("HandlerSystem", "error_mode"): (
        "handler_error_mode",
        lambda: "raise",
        lambda api: api.handler_system.error_mode,
    ),
    ("HandlerSystem", "handler_timeout"): (
        "handler_timeout",
        lambda: 0.25,
        lambda api: api.handler_system.handler_timeout,
    ),
    ("FSMManager", "llm_interface"): (
        "llm_interface",
        None,  # identity: the mock passed to every API below
        lambda api: api.fsm_manager.llm_interface,
    ),
    ("FSMManager", "max_history_size"): (
        "max_history_size",
        lambda: 7,
        lambda api: api.fsm_manager.max_history_size,
    ),
    ("FSMManager", "max_message_length"): (
        "max_message_length",
        lambda: 1234,
        lambda api: api.fsm_manager.max_message_length,
    ),
    ("FSMManager", "max_fsm_cache_size"): (
        "max_fsm_cache_size",
        lambda: 3,
        lambda api: api.fsm_manager._max_fsm_cache_size,
    ),
    ("FSMManager", "transition_evaluator"): (
        "transition_config",
        TransitionEvaluatorConfig,
        lambda api: api.fsm_manager.transition_evaluator.config,
    ),
}

_EXEMPT = {
    ("FSMManager", "fsm_loader"): "API builds its own loader over fsm_definition "
    "and the push_fsm temp definitions; a caller loader would bypass both",
    ("FSMManager", "handler_system"): "API builds one HandlerSystem from "
    "handler_error_mode + handler_timeout and shares it (both reachable above)",
    ("FSMManager", "handler_error_mode"): "ignored when handler_system is "
    "given, which API always does; reached via API(handler_error_mode=...)",
    ("FSMManager", "data_extraction_prompt_builder"): "API constructs "
    "default-config prompt builders; custom builders need FSMManager directly",
    ("FSMManager", "response_generation_prompt_builder"): "API constructs "
    "default-config prompt builders; custom builders need FSMManager directly",
    ("FSMManager", "field_extraction_prompt_builder"): "API constructs "
    "default-config prompt builders; custom builders need FSMManager directly",
}


def _init_kwargs(cls: type) -> set[str]:
    return {
        n
        for n, p in inspect.signature(cls.__init__).parameters.items()
        if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _sweep_fsm() -> dict[str, Any]:
    return {
        "name": "sweep",
        "description": "d",
        "initial_state": "a",
        "states": {
            "a": {
                "id": "a",
                "description": "a",
                "purpose": "a",
                "response_instructions": "hi",
            }
        },
    }


class TestConfigReachability:
    def test_every_option_is_classified(self):
        from fsm_llm.fsm import FSMManager
        from fsm_llm.handlers import HandlerSystem

        declared = {
            (cls.__name__, kw)
            for cls in (HandlerSystem, FSMManager)
            for kw in _init_kwargs(cls)
        }
        classified = set(_REACHABLE) | set(_EXEMPT)
        assert declared == classified, (
            f"unclassified: {sorted(declared - classified)}; "
            f"stale: {sorted(classified - declared)}"
        )
        assert set(_REACHABLE).isdisjoint(_EXEMPT)
        assert all(reason.strip() for reason in _EXEMPT.values())

    @pytest.mark.parametrize(
        "option", sorted(_REACHABLE), ids=lambda o: f"{o[0]}.{o[1]}"
    )
    def test_reachable_option_round_trips(self, option):
        from fsm_llm.api import API
        from fsm_llm.llm import LLMInterface

        api_kw, make, read_back = _REACHABLE[option]
        assert api_kw in _init_kwargs(API)
        llm = MagicMock(spec=LLMInterface)
        value = llm if make is None else make()
        kwargs = {"fsm_definition": _sweep_fsm(), "llm_interface": llm, api_kw: value}
        assert read_back(API(**kwargs)) is value


# ---------------------------------------------------------------------------
# 4. Tag sanitiser invariants
# ---------------------------------------------------------------------------

_SAFE = BasePromptBuilder._SAFE_TAGS
_NAME = r"([A-Za-z_!?][A-Za-z0-9._:-]*)"
# A complete tag with no nested `<`: what a reader of the prompt sees as markup.
_LIVE_TAG = re.compile(r"<(?:\s*/)?\s*" + _NAME + r"[^<>]*>")
_RAW_CLOSER = re.compile(r"<\s*/\s*" + _NAME)
_RAW_OPENER = re.compile(r"<\s*" + _NAME)

# Structural tokens are single symbols so length 5 still reaches `</task>`.
_ALPHABET = ["<", ">", "/", " ", "\n", "a", "1", "b", "task", "!"]
_MAX_LEN = 5


def _sanitize(text: str) -> str:
    return BasePromptBuilder()._sanitize_text_for_prompt(text)


def _violations(text: str, out: str) -> list[str]:
    """Return the name of every stated invariant ``out`` breaks for ``text``."""
    broken = []
    flat = text.replace("\n", " ").replace("\r", " ")
    if "\n" in out or "\r" in out:
        broken.append("newline survived")  # docstring: strips newlines
    if html.unescape(out) != flat:  # D-047 amendment: escape, never consume
        broken.append("text lost or altered beyond escaping")
    for m in _LIVE_TAG.finditer(out):  # D-017: allow-list of inert tags only
        if m.group(1).lower() not in _SAFE:
            broken.append(f"live tag {m.group(0)!r}")
    for m in _RAW_CLOSER.finditer(out):  # D-026/D-040: a closer is never raw
        if m.group(1).lower() not in _SAFE:
            broken.append(f"raw closer {m.group(0)!r}")
    for m in _RAW_OPENER.finditer(out):  # D-054: raw opener only with no `>` after
        if m.group(1).lower() not in _SAFE and ">" in out[m.end() :]:
            broken.append(f"raw opener {m.group(0)!r} before a '>'")
    if ">" not in flat and "/" not in flat and out != flat:
        broken.append("prose without '>' or '/' was changed")  # D-040 opener rule
    return broken


class TestSanitizerSweep:
    def test_exhaustive_small_alphabet(self):
        failures = []
        count = 0
        for length in range(_MAX_LEN + 1):
            for symbols in itertools.product(_ALPHABET, repeat=length):
                text = "".join(symbols)
                count += 1
                broken = _violations(text, _sanitize(text))
                if broken:
                    failures.append((text, broken))
        assert count == sum(len(_ALPHABET) ** n for n in range(_MAX_LEN + 1))
        assert not failures, failures[:10]

    @pytest.mark.parametrize("sample", ["a < b", "x<5", "1 <2", "if a < b then"])
    def test_comparison_prose_unchanged(self, sample):
        assert _sanitize(sample) == sample

    @pytest.mark.parametrize("sample", ["<b>x</b>", "<i>y</i>", "<B>z</B>"])
    def test_safe_tags_pass_through(self, sample):
        assert _sanitize(sample) == sample

    @pytest.mark.parametrize("pad", [255, 256, 257, 300])
    def test_padded_closer_is_escaped(self, pad):
        text = "</task" + " " * pad + ">"
        out = _sanitize(text)
        assert "</task" not in out
        assert not _violations(text, out)

    @pytest.mark.parametrize("pad", [255, 256, 257, 300])
    def test_padded_spaced_opener_is_escaped(self, pad):
        # D-054: `< name` + padding + `>` is a padded OPENER, not a comparison.
        text = "< task" + "x" * pad + " >"
        out = _sanitize(text)
        assert out.startswith("&lt;")
        assert not _violations(text, out)

    @pytest.mark.parametrize("pad", [255, 256, 257, 300])
    def test_padded_unterminated_closer_is_escaped(self, pad):
        text = "bye </task" + " " * pad
        out = _sanitize(text)
        assert "</task" not in out
        assert not _violations(text, out)

    @pytest.mark.parametrize(
        ("pad", "escaped"), [(255, False), (256, False), (257, True), (300, True)]
    )
    def test_unterminated_opener_boundary(self, pad, escaped):
        # D-040: an end-arm opener stays raw while at most 257 characters
        # follow its first name character; past that the overflow arm escapes
        # only `<y`, never the tail (D-047 amendment).
        text = "x<y" + " " * pad
        out = _sanitize(text)
        assert out == (("x&lt;y" + " " * pad) if escaped else text)

    @pytest.mark.parametrize("pad", [256, 257, 300])
    def test_padded_comparison_prose_unchanged(self, pad):
        # D-047/D-054: space after `<`, no `/`, no later `>`: a comparison.
        text = "latency < threshold" + " " * pad
        assert _sanitize(text) == text
