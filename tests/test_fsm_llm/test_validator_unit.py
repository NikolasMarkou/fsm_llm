"""
Unit tests for fsm_llm.validator module.

Tests cover:
- FSMValidationResult: error/warning/info accumulation, is_valid property
- FSMValidator.validate: valid FSMs, orphaned states, unreachable terminals, missing initial
- Cycle detection: simple cycles, no cycles, self-loops
- Complex FSM with multiple issues
"""

import copy
import importlib
import inspect
import io
import json
import sys
import typing
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from annotated_types import Ge, Le, MaxLen, MinLen
from loguru import logger
from pydantic import BaseModel, ValidationError

import fsm_llm.logging as fsm_llm_logging
from fsm_llm.api import API
from fsm_llm.definitions import FSMDefinition
from fsm_llm.validator import (
    FSMValidationResult,
    FSMValidator,
    main_cli,
    validate_fsm_from_file,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _valid_fsm_data():
    """A simple valid FSM: start -> middle -> end.

    Carries every field the strict Pydantic `FSMDefinition` model requires
    (top-level `description`, per-state `id`) so that this fixture means what
    its name says: an FSM that BOTH `FSMValidator` and `API.from_file` accept.
    Pinned by `test_valid_fsm_data_is_pydantic_clean` -- do not remove fields.
    """
    return {
        "name": "valid_fsm",
        "description": "A simple valid FSM",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "Start",
                "purpose": "Begin",
                "transitions": [
                    {"target_state": "middle", "description": "Go to middle"}
                ],
            },
            "middle": {
                "id": "middle",
                "description": "Middle",
                "purpose": "Process",
                "transitions": [{"target_state": "end", "description": "Go to end"}],
            },
            "end": {
                "id": "end",
                "description": "End",
                "purpose": "Finish",
                "transitions": [],
            },
        },
    }


class TestFixtureHygiene:
    def test_valid_fsm_data_is_pydantic_clean(self):
        """SC-05: `_valid_fsm_data()` must be loadable, not merely validator-clean.

        This fixture is used across the file as the "a valid FSM" backdrop. It
        previously omitted the top-level `description` and every state `id`,
        carrying four `missing`-class pydantic errors -- so it was only "valid"
        under the validator's leniency, not under the load path. Asserting the
        construction here pins the cleanliness so it cannot silently rot back.
        """
        assert FSMDefinition(**_valid_fsm_data()) is not None


# ==================================================================
# FSMValidationResult
# ==================================================================


class TestFSMValidationResult:
    def test_initially_valid(self):
        result = FSMValidationResult("test")
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.info == []

    def test_add_error_invalidates(self):
        result = FSMValidationResult("test")
        result.add_error("something broke")
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        result = FSMValidationResult("test")
        result.add_warning("watch out")
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_add_info(self):
        result = FSMValidationResult("test")
        result.add_info("FYI")
        assert result.is_valid is True
        assert len(result.info) == 1

    def test_as_dict(self):
        result = FSMValidationResult("my_fsm")
        result.add_error("err1")
        result.add_warning("warn1")
        d = result.as_dict()
        assert d["fsm_name"] == "my_fsm"
        assert d["is_valid"] is False
        assert "err1" in d["errors"]
        assert "warn1" in d["warnings"]

    def test_str_representation(self):
        result = FSMValidationResult("demo")
        result.add_error("bad thing")
        text = str(result)
        assert "demo" in text
        assert "INVALID" in text
        assert "bad thing" in text


# ==================================================================
# FSMValidator — valid FSM
# ==================================================================


class TestValidatorValidFSM:
    def test_valid_linear_fsm(self):
        result = FSMValidator(_valid_fsm_data()).validate()
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_valid_single_state_terminal(self):
        """An FSM where the initial state is also the terminal state."""
        data = {
            "name": "one_state",
            "description": "An FSM whose only state is terminal",
            "initial_state": "only",
            "states": {
                "only": {
                    "id": "only",
                    "description": "Only state",
                    "purpose": "Everything",
                    "transitions": [],
                }
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True


# ==================================================================
# FSMValidator — missing / bad initial state
# ==================================================================


class TestValidatorInitialState:
    def test_missing_initial_state(self):
        data = _valid_fsm_data()
        data["initial_state"] = ""
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("initial state" in e.lower() for e in result.errors)

    def test_initial_state_not_in_states(self):
        data = _valid_fsm_data()
        data["initial_state"] = "nonexistent"
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("nonexistent" in e for e in result.errors)


# ==================================================================
# FSMValidator — orphaned states
# ==================================================================


class TestValidatorOrphanedStates:
    def test_orphaned_state_detected(self):
        data = _valid_fsm_data()
        data["states"]["orphan"] = {
            "description": "Orphan",
            "purpose": "Unreachable",
            "transitions": [],
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("orphan" in e.lower() for e in result.errors)


# ==================================================================
# FSMValidator — unreachable terminal
# ==================================================================


class TestValidatorUnreachableTerminal:
    def test_no_terminal_state(self):
        """Every state has transitions, so there is no terminal state."""
        data = {
            "name": "no_terminal",
            "description": "An FSM with no terminal state",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "A",
                    "purpose": "A",
                    "transitions": [{"target_state": "b", "description": "to b"}],
                },
                "b": {
                    "id": "b",
                    "description": "B",
                    "purpose": "B",
                    "transitions": [{"target_state": "a", "description": "to a"}],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("terminal" in e.lower() for e in result.errors)


# ==================================================================
# FSMValidator — cycle detection
# ==================================================================


class TestValidatorCycleDetection:
    def test_no_cycle_linear(self):
        """Linear FSM should report no cycles."""
        result = FSMValidator(_valid_fsm_data()).validate()
        has_cycle_info = any("cycle" in i.lower() for i in result.info)
        has_no_cycle_info = any("no cycle" in i.lower() for i in result.info)
        # Either no mention of cycles, or explicit "no cycles" message
        assert has_no_cycle_info or not has_cycle_info

    def test_simple_cycle_detected(self):
        data = {
            "name": "cyclic",
            "description": "An FSM containing the cycle a -> b -> a",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "A",
                    "purpose": "A",
                    "transitions": [
                        {"target_state": "b", "description": "to b"},
                        {"target_state": "done", "description": "to done"},
                    ],
                },
                "b": {
                    "id": "b",
                    "description": "B",
                    "purpose": "B",
                    "transitions": [{"target_state": "a", "description": "back to a"}],
                },
                "done": {
                    "id": "done",
                    "description": "Done",
                    "purpose": "End",
                    "transitions": [],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        # The cycle a->b->a should be detected in info or warnings
        all_messages = result.info + result.warnings
        assert any("cycle" in m.lower() for m in all_messages)

    def test_self_loop_detected(self):
        data = {
            "name": "self_loop",
            "description": "An FSM containing the self-loop a -> a",
            "initial_state": "a",
            "states": {
                "a": {
                    "id": "a",
                    "description": "A",
                    "purpose": "A",
                    "transitions": [
                        {"target_state": "a", "description": "stay in a"},
                        {"target_state": "end", "description": "to end"},
                    ],
                },
                "end": {
                    "id": "end",
                    "description": "End",
                    "purpose": "End",
                    "transitions": [],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        all_messages = result.info + result.warnings
        assert any("cycle" in m.lower() for m in all_messages)


# ==================================================================
# Complex FSM with multiple issues
# ==================================================================


class TestValidatorComplexFSM:
    def test_multiple_issues(self):
        data = {
            "name": "complex_issues",
            "initial_state": "start",
            "states": {
                "start": {
                    "description": "Start",
                    "purpose": "Begin",
                    "transitions": [{"target_state": "middle", "description": "go"}],
                },
                "middle": {
                    "description": "Middle",
                    "purpose": "Work",
                    "transitions": [{"target_state": "end", "description": "finish"}],
                },
                "end": {
                    "description": "End",
                    "purpose": "Done",
                    "transitions": [],
                },
                "orphan": {
                    "description": "Orphan",
                    "purpose": "Nowhere",
                    "transitions": [],
                },
            },
        }
        result = FSMValidator(data).validate()
        # Orphan should trigger an error
        assert result.is_valid is False
        assert any("orphan" in e.lower() for e in result.errors)

    def test_transition_to_nonexistent_state(self):
        data = {
            "name": "bad_target",
            "initial_state": "start",
            "states": {
                "start": {
                    "description": "Start",
                    "purpose": "Begin",
                    "transitions": [
                        {"target_state": "ghost", "description": "to ghost"}
                    ],
                },
            },
        }
        result = FSMValidator(data).validate()
        assert result.is_valid is False
        assert any("ghost" in e for e in result.errors)


# ------------------------------------------------------------------
# S4 regression: validator verdict must AGREE with FSMDefinition(**data)
# ------------------------------------------------------------------


def _pydantic_clean_fsm_data():
    """An FSM dict that FSMDefinition(**data) accepts outright.

    Unlike _valid_fsm_data() this carries the keys the strict Pydantic model
    requires (top-level `description`, per-state `id`), so that a semantic
    validator is actually REACHED rather than short-circuited by an earlier
    `missing`-type structural error.
    """
    return {
        "name": "clean_fsm",
        "description": "A pydantic-clean FSM",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "Start",
                "purpose": "Begin",
                "transitions": [{"target_state": "end", "description": "Go to end"}],
            },
            "end": {
                "id": "end",
                "description": "End",
                "purpose": "Finish",
                "transitions": [],
            },
        },
    }


def _missing_required_fields_fsm_data():
    """`_valid_fsm_data()` with EXACTLY the pydantic-required fields removed.

    ADVERSARIALLY PAIRED with `_valid_fsm_data()`: this dict is *derived* from
    it by deleting the top-level `description` and each state's `id` and
    nothing else, so the two differ in precisely the fields under test. A
    "structurally invalid FSM" written from scratch would differ in many ways
    and could not attribute a verdict change to those fields.

    Its only pydantic complaints are `missing`-class -- pinned by
    `test_missing_fields_fixture_is_rejected_only_as_missing_class`.
    """
    data = _valid_fsm_data()
    del data["description"]
    for state in data["states"].values():
        del state["id"]
    return data


def _fsm_with_bad_fallback_intent():
    """FSM whose classification_extractions fallback_intent is not in intents.

    Reproducer from findings/schema-validation.md #2: the
    ClassificationExtractionConfig.validate_fallback_in_intents model_validator
    raises ValueError, so FSMDefinition(**data) fails while FSMValidator used to
    report is_valid=True with zero errors (the S4 false green).
    """
    data = _pydantic_clean_fsm_data()
    data["states"]["start"]["classification_extractions"] = [
        {
            "field_name": "user_intent",
            "intents": [
                {"name": "buy", "description": "User wants to purchase"},
                {"name": "browse", "description": "User is just looking"},
            ],
            "fallback_intent": "not_a_declared_intent",
        }
    ]
    return data


def _fsm_with_n_intents(n):
    """`_pydantic_clean_fsm_data()` plus a classification_extractions block of
    exactly `n` intents, and nothing else changed.

    ADVERSARIALLY PAIRED across n: the only thing that varies between the
    accepted and rejected cases is the list length, so a verdict change is
    attributable to the length bound alone.
    """
    data = _pydantic_clean_fsm_data()
    data["states"]["start"]["classification_extractions"] = [
        {
            "field_name": "user_intent",
            "intents": [
                {"name": f"intent_{i}", "description": f"The intent_{i} intent"}
                for i in range(n)
            ],
            "fallback_intent": "intent_0",
        }
    ]
    return data


def _fsm_violating(path, value):
    """`_pydantic_clean_fsm_data()` with exactly ONE field overwritten.

    Interface contract:
      - `path`: a tuple of dict keys / list indices addressing a field inside the
        FSM dict, e.g. `("states", "start", "transitions", 0, "priority")`.
      - `value`: the constraint-violating value to write there.
      - Returns: a fresh dict. The base is never mutated.
      - Failure mode: `KeyError`/`IndexError` if `path` does not exist in the
        base fixture -- deliberately loud, since a silently-misaddressed path
        would produce a fixture that violates nothing and passes vacuously.

    ADVERSARIALLY PAIRED with the base by construction: the base is pydantic-clean,
    so any verdict change is attributable to this single field and nothing else.
    """
    import copy

    data = copy.deepcopy(_pydantic_clean_fsm_data())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


def _unknown_type_validation_error(type_name="some_future_pydantic_error_type"):
    """A real `ValidationError` carrying a type name pydantic itself never emits.

    Interface contract:
      - `type_name`: any string. Callers should pass a name that is NOT a real
        pydantic error type, which is the whole point of the helper.
      - Returns: a `pydantic.ValidationError` whose single entry reports
        `type_name`. Raising it from a patched `FSMDefinition` drives
        `_validate_pydantic_schema` down its `else: add_warning` fail-safe branch.
      - Failure mode: none -- `PydanticCustomError` accepts arbitrary type names,
        unlike `ValidationError.from_exception_data`, which raises
        `KeyError: Invalid error type` for anything outside pydantic's own table.
        That restriction is exactly why this helper exists.
    """
    from pydantic_core import InitErrorDetails, PydanticCustomError

    return ValidationError.from_exception_data(
        "FSMDefinition",
        [
            InitErrorDetails(
                type=PydanticCustomError(type_name, "synthetic future error"),
                loc=("name",),
                input="x",
            )
        ],
    )


class TestValidatorAgreesWithFSMDefinition:
    """The validator must not report is_valid=True for an FSM that the load
    path (API.from_file -> FSMDefinition(**data)) hard-rejects on a semantic
    rule the framework itself authored."""

    def test_bad_fallback_intent_rejected_by_pydantic(self):
        """Precondition: the reproducer really does break the load path."""
        with pytest.raises(ValidationError) as exc_info:
            FSMDefinition(**_fsm_with_bad_fallback_intent())
        assert any(
            e["type"] in ("value_error", "assertion_error")
            for e in exc_info.value.errors()
        )

    def test_bad_fallback_intent_rejected_by_validator(self):
        """SC-6: the false-green is gone."""
        result = FSMValidator(_fsm_with_bad_fallback_intent()).validate()
        assert result.is_valid is False
        assert any("not_a_declared_intent" in e for e in result.errors)

    def test_missing_fields_fixture_is_rejected_only_as_missing_class(self):
        """Precondition for the widening's coverage.

        The adversarial negative must exercise the `missing` promotion and
        NOTHING else. If it also carried a `value_error`, every test built on
        it would pass under the pre-widening implementation too, and would
        prove nothing about `missing`.
        """
        with pytest.raises(ValidationError) as exc_info:
            FSMDefinition(**_missing_required_fields_fsm_data())
        types = {e["type"] for e in exc_info.value.errors()}
        assert types == {"missing"}, f"fixture is contaminated: {types}"

    def test_agreement_property_across_fixtures(self):
        """AGREEMENT PROPERTY (the real contract).

        Maintainer direction, verbatim -- this inversion is AUTHORIZED, not
        drift:

            "api from_file is the truth always, make sure fsm-llm-validate
             aligns with it"

        This test previously asserted the OPPOSITE for structural failures:
        that "simplified-dict fixtures that pydantic rejects only structurally
        (or accepts) must still validate True". That proposition is now
        explicitly repudiated. `API.from_file` -> `FSMDefinition(**data)` is
        the reference implementation, and the validator is a diagnostic that
        must agree with it across ALL pydantic rejection classes -- not only
        the semantic ones.
        """
        cases = [
            ("simplified_valid", _valid_fsm_data()),
            ("pydantic_clean", _pydantic_clean_fsm_data()),
            ("bad_fallback_intent", _fsm_with_bad_fallback_intent()),
            ("missing_required_fields", _missing_required_fields_fsm_data()),
            # F-09/D-013: the LIST-length classes. `too_long` is the class F-09
            # newly makes reachable; `too_short` was already reachable and
            # already false-greening. Both sides of the bound plus an in-range
            # control, so this fixture set cannot pass under a validator that
            # ignores length errors OR under one that rejects every intent list.
            ("intents_too_short", _fsm_with_n_intents(1)),
            ("intents_in_range", _fsm_with_n_intents(15)),
            ("intents_too_long", _fsm_with_n_intents(16)),
            # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE] -- VACUITY REPAIR.
            # The 7 fixtures above touch only `value_error`, `missing` and the
            # LIST-length classes: i.e. exactly the classes that were ALREADY
            # promoted. They therefore could not fail no matter how large the
            # false-green surface got, and in fact all 7 passed while 40 of 51
            # measured constraint violations were false greens. Every fixture
            # below exercises a class that WAS false-green. Do not delete these
            # to "simplify" the case list -- that restores the vacuity.
            ("state_id_pattern", _fsm_violating(("states", "start", "id"), "café")),
            ("state_id_too_long", _fsm_violating(("states", "start", "id"), "s" * 101)),
            ("name_too_short", _fsm_violating(("name",), "")),
            ("name_too_long", _fsm_violating(("name",), "n" * 101)),
            (
                "description_too_long",
                _fsm_violating(("states", "start", "description"), "d" * 301),
            ),
            (
                "priority_below_ge",
                _fsm_violating(("states", "start", "transitions", 0, "priority"), -1),
            ),
            (
                "priority_above_le",
                _fsm_violating(("states", "start", "transitions", 0, "priority"), 1001),
            ),
            (
                "priority_int_parsing",
                _fsm_violating(
                    ("states", "start", "transitions", 0, "priority"), "not-an-int"
                ),
            ),
        ]
        loader_rejects = {}
        for name, data in cases:
            try:
                FSMDefinition(**data)
                loader_rejects[name] = False
            except ValidationError:
                loader_rejects[name] = True

            result = FSMValidator(data).validate()
            if loader_rejects[name]:
                assert result.is_valid is False, (
                    f"{name}: the loader rejects this FSM but the validator "
                    f"reports is_valid=True (false green -- fsm-llm-validate "
                    f"would green-light a file API.from_file crashes on)"
                )
            else:
                assert result.is_valid is True, (
                    f"{name}: the loader accepts this FSM, so the validator "
                    f"must not report it invalid (false red)"
                )

        # Vacuity guard: with every fixture on one side of the branch, the
        # assertions above would hold under a validator that ignored pydantic
        # entirely. Both branches must actually be taken.
        assert any(loader_rejects.values()), "no case exercises the reject branch"
        assert not all(loader_rejects.values()), "no case exercises the accept branch"

        # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE] -- second vacuity guard.
        # "Both branches were taken" is too weak: the original 7 fixtures
        # satisfied it while covering only already-promoted classes. Assert the
        # case list actually SPANS the classes that were false-green, so deleting
        # a fixture above fails HERE rather than silently shrinking coverage.
        covered = set()
        for _name, data in cases:
            try:
                FSMDefinition(**data)
            except ValidationError as exc:
                covered.update(e["type"] for e in exc.errors())
        for required in (
            "string_pattern_mismatch",
            "string_too_short",
            "string_too_long",
            "greater_than_equal",
            "less_than_equal",
            "int_parsing",
            "value_error",
            "missing",
            "too_short",
            "too_long",
        ):
            assert required in covered, (
                f"no fixture in this case list produces a {required!r} error, so "
                f"the test is vacuous over that class"
            )

    def test_type_coercion_failure_agrees_with_the_loader(self):
        """DELIBERATELY INVERTED. Was `test_simplified_dict_leniency_preserved`.

        # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE]
        This test previously asserted the OPPOSITE: that `priority: "not-an-int"`
        (an `int_parsing` error) must stay a WARNING and leave `is_valid=True`,
        under the rationale "type-coercion leniency is NOT part of the `missing`
        widening". That rationale was FALSIFIED BY MEASUREMENT, not by taste:

            API.from_file(<fsm with priority="not-an-int">)
            -> ValueError: Input should be a valid integer ... [type=int_parsing]

        `API.from_file` hard-refuses this file exactly as it refuses `id: "café"`.
        Under the standing invariant "api from_file is the truth always", a
        validator that returns is_valid=True here is emitting a FALSE GREEN --
        `fsm-llm-validate` exits 0 on a file the loader crashes on. The old
        assertion was therefore pinning a live defect in place.

        Do NOT re-invert this test to "restore" coercion leniency. Doing so
        reintroduces the false green AND puts the suite back in contradiction
        with `test_constraint_sweep_validator_agrees_with_loader`, which derives
        the same conclusion mechanically from `definitions.py`. If deliberate
        coercion leniency is ever genuinely wanted, it has to be implemented in
        the LOADER first -- the validator only ever mirrors the loader.
        """
        data = _valid_fsm_data()
        data["states"]["start"]["transitions"][0]["priority"] = "not-an-int"

        # Precondition: the loader really does refuse this, and ONLY on the
        # coercion class -- so the verdict below is attributable to `int_parsing`
        # alone and not to some other contamination in the fixture.
        with pytest.raises(ValidationError) as exc_info:
            FSMDefinition(**data)
        assert {e["type"] for e in exc_info.value.errors()} == {"int_parsing"}

        result = FSMValidator(data).validate()
        schema_errors = [e for e in result.errors if e.startswith("Schema:")]
        assert schema_errors, (
            "int_parsing stayed a warning: fsm-llm-validate would exit 0 on a "
            "file API.from_file hard-refuses (false green)"
        )
        assert result.is_valid is False

    def test_unknown_error_type_falls_through_to_warning(self):
        """Fail-safe: an unrecognized/future pydantic error type must warn, not error.

        The promotion set is an ALLOW-list, so a new pydantic version can never
        make the validator accidentally stricter than the loader. Widening it to
        cover `missing` must NOT have flipped it into a deny-list -- if it had,
        this synthetic unknown type would be promoted to an error.

        # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE]
        The synthetic stand-in used to be `string_too_short`. That name is now a
        RECOGNIZED, promoted member of the allow-list (it is loader-raising and
        was a measured false green), so it can no longer stand in for "a type
        name this validator has never heard of" -- it would test the promotion
        branch, not the fail-safe branch. Only the stand-in changed; every
        assertion below is the original.

        The stand-in MUST be a name pydantic does not emit. Do NOT swap in a real
        pydantic type name to "fix" a failure here: that silently converts this
        test into a duplicate of the promotion tests and leaves the fail-safe
        branch -- the HARD invariant that the validator can never become stricter
        than the loader -- with no coverage at all.
        """
        validator = FSMValidator(_valid_fsm_data())
        fake = _unknown_type_validation_error()
        with patch("fsm_llm.validator.FSMDefinition", side_effect=fake):
            validator._validate_pydantic_schema()
        assert validator.result.errors == []
        assert validator.result.is_valid is True
        assert any("Schema:" in w for w in validator.result.warnings)

    def test_promotion_matches_exact_names_not_prefixes(self):
        """DELIBERATELY INVERTED. Was
        `test_list_length_promotion_did_not_widen_to_string_lengths`.

        # DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE]
        The original asserted that `string_too_short`/`string_too_long` must NOT
        be promoted -- the D-013 "the string classes are deliberately excluded"
        carve-out. The constraint sweep measured that carve-out as a live false
        green in both directions (`FSMDefinition.name`, `State.description`,
        `State.purpose`, `Transition.description`, and 8 more all raise
        `string_too_*` and the loader hard-refuses every one), so the carve-out
        is gone and both names are now promoted.

        The original's REAL purpose survives and is what this test now pins:
        membership must be by EXACT name, never by prefix. A `startswith("too_")`
        or `"too_" in name` "simplification" would look equivalent and would
        silently swallow any future `too_*` name pydantic invents -- turning the
        allow-list into a partial deny-list and defeating the fail-safe branch.
        So: the two real names are promoted, and a synthetic name sharing their
        prefixes still WARNS.
        """
        for type_name, ctx in (
            ("string_too_short", {"min_length": 1}),
            ("string_too_long", {"max_length": 1}),
        ):
            validator = FSMValidator(_valid_fsm_data())
            fake = ValidationError.from_exception_data(
                "FSMDefinition",
                [{"type": type_name, "loc": ("name",), "input": "x", "ctx": ctx}],
            )
            with patch("fsm_llm.validator.FSMDefinition", side_effect=fake):
                validator._validate_pydantic_schema()
            assert validator.result.errors, (
                f"{type_name} stayed a warning; the loader hard-refuses this "
                f"class, so leaving it unpromoted is a false green"
            )

        # Vacuity/over-correction guard: prefix-shaped impostors must NOT be
        # promoted. Without this the block above is satisfied by a substring match.
        for impostor in ("too_short_for_comfort", "string_too_wide", "not_missing"):
            validator = FSMValidator(_valid_fsm_data())
            with patch(
                "fsm_llm.validator.FSMDefinition",
                side_effect=_unknown_type_validation_error(impostor),
            ):
                validator._validate_pydantic_schema()
            assert validator.result.errors == [], (
                f"{impostor!r} was promoted: membership is matching on a prefix "
                f"or substring instead of the exact type name"
            )
            assert any("Schema:" in w for w in validator.result.warnings)


# ------------------------------------------------------------------
# Mechanical validator/loader agreement sweep.
#
# DECISION plan-2026-07-20T040150-876e7164/D-001 [STALE]
# THE LIST IN validator.py IS NOT THE CONTROL. THIS IS.
#
# Three consecutive plans "fixed" the validator/loader disagreement by adding
# the type names they happened to have looked at, and each time the remaining
# false greens survived because no test constructed the exact input shape that
# would expose them. The 7-fixture `test_agreement_property_across_fixtures`
# passed throughout while 40 of the 51 constraint violations below were live
# false greens.
#
# So this sweep does NOT enumerate type names or fields by hand. It walks
# `definitions.py` reflectively from `FSMDefinition`, derives one violating
# fixture per declared constraint, and asserts agreement as an IFF. A future
# `Field(...)` constraint that introduces an unlisted pydantic type name fails
# HERE, loudly, on the commit that adds it.
#
# Do NOT "simplify" this into a hardcoded list of cases. A hardcoded list can
# only ever see the vocabulary its author already knew about, which is the
# exact failure mode this file has now hit four times.
# ------------------------------------------------------------------


def _definitions_namespace():
    import fsm_llm.definitions as definitions_module

    return vars(definitions_module)


def _resolve_annotation(annotation):
    """Resolve str / `ForwardRef` annotations against `fsm_llm.definitions`.

    Load-bearing, not defensive: `ClassificationExtractionConfig.intents` is
    stored by pydantic as an UNRESOLVED `ForwardRef('list[IntentDefinition]')`.
    Skipping resolution silently hides `IntentDefinition` from the model walk AND
    misclassifies `intents` as a scalar, which fabricates a bogus `list_type`
    disagreement instead of the real `too_short`/`too_long` one. Both mistakes
    were observed while building this sweep.
    """
    namespace = _definitions_namespace()
    if isinstance(annotation, str):
        return eval(annotation, namespace)
    if isinstance(annotation, typing.ForwardRef):
        return eval(annotation.__forward_arg__, namespace)
    return annotation


def _nested_models(annotation):
    """Every `BaseModel` subclass appearing anywhere inside an annotation."""
    found, stack = [], [annotation]
    while stack:
        current = _resolve_annotation(stack.pop())
        if isinstance(current, type) and issubclass(current, BaseModel):
            found.append(current)
            continue
        stack.extend(typing.get_args(current))
    return found


def _reachable_models(model=None, seen=None):
    """Every pydantic model reachable from `FSMDefinition` by field traversal."""
    model = FSMDefinition if model is None else model
    seen = {} if seen is None else seen
    if model.__name__ in seen:
        return seen
    seen[model.__name__] = model
    for field in model.model_fields.values():
        for nested in _nested_models(field.annotation):
            _reachable_models(nested, seen)
    return seen


def _container_kind(annotation):
    """`list`, `dict`, or None -- the container origin inside an annotation."""
    resolved = _resolve_annotation(annotation)
    origin = typing.get_origin(resolved)
    if origin in (list, dict):
        return origin
    for arg in typing.get_args(resolved):
        kind = _container_kind(arg)
        if kind is not None:
            return kind
    return None


def _literal_values(annotation):
    values, stack = [], [_resolve_annotation(annotation)]
    while stack:
        current = stack.pop()
        if typing.get_origin(current) is typing.Literal:
            values.extend(typing.get_args(current))
        else:
            stack.extend(typing.get_args(current))
    return values


def _maximal_fsm_data():
    """A VALID FSM that instantiates every model reachable from `FSMDefinition`.

    Every model in `_MODEL_ANCHORS` must have a live instance in here, otherwise
    the sweep has nowhere to inject that model's constraint violations. Pinned by
    `test_maximal_fixture_is_accepted_by_both_layers`.
    """
    return {
        "name": "SweepFSM",
        "description": "A maximal FSM exercising every reachable nested model.",
        "version": "4.1",
        "persona": "A neutral test persona.",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "The start state.",
                "purpose": "Begin the conversation.",
                "extraction_instructions": "Extract the user name.",
                "response_instructions": "Greet the user.",
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "Move on once the name is known.",
                        "priority": 100,
                        "conditions": [
                            {
                                "description": "The name is present.",
                                "requires_context_keys": ["name"],
                                "evaluation_priority": 100,
                            }
                        ],
                    }
                ],
                "field_extractions": [
                    {
                        "field_name": "name",
                        "field_type": "str",
                        "extraction_instructions": "Extract the user's name.",
                    }
                ],
                "classification_extractions": [
                    {
                        "field_name": "user_intent",
                        "intents": [
                            {"name": "buy", "description": "Wants to purchase."},
                            {"name": "browse", "description": "Just looking."},
                        ],
                        "fallback_intent": "browse",
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "The terminal state.",
                "purpose": "Close the conversation.",
                "extraction_instructions": "Nothing to extract.",
                "response_instructions": "Say goodbye.",
                "transitions": [],
            },
        },
    }


# Where an instance of each reachable model lives inside `_maximal_fsm_data()`.
# Coverage is asserted against the reflective walk, so a NEW model added to
# definitions.py fails the suite until it is anchored here.
_MODEL_ANCHORS = {
    "FSMDefinition": (),
    "State": ("states", "start"),
    "Transition": ("states", "start", "transitions", 0),
    "TransitionCondition": ("states", "start", "transitions", 0, "conditions", 0),
    "FieldExtractionConfig": ("states", "start", "field_extractions", 0),
    "ClassificationExtractionConfig": (
        "states",
        "start",
        "classification_extractions",
        0,
    ),
    "IntentDefinition": (
        "states",
        "start",
        "classification_extractions",
        0,
        "intents",
        0,
    ),
}

# One violating mutation per `model_validator`/`field_validator`. These cannot be
# derived reflectively -- the rule lives in the function body -- so they are
# hand-authored, and `test_every_validator_has_a_violating_fixture` fails the
# moment definitions.py grows a validator with no entry here.
_VALIDATOR_VIOLATIONS = {
    ("FSMDefinition", "validate_fsm_structure"): (
        ("initial_state",),
        "a_state_that_does_not_exist",
    ),
    ("FieldExtractionConfig", "validate_validation_rule_keys"): (
        ("states", "start", "field_extractions", 0, "validation_rules"),
        {"totally_bogus_rule_key": 1},
    ),
    ("ClassificationExtractionConfig", "validate_fallback_in_intents"): (
        ("states", "start", "classification_extractions", 0, "fallback_intent"),
        "not_a_declared_intent",
    ),
    ("IntentDefinition", "validate_name_format"): (
        ("states", "start", "classification_extractions", 0, "intents", 0, "name"),
        "café",
    ),
    # H8: an unknown JsonLogic operator in a condition's `logic` is rejected by
    # `TransitionCondition._validate_logic` (ValueError -> value_error), so the
    # loader raises and the validator promotes to ERROR -- both agree.
    ("TransitionCondition", "_validate_logic"): (
        ("states", "start", "transitions", 0, "conditions", 0, "logic"),
        {"eqauls": [{"var": "name"}, 1]},
    ),
}


def _node_at(data, path):
    node = data
    for key in path:
        node = node[key]
    return node


def _with_violation(path, value):
    data = copy.deepcopy(_maximal_fsm_data())
    _node_at(data, path[:-1])[path[-1]] = value
    return data


def _constraint_cases():
    """(label, fsm_dict) for every declared constraint, derived reflectively."""
    cases = []
    for model_name, model in _reachable_models().items():
        anchor = _MODEL_ANCHORS[model_name]
        for field_name, field in model.model_fields.items():
            path = (*anchor, field_name)
            container = _container_kind(field.annotation)
            current = _node_at(_maximal_fsm_data(), anchor).get(field_name)

            for constraint in field.metadata:
                label = f"{model_name}.{field_name}"
                if isinstance(constraint, MinLen):
                    if container is list:
                        bad = (current or [])[: max(0, constraint.min_length - 1)]
                    elif container is dict:
                        bad = {}
                    else:
                        bad = ""
                    cases.append((f"{label} MinLen", _with_violation(path, bad)))
                elif isinstance(constraint, MaxLen):
                    if container is dict:
                        continue  # no declared dict upper bound to violate
                    if container is list:
                        template = (current or [{}])[0]
                        bad = [
                            copy.deepcopy(template)
                            for _ in range(constraint.max_length + 1)
                        ]
                        for index, element in enumerate(bad):
                            if isinstance(element, dict) and "name" in element:
                                element["name"] = f"element_{index}"
                        data = _with_violation(path, bad)
                        # Keep the ONLY violation the length bound: a cloned
                        # intent list would otherwise also break the
                        # fallback-in-intents rule and contaminate the case.
                        if model_name == "ClassificationExtractionConfig":
                            _node_at(data, anchor)["fallback_intent"] = "element_0"
                        cases.append((f"{label} MaxLen", data))
                        continue
                    bad = "x" * (constraint.max_length + 1)
                    cases.append((f"{label} MaxLen", _with_violation(path, bad)))
                elif isinstance(constraint, Ge):
                    cases.append(
                        (f"{label} Ge", _with_violation(path, constraint.ge - 1))
                    )
                elif isinstance(constraint, Le):
                    cases.append(
                        (f"{label} Le", _with_violation(path, constraint.le + 1))
                    )
                elif getattr(constraint, "pattern", None):
                    cases.append((f"{label} pattern", _with_violation(path, "café")))

            if _literal_values(field.annotation):
                cases.append(
                    (
                        f"{model_name}.{field_name} Literal",
                        _with_violation(path, "definitely_not_a_valid_literal"),
                    )
                )
    return cases


def _validator_cases():
    return [
        (f"{model_name}.{validator_name} validator", _with_violation(path, value))
        for (model_name, validator_name), (path, value) in _VALIDATOR_VIOLATIONS.items()
    ]


def _all_agreement_cases():
    return _constraint_cases() + _validator_cases()


class TestConstraintSweepAgreement:
    """SC-1: `fsm-llm-validate` agrees with `API.from_file` on EVERY constraint
    mechanically enumerated from `definitions.py`, in BOTH directions."""

    def test_maximal_fixture_is_accepted_by_both_layers(self):
        """Precondition. If the base fixture were already invalid, every case
        below would 'agree' trivially (both reject) and prove nothing."""
        FSMDefinition(**_maximal_fsm_data())  # must not raise
        assert FSMValidator(_maximal_fsm_data()).validate().is_valid is True

    def test_every_reachable_model_is_anchored(self):
        """A new model in definitions.py must fail HERE, not slip through.

        Without this, a model added to `definitions.py` and wired into
        `FSMDefinition` would simply never be swept -- its constraints would be
        silently unmeasured, which is precisely how the previous false-green
        surface stayed invisible across three plans.
        """
        discovered = set(_reachable_models())
        assert discovered == set(_MODEL_ANCHORS), (
            f"model set changed: unanchored={sorted(discovered - set(_MODEL_ANCHORS))}, "
            f"stale={sorted(set(_MODEL_ANCHORS) - discovered)}. Add an anchor path "
            f"into _maximal_fsm_data() for each unanchored model."
        )

    def test_every_validator_has_a_violating_fixture(self):
        """Same guard for `model_validator`/`field_validator` rules, which cannot
        be derived reflectively and so must be hand-authored."""
        declared = set()
        for model_name, model in _reachable_models().items():
            decorators = model.__pydantic_decorators__
            for validator_name in decorators.model_validators:
                declared.add((model_name, validator_name))
            for validator_name in decorators.field_validators:
                declared.add((model_name, validator_name))
        assert declared == set(_VALIDATOR_VIOLATIONS), (
            f"validator set changed: uncovered="
            f"{sorted(declared - set(_VALIDATOR_VIOLATIONS))}, stale="
            f"{sorted(set(_VALIDATOR_VIOLATIONS) - declared)}"
        )

    def test_sweep_is_not_vacuous(self):
        """The sweep must actually construct a meaningful number of cases and
        every one must genuinely break the loader. A generator bug that produced
        zero cases -- or cases the loader happily accepts -- would make the
        agreement assertion below pass while testing nothing."""
        cases = _all_agreement_cases()
        assert len(cases) >= 45, f"sweep collapsed to {len(cases)} cases"
        inert = []
        for label, data in cases:
            try:
                FSMDefinition(**data)
            except ValidationError:
                continue
            inert.append(label)
        assert not inert, (
            f"{len(inert)} generated fixture(s) do not actually violate anything, "
            f"so their agreement assertion is vacuous: {inert}"
        )

    def test_constraint_sweep_validator_agrees_with_loader(self):
        """SC-1, the IFF. Validator `is_valid is False` <=> loader raises.

        Measured 40 disagreements out of 51 cases before the allow-list widening;
        this must stay at 0. Both directions are asserted, so it is NOT
        satisfiable by "promote everything and reject valid files" -- the false-RED
        direction is covered by
        `TestValidatorIsNeverStricterThanTheLoader` below.
        """
        disagreements = []
        for label, data in _all_agreement_cases():
            try:
                FSMDefinition(**data)
                loader_raises = False
                raised_types = []
            except ValidationError as exc:
                loader_raises = True
                raised_types = sorted({e["type"] for e in exc.errors()})

            validator_rejects = FSMValidator(data).validate().is_valid is False
            if loader_raises != validator_rejects:
                disagreements.append(
                    f"{label}: loader_raises={loader_raises} "
                    f"validator_rejects={validator_rejects} types={raised_types}"
                )

        assert not disagreements, (
            f"{len(disagreements)} validator/loader disagreement(s). Each is a "
            f"file fsm-llm-validate green-lights and API.from_file crashes on "
            f"(or the reverse). If the type name is new, add it to the allow-list "
            f"in validator.py::_validate_pydantic_schema:\n" + "\n".join(disagreements)
        )


def _fsm_with_condition_logic(logic):
    """`_valid_fsm_data()` where start's single transition carries one condition
    with the given `logic` dict. Used to drive H8/H6 load-time rejection."""
    data = copy.deepcopy(_valid_fsm_data())
    data["states"]["start"]["transitions"][0]["conditions"] = [
        {"description": "gate", "logic": logic}
    ]
    return data


class TestLogicOperatorValidation:
    """H8/H6: a `TransitionCondition.logic` with an unknown operator or an empty
    (or nested-empty) dict is rejected at LOAD time by BOTH `FSMDefinition(**data)`
    (raises) and `FSMValidator(...).validate()` (`is_valid is False`), preserving
    the validator/loader agreement IFF. Valid nested logic still loads clean."""

    def test_unknown_operator_rejected_by_loader_and_validator(self):
        data = _fsm_with_condition_logic({"eqauls": [{"var": "email"}, 1]})
        with pytest.raises(ValidationError):
            FSMDefinition(**data)
        assert FSMValidator(data).validate().is_valid is False

    def test_empty_logic_dict_rejected_by_loader_and_validator(self):
        data = _fsm_with_condition_logic({})
        with pytest.raises(ValidationError):
            FSMDefinition(**data)
        assert FSMValidator(data).validate().is_valid is False

    def test_nested_empty_logic_dict_rejected(self):
        data = _fsm_with_condition_logic({"and": [{}]})
        with pytest.raises(ValidationError):
            FSMDefinition(**data)
        assert FSMValidator(data).validate().is_valid is False

    def test_valid_nested_logic_still_loads_and_validates(self):
        # `["a", "b"]` are DATA args of `in`, not operators — the walk must not
        # reject them. `and`/`==`/`in`/`var` are all allow-listed.
        data = _fsm_with_condition_logic(
            {
                "and": [
                    {"==": [{"var": "email"}, 1]},
                    {"in": [{"var": "kind"}, ["a", "b"]]},
                ]
            }
        )
        FSMDefinition(**data)  # must not raise
        assert FSMValidator(data).validate().is_valid is True


class TestValidatorIsNeverStricterThanTheLoader:
    """I-2 / SC-2: zero false REDs. The validator must never reject an FSM the
    loader accepts -- widening the allow-list must not have over-corrected."""

    def test_loader_accepted_fixtures_stay_valid(self):
        accepted = [
            ("valid_fsm", _valid_fsm_data()),
            ("pydantic_clean", _pydantic_clean_fsm_data()),
            ("maximal", _maximal_fsm_data()),
            ("intents_in_range", _fsm_with_n_intents(15)),
            ("intents_at_lower_bound", _fsm_with_n_intents(2)),
        ]
        for name, data in accepted:
            FSMDefinition(**data)  # precondition: the loader really accepts it
            assert FSMValidator(data).validate().is_valid is True, (
                f"{name}: FALSE RED -- the loader accepts this FSM but "
                f"fsm-llm-validate rejects it"
            )

    def test_boundary_values_are_accepted_on_the_legal_side(self):
        """Off-by-one guard. Every bound the sweep violates by +1/-1 must still
        be ACCEPTED exactly at the limit, in both layers. Without this, an
        allow-list widening paired with an off-by-one bound would look green in
        the sweep while rejecting legal files."""
        at_the_limit = [
            ("name at max_length", ("name",), "n" * 100),
            ("state id at max_length", ("states", "start", "id"), "s" * 100),
            ("priority at ge", ("states", "start", "transitions", 0, "priority"), 0),
            ("priority at le", ("states", "start", "transitions", 0, "priority"), 1000),
        ]
        for label, path, value in at_the_limit:
            data = _with_violation(path, value)
            if path == ("states", "start", "id"):
                # `id` is also the states-dict key; keep the FSM self-consistent
                # so the verdict is attributable to the length bound alone.
                data["states"][value] = data["states"].pop("start")
                data["initial_state"] = value
                data["states"][value]["transitions"][0]["target_state"] = "done"
            FSMDefinition(**data)  # precondition: legal at the boundary
            assert FSMValidator(data).validate().is_valid is True, (
                f"{label}: FALSE RED at the legal boundary"
            )


class TestUnknownKeyWarnings:
    """H7: a misspelled OPTIONAL FSM-JSON key surfaces as a `fsm-llm-validate`
    WARNING naming the key, WITHOUT flipping `is_valid` (the loader ignores
    extras via pydantic `extra="ignore"`, so both layers still agree)."""

    def test_state_typo_key_warns_and_stays_valid(self):
        data = _valid_fsm_data()
        data["states"]["start"]["requred_context_keys"] = ["email"]
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        assert any("requred_context_keys" in w for w in result.warnings), (
            f"no unknown-key warning naming the typo: {result.warnings}"
        )

    def test_free_form_logic_keys_do_not_warn(self):
        """Vacuity / false-positive guard (Pre-Mortem 3): operator keys inside a
        condition's `logic` are DATA, not model field names -- they must NOT
        produce unknown-key warnings."""
        data = _fsm_with_condition_logic({"==": [{"var": "email"}, 1]})
        result = FSMValidator(data).validate()
        assert result.is_valid is True
        assert not any("Unknown key" in w for w in result.warnings), result.warnings


# ------------------------------------------------------------------
# F-15 / F-23: required_context_keys cross-check and dead-branch removal
# ------------------------------------------------------------------


def _fsm_with_required_keys(declared, gated):
    """`_pydantic_clean_fsm_data()` where `start` declares `declared` as
    required_context_keys and its single transition condition gates on `gated`.

    ADVERSARIALLY PAIRED: the two SC-14 cases differ ONLY in the `gated` list,
    so the presence/absence of the warning is attributable to the cross-check
    and to nothing else in the fixture.
    """
    data = _pydantic_clean_fsm_data()
    start = data["states"]["start"]
    start["required_context_keys"] = list(declared)
    start["transitions"][0]["conditions"] = [
        {
            "description": "gate",
            "requires_context_keys": list(gated),
        }
    ]
    return data


def _required_key_warnings(result):
    return [w for w in result.warnings if "required_context_keys" in w]


class TestRequiredContextKeysCrossCheck:
    """F-15 / SC-14.

    The check used to ask only "does ANY transition have ANY non-empty
    `requires_context_keys`?", so gating on a completely different key silenced
    it -- the exact defect it exists to catch.
    """

    def test_declared_key_gated_by_a_different_key_warns(self):
        result = FSMValidator(
            _fsm_with_required_keys(declared=["email"], gated=["phone"])
        ).validate()
        warnings = _required_key_warnings(result)
        assert warnings, "mismatched required key produced no warning"
        assert any("email" in w for w in warnings)
        # WARNING tier only -- never promoted to ERROR.
        assert result.is_valid is True

    def test_declared_key_actually_gated_does_not_warn(self):
        """MANDATORY vacuity guard.

        Without this, the test above is satisfied by an implementation that
        warns unconditionally on every state with required_context_keys.
        """
        result = FSMValidator(
            _fsm_with_required_keys(declared=["email"], gated=["email"])
        ).validate()
        assert _required_key_warnings(result) == []

    def test_both_branches_are_actually_exercised(self):
        """Proves the pair above straddles the branch rather than sharing a side."""
        mismatched = FSMValidator(
            _fsm_with_required_keys(declared=["email"], gated=["phone"])
        ).validate()
        matched = FSMValidator(
            _fsm_with_required_keys(declared=["email"], gated=["email"])
        ).validate()
        assert bool(_required_key_warnings(mismatched)) is True
        assert bool(_required_key_warnings(matched)) is False

    def test_partial_match_warns_only_about_the_ungated_key(self):
        """The report is per-key, not all-or-nothing."""
        result = FSMValidator(
            _fsm_with_required_keys(declared=["email", "phone"], gated=["email"])
        ).validate()
        warnings = _required_key_warnings(result)
        assert len(warnings) == 1
        # Only the UNGATED key is reported as the problem; `email` appears only
        # in the trailing "conditions require ..." context, never as a complaint.
        assert "declares required_context_keys ['phone']" in warnings[0]

    def test_no_conditions_at_all_still_warns(self):
        """The original all-or-nothing case must keep its original message."""
        data = _pydantic_clean_fsm_data()
        data["states"]["start"]["required_context_keys"] = ["email"]
        result = FSMValidator(data).validate()
        warnings = _required_key_warnings(result)
        assert warnings
        assert "no transitions with conditions requiring these keys" in warnings[0]


class TestNoStatesDefinedBranchRemoved:
    """F-15's neighbour F-23 / SC-15.

    The removed branch was unreachable: an empty `states` dict can never contain
    a truthy `initial_state`, so the earlier arm always returns first.
    """

    def test_empty_states_reports_the_initial_state_error(self):
        result = FSMValidator(
            {"name": "x", "initial_state": "start", "states": {}}
        ).validate()
        assert any("not found in states" in e for e in result.errors)
        assert not any("No states defined" in e for e in result.errors)

    def test_forcing_empty_states_post_construction_is_unchanged(self):
        """The finding's own probe shape: even bypassing construction, the
        deleted branch was never the one that fired."""
        validator = FSMValidator(_pydantic_clean_fsm_data())
        validator.states = {}
        validator._validate_fsm_structure()
        assert validator.result.errors == ["Initial state 'start' not found in states"]

    def test_absent_initial_state_still_reports_its_own_error(self):
        """Vacuity guard for the arm above: the two error paths stay distinct."""
        result = FSMValidator({"name": "x", "states": {}}).validate()
        assert any("No initial state defined" in e for e in result.errors)


# ------------------------------------------------------------------
# D-018 / D-006: fsm-llm-validate must agree with API.from_file
# ------------------------------------------------------------------


def _write_fsm(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestValidatorAgreesWithFromFile:
    """The pair below IS the defect being closed.

    Either half alone is insufficient: `is_valid is False` on its own does not
    show the file was ever loadable-in-question, and `from_file` raising on its
    own does not show the validator noticed. Only running the SAME bytes
    through BOTH paths demonstrates the divergence is gone.
    """

    def test_validator_and_loader_agree_on_rejecting_missing_fields(self, tmp_path):
        """SC-07: same file, both paths, both reject."""
        path = _write_fsm(tmp_path, "broken.json", _missing_required_fields_fsm_data())

        result = validate_fsm_from_file(str(path))
        assert result.is_valid is False
        assert any(e.startswith("Schema:") for e in result.errors)

        with pytest.raises(ValueError, match="validation errors for FSMDefinition"):
            API.from_file(str(path))

    def test_validator_and_loader_agree_on_accepting_the_paired_control(self, tmp_path):
        """The adversarial control: the SAME file with only `id`/`description`
        restored must be accepted by BOTH paths.

        Without this half, a validator that rejected everything would pass the
        test above.
        """
        path = _write_fsm(tmp_path, "clean.json", _valid_fsm_data())

        assert validate_fsm_from_file(str(path)).is_valid is True
        assert API.from_file(str(path)) is not None

    def test_cli_exit_code_matches_loader_verdict(self, tmp_path):
        """SC-10: end-to-end through the real `fsm-llm-validate` console script.

        The user-visible property is the CLI's EXIT CODE, not an internal
        boolean -- external CI gates on the exit status. Adversarially paired:
        the two files differ by exactly the `description`/`id` fields.
        """
        broken = _write_fsm(
            tmp_path, "cli_broken.json", _missing_required_fields_fsm_data()
        )
        clean = _write_fsm(tmp_path, "cli_clean.json", _valid_fsm_data())

        with patch.object(sys, "argv", ["fsm-llm-validate", "--fsm", str(broken)]):
            with pytest.raises(SystemExit) as exc_info:
                main_cli()
        assert exc_info.value.code == 1

        with patch.object(sys, "argv", ["fsm-llm-validate", "--fsm", str(clean)]):
            with pytest.raises(SystemExit) as exc_info:
                main_cli()
        assert exc_info.value.code == 0


# ------------------------------------------------------------------
# F-04 / SC-16: the CLI must actually SAY something
# ------------------------------------------------------------------


@contextmanager
def _cli_capture():
    """Run a console-script body from the library's real default state.

    `logging.py` calls `logger.disable("fsm_llm")` at import; this restores
    that state first, so the capture measures whether `main()` opts BACK IN --
    which is exactly the defect. A sink is attached instead of using capsys
    because loguru binds `sys.stderr` at handler-add time and never sees
    pytest's replacement.
    """
    buffer = io.StringIO()
    logger.disable("fsm_llm")
    sink_id = logger.add(buffer, format="{message}", level="DEBUG")
    try:
        yield buffer
    finally:
        logger.remove(sink_id)
        logger.disable("fsm_llm")


class TestValidateCliEmitsOutput:
    """F-04. `test_cli_exit_code_matches_loader_verdict` above pins the exit
    code and was BLIND to this: the tool wrote ZERO bytes on both the success
    and the failure path, so an exit-code-only assertion is satisfied by a
    strictly worse outcome (a silent tool). These assert CONTENT.
    """

    def test_valid_fsm_reports_the_verdict(self, tmp_path):
        clean = _write_fsm(tmp_path, "out_clean.json", _valid_fsm_data())

        with _cli_capture() as buffer:
            with patch.object(sys, "argv", ["fsm-llm-validate", "--fsm", str(clean)]):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()

        output = buffer.getvalue()
        assert exc_info.value.code == 0, "exit code must be unchanged by the fix"
        assert "Validation result:" in output
        assert '"is_valid": true' in output, (
            f"the verdict itself must reach the user, got: {output!r}"
        )

    def test_invalid_fsm_reports_the_failing_verdict(self, tmp_path):
        broken = _write_fsm(
            tmp_path, "out_broken.json", _missing_required_fields_fsm_data()
        )

        with _cli_capture() as buffer:
            with patch.object(sys, "argv", ["fsm-llm-validate", "--fsm", str(broken)]):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()

        output = buffer.getvalue()
        assert exc_info.value.code == 1, "exit code must be unchanged by the fix"
        assert '"is_valid": false' in output
        assert "Schema:" in output, (
            f"the user must be told WHY it failed, got: {output!r}"
        )

    def test_missing_file_names_the_missing_path(self, tmp_path):
        """SC-16's failure path: the user must learn WHICH file was not found,
        not merely that the process exited 1."""
        missing = tmp_path / "definitely_absent.json"

        with _cli_capture() as buffer:
            with patch.object(sys, "argv", ["fsm-llm-validate", "--fsm", str(missing)]):
                with pytest.raises(SystemExit) as exc_info:
                    main_cli()

        output = buffer.getvalue()
        assert exc_info.value.code == 1
        assert str(missing) in output, (
            f"the missing path must be named, got: {output!r}"
        )
        assert "not found" in output

    def test_library_import_still_leaves_logging_disabled(self):
        """Over-correction guard. The fix must stay INSIDE `main()`.

        If anyone "simplifies" it by deleting `logging.py`'s import-time
        `logger.disable("fsm_llm")`, or by hoisting the enable to module
        scope, importing the library would start spamming a host
        application's stderr. That is the root cause the fix deliberately
        does NOT touch, so this fails if it is removed.
        """
        source = inspect.getsource(fsm_llm_logging)
        assert 'logger.disable("fsm_llm")' in source

        buffer = io.StringIO()
        logger.disable("fsm_llm")
        sink_id = logger.add(buffer, format="{message}", level="DEBUG")
        try:
            importlib.reload(fsm_llm_logging)
            FSMValidator(_valid_fsm_data()).validate()
        finally:
            logger.remove(sink_id)
            logger.disable("fsm_llm")

        assert buffer.getvalue() == "", (
            "merely using the library (no CLI) must stay silent; "
            f"got: {buffer.getvalue()!r}"
        )
