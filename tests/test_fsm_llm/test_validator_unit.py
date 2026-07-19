"""
Unit tests for fsm_llm.validator module.

Tests cover:
- FSMValidationResult: error/warning/info accumulation, is_valid property
- FSMValidator.validate: valid FSMs, orphaned states, unreachable terminals, missing initial
- Cycle detection: simple cycles, no cycles, self-loops
- Complex FSM with multiple issues
"""

import importlib
import inspect
import io
import json
import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from loguru import logger
from pydantic import ValidationError

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

    def test_simplified_dict_leniency_preserved(self):
        """Type-coercion leniency is NOT part of the `missing` widening.

        A wrong scalar type (`priority: "not-an-int"`) raises `int_parsing`,
        which is outside the promotion allow-list and must stay a warning. The
        FSM is otherwise pydantic-clean, so the verdict is attributable to the
        coercion failure alone.
        """
        data = _valid_fsm_data()
        data["states"]["start"]["priority"] = "not-an-int"
        data["states"]["start"]["transitions"][0]["priority"] = "also-not-an-int"
        result = FSMValidator(data).validate()
        schema_errors = [e for e in result.errors if e.startswith("Schema:")]
        assert schema_errors == [], (
            f"structural/type mismatch was promoted to an error: {schema_errors}"
        )
        assert result.is_valid is True

    def test_unknown_error_type_falls_through_to_warning(self):
        """Fail-safe: an unrecognized/future pydantic error type must warn, not error.

        The promotion set is an ALLOW-list, so a new pydantic version can never
        make the validator accidentally stricter than the loader. Widening it to
        cover `missing` must NOT have flipped it into a deny-list -- if it had,
        this synthetic unknown type would be promoted to an error.
        """
        validator = FSMValidator(_valid_fsm_data())
        fake = ValidationError.from_exception_data(
            "FSMDefinition",
            [
                {
                    "type": "string_too_short",
                    "loc": ("name",),
                    "input": "",
                    "ctx": {"min_length": 1},
                }
            ],
        )
        with patch("fsm_llm.validator.FSMDefinition", side_effect=fake):
            validator._validate_pydantic_schema()
        assert validator.result.errors == []
        assert validator.result.is_valid is True
        assert any("Schema:" in w for w in validator.result.warnings)

    def test_list_length_promotion_did_not_widen_to_string_lengths(self):
        """D-013 over-correction guard.

        `too_short`/`too_long` (LIST bounds) were promoted. The distinct
        `string_too_short`/`string_too_long` type names were NOT, and must keep
        falling through to WARNING. Without this the promotion could be
        "simplified" to a `startswith("too_")` match, which would silently
        capture the string classes too.
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
            assert validator.result.errors == [], (
                f"{type_name} was promoted to an error; only the LIST-length "
                f"classes are in the allow-list"
            )
            assert any("Schema:" in w for w in validator.result.warnings)


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
