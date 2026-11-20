"""Tests for ``fsm_llm.core.codes``: the status, reason and completion-kind vocabulary.

The pinned modules (``synthesis.constants``, ``agents.constants``) are only
read here, never edited: a bench manifest pins their hashes. This file is what
stops the vocabulary from drifting away from them.
"""

from __future__ import annotations

import ast
import json
from enum import Enum
from pathlib import Path
from typing import ClassVar, get_args

import pytest

from fsm_llm.agents.constants import StopReason, ToolRunStatus
from fsm_llm.core import codes
from fsm_llm.core.codes import (
    CompletionKind,
    Lifecycle,
    MessageSource,
    Reason,
    Status,
    approval_code,
    status_of_stop_reason,
    status_of_tool_status,
)
from fsm_llm.core.definitions import CompletionResponse
from fsm_llm.core.definitions import fsm as definitions
from fsm_llm.synthesis import constants as synth_constants
from fsm_llm.synthesis import tools as synth_tools

_ENUMS = [Status, Reason, CompletionKind, Lifecycle, MessageSource]


def _public_values(cls: type) -> dict[str, str]:
    """The wire strings of a str enum (its members) or of a plain constants class
    (its public string attributes, no dunders, no sets), name to value."""
    if issubclass(cls, Enum):
        return {member.name: member.value for member in cls}
    return {
        name: value
        for name, value in vars(cls).items()
        if not name.startswith("_") and isinstance(value, str)
    }


class TestMembers:
    def test_status_members_and_values(self):
        assert [m.name for m in Status] == [
            "OK",
            "FAILED",
            "ABORTED",
            "RUNNING",
            "ABSTAINED",
        ]
        assert [m.value for m in Status] == [
            "ok",
            "failed",
            "aborted",
            "running",
            "abstained",
        ]

    def test_reason_values_are_the_pinned_tuple_then_two_more(self):
        values = [m.value for m in Reason]
        assert len(values) == 17
        assert values[:15] == list(synth_constants.REASONS)
        assert values[15] == "needs_user_input"
        assert values[16] == "cancelled"

    def test_cancelled_is_outside_the_pinned_reasons(self):
        # `cancelled` is run-level vocabulary (roadmap 0.16, 4.8): no synthesis
        # string names it, so the pinned tuple stays 15 long.
        assert Reason.CANCELLED.value == "cancelled"
        assert "cancelled" not in synth_constants.REASONS
        assert Reason.from_string("cancelled") is Reason.CANCELLED

    def test_needs_user_input_is_outside_the_pinned_reasons(self):
        # `needs_user_input` is written by `synthesis.tools` but is not in the
        # pinned REASONS tuple (a bench manifest pins that file). If a later
        # change adds it there, this assertion is the notice to reconcile.
        assert synth_tools._NEEDS_USER_INPUT == "needs_user_input"
        assert Reason.NEEDS_USER_INPUT.value == synth_tools._NEEDS_USER_INPUT
        assert "needs_user_input" not in synth_constants.REASONS

    def test_completion_kind_members_and_values(self):
        assert [m.name for m in CompletionKind] == ["CALLS", "FINAL", "MALFORMED"]
        assert [m.value for m in CompletionKind] == ["calls", "final", "malformed"]

    def test_lifecycle_members_and_values(self):
        assert [m.name for m in Lifecycle] == [
            "DRAFT",
            "VERIFIED",
            "TRUSTED",
            "DEPRECATED",
            "BUILTIN",
        ]
        assert [m.value for m in Lifecycle] == [
            "draft",
            "verified",
            "trusted",
            "deprecated",
            "builtin",
        ]

    def test_message_source_members_and_values(self):
        assert [m.name for m in MessageSource] == [
            "USER",
            "RUN",
            "REMOTE_CLIENT",
            "SIMULATOR",
        ]
        assert [m.value for m in MessageSource] == [
            "user",
            "run",
            "remote_client",
            "simulator",
        ]

    @pytest.mark.parametrize("enum", _ENUMS)
    def test_members_are_snake_case_with_upper_names(self, enum):
        for member in enum:
            assert member.name == member.name.upper()
            assert member.value == member.name.lower()


class TestFromString:
    @pytest.mark.parametrize("enum", _ENUMS)
    def test_accepts_members_and_exact_strings(self, enum):
        for member in enum:
            assert enum.from_string(member) is member
            assert enum.from_string(member.value) is member

    @pytest.mark.parametrize("enum", _ENUMS)
    @pytest.mark.parametrize("bad", ["bogus", "", None, 3, 1.5])
    def test_refuses_everything_else(self, enum, bad):
        with pytest.raises(ValueError) as exc:
            enum.from_string(bad)
        message = str(exc.value)
        assert repr(bad) in message
        assert str([m.value for m in enum]) in message
        assert f"{enum.__name__}." not in message

    def test_refuses_a_near_miss(self):
        for bad in ("OK", " ok", "Ok"):
            with pytest.raises(ValueError):
                Status.from_string(bad)


class TestDropIn:
    @pytest.mark.parametrize("enum", _ENUMS)
    def test_equal_hash_and_json_like_the_plain_string(self, enum):
        for member in enum:
            assert member == member.value
            assert hash(member) == hash(member.value)
            assert json.dumps({"s": member}) == json.dumps({"s": member.value})

    def test_usable_as_a_dict_key_with_the_plain_string(self):
        table = {Status.OK: 1}
        assert table["ok"] == 1
        assert {"ok": 1}[Status.OK] == 1
        assert json.dumps({"s": Status.OK}) == '{"s": "ok"}'


class TestAgainstThePinnedVocabularies:
    def test_status_equals_the_synthesis_status_constants(self):
        assert Status.OK == synth_constants.STATUS_OK
        assert Status.FAILED == synth_constants.STATUS_FAILED
        assert Status.ABORTED == synth_constants.STATUS_ABORTED
        assert Status.RUNNING == synth_constants.STATUS_RUNNING

    def test_lifecycle_starts_with_the_synthesis_lifecycles_in_order(self):
        values = [m.value for m in Lifecycle]
        assert values[:4] == list(synth_constants.LIFECYCLES)
        assert values[4] == "builtin"
        assert "builtin" not in synth_constants.LIFECYCLES

    def test_completion_kind_equals_the_response_literal(self):
        annotation = CompletionResponse.model_fields["kind"].annotation
        assert [m.value for m in CompletionKind] == list(get_args(annotation))

    def test_the_synthesize_kind_samples_are_the_status_strings(self):
        """The samples live in the `state_kinds` leaf, which cannot import
        `codes.Status`; the loader's name for them is the same tuple, and the
        one status the hook compares against is the enum's value."""
        from fsm_llm.core import state_kinds

        assert definitions._SYNTHESIZE_SAMPLES is state_kinds.SYNTHESIZE_SAMPLES
        assert state_kinds._RUNNING == Status.RUNNING.value

    def test_synthesize_samples_use_status_values(self):
        statuses = {
            sample["status"]
            for _, sample in definitions._SYNTHESIZE_SAMPLES
            if sample is not None
        }
        assert statuses == {"running", "failed", "aborted"}
        assert all(Status.from_string(s) in Status for s in statuses)
        reasons = {
            sample["reason"]
            for _, sample in definitions._SYNTHESIZE_SAMPLES
            if sample is not None and "reason" in sample
        }
        assert all(Reason.from_string(r) in Reason for r in reasons)


class TestStatusOfStopReason:
    _EXPECTED: ClassVar[dict] = {
        StopReason.ANSWERED: (Status.OK, None),
        StopReason.EVIDENCE: (Status.OK, None),
        StopReason.MAX_ITERATIONS: (Status.ABORTED, Reason.BUDGET_SPENT),
        StopReason.ENDED: (Status.ABORTED, None),
        StopReason.FORCED_PASS: (Status.FAILED, None),
        StopReason.STALLED: (Status.FAILED, None),
        StopReason.VERIFICATION_FAILED: (Status.FAILED, None),
        StopReason.NO_RESULT: (Status.FAILED, None),
        StopReason.GATE_FAILED: (Status.FAILED, None),
        StopReason.TRUNCATED: (Status.FAILED, None),
    }

    def test_every_public_stop_reason_is_mapped(self):
        public = set(_public_values(StopReason).values())
        assert public == set(self._EXPECTED)

    @pytest.mark.parametrize("stop_reason", sorted(_EXPECTED))
    def test_mapping(self, stop_reason):
        assert status_of_stop_reason(stop_reason) == self._EXPECTED[stop_reason]

    def test_every_public_stop_reason_maps(self):
        for value in _public_values(StopReason).values():
            status, reason = status_of_stop_reason(value)
            assert isinstance(status, Status)
            assert reason is None or isinstance(reason, Reason)

    @pytest.mark.parametrize("bad", ["bogus", "", "OK"])
    def test_unknown_value_raises(self, bad):
        with pytest.raises(ValueError, match="stop_reason"):
            status_of_stop_reason(bad)


class TestStatusOfToolStatus:
    _EXPECTED: ClassVar[dict] = {
        ToolRunStatus.SUCCESS: (Status.OK, None),
        ToolRunStatus.FAILED: (Status.FAILED, None),
        ToolRunStatus.SKIPPED: (Status.ABSTAINED, None),
        ToolRunStatus.REJECTED: (Status.FAILED, Reason.APPROVAL_DENIED),
        ToolRunStatus.AWAITING_APPROVAL: (Status.RUNNING, Reason.APPROVAL_REQUIRED),
    }

    def test_every_public_status_but_unknown_is_mapped(self):
        public = set(_public_values(ToolRunStatus).values())
        assert public - {ToolRunStatus.UNKNOWN} == set(self._EXPECTED)

    @pytest.mark.parametrize("tool_status", sorted(_EXPECTED))
    def test_mapping(self, tool_status):
        assert status_of_tool_status(tool_status) == self._EXPECTED[tool_status]

    def test_unknown_is_refused_not_reported_as_failed(self):
        # `agents/handlers/executor.py` D-043: a timed-out call may still be running and
        # may still take effect, so "failed" (safe to run again) would be a lie.
        with pytest.raises(ValueError) as exc:
            status_of_tool_status(ToolRunStatus.UNKNOWN)
        assert "unknown outcome must not be reported as failed" in str(exc.value)

    @pytest.mark.parametrize("bad", ["bogus", ""])
    def test_other_unknown_values_raise(self, bad):
        with pytest.raises(ValueError, match="tool_status"):
            status_of_tool_status(bad)


class TestApprovalCode:
    @pytest.mark.parametrize(
        "approved, errored, expected",
        [
            (True, False, (Status.OK, None)),
            (False, False, (Status.FAILED, Reason.APPROVAL_DENIED)),
            (None, False, (Status.FAILED, Reason.APPROVAL_REQUIRED)),
            (True, True, (Status.FAILED, Reason.REVIEWER_ERROR)),
            (False, True, (Status.FAILED, Reason.REVIEWER_ERROR)),
            (None, True, (Status.FAILED, Reason.REVIEWER_ERROR)),
        ],
    )
    def test_table(self, approved, errored, expected):
        assert approval_code(approved, errored=errored) == expected

    def test_errored_is_keyword_only(self):
        with pytest.raises(TypeError):
            approval_code(True, True)  # type: ignore[misc]


class TestSource:
    """``codes.py`` stays stdlib-only and never formats a member into text."""

    _SOURCE = Path(codes.__file__).read_text(encoding="utf-8")

    def test_imports_nothing_from_the_package(self):
        tree = ast.parse(self._SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import in codes.py"
                assert not (node.module or "").startswith("fsm_llm")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("fsm_llm")

    def test_messages_use_value_not_member_formatting(self):
        tree = ast.parse(self._SOURCE)
        member_names = {
            "Status",
            "Reason",
            "CompletionKind",
            "Lifecycle",
            "MessageSource",
        }

        def _names_a_member(node: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id in member_names
                and n.attr.isupper()
                for n in ast.walk(node)
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.FormattedValue):
                assert not _names_a_member(node.value), ast.dump(node)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "str"
            ):
                assert not any(_names_a_member(a) for a in node.args), ast.dump(node)

    def test_all_is_static_and_complete(self):
        assert sorted(codes.__all__) == sorted(
            [
                "CompletionKind",
                "Lifecycle",
                "MessageSource",
                "Reason",
                "Status",
                "approval_code",
                "status_of_stop_reason",
                "status_of_tool_status",
            ]
        )
