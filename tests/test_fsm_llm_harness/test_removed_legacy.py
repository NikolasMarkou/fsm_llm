"""Absence tests for the harness names and read paths removed as legacy.

Each test fails on the commit before the removal: the four unread
``Defaults`` constants and the ``storage._atomic_write_text`` alias existed,
and the cross-plan reader and the tool routing hint accepted the retired
``plan_YYYY-MM-DD_hex8`` directory name.
"""

from __future__ import annotations

import pytest

import fsm_llm.harness as harness
from fsm_llm.harness import artifacts, storage
from fsm_llm.harness.artifacts import ConsolidatedDoc
from fsm_llm.harness.constants import Defaults
from fsm_llm.harness.tools import _addresses_plan_memory

CURRENT_PLAN_ID = "plan-2026-07-21T191807-bf7ffe24"
LEGACY_PLAN_ID = "plan_2026-05-07_7556fb98"
COMMIT_TAG_FORM = "plan-2026-07-21-bf7ffe24"

CONSOLIDATED_MD = "\n\n".join(
    [
        "# Consolidated Findings",
        "*Cross-plan findings archive. Newest first.*",
        f"## {CURRENT_PLAN_ID}\n### Index\n\n1. harness tier — `findings/tier.md`",
        f"## {LEGACY_PLAN_ID}\n### Index\n\n1. old layout — `findings/old.md`",
        f"## {COMMIT_TAG_FORM}\n### Index\n\n1. tag form — `findings/tag.md`",
    ]
)


class TestRemovedLegacy:
    @pytest.mark.parametrize(
        "name",
        [
            "DECISIONS_COMPRESS_LINES",
            "CHANGELOG_COMPRESS_LINES",
            "LESSONS_IMPORTANCE_MIN",
            "LESSONS_IMPORTANCE_MAX",
        ],
    )
    def test_unread_defaults_are_gone(self, name: str) -> None:
        assert not hasattr(Defaults, name)

    def test_live_compression_constants_are_kept(self) -> None:
        assert ConsolidatedDoc.COMPRESS_LINES == Defaults.CONSOLIDATED_COMPRESS_LINES
        assert Defaults.LESSONS_PROTECTED_IMPORTANCE == 5

    def test_storage_has_no_private_atomic_write_alias(self) -> None:
        assert not hasattr(storage, "_atomic_write_text")

    def test_plan_id_shape_is_defined_once(self) -> None:
        assert harness.PLAN_ID_RE is artifacts.PLAN_ID_RE
        assert not hasattr(storage, "PLAN_ID_RE")
        assert "PLAN_ID_RE" in artifacts.__all__
        assert "PLAN_ID_RE" not in storage.__all__
        assert not hasattr(artifacts, "_PLAN_ID_RE")

    @pytest.mark.parametrize("plan_id", [LEGACY_PLAN_ID, COMMIT_TAG_FORM])
    def test_plan_id_shape_refuses_retired_forms(self, plan_id: str) -> None:
        assert harness.PLAN_ID_RE.match(plan_id) is None

    def test_consolidated_reader_lists_only_minted_plan_ids(self) -> None:
        doc = ConsolidatedDoc.from_markdown(CONSOLIDATED_MD)
        assert doc.plan_ids() == [CURRENT_PLAN_ID]

    def test_legacy_plan_directory_head_is_not_plan_memory(self) -> None:
        assert _addresses_plan_memory(f"{LEGACY_PLAN_ID}/state.md") is False
        assert _addresses_plan_memory(f"{CURRENT_PLAN_ID}/state.md") is True
