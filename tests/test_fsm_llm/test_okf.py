"""Tests for the Open Knowledge Format (OKF) support.

This file grows with the OKF plan; the first class guards the one
path-confinement helper that the OKF loader and writer share with the harness
workspace, the rest cover the tolerant parser, trust tiers and staleness.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

import fsm_llm.okf as okf_module
from fsm_llm.constants import (
    OKF_MAX_DEPTH,
    OKF_MAX_FILE_BYTES,
    OKF_MAX_FILES,
    OKF_MAX_FRONTMATTER_BYTES,
    OKF_MAX_TOTAL_BYTES,
)
from fsm_llm.definitions import FSMError, OKFError
from fsm_llm.okf import (
    OKFBundle,
    OKFConcept,
    OKFIssue,
    OKFLimits,
    TrustTier,
    check_bundle,
    load_bundle,
    parse_concept_text,
    parse_okf_datetime,
    strip_unsafe_text,
)
from fsm_llm.security import resolve_within


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - platform without symlink support
        pytest.skip("symlinks are not supported here")


class TestResolveWithin:
    """``security.resolve_within`` resolves first and compares components."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "ws").mkdir()
        return (tmp_path / "ws").resolve()

    def test_a_plain_relative_path_resolves_inside(self, root: Path) -> None:
        assert resolve_within(root, "a/b.txt") == root / "a" / "b.txt"

    def test_dot_returns_the_root_itself(self, root: Path) -> None:
        assert resolve_within(root, ".") == root

    def test_a_non_existent_target_inside_is_allowed(self, root: Path) -> None:
        result = resolve_within(root, "does/not/exist.md")
        assert result == root / "does" / "not" / "exist.md"
        assert not result.exists()

    def test_a_path_like_argument_is_accepted(self, root: Path) -> None:
        assert resolve_within(root, Path("a") / "b") == root / "a" / "b"

    def test_dotdot_escape_is_refused(self, root: Path) -> None:
        assert resolve_within(root, "..") is None
        assert resolve_within(root, "../x") is None

    def test_a_nested_dotdot_escape_is_refused(self, root: Path) -> None:
        assert resolve_within(root, "a/../../x") is None

    def test_a_dotdot_that_stays_inside_is_allowed(self, root: Path) -> None:
        assert resolve_within(root, "a/../b") == root / "b"

    def test_an_absolute_path_outside_is_refused(
        self, root: Path, tmp_path: Path
    ) -> None:
        assert resolve_within(root, str(tmp_path / "elsewhere.txt")) is None
        assert resolve_within(root, "/etc/passwd") is None

    def test_an_absolute_path_inside_is_allowed(self, root: Path) -> None:
        inside = root / "a.txt"
        assert resolve_within(root, str(inside)) == inside

    def test_a_sibling_with_a_shared_prefix_is_refused(
        self, root: Path, tmp_path: Path
    ) -> None:
        """``ws-evil`` startswith ``ws``; the comparison is on components."""
        evil = tmp_path / "ws-evil"
        evil.mkdir()
        (evil / "loot.txt").write_text("loot\n")
        _symlink(root / "link", evil)

        assert resolve_within(root, "link/loot.txt") is None
        assert resolve_within(root, "../ws-evil/loot.txt") is None

    def test_a_symlink_pointing_out_is_refused(
        self, root: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret\n")
        _symlink(root / "escape", outside)

        assert resolve_within(root, "escape") is None
        assert resolve_within(root, "escape/secret.txt") is None
        # A lexical `link/../..` fold would look contained and is not.
        assert resolve_within(root, "escape/../x") is None

    def test_a_symlink_staying_inside_is_allowed(self, root: Path) -> None:
        (root / "real").mkdir()
        (root / "real" / "f.txt").write_text("x\n")
        _symlink(root / "alias", root / "real")

        assert resolve_within(root, "alias/f.txt") == root / "real" / "f.txt"

    def test_a_nul_byte_raises_value_error(self, root: Path) -> None:
        with pytest.raises(ValueError):
            resolve_within(root, "a\x00b")

    def test_a_symlink_loop_raises(self, root: Path) -> None:
        _symlink(root / "a", root / "b")
        _symlink(root / "b", root / "a")

        with pytest.raises((RuntimeError, OSError)):
            resolve_within(root, "a")

    def test_an_unresolved_root_fails_closed(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        alias = tmp_path / "alias"
        _symlink(alias, real)

        # The root is reached through a symlink, so it is not resolved: the
        # resolved candidate never equals it nor has it as a parent.
        assert resolve_within(alias, "a.txt") is None
        assert resolve_within(alias, ".") is None


UTC = timezone.utc


def _parse(frontmatter: str, body: str = "body\n") -> tuple[OKFConcept, list[str]]:
    """Parse ``---\n<frontmatter>---\n<body>``; return the concept and issue codes."""
    concept, issues = parse_concept_text("a/b", f"---\n{frontmatter}---\n{body}")
    return concept, [i.code for i in issues]


class TestFrontmatter:
    """Splitting and loading the frontmatter block."""

    def test_a_minimal_concept_is_conformant(self) -> None:
        concept, codes = _parse("type: Table\n")

        assert concept.id == "a/b"
        assert concept.type == "Table"
        assert concept.body == "body\n"
        assert codes == []

    def test_every_known_field_is_read(self) -> None:
        concept, codes = _parse(
            "type: Table\ntitle: Customers\ndescription: One row each.\n"
            "resource: bq://p.d.t\ntags: [a, b]\nstatus: stable\n"
            "sources:\n  - resource: /x.md\n    id: s1\n"
            "generated: {by: tool/1, at: '2026-01-01T00:00:00Z'}\n"
            "verified:\n  - {by: 'human:ann', at: '2026-01-02T00:00:00Z'}\n"
            "stale_after: '2027-01-01T00:00:00Z'\n"
        )

        assert codes == []
        assert (concept.title, concept.description) == ("Customers", "One row each.")
        assert concept.resource == "bq://p.d.t"
        assert concept.tags == ["a", "b"]
        assert concept.sources == [{"resource": "/x.md", "id": "s1"}]
        assert concept.generated == {"by": "tool/1", "at": "2026-01-01T00:00:00Z"}
        assert concept.stale_after == "2027-01-01T00:00:00Z"
        assert concept.extra == {}

    def test_unknown_keys_are_kept_in_order(self) -> None:
        concept, codes = _parse("zeta: 1\ntype: T\nalpha: {x: [1, 2]}\ntimestamp: x\n")

        assert codes == []
        assert list(concept.extra) == ["zeta", "alpha", "timestamp"]
        assert concept.extra["alpha"] == {"x": [1, 2]}

    def test_bom_and_crlf_are_handled(self) -> None:
        text = "\ufeff---\r\ntype: T\r\ntitle: Hi\r\n---\r\nline one\r\nline two\r\n"

        concept, issues = parse_concept_text("c", text)

        assert (concept.type, concept.title) == ("T", "Hi")
        assert concept.body == "line one\nline two\n"
        assert issues == []

    def test_a_closing_marker_may_carry_trailing_blanks(self) -> None:
        concept, issues = parse_concept_text("c", "---\ntype: T\n---  \nbody")

        assert (concept.type, concept.body, issues) == ("T", "body", [])

    def test_empty_frontmatter_is_an_issue_not_an_error(self) -> None:
        concept, issues = parse_concept_text("c", "---\n---\nbody\n")

        assert concept.body == "body\n"
        assert [i.code for i in issues] == ["frontmatter-empty", "type-missing"]

    def test_missing_frontmatter_keeps_the_text_as_body(self) -> None:
        concept, issues = parse_concept_text("c", "# Just prose\n")

        assert concept.body == "# Just prose\n"
        assert [i.code for i in issues] == ["frontmatter-missing", "type-missing"]

    def test_unterminated_frontmatter_keeps_the_whole_text(self) -> None:
        text = "---\ntype: T\nno closing marker\n"

        concept, issues = parse_concept_text("c", text)

        assert concept.body == text
        assert concept.type == ""
        assert issues[0].code == "frontmatter-unterminated"
        assert issues[0].severity == "error"

    @pytest.mark.parametrize("payload", ["- a\n- b\n", "just a string\n", "42\n"])
    def test_non_mapping_frontmatter_is_an_issue(self, payload: str) -> None:
        concept, codes = _parse(payload)

        assert concept.type == ""
        assert codes == ["frontmatter-not-mapping", "type-missing"]
        assert concept.body == "body\n"

    def test_bad_yaml_is_an_issue_and_the_body_survives(self) -> None:
        concept, codes = _parse("type: [unclosed\n")

        assert codes[0] == "frontmatter-invalid"
        assert concept.body == "body\n"

    def test_two_documents_in_one_block_are_an_issue(self) -> None:
        _, codes = _parse("type: T\n...\nother: 1\n")

        assert "frontmatter-invalid" in codes

    def test_duplicate_keys_warn_and_the_last_wins(self) -> None:
        concept, codes = _parse("type: First\ntype: Second\n")

        assert concept.type == "Second"
        assert codes == ["key-duplicate"]

    def test_non_string_keys_are_read_as_text(self) -> None:
        concept, codes = _parse("type: T\n1: one\nfalse: yes\n")

        assert codes == ["key-not-string", "key-not-string"]
        assert concept.extra == {"1": "one", "False": True}

    def test_a_control_character_in_the_stream_is_an_issue(self) -> None:
        _, codes = _parse("type: T\x00\n")

        assert codes[0] == "frontmatter-invalid"

    def test_explicit_bad_scalars_never_raise(self) -> None:
        for payload in (
            "!!int abc",
            "!!float x",
            "!!timestamp 2026-13-45",
            "!!binary %",
        ):
            concept, codes = _parse(f"type: T\nk: {payload}\n")
            assert codes and concept.body == "body\n"

    def test_an_explicit_timestamp_tag_is_read_as_text(self) -> None:
        concept, codes = _parse("type: T\nstale_after: !!timestamp 2026-06-01\n")

        assert concept.stale_after == "2026-06-01"
        assert "value-coerced" in codes


class TestYamlSafety:
    """Untrusted YAML never runs code, expands, or recurses without bound."""

    def test_a_python_object_payload_does_not_execute(self, tmp_path: Path) -> None:
        marker = tmp_path / "pwned"
        payload = f"type: !!python/object/apply:os.system ['touch {marker}']\n"

        concept, codes = _parse(payload)

        assert not marker.exists()
        assert codes[0] == "frontmatter-invalid"
        assert concept.type == ""

    def test_python_name_and_module_tags_are_refused(self, tmp_path: Path) -> None:
        for tag in (
            "python/name:os.system",
            "python/module:os",
            "python/object:os.path",
        ):
            _, codes = _parse(f"type: T\nk: !!{tag} x\n")
            assert codes[0] == "frontmatter-invalid"

    def test_an_alias_bomb_is_refused_before_it_expands(self) -> None:
        refs = ", ".join(["*a"] * 9)
        levels = "".join(
            f"{name}: &{name} [{', '.join(['*' + prev] * 9)}]\n"
            for prev, name in zip("abcdefgh", "bcdefghi", strict=True)
        )
        text = f"type: T\na: &a [x, x, x, x, x, x, x, x, x]\n{levels}j: [{refs}]\n"

        started = time.perf_counter()
        concept, codes = _parse(text)

        assert time.perf_counter() - started < 1.0
        assert codes[0] == "frontmatter-invalid"
        assert concept.extra == {}

    def test_a_plain_alias_and_a_lone_anchor_are_refused(self) -> None:
        for body in ("type: T\na: &x 1\nb: *x\n", "type: &t T\n"):
            _, codes = _parse(body)
            assert codes[0] == "frontmatter-invalid"

    def test_deeply_nested_flow_sequences_give_an_issue_quickly(self) -> None:
        text = "type: T\nk: " + "[" * 5000 + "]" * 5000 + "\n"

        started = time.perf_counter()
        concept, codes = _parse(text)

        assert time.perf_counter() - started < 1.0
        assert codes[0] == "frontmatter-too-deep"
        assert concept.body == "body\n"

    def test_an_unclosed_deep_flow_sequence_gives_an_issue_quickly(self) -> None:
        started = time.perf_counter()
        _, codes = _parse("type: T\nk: " + "{a: [" * 2500 + "\n")

        assert time.perf_counter() - started < 1.0
        assert codes[0] == "frontmatter-too-deep"

    def test_deep_block_nesting_gives_an_issue_not_a_recursion_error(self) -> None:
        lines = "".join(" " * i + f"k{i}:\n" for i in range(700))

        started = time.perf_counter()
        _, codes = _parse("type: T\n" + lines + " " * 700 + "v: 1\n")

        assert time.perf_counter() - started < 1.0
        # Block nesting deep enough to recurse is far above the frontmatter size
        # bound, so that bound is what stops it.
        assert "frontmatter-too-large" in codes

    def test_global_safe_load_still_resolves_timestamps(self) -> None:
        # The OKF loader copies the resolver table; mutating the shared one would
        # turn every other safe_load in the process into strings.
        assert yaml.safe_load("2026-01-01") == date(2026, 1, 1)
        assert yaml.safe_load("2026-01-01T00:00:00Z") == datetime(
            2026, 1, 1, tzinfo=UTC
        )
        _parse("type: T\nk: 2026-01-01\n")
        assert yaml.safe_load("2026-01-01") == date(2026, 1, 1)

    def test_okf_timestamps_stay_strings(self) -> None:
        concept, _ = _parse(
            "type: T\nstale_after: 2026-06-30T14:00:00Z\nk: 2026-01-01\n"
        )

        assert concept.stale_after == "2026-06-30T14:00:00Z"
        assert concept.extra["k"] == "2026-01-01"


class TestScalarCoercion:
    """Odd shapes of the known keys become issues, never exceptions."""

    @pytest.mark.parametrize(
        ("frontmatter", "code"),
        [
            ("title: x\n", "type-missing"),
            ("type:\n", "type-empty"),
            ("type: ''\n", "type-empty"),
            ("type: '   '\n", "type-empty"),
            ("type: [a]\n", "value-invalid"),
            ("type: {a: b}\n", "value-invalid"),
        ],
    )
    def test_a_missing_or_empty_type_is_an_error_issue(
        self, frontmatter: str, code: str
    ) -> None:
        concept, issues = parse_concept_text("c", f"---\n{frontmatter}---\n")

        assert concept.type == ""
        assert code in [i.code for i in issues]
        assert any(i.severity == "error" for i in issues)

    def test_a_non_string_type_is_read_as_text(self) -> None:
        concept, codes = _parse("type: 7\n")

        assert concept.type == "7"
        assert codes == ["value-coerced"]

    def test_the_type_is_stripped(self) -> None:
        concept, _ = _parse("type: '  Table  '\n")

        assert concept.type == "Table"

    def test_scalar_fields_that_are_not_strings(self) -> None:
        concept, codes = _parse(
            "type: T\ntitle: 2026\ndescription: [a]\nresource: 1.5\n"
        )

        assert (concept.title, concept.description, concept.resource) == (
            "2026",
            None,
            "1.5",
        )
        assert codes == ["value-coerced", "value-invalid", "value-coerced"]

    def test_a_comma_string_of_tags_is_split(self) -> None:
        concept, codes = _parse("type: T\ntags: 'a, b ,, c'\n")

        assert concept.tags == ["a", "b", "c"]
        assert codes == ["tags-string"]

    def test_tags_with_non_strings_are_coerced(self) -> None:
        concept, codes = _parse("type: T\ntags: [a, 2, null, ' ']\n")

        assert concept.tags == ["a", "2"]
        assert codes == ["value-coerced"]

    def test_tags_that_are_a_mapping_are_ignored(self) -> None:
        concept, codes = _parse("type: T\ntags: {a: b}\n")

        assert concept.tags == []
        assert codes == ["tags-invalid"]

    def test_a_bare_verified_mapping_is_a_one_element_list(self) -> None:
        concept, codes = _parse(
            "type: T\nverified: {by: 'process:ci', at: '2026-01-01T00:00:00Z'}\n"
        )

        assert concept.verified == [{"by": "process:ci", "at": "2026-01-01T00:00:00Z"}]
        assert codes == []

    def test_verified_and_sources_drop_non_mapping_entries(self) -> None:
        concept, codes = _parse("type: T\nverified: [a, {by: x}]\nsources: 5\n")

        assert concept.verified == [{"by": "x"}]
        assert concept.sources == []
        assert codes == ["verified-invalid", "sources-invalid"]

    def test_generated_that_is_not_a_mapping_is_ignored(self) -> None:
        concept, codes = _parse("type: T\ngenerated: yesterday\n")

        assert concept.generated is None
        assert codes == ["generated-invalid"]

    def test_an_unknown_status_warns_but_is_kept(self) -> None:
        concept, codes = _parse("type: T\nstatus: Frozen\n")

        assert concept.status == "Frozen"
        assert codes == ["status-unknown"]
        assert not concept.is_deprecated and not concept.is_draft

    @pytest.mark.parametrize("status", ["deprecated", " Deprecated "])
    def test_deprecated_and_draft_flags(self, status: str) -> None:
        assert OKFConcept(id="a", type="T", status=status).is_deprecated
        assert OKFConcept(id="a", type="T", status="draft").is_draft

    def test_issue_text_from_the_document_is_stripped(self) -> None:
        _, issues = parse_concept_text(
            "a\x1b[31m/b\u202e", "---\n\x1b[31m\u202e: x\n---\n"
        )

        assert all("\x1b" not in (i.message + (i.id or "")) for i in issues)
        assert all("\u202e" not in (i.message + (i.id or "")) for i in issues)
        assert issues[0].id == "a/b"


class TestTimestamp:
    """``parse_okf_datetime`` is 3.10-safe and documents every warning."""

    @pytest.mark.parametrize(
        ("text", "expected", "code"),
        [
            ("2026-01-01T00:00:00Z", datetime(2026, 1, 1, tzinfo=UTC), None),
            ("2026-01-01t00:00:00z", datetime(2026, 1, 1, tzinfo=UTC), None),
            ("2026-01-01 00:00:00+00:00", datetime(2026, 1, 1, tzinfo=UTC), None),
            (
                "2026-01-01T00:00:00.5Z",
                datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=UTC),
                None,
            ),
            (
                "2026-01-01T00:00:00.123456789Z",
                datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=UTC),
                None,
            ),
            (
                "2026-01-01T00:00:00,25Z",
                datetime(2026, 1, 1, 0, 0, 0, 250000, tzinfo=UTC),
                None,
            ),
            ("2026-01-01T00:00Z", datetime(2026, 1, 1, tzinfo=UTC), None),
            ("2026-01-01T10:00:00+02:00", datetime(2026, 1, 1, 8, tzinfo=UTC), None),
            (
                "2026-01-01T10:00:00-0530",
                datetime(2026, 1, 1, 15, 30, tzinfo=UTC),
                None,
            ),
            ("2026-01-01T10:00:00+05", datetime(2026, 1, 1, 5, tzinfo=UTC), None),
            ("  2026-01-01T00:00:00Z  ", datetime(2026, 1, 1, tzinfo=UTC), None),
            ("2026-01-01", datetime(2026, 1, 1, tzinfo=UTC), "date-only"),
            ("2026-01-01T10:30:00", datetime(2026, 1, 1, 10, 30, tzinfo=UTC), "naive"),
            ("2026-01-01T10:30", datetime(2026, 1, 1, 10, 30, tzinfo=UTC), "naive"),
            (date(2026, 1, 1), datetime(2026, 1, 1, tzinfo=UTC), "date-only"),
            (datetime(2026, 1, 1, 9), datetime(2026, 1, 1, 9, tzinfo=UTC), "naive"),
            (
                datetime(2026, 1, 1, 9, tzinfo=UTC),
                datetime(2026, 1, 1, 9, tzinfo=UTC),
                None,
            ),
        ],
    )
    def test_accepted_forms(
        self, text: Any, expected: datetime, code: str | None
    ) -> None:
        parsed, warning = parse_okf_datetime(text)

        assert parsed == expected
        assert parsed is not None and parsed.utcoffset() is not None
        assert warning == code

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "garbage",
            "2026-13-01",
            "2026-02-30",
            "2026-01-01T25:00:00Z",
            "2026-01-01T00:00:00+25:00",
            "2026-1-1",
            "2026-01-01T00:00:00Zjunk",
            5,
            None,
            ["x"],
        ],
    )
    def test_rejected_forms(self, text: Any) -> None:
        assert parse_okf_datetime(text) == (None, "invalid")

    def test_each_non_conformant_timestamp_field_warns_once(self) -> None:
        _, codes = _parse(
            "type: T\nstale_after: 2027-01-01\ngenerated: {by: x, at: '2026-01-01T00:00:00'}\n"
            "verified:\n  - {by: x, at: nope}\n"
        )

        assert codes == ["timestamp-date-only", "timestamp-naive", "timestamp-invalid"]

    def test_absent_or_conformant_timestamps_do_not_warn(self) -> None:
        _, codes = _parse("type: T\ngenerated: {by: x}\nverified: [{by: x}]\n")

        assert codes == []


class TestTrust:
    """The tier follows ``verified[].by`` only and is a claim, not authority."""

    @pytest.mark.parametrize(
        ("verified", "tier"),
        [
            ([], TrustTier.UNVERIFIED),
            ([{"by": "process:ci"}], TrustTier.MACHINE_CONFIRMED),
            (
                [{"by": "agent/gemini-2.5"}, {"by": "process:ci"}],
                TrustTier.MACHINE_CONFIRMED,
            ),
            ([{"by": "process:ci"}, {"by": "human:ann"}], TrustTier.HUMAN_REVIEWED),
            ([{"by": " human:ann "}], TrustTier.HUMAN_REVIEWED),
            ([{"by": "Human:ann"}], TrustTier.MACHINE_CONFIRMED),
            ([{"by": "humanoid:x"}], TrustTier.MACHINE_CONFIRMED),
            ([{"at": "2026-01-01T00:00:00Z"}], TrustTier.UNVERIFIED),
            ([{"by": ""}, {"by": None}, {"by": 7}], TrustTier.UNVERIFIED),
        ],
    )
    def test_the_tier_table(
        self, verified: list[dict[Any, Any]], tier: TrustTier
    ) -> None:
        assert OKFConcept(id="a", type="T", verified=verified).trust is tier

    def test_a_bare_verified_mapping_counts(self) -> None:
        concept, _ = _parse("type: T\nverified: {by: 'human:ann'}\n")

        assert concept.trust is TrustTier.HUMAN_REVIEWED

    def test_tier_values_are_plain_strings(self) -> None:
        assert TrustTier.HUMAN_REVIEWED == "human-reviewed"
        assert {t.value for t in TrustTier} == {
            "unverified",
            "machine-confirmed",
            "human-reviewed",
        }

    def test_the_tier_ignores_everything_but_verified(self) -> None:
        concept = OKFConcept(
            id="a", type="T", generated={"by": "human:ann"}, tags=["human:x"]
        )

        assert concept.trust is TrustTier.UNVERIFIED


class TestStaleness:
    """``now >= stale_after`` is stale; anything unusable is not."""

    NOW = datetime(2026, 6, 1, tzinfo=UTC)

    def _concept(self, stale_after: Any) -> OKFConcept:
        return OKFConcept(id="a", type="T", stale_after=stale_after)

    def test_the_boundary_instant_is_stale(self) -> None:
        concept = self._concept("2026-06-01T00:00:00Z")

        assert concept.is_stale(self.NOW)
        assert not concept.is_stale(self.NOW - timedelta(microseconds=1))
        assert concept.is_stale(self.NOW + timedelta(microseconds=1))

    def test_offsets_are_compared_as_instants(self) -> None:
        concept = self._concept("2026-06-01T02:00:00+02:00")

        assert concept.is_stale(self.NOW)
        assert not concept.is_stale(self.NOW - timedelta(seconds=1))

    def test_a_date_only_value_is_midnight_utc(self) -> None:
        concept = self._concept("2026-06-01")

        assert concept.is_stale(self.NOW)
        assert not concept.is_stale(self.NOW - timedelta(hours=1))

    def test_a_naive_value_and_a_naive_now_are_read_as_utc(self) -> None:
        concept = self._concept("2026-06-01T00:00:00")

        assert concept.is_stale(datetime(2026, 6, 1))
        assert not concept.is_stale(datetime(2026, 5, 31, 23, 59))

    def test_a_datetime_object_is_accepted(self) -> None:
        assert self._concept(self.NOW).is_stale(self.NOW)

    @pytest.mark.parametrize("value", [None, "", "soon", "2026-13-40", 5])
    def test_absent_or_unparseable_is_not_stale(self, value: Any) -> None:
        concept = OKFConcept.model_construct(id="a", type="T", stale_after=value)

        assert not concept.is_stale(self.NOW)

    def test_the_default_now_is_the_wall_clock(self) -> None:
        assert self._concept("2000-01-01T00:00:00Z").is_stale()
        assert not self._concept("2999-01-01T00:00:00Z").is_stale()

    def test_staleness_does_not_depend_on_status_or_trust(self) -> None:
        concept = OKFConcept(
            id="a",
            type="T",
            stale_after="2000-01-01",
            status="stable",
            verified=[{"by": "human:ann"}],
        )

        assert concept.is_stale(self.NOW)


class TestStripUnsafeText:
    """One helper strips what could drive a terminal or hide text."""

    def test_ansi_sequences_are_removed_whole(self) -> None:
        assert (
            strip_unsafe_text("a\x1b[31;1mred\x1b[0m b\x1b]0;title\x07c") == "ared bc"
        )

    def test_control_zero_width_and_bidi_characters_are_dropped(self) -> None:
        dirty = "a\x00b\x7fc\u200bd\u200fe\u202ef\u2066g\ufeffh\x9bi\u00adj"

        assert strip_unsafe_text(dirty) == "abcdefghij"

    def test_line_breaks_become_spaces_unless_multiline(self) -> None:
        text = "a\nb\tc\r\nd\u2028e"

        assert strip_unsafe_text(text) == "a b c  d e"
        assert strip_unsafe_text(text, multiline=True) == "a\nb\tc \nd e"

    def test_ordinary_unicode_survives(self) -> None:
        assert (
            strip_unsafe_text("caf\u00e9 \u65e5\u672c \U0001f600")
            == "caf\u00e9 \u65e5\u672c \U0001f600"
        )

    def test_empty_text(self) -> None:
        assert strip_unsafe_text("") == ""


class TestModelsAndErrors:
    """Limits, issues, titles and the error class."""

    def test_limit_defaults_come_from_the_constants(self) -> None:
        limits = OKFLimits()

        assert (limits.max_files, limits.max_file_bytes) == (
            OKF_MAX_FILES,
            OKF_MAX_FILE_BYTES,
        )
        assert (limits.max_total_bytes, limits.max_depth) == (
            OKF_MAX_TOTAL_BYTES,
            OKF_MAX_DEPTH,
        )
        assert (
            OKF_MAX_FILES,
            OKF_MAX_FILE_BYTES,
            OKF_MAX_TOTAL_BYTES,
            OKF_MAX_DEPTH,
        ) == (5_000, 262_144, 33_554_432, 16)

    @pytest.mark.parametrize(
        "field", ["max_files", "max_file_bytes", "max_total_bytes", "max_depth"]
    )
    @pytest.mark.parametrize("value", [0, -1])
    def test_limits_must_be_positive(self, field: str, value: int) -> None:
        with pytest.raises(ValueError):
            OKFLimits(**{field: value})

    def test_limits_are_frozen(self) -> None:
        with pytest.raises(ValueError):
            OKFLimits().max_files = 1  # type: ignore[misc]

    def test_an_issue_is_frozen_and_cleans_its_text(self) -> None:
        issue = OKFIssue(
            severity="error",
            code="c",
            message="x\x1b[1my" + "z" * 400,
            path="p\u202eq",
            id="i\nj",
        )

        assert issue.message == "xy" + "z" * 298
        assert (issue.path, issue.id) == ("pq", "i j")
        with pytest.raises(ValueError):
            issue.code = "d"  # type: ignore[misc]

    def test_an_issue_severity_is_checked(self) -> None:
        with pytest.raises(ValueError):
            OKFIssue(severity="fatal", code="c", message="m")  # type: ignore[arg-type]

    def test_display_title_prefers_the_title_and_strips_it(self) -> None:
        assert (
            OKFConcept(
                id="t/customers", type="T", title="  Cust\x1b[1momers\u202e "
            ).display_title
            == "Customers"
        )

    @pytest.mark.parametrize("title", [None, "", "   \u200b"])
    def test_display_title_falls_back_to_the_last_id_segment(
        self, title: str | None
    ) -> None:
        assert (
            OKFConcept(id="tables/customers", type="T", title=title).display_title
            == "customers"
        )

    def test_okf_error_is_an_fsm_error(self) -> None:
        assert issubclass(OKFError, FSMError)
        assert OKFError("x", details={"k": 1}).details == {"k": 1}

    def test_the_module_does_not_import_litellm(self) -> None:
        source = (Path(__file__).parents[2] / "src" / "fsm_llm" / "okf.py").read_text()

        assert "litellm" not in source


# --------------------------------------------------------------
# Bundle loader, links, search, conformance (plan step 3)
# --------------------------------------------------------------


def _write(root: Path, files: dict[str, str | bytes]) -> Path:
    """Write ``files`` (relative path -> text or bytes) under ``root``."""
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return root


def _doc(type_: str = "Note", body: str = "text\n", **keys: str) -> str:
    front = "".join(f"{k}: {v}\n" for k, v in keys.items())
    return f"---\ntype: {type_}\n{front}---\n{body}"


def _mem(**bodies: str) -> OKFBundle:
    """In-memory bundle; ``a__b`` keyword names become ids ``a/b``."""
    return OKFBundle(
        OKFConcept(id=name.replace("__", "/"), type="Note", body=body)
        for name, body in bodies.items()
    )


def _codes(bundle: OKFBundle) -> list[str]:
    return [i.code for i in bundle.issues]


class TestFrontmatterSizeBound:
    """A frontmatter block over the bound is not YAML-parsed (DoS guard)."""

    def test_a_20kb_frontmatter_is_refused_fast_and_keeps_the_body(self) -> None:
        raw = "".join(f"k{i:05d}: v\n" for i in range(2_100))
        assert len(raw) > 20_000

        started = time.perf_counter()
        concept, issues = parse_concept_text("a", f"---\ntype: T\n{raw}---\nbody\n")

        assert time.perf_counter() - started < 0.1
        assert concept.type == ""
        assert concept.body == "body\n"
        errors = {i.code: i.severity for i in issues}
        assert errors["frontmatter-too-large"] == "error"

    def test_the_bound_is_exact_in_bytes(self) -> None:
        head = "type: T\n"
        fill = OKF_MAX_FRONTMATTER_BYTES - len(head.encode())
        at_limit = head + "n: " + "x" * (fill - len("n: \n")) + "\n"
        assert len(at_limit.encode()) == OKF_MAX_FRONTMATTER_BYTES

        ok, ok_issues = parse_concept_text("a", f"---\n{at_limit}---\nb\n")
        over, over_issues = parse_concept_text("a", f"---\n{at_limit}x: 1\n---\nb\n")

        assert ok.type == "T"
        assert "frontmatter-too-large" not in [i.code for i in ok_issues]
        assert over.type == ""
        assert "frontmatter-too-large" in [i.code for i in over_issues]

    def test_multibyte_text_counts_bytes_not_characters(self) -> None:
        text = "---\ntype: T\nn: " + "\u00e9" * 9_000 + "\n---\n"  # 18 KB of bytes
        concept, issues = parse_concept_text("a", text)

        assert concept.type == ""
        assert "frontmatter-too-large" in [i.code for i in issues]

    def test_a_flood_of_duplicate_keys_is_bounded(self) -> None:
        raw = "type: T\n" + "a: 1\n" * 3_000
        started = time.perf_counter()
        concept, issues = parse_concept_text("a", f"---\n{raw}---\n")

        warnings = [i for i in issues if i.severity == "warning"]
        assert concept.type == "T"
        assert len(warnings) == 25
        assert [i.code for i in issues].count("issues-truncated") == 1
        assert time.perf_counter() - started < 0.5


class TestLoader:
    def test_ids_are_the_path_minus_md_in_sorted_order(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "zeta.md": _doc(),
                "tables/orders.md": _doc("Table"),
                "tables/deep/x.md": _doc(),
                "alpha.md": _doc(),
                "notes.txt": "ignored",
                "readme.MD": _doc(),
            },
        )

        bundle = load_bundle(tmp_path)

        assert [c.id for c in bundle] == [
            "alpha",
            "tables/deep/x",
            "tables/orders",
            "zeta",
        ]
        assert bundle.root == tmp_path.resolve()
        assert len(bundle) == 4
        assert "alpha" in bundle and "notes" not in bundle
        assert bundle.get("tables/orders") is not None
        assert bundle.get("tables/orders").type == "Table"  # type: ignore[union-attr]
        assert bundle.issues == []

    def test_a_missing_root_is_an_error_and_is_not_created(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nope"

        with pytest.raises(OKFError):
            load_bundle(missing)

        assert not missing.exists()

    def test_a_file_as_root_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.md").write_text("x")

        with pytest.raises(OKFError):
            load_bundle(tmp_path / "f.md")

    def test_an_empty_directory_is_an_empty_bundle(self, tmp_path: Path) -> None:
        bundle = load_bundle(tmp_path)

        assert len(bundle) == 0
        assert bundle.issues == []
        assert bundle.okf_version is None

    def test_index_and_log_are_never_concepts_at_any_level(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path,
            {
                "index.md": "# Index\n\n* [A](/a.md) - the a\n",
                "log.md": "## 2026-05-22\n* **Update**: x\n",
                "a.md": _doc(),
                "sub/index.md": "# Sub\n",
                "sub/log.md": "## 2026-01-01\n",
                "sub/b.md": _doc(),
            },
        )

        bundle = load_bundle(tmp_path)

        assert [c.id for c in bundle] == ["a", "sub/b"]
        assert bundle.issues == []

    def test_a_root_index_supplies_okf_version_and_capped_text(
        self, tmp_path: Path
    ) -> None:
        body = "# Index\n\x1b[31m* [A](/a.md)\n" + "x" * 10_000
        _write(tmp_path, {"index.md": f'---\nokf_version: "0.2"\n---\n{body}'})

        bundle = load_bundle(tmp_path)

        assert bundle.okf_version == "0.2"
        assert bundle.index_text.startswith("# Index\n")
        assert "\x1b" not in bundle.index_text
        assert len(bundle.index_text) <= 4_000
        assert bundle.issues == []

    def test_an_unquoted_okf_version_number_is_read_as_text(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, {"index.md": "---\nokf_version: 0.2\n---\n# I\n"})

        assert load_bundle(tmp_path).okf_version == "0.2"

    def test_reserved_file_problems_are_warnings_and_never_load(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path,
            {
                "index.md": "---\nokf_version: '0.2'\nowner: me\n---\n",
                "sub/index.md": "---\ntype: Index\n---\n",
                "log.md": "## yesterday\nstuff\n",
                "a.md": _doc(),
            },
        )

        bundle = load_bundle(tmp_path)

        assert [c.id for c in bundle] == ["a"]
        assert sorted(_codes(bundle)) == [
            "index-frontmatter",
            "index-frontmatter",
            "log-heading",
        ]
        assert all(i.severity == "warning" for i in bundle.issues)

    def test_dot_directories_are_not_entered(self, tmp_path: Path) -> None:
        _write(tmp_path, {".git/notes.md": _doc(), "a.md": _doc()})

        assert [c.id for c in load_bundle(tmp_path)] == ["a"]

    def test_bom_and_crlf_files_load(self, tmp_path: Path) -> None:
        raw = "\ufeff---\r\ntype: Note\r\ntitle: T\r\n---\r\nline\r\n"
        _write(tmp_path, {"a.md": raw.encode("utf-8")})

        concept = load_bundle(tmp_path).get("a")

        assert concept is not None
        assert (concept.type, concept.title, concept.body) == ("Note", "T", "line\n")

    def test_non_utf8_is_skipped_with_an_error_issue(self, tmp_path: Path) -> None:
        _write(tmp_path, {"bad.md": b"---\ntype: T\n---\n\xff\xfe", "ok.md": _doc()})

        bundle = load_bundle(tmp_path)

        assert [c.id for c in bundle] == ["ok"]
        (issue,) = bundle.issues
        assert (issue.code, issue.severity, issue.path) == (
            "not-utf8",
            "error",
            "bad.md",
        )

    def test_a_file_name_with_control_characters_is_skipped(
        self, tmp_path: Path
    ) -> None:
        for name in ("evil\x1b[2Jname.md", "bidi\u202ename.md"):
            try:
                (tmp_path / name).write_text(_doc())
            except OSError:  # pragma: no cover - filesystem refuses the name
                pytest.skip("file system refuses control characters in names")
        _write(tmp_path, {"fine name.md": _doc()})

        bundle = load_bundle(tmp_path)

        assert [c.id for c in bundle] == ["fine name"]
        assert _codes(bundle) == ["id-unsafe", "id-unsafe"]
        assert all("\x1b" not in (i.path or "") for i in bundle.issues)

    def test_binary_garbage_with_an_md_name_never_raises(self, tmp_path: Path) -> None:
        _write(
            tmp_path, {"junk.md": bytes(range(256)) * 4, "nul.md": b"---\n\x00\n---\n"}
        )

        bundle = load_bundle(tmp_path)

        assert {c.id for c in bundle} <= {"junk", "nul"}  # whichever decode; no raise

    def test_a_document_without_frontmatter_loads_with_an_error_issue(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, {"plain.md": "just prose\n"})

        bundle = load_bundle(tmp_path)

        assert bundle.get("plain") is not None
        assert "type-missing" in _codes(bundle)

    def test_the_loader_accepts_a_str_path_and_resolves_it_once(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / "real", {"a.md": _doc()})
        _symlink(tmp_path / "alias", tmp_path / "real")

        bundle = load_bundle(str(tmp_path / "alias"))

        assert [c.id for c in bundle] == ["a"]
        assert bundle.root == (tmp_path / "real").resolve()

    def test_500_small_concepts_load_in_under_2_seconds(self, tmp_path: Path) -> None:
        for i in range(500):
            _write(
                tmp_path,
                {
                    f"d{i % 10}/c{i}.md": _doc(
                        "Note", f"Body {i} [n](/d0/c0.md)\n", title=f"T{i}"
                    )
                },
            )

        started = time.perf_counter()
        bundle = load_bundle(tmp_path)

        assert time.perf_counter() - started < 2.0
        assert len(bundle) == 500
        assert bundle.issues == []


class TestLimits:
    def test_max_files_stops_with_an_error_issue(self, tmp_path: Path) -> None:
        _write(tmp_path, {f"c{i:02d}.md": _doc() for i in range(60)})

        bundle = load_bundle(tmp_path, limits=OKFLimits(max_files=50))

        assert len(bundle) == 50
        assert [c.id for c in bundle][-1] == "c49"  # sorted order, deterministic
        (issue,) = bundle.issues
        assert (issue.code, issue.severity) == ("max-files", "error")

    def test_reserved_files_do_not_count_toward_max_files(self, tmp_path: Path) -> None:
        _write(tmp_path, {"index.md": "# I\n", "log.md": "## 2026-01-01\n"})
        _write(tmp_path, {f"c{i}.md": _doc() for i in range(3)})

        bundle = load_bundle(tmp_path, limits=OKFLimits(max_files=3))

        assert len(bundle) == 3
        assert bundle.issues == []

    def test_an_oversize_file_is_skipped_and_the_rest_loads(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path,
            {"big.md": _doc(body="x" * 5_000), "small.md": _doc()},
        )

        bundle = load_bundle(tmp_path, limits=OKFLimits(max_file_bytes=1_000))

        assert [c.id for c in bundle] == ["small"]
        (issue,) = bundle.issues
        assert (issue.code, issue.path) == ("file-too-large", "big.md")

    def test_the_default_per_file_cap_is_the_constant(self) -> None:
        assert OKFLimits().max_file_bytes == OKF_MAX_FILE_BYTES

    def test_a_file_that_grows_past_the_cap_while_reading_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, {"a.md": _doc()})
        monkeypatch.setattr(
            okf_module, "_read_bytes", lambda path, limit: b"x" * (limit + 1)
        )

        bundle = load_bundle(tmp_path)

        assert len(bundle) == 0
        assert _codes(bundle) == ["file-too-large"]

    def test_max_total_bytes_stops_the_load(self, tmp_path: Path) -> None:
        _write(tmp_path, {f"c{i}.md": _doc(body="y" * 200) for i in range(10)})
        one = len(_doc(body="y" * 200).encode())

        bundle = load_bundle(tmp_path, limits=OKFLimits(max_total_bytes=one * 3 + 5))

        assert [c.id for c in bundle] == ["c0", "c1", "c2"]
        assert _codes(bundle) == ["max-total-bytes"]
        assert bundle.issues[0].severity == "error"

    def test_depth_beyond_max_depth_is_not_entered(self, tmp_path: Path) -> None:
        parts = [f"d{i}" for i in range(6)]
        for n in range(len(parts) + 1):
            _write(tmp_path, {"/".join([*parts[:n], f"f{n}.md"]): _doc()})

        bundle = load_bundle(tmp_path, limits=OKFLimits(max_depth=3))

        assert [c.id for c in bundle] == [
            "d0/d1/d2/f3",
            "d0/d1/f2",
            "d0/f1",
            "f0",
        ]
        assert _codes(bundle) == ["max-depth"]
        assert bundle.issues[0].path == "d0/d1/d2/d3"

    def test_default_depth_is_the_constant(self) -> None:
        assert OKFLimits().max_depth == OKF_MAX_DEPTH
        assert OKFLimits().max_files == OKF_MAX_FILES
        assert OKFLimits().max_total_bytes == OKF_MAX_TOTAL_BYTES

    def test_a_directory_with_far_too_many_entries_is_truncated(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, {f"c{i:02d}.md": _doc() for i in range(12)})

        bundle = load_bundle(tmp_path, limits=OKFLimits(max_files=3))

        assert len(bundle) == 3
        assert "dir-truncated" in _codes(bundle)
        assert "max-files" in _codes(bundle)

    def test_the_issue_list_is_capped_errors_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(okf_module, "OKF_MAX_ISSUES", 5)
        _write(
            tmp_path, {f"w{i}.md": "---\ntype: T\nstatus: odd\n---\n" for i in range(8)}
        )
        _write(tmp_path, {"bad.md": b"\xff"})

        bundle = load_bundle(tmp_path)

        assert len(bundle.issues) == 6
        assert bundle.issues[0].code == "not-utf8"  # error ranked first
        assert bundle.issues[-1].code == "issues-truncated"


class TestSymlink:
    def test_a_symlinked_file_to_an_outside_secret_is_not_loaded(
        self, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        _write(outside, {"secret.md": _doc(body="TOP-SECRET-MARKER\n")})
        bundle_dir = tmp_path / "bundle"
        _write(bundle_dir, {"ok.md": _doc()})
        _symlink(bundle_dir / "leak.md", outside / "secret.md")

        bundle = load_bundle(bundle_dir)

        assert [c.id for c in bundle] == ["ok"]
        assert _codes(bundle) == ["symlink-skipped"]
        assert "TOP-SECRET-MARKER" not in repr([c.body for c in bundle])

    def test_a_symlink_to_a_file_inside_the_bundle_is_skipped_too(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, {"a.md": _doc()})
        _symlink(tmp_path / "alias.md", tmp_path / "a.md")

        bundle = load_bundle(tmp_path)

        assert [c.id for c in bundle] == ["a"]
        assert _codes(bundle) == ["symlink-skipped"]

    def test_a_symlinked_directory_is_not_followed(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        _write(outside, {"x.md": _doc(body="OUTSIDE\n")})
        bundle_dir = tmp_path / "bundle"
        _write(bundle_dir, {"ok.md": _doc()})
        _symlink(bundle_dir / "linked", outside)

        bundle = load_bundle(bundle_dir)

        assert [c.id for c in bundle] == ["ok"]
        assert _codes(bundle) == ["symlink-skipped"]

    def test_a_symlink_loop_terminates(self, tmp_path: Path) -> None:
        _write(tmp_path, {"d/a.md": _doc()})
        _symlink(tmp_path / "d" / "loop", tmp_path / "d")
        _symlink(tmp_path / "up", tmp_path)

        started = time.perf_counter()
        bundle = load_bundle(tmp_path)

        assert time.perf_counter() - started < 2.0
        assert [c.id for c in bundle] == ["d/a"]
        assert _codes(bundle).count("symlink-skipped") == 2

    def test_a_dangling_symlink_is_skipped_without_error(self, tmp_path: Path) -> None:
        _symlink(tmp_path / "gone.md", tmp_path / "nowhere")

        bundle = load_bundle(tmp_path)

        assert len(bundle) == 0
        assert _codes(bundle) == ["symlink-skipped"]

    def test_defence_in_depth_refuses_a_path_the_helper_puts_outside(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, {"a.md": _doc()})
        monkeypatch.setattr(okf_module, "resolve_within", lambda root, rel: None)

        bundle = load_bundle(tmp_path)

        assert len(bundle) == 0
        assert _codes(bundle) == ["path-indirect"]

    def test_resolve_errors_become_issues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, {"a.md": _doc(), "b.md": _doc()})

        def boom(root: Path, rel: str) -> Path:
            raise RuntimeError("symlink loop")

        monkeypatch.setattr(okf_module, "resolve_within", boom)

        bundle = load_bundle(tmp_path)

        assert len(bundle) == 0
        assert _codes(bundle) == ["file-unreadable", "file-unreadable"]


class TestLinkResolution:
    """Lexical, in id space: no filesystem access."""

    @pytest.fixture
    def bundle(self) -> OKFBundle:
        return _mem(
            **{
                "x": "",
                "d__x": "",
                "d__e": "",
                "d__sub__deep": "",
                "tables__orders": "",
            }
        )

    @pytest.mark.parametrize(
        ("target", "from_id", "expected"),
        [
            ("/tables/orders.md", "d/e", "tables/orders"),
            ("tables/orders.md", "d/e", "tables/orders"),  # root fallback
            ("./x.md", "d/e", "d/x"),
            ("x.md", "d/e", "d/x"),  # file-relative wins over root-relative
            ("/x.md", "d/e", "x"),
            ("../x.md", "d/e", "x"),
            ("sub/deep.md", "d/e", "d/sub/deep"),
            ("../d/sub/../sub/deep.md", "d/e", "d/sub/deep"),
            ("x.md#heading", "d/e", "d/x"),
            ("x.md?raw=1", "d/e", "d/x"),
            ("x.md", "root", "x"),
            ("%2e%2e/x.md", "d/e", "x"),  # percent-encoded ".." decoded once
            ("tables%2Forders.md", "d/e", "tables/orders"),
            ("<x.md>", "d/e", None),  # angle form is only valid inside a link
            ("  /x.md  ", "d/e", "x"),
        ],
    )
    def test_internal_targets(
        self, bundle: OKFBundle, target: str, from_id: str, expected: str | None
    ) -> None:
        assert bundle.resolve_link(target, from_id) == expected

    @pytest.mark.parametrize(
        "target",
        [
            "../../x.md",  # climbs above the root
            "/../x.md",
            "%2e%2e/%2e%2e/x.md",
            "https://example.com/x.md",
            "HTTP://example.com/x.md",
            "mailto:me@example.com",
            "ftp:x.md",
            "C:/x.md",
            "//host/x.md",
            "#frag",
            "?q=1",
            "",
            "x.txt",
            "x",
            "x.md.bak",
            "index.md",
            "/log.md",
            "d/index.md",
            "x%00.md",
            "x%0a.md",
            "x%e2%80%aey.md",  # bidi override
            "d\\x.md",
            "d%5Cx.md",
            "missing.md",
            "a" * 5_000 + ".md",
        ],
    )
    def test_everything_else_resolves_to_nothing(
        self, bundle: OKFBundle, target: str
    ) -> None:
        assert bundle.resolve_link(target, "d/e") is None

    def test_non_string_targets_resolve_to_nothing(self, bundle: OKFBundle) -> None:
        for bad in (None, 3, ["x.md"], {"a": 1}, b"x.md"):
            assert bundle.resolve_link(bad, "d/e") is None
            assert bundle.resolve_resource(bad, "d/e") is None

    def test_root_relative_fallback_applies_only_when_file_relative_is_missing(
        self,
    ) -> None:
        only_root = _mem(x="", d__e="")
        both = _mem(x="", d__x="", d__e="")

        assert only_root.resolve_link("x.md", "d/e") == "x"
        assert both.resolve_link("x.md", "d/e") == "d/x"
        assert both.resolve_link("/x.md", "d/e") == "x"  # "/" never goes file-relative

    def test_resolve_resource_uses_the_same_orders(self, bundle: OKFBundle) -> None:
        assert bundle.resolve_resource("references/none.md", "d/e") is None
        assert bundle.resolve_resource("tables/orders.md", "d/e") == "tables/orders"
        assert bundle.resolve_resource("https://x.test/a.md", "d/e") is None

    def test_resolution_never_touches_the_filesystem(
        self, bundle: OKFBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("filesystem access during link resolution")

        monkeypatch.setattr(Path, "resolve", forbidden)
        monkeypatch.setattr(Path, "exists", forbidden)
        monkeypatch.setattr(os, "stat", forbidden)

        assert bundle.resolve_link("../x.md", "d/e") == "x"
        assert bundle.resolve_link("../../etc/passwd.md", "d/e") is None


class TestOutgoingAndBacklinks:
    def test_body_links_in_order_without_duplicates_or_self(self) -> None:
        bundle = _mem(
            a="[b](/b.md) [c](./c.md) [b again](/b.md#h) [self](/a.md)\n",
            b="",
            c="",
        )

        assert bundle.outgoing("a") == ["b", "c"]

    def test_links_in_code_are_ignored(self) -> None:
        body = (
            "[real](/b.md)\n"
            "```\n[fenced](/c.md)\n```\n"
            "~~~sql\n[tilde](/d.md)\n~~~\n"
            "inline `[code](/e.md)` span\n"
            "````\n```\n[nested](/f.md)\n````\n"
            "[after](/g.md)\n"
        )
        bundle = _mem(a=body, b="", c="", d="", e="", f="", g="")

        assert bundle.outgoing("a") == ["b", "g"]

    def test_an_unclosed_fence_hides_the_rest_of_the_document(self) -> None:
        bundle = _mem(a="[b](/b.md)\n```\n[c](/c.md)\n[d](/d.md)\n", b="", c="", d="")

        assert bundle.outgoing("a") == ["b"]

    def test_link_syntax_variants(self) -> None:
        body = (
            "[titled](/b.md \"A title\") [single](/c.md 't') [angle](</d.md>) "
            "![image](/e.md) [empty]() [ext](https://x.test/f.md)\n"
        )
        bundle = _mem(a=body, b="", c="", d="", e="", f="")

        assert bundle.outgoing("a") == ["b", "c", "d", "e"]

    def test_resource_and_sources_resource_count_as_outgoing(self) -> None:
        bundle = OKFBundle(
            [
                OKFConcept(
                    id="a",
                    type="T",
                    resource="/r.md",
                    sources=[{"resource": "/s.md"}, {"resource": "all queries"}, {}],
                ),
                OKFConcept(id="r", type="T"),
                OKFConcept(id="s", type="T"),
            ]
        )

        assert bundle.outgoing("a") == ["r", "s"]

    def test_unknown_id_has_no_links(self) -> None:
        assert _mem(a="[b](/b.md)").outgoing("nope") == []
        assert _mem(a="[b](/b.md)").backlinks("nope") == []

    def test_broken_and_escaping_links_are_tolerated(self) -> None:
        bundle = _mem(
            a="[gone](/gone.md) [up](../../../etc/x.md) [e](https://x.test)\n"
        )

        assert bundle.outgoing("a") == []

    def test_backlinks_are_the_inverse_in_id_order(self) -> None:
        bundle = _mem(
            z="[t](/t.md)\n",
            a="[t](/t.md) [z](/z.md)\n",
            m__x="[t](/t.md)\n",
            t="",
        )

        assert bundle.backlinks("t") == ["a", "m/x", "z"]
        assert bundle.backlinks("z") == ["a"]
        assert bundle.backlinks("a") == []

    def test_put_invalidates_the_link_caches(self) -> None:
        bundle = _mem(a="[b](/b.md)\n")
        assert bundle.outgoing("a") == []
        assert bundle.backlinks("b") == []

        bundle.put(OKFConcept(id="b", type="T"))
        assert bundle.outgoing("a") == ["b"]
        assert bundle.backlinks("b") == ["a"]

        bundle.put(OKFConcept(id="a", type="T", body="[c](/c.md)\n"))
        assert bundle.outgoing("a") == []
        assert bundle.backlinks("b") == []

    def test_results_are_copies(self) -> None:
        bundle = _mem(a="[b](/b.md)\n", b="")
        bundle.outgoing("a").append("evil")
        bundle.backlinks("b").append("evil")

        assert bundle.outgoing("a") == ["b"]
        assert bundle.backlinks("b") == ["a"]

    def test_a_hostile_body_is_scanned_in_linear_time(self) -> None:
        bodies = ["[](" * 60_000, "[" * 200_000, "](" * 100_000, "[a](" + "b" * 200_000]
        started = time.perf_counter()
        for body in bodies:
            _mem(a=body).outgoing("a")

        assert time.perf_counter() - started < 2.0

    def test_the_number_of_links_read_from_one_body_is_capped(self) -> None:
        body = " ".join(f"[l](/c{i}.md)" for i in range(1_500))
        bundle = OKFBundle(
            [OKFConcept(id="a", type="T", body=body)]
            + [OKFConcept(id=f"c{i}", type="T") for i in range(1_500)]
        )

        assert len(bundle.outgoing("a")) == 1_000


class TestListAndSearch:
    @pytest.fixture
    def bundle(self) -> OKFBundle:
        return OKFBundle(
            [
                OKFConcept(
                    id="tables/orders",
                    type="BigQuery Table",
                    title="Customer Orders",
                    description="One row per order",
                    tags=["sales"],
                    body="Revenue is booked here.\n",
                ),
                OKFConcept(
                    id="tables/customers",
                    type="BigQuery Table",
                    title="Customers",
                    body="Mentions orders once.\n",
                ),
                OKFConcept(
                    id="playbooks/freshness",
                    type="Playbook",
                    title="Freshness alert",
                    body="Check the orders table.\n",
                ),
                OKFConcept(
                    id="tables/old_orders",
                    type="BigQuery Table",
                    title="Old orders",
                    status="deprecated",
                ),
            ]
        )

    def test_list_filters_by_type_and_prefix_and_hides_deprecated(
        self, bundle: OKFBundle
    ) -> None:
        assert [c.id for c in bundle.list()] == [
            "playbooks/freshness",
            "tables/customers",
            "tables/orders",
        ]
        assert [c.id for c in bundle.list(type="playbook")] == ["playbooks/freshness"]
        assert [c.id for c in bundle.list(prefix="tables/")] == [
            "tables/customers",
            "tables/orders",
        ]
        everything = bundle.list(include_deprecated=True)
        assert "tables/old_orders" in [c.id for c in everything]
        assert bundle.list(type="Nope") == []

    def test_title_and_tag_matches_outrank_body_matches(
        self, bundle: OKFBundle
    ) -> None:
        hits = bundle.search("orders")

        assert [c.id for c, _ in hits] == [
            "tables/orders",
            "playbooks/freshness",
            "tables/customers",
        ]

    def test_scores_are_in_the_open_unit_interval_and_non_increasing(
        self, bundle: OKFBundle
    ) -> None:
        scores = [score for _, score in bundle.search("customer orders revenue")]

        assert scores and all(0 < s <= 1 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_a_concept_matching_every_token_everywhere_scores_one(self) -> None:
        bundle = OKFBundle(
            [OKFConcept(id="widget", type="T", title="widget", body="widget")]
        )

        assert bundle.search("widget")[0][1] == 1.0

    def test_ties_break_by_id(self) -> None:
        bundle = OKFBundle(
            [OKFConcept(id=i, type="T", body="shared words") for i in ("c", "a", "b")]
        )

        assert [c.id for c, _ in bundle.search("shared")] == ["a", "b", "c"]

    def test_deprecated_is_excluded_unless_asked(self, bundle: OKFBundle) -> None:
        assert "tables/old_orders" not in [c.id for c, _ in bundle.search("old")]
        found = bundle.search("old", include_deprecated=True)
        assert [c.id for c, _ in found] == ["tables/old_orders"]

    def test_search_is_case_insensitive_and_matches_ids(
        self, bundle: OKFBundle
    ) -> None:
        assert [c.id for c, _ in bundle.search("PLAYBOOKS")] == ["playbooks/freshness"]

    def test_short_tokens_and_odd_queries_give_nothing(self, bundle: OKFBundle) -> None:
        assert bundle.search("an of to") == []
        assert bundle.search("") == []
        assert bundle.search("   ") == []
        assert bundle.search(None) == []  # type: ignore[arg-type]
        assert bundle.search(123) == []  # type: ignore[arg-type]
        assert bundle.search("zzzzqq") == []

    def test_limit_is_respected_and_clamped(self, bundle: OKFBundle) -> None:
        assert len(bundle.search("orders", limit=1)) == 1
        assert bundle.search("orders", limit=0) == []
        assert bundle.search("orders", limit=-5) == []
        assert len(bundle.search("orders", limit=1_000)) == 3

    def test_a_huge_query_is_bounded(self, bundle: OKFBundle) -> None:
        started = time.perf_counter()
        bundle.search("orders " * 100_000)

        assert time.perf_counter() - started < 0.5

    def test_put_refreshes_the_search_index(self, bundle: OKFBundle) -> None:
        assert bundle.search("zebra") == []

        bundle.put(OKFConcept(id="new", type="T", body="a zebra appears"))

        assert [c.id for c, _ in bundle.search("zebra")] == ["new"]

    def test_search_is_deterministic(self, bundle: OKFBundle) -> None:
        assert bundle.search("orders table") == bundle.search("orders table")


class TestCheckBundle:
    def test_a_conformant_bundle_has_no_error_or_warning(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "index.md": '---\nokf_version: "0.2"\n---\n# I\n',
                "a.md": _doc(
                    "Note", "[b](/b.md)\n", stale_after="2030-01-01T00:00:00Z"
                ),
                "b.md": _doc(),
            },
        )

        issues = check_bundle(load_bundle(tmp_path))

        assert [i for i in issues if i.severity != "info"] == []

    def test_non_conformant_documents_are_reported(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "notype.md": "---\ntitle: x\n---\nbody\n",
                "emptytype.md": '---\ntype: ""\n---\n',
                "nofront.md": "prose only\n",
                "badyaml.md": "---\ntype: [unclosed\n---\n",
                "naive.md": _doc(stale_after="2030-01-01T00:00:00"),
                "dateonly.md": _doc(stale_after="2030-01-01"),
                "bin.md": b"\xff\xfe",
            },
        )

        issues = check_bundle(load_bundle(tmp_path))

        by_id = {(i.id or i.path, i.code): i.severity for i in issues}
        assert by_id[("notype", "type-missing")] == "error"
        assert by_id[("emptytype", "type-empty")] == "error"
        assert by_id[("nofront", "frontmatter-missing")] == "error"
        assert by_id[("badyaml", "frontmatter-invalid")] == "error"
        assert by_id[("naive", "timestamp-naive")] == "warning"
        assert by_id[("dateonly", "timestamp-date-only")] == "warning"
        assert by_id[("bin.md", "not-utf8")] == "error"

    def test_load_issues_are_not_duplicated(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "notype.md": "---\ntitle: x\n---\n",
                "n.md": _doc(stale_after="2030-01-01"),
            },
        )
        bundle = load_bundle(tmp_path)

        issues = check_bundle(bundle)

        assert issues == bundle.issues
        assert [i.code for i in issues].count("type-missing") == 1
        assert [i.code for i in issues].count("type-empty") == 0

    def test_concepts_added_in_memory_are_checked_too(self) -> None:
        bundle = OKFBundle(
            [
                OKFConcept(id="a", type="", stale_after="soon"),
                OKFConcept(id="index", type="T"),
                OKFConcept(id="d/log", type="T"),
                OKFConcept(
                    id="b",
                    type="T",
                    generated={"by": "x", "at": "2026-01-01"},
                    verified=[{"by": "human:me", "at": "yesterday"}],
                ),
            ]
        )

        issues = check_bundle(bundle)
        found = {(i.id, i.code) for i in issues}

        assert ("a", "type-empty") in found
        assert ("a", "timestamp-invalid") in found
        assert ("index", "reserved-id") in found
        assert ("d/log", "reserved-id") in found
        assert ("b", "timestamp-date-only") in found
        assert ("b", "timestamp-invalid") in found

    def test_broken_links_are_info_and_escaping_or_external_are_not_reported(
        self,
    ) -> None:
        bundle = _mem(a="[x](/gone.md) [y](https://x.test/y.md) [z](../../z.md)\n")

        issues = check_bundle(bundle)

        assert [(i.code, i.severity) for i in issues] == [("link-broken", "info")]

    def test_okf_version_unknown_is_info_and_known_is_silent(self) -> None:
        unknown = check_bundle(OKFBundle(okf_version="9.9"))
        known = check_bundle(OKFBundle(okf_version="0.1"))

        assert [(i.code, i.severity) for i in unknown] == [
            ("okf-version-unknown", "info")
        ]
        assert known == []

    def test_check_does_not_mutate_the_bundle_issues(self) -> None:
        bundle = OKFBundle([OKFConcept(id="a", type="")])

        check_bundle(bundle)

        assert bundle.issues == []

    def test_check_never_raises_on_odd_models(self) -> None:
        bundle = OKFBundle(
            [
                OKFConcept(
                    id="a", type="T", generated={1: 2}, verified=[{"at": object()}]
                )
            ]
        )

        assert isinstance(check_bundle(bundle), list)

    def test_the_result_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(okf_module, "OKF_MAX_ISSUES", 3)
        bundle = OKFBundle(OKFConcept(id=f"c{i}", type="") for i in range(10))

        issues = check_bundle(bundle)

        assert len(issues) == 4
        assert issues[-1].code == "issues-truncated"


class TestInteropFixture:
    """Hand-written bundle copied from the spec's own examples.

    Section numbers refer to SPEC.md v0.2 at commit ad30107: 4.3 (a concept
    bound to a resource, L225-250), 4.4 (a concept not bound to a resource),
    6.1 links (L441-466), 8 index (L505-529, only the root index carries
    ``okf_version``), 9 log (L533-552), 10.2 (an Attested Computation,
    L606-650), 11 conformance (L742-762), 12 versioning (L766-780).
    """

    FILES: ClassVar[dict[str, str]] = {
        # Section 8 and 12: the only index.md that may carry frontmatter.
        "index.md": (
            '---\nokf_version: "0.2"\n---\n'
            "# Tables\n\n"
            "* [Customer Orders](/tables/orders.md) - One row per completed order.\n"
            "* [Subdirectory](playbooks/) - Runbooks.\n"
        ),
        # Section 9: ISO date headings, newest first.
        "log.md": (
            "# Directory Update Log\n\n"
            "## 2026-05-22\n"
            "* **Update**: Added [Customer Orders](/tables/orders.md).\n\n"
            "## 2026-05-15\n"
            "* **Initialization**: Created foundational directory structure.\n"
        ),
        # Section 4.3: a concept bound to a resource, with a table and links.
        "tables/orders.md": (
            "---\n"
            "type: BigQuery Table\n"
            "title: Customer Orders\n"
            "description: One row per completed customer order across all channels.\n"
            "resource: https://console.cloud.google.com/bigquery?p=acme&d=sales&t=orders\n"
            "tags: [sales, orders, revenue]\n"
            "generated: { by: reference_agent/gemini-2.5-pro, at: 2026-05-28T14:30:00Z }\n"
            "---\n\n"
            "# Schema\n\n"
            "| Column        | Type      | Description                              |\n"
            "|---------------|-----------|------------------------------------------|\n"
            "| `order_id`    | STRING    | Globally unique order identifier.        |\n"
            "| `customer_id` | STRING    | Foreign key into [customers](/tables/customers.md). |\n\n"
            "# Joins\n\n"
            "Joined with [customers](/tables/customers.md) on `customer_id`.\n"
        ),
        "tables/customers.md": (
            "---\ntype: BigQuery Table\ntitle: Customers\n---\n"
            "See [orders](./orders.md).\n"
        ),
        # Section 4.4: no resource; `generated.by` uses the human: actor form.
        "playbooks/freshness.md": (
            "---\n"
            "type: Playbook\n"
            'title: "Incident response: data freshness alert"\n'
            "description: Steps to triage a freshness alert on the orders pipeline.\n"
            "tags: [oncall, incident]\n"
            "generated: { by: human:ahormati, at: 2026-04-12T09:00:00Z }\n"
            "---\n\n"
            "# Trigger\n\n"
            "See the [orders table](/tables/orders.md).\n"
        ),
        # Section 5: sources / generated / verified (bare mapping, a MUST to accept),
        # per-claim footnote keyed by sources[].id (L350-364).
        "metrics/revenue.md": (
            "---\n"
            "type: Metric\n"
            "title: Revenue\n"
            "sources:\n"
            "  - id: rev-policy\n"
            "    resource: https://wiki.acme/finance/revenue-recognition\n"
            "    title: Revenue recognition policy\n"
            "generated: { by: reference_agent/gemini-2.5-pro, at: 2026-06-20T22:53:05Z }\n"
            "verified: { by: human:ahormati, at: 2026-06-25T09:00:00Z }\n"
            "stale_after: 2030-09-23T00:00:00Z\n"
            "status: stable\n"
            "---\n\n"
            "Recognized revenue.[^rev-policy]\n\n"
            "[^rev-policy]: Revenue recognition policy\n"
        ),
        # Section 10.2: an Attested Computation; executor / attester are plain
        # data here and are never run.
        "computations/revenue_fy.md": (
            "---\n"
            "type: Attested Computation\n"
            "title: Revenue for fiscal year\n"
            "description: Recognized revenue for a fiscal year, per Finance's definition.\n"
            "status: stable\n"
            "runtime: bigquery\n"
            "parameters:\n"
            "  - { name: year, type: integer, required: true }\n"
            "executor:\n"
            "  resource: references/skills/run-on-bq.md\n"
            "  receipt: [job_id, executed_sql, result]\n"
            "attester:\n"
            "  resource: references/attesters/revenue.py\n"
            "generated: { by: reference_agent/gemini-2.5-pro, at: 2026-06-20T22:53:05Z }\n"
            "verified: { by: human:ahormati, at: 2026-06-25T09:00:00Z }\n"
            "stale_after: 2026-09-23T00:00:00Z\n"
            "sources:\n"
            "  - id: rev-policy\n"
            "    resource: https://wiki.acme/finance/revenue-recognition\n"
            "    title: Revenue recognition policy\n"
            "---\n\n"
            "# Computation\n\n"
            "    SELECT SUM(amount) AS revenue\n"
            "    FROM finance.recognized_revenue\n"
            "    WHERE fiscal_year = @year\n\n"
            "Used by the [Revenue](/metrics/revenue.md) metric.\n"
        ),
        # Non-.md files coexist with concepts and are never concepts (section 3).
        "references/attesters/revenue.py": "print('never executed')\n",
        # Section 11: a concept carrying just `type` is fully conformant.
        "minimal.md": "---\ntype: Anything\n---\n",
    }

    @pytest.fixture
    def bundle(self, tmp_path: Path) -> OKFBundle:
        return load_bundle(_write(tmp_path, dict(self.FILES)))

    def test_it_loads_with_zero_error_issues(self, bundle: OKFBundle) -> None:
        assert [i for i in check_bundle(bundle) if i.severity == "error"] == []
        assert bundle.okf_version == "0.2"

    def test_reserved_and_non_markdown_files_are_not_concepts(
        self, bundle: OKFBundle
    ) -> None:
        assert [c.id for c in bundle] == [
            "computations/revenue_fy",
            "metrics/revenue",
            "minimal",
            "playbooks/freshness",
            "tables/customers",
            "tables/orders",
        ]

    def test_computation_fields_are_data_and_unknown_keys_survive(
        self, bundle: OKFBundle
    ) -> None:
        concept = bundle.get("computations/revenue_fy")

        assert concept is not None
        assert concept.type == "Attested Computation"
        assert concept.extra["runtime"] == "bigquery"
        assert concept.extra["attester"] == {
            "resource": "references/attesters/revenue.py"
        }
        assert concept.extra["executor"]["receipt"] == [
            "job_id",
            "executed_sql",
            "result",
        ]
        assert concept.trust is TrustTier.HUMAN_REVIEWED
        assert concept.is_stale(datetime(2026, 9, 23, tzinfo=timezone.utc))
        assert not concept.is_stale(datetime(2026, 9, 22, tzinfo=timezone.utc))

    def test_a_bare_verified_mapping_is_a_one_element_list(
        self, bundle: OKFBundle
    ) -> None:
        concept = bundle.get("metrics/revenue")

        assert concept is not None
        assert concept.verified == [
            {"by": "human:ahormati", "at": "2026-06-25T09:00:00Z"}
        ]

    def test_both_link_forms_resolve(self, bundle: OKFBundle) -> None:
        assert bundle.outgoing("tables/orders") == ["tables/customers"]
        assert bundle.outgoing("tables/customers") == ["tables/orders"]  # "./orders.md"
        assert bundle.outgoing("playbooks/freshness") == ["tables/orders"]
        assert bundle.backlinks("tables/orders") == [
            "playbooks/freshness",
            "tables/customers",
        ]
        assert bundle.backlinks("metrics/revenue") == ["computations/revenue_fy"]

    def test_external_resources_resolve_to_nothing(self, bundle: OKFBundle) -> None:
        orders = bundle.get("tables/orders")

        assert orders is not None and orders.resource is not None
        assert bundle.resolve_resource(orders.resource, orders.id) is None
        assert bundle.resolve_resource("references/attesters/revenue.py", "x") is None

    def test_the_index_text_is_kept(self, bundle: OKFBundle) -> None:
        assert "Customer Orders" in bundle.index_text


class TestLayering:
    def test_importing_okf_pulls_in_no_subpackage_and_core_does_not_import_okf(
        self,
    ) -> None:
        probe = (
            "import sys\n"
            "import fsm_llm\n"
            "assert 'fsm_llm.okf' not in sys.modules, 'core imports okf'\n"
            "import fsm_llm.okf\n"
            "bad = [m for m in sys.modules if m.split('.')[:2] in "
            "(['fsm_llm', 'agents'], ['fsm_llm', 'harness'], ['fsm_llm', 'reasoning'], "
            "['fsm_llm', 'workflows'], ['fsm_llm', 'monitor'], ['fsm_llm', 'eval'])]\n"
            "assert not bad, bad\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[2],
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
