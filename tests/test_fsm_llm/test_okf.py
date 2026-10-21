"""Tests for the Open Knowledge Format (OKF) support.

This file grows with the OKF plan; the first class guards the one
path-confinement helper that the OKF loader and writer share with the harness
workspace, the rest cover the tolerant parser, trust tiers and staleness.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from fsm_llm.constants import (
    OKF_MAX_DEPTH,
    OKF_MAX_FILE_BYTES,
    OKF_MAX_FILES,
    OKF_MAX_TOTAL_BYTES,
)
from fsm_llm.definitions import FSMError, OKFError
from fsm_llm.okf import (
    OKFConcept,
    OKFIssue,
    OKFLimits,
    TrustTier,
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
        assert "frontmatter-too-deep" in codes

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
