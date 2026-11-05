"""Step 9 of plan 5980fbb7: OKF check deadline, frontmatter depth guard, the shared
validate report of ``fsm-llm-okf`` and ``fsm-llm-skills``."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import fsm_llm.agent_skills_cli as skills_cli
import fsm_llm.okf_cli as okf_cli
from fsm_llm import cli_support
from fsm_llm import okf as okf_module
from fsm_llm.agent_skills import parse_skill_text
from fsm_llm.constants import OKF_MAX_LOAD_SECONDS, SKILL_MAX_LOAD_SECONDS
from fsm_llm.frontmatter import load_yaml
from fsm_llm.okf import OKFBundle, OKFConcept, OKFIssue, check_bundle


class _Clock:
    """Stand-in for ``time`` inside ``fsm_llm.okf``: moves only when ticked."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class TestCheckBundleReportsTheDeadlineOnce:
    """F-OK-06: ``backlinks`` already stored the frozen deadline issue."""

    def test_a_deadline_left_by_backlinks_is_not_added_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = OKFBundle(
            (OKFConcept(id=f"c{i}", type="Note", body="[](/c0.md)") for i in range(6)),
            max_seconds=10,
        )
        clock = _Clock()
        monkeypatch.setattr(okf_module, "time", clock)
        real = OKFBundle.outgoing

        def slow(self: OKFBundle, cid: str) -> Any:
            clock.now += 4.0
            return real(self, cid)

        monkeypatch.setattr(OKFBundle, "outgoing", slow)
        bundle.backlinks("c0")  # spends the budget; leaves check-deadline behind
        assert [i.code for i in bundle.issues] == ["check-deadline"]

        clock.now = 0.0
        stamps = okf_module._check_timestamps

        def tick(*args: Any, **kwargs: Any) -> Any:
            clock.now += 4.0  # the check itself runs past the budget too
            return stamps(*args, **kwargs)

        monkeypatch.setattr(okf_module, "_check_timestamps", tick)
        monkeypatch.setattr(OKFBundle, "outgoing", real)

        issues = check_bundle(bundle)

        assert [i.code for i in issues].count("check-deadline") == 1


class TestQuotedBracketsAreNotNesting:
    """F-OK-07: brackets inside a quoted scalar or a comment are text."""

    @staticmethod
    def _load(raw: str, max_depth: int = 64) -> tuple[dict[Any, Any], list[str]]:
        codes: list[str] = []
        data = load_yaml(
            raw,
            lambda sev, code, msg: codes.append(code),
            max_bytes=16_384,
            max_depth=max_depth,
        )
        return data, codes

    @pytest.mark.parametrize("quote", ['"', "'"])
    @pytest.mark.parametrize("bracket", ["[", "{"])
    def test_a_flat_quoted_string_with_many_brackets_loads(
        self, quote: str, bracket: str
    ) -> None:
        raw = f"description: {quote}uses {bracket * 70} as a literal{quote}\n"

        data, codes = self._load(raw)

        assert codes == []
        assert data == {"description": f"uses {bracket * 70} as a literal"}

    def test_a_quoted_string_inside_a_flow_list_loads(self) -> None:
        data, codes = self._load("k: ['" + "[" * 70 + "', \"" + "{" * 70 + '"]\n')

        assert codes == []
        assert data["k"] == ["[" * 70, "{" * 70]

    def test_a_comment_with_brackets_is_not_nesting(self) -> None:
        data, codes = self._load("a: 1  # " + "[" * 100 + "\nb: 2\n")

        assert codes == [] and data == {"a": 1, "b": 2}

    def test_the_real_limit_still_holds_for_nested_flow_collections(self) -> None:
        for raw in (
            "k: " + "[" * 70 + "]" * 70,
            "k: " + "{a: " * 70 + "1" + "}" * 70,
            "k: 'x'\nj: " + "[" * 70 + "]" * 70,  # a quote before the nesting
            "k: ['a', " + "[" * 70 + "]" * 70 + "]",
        ):
            data, codes = self._load(raw)
            assert (data, codes) == ({}, ["frontmatter-too-deep"]), raw

    @pytest.mark.parametrize(
        "raw",
        [
            "a: [it's, " + "[" * 300 + "]" * 300 + "]\n",
            "a: [foo 'bar, " + "[" * 300 + "]" * 300 + "]\n",  # a false quote span
        ],
    )
    def test_an_apostrophe_cannot_hide_the_nesting(self, raw: str) -> None:
        # A quote-like character in a plain scalar must not open a span that
        # swallows a genuinely nested flow collection.
        assert self._load(raw)[1] == ["frontmatter-too-deep"]

    def test_a_flood_of_quoted_brackets_is_still_refused(self) -> None:
        raw = "k: '" + "[" * 5000 + "'\n"

        assert self._load(raw)[1] == ["frontmatter-too-deep"]

    def test_parse_skill_text_accepts_the_flat_string(self) -> None:
        text = (
            '---\nname: ok\ndescription: "uses ' + "[" * 70 + ' literally"\n---\nbody\n'
        )

        skill, issues = parse_skill_text(text, "ok")

        assert skill is not None
        assert "frontmatter-too-deep" not in [i.code for i in issues]


class _Issue:
    def __init__(self, severity: str, code: str, path: str = "p") -> None:
        self.severity, self.code, self.path = severity, code, path
        self.message = "m\x1b[31m"


class TestOneValidateReport:
    """F-OK-09, F-XD-11, F-OK-08: the shared report and ``--max-seconds``."""

    @staticmethod
    def _report(issues: list[Any], *, strict: bool = False) -> Callable[[], int]:
        return lambda: cli_support.report_issues(
            issues,
            label="things",
            count=3,
            strict=strict,
            where=lambda issue: issue.path,
        )

    def test_the_report_lines_the_summary_and_the_exit_code(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        issues = [_Issue("warning", "w"), _Issue("info", "i"), _Issue("error", "e")]

        code = self._report(issues)()

        out, err = capsys.readouterr()
        assert code == 1
        assert out == "warning w p m\ninfo i p m\nerror e p m\n"
        assert err == "3 things, 1 errors, 1 warnings, 1 info\n"

    def test_a_warning_fails_only_under_strict(self) -> None:
        warnings = [_Issue("warning", "w")]

        assert self._report(warnings)() == 0
        assert self._report(warnings, strict=True)() == 1
        assert self._report([])() == 0

    @pytest.mark.parametrize("code", ["load-deadline", "check-deadline"])
    def test_both_deadline_codes_hint_the_flag(
        self, capsys: pytest.CaptureFixture[str], code: str
    ) -> None:
        self._report([_Issue("error", code)])()

        assert capsys.readouterr().out.endswith("m (raise with --max-seconds)\n")

    def test_other_codes_have_no_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        self._report([_Issue("error", "type-missing")])()

        assert "--max-seconds" not in capsys.readouterr().out

    def test_the_okf_cli_hints_a_check_deadline(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "a.md").write_text("---\ntype: Note\n---\nx\n", encoding="utf-8")
        deadline = OKFIssue(severity="error", code="check-deadline", message="late")
        monkeypatch.setattr(okf_cli, "check_bundle", lambda bundle: [deadline])

        code = okf_cli.main(["validate", str(tmp_path)])

        out = capsys.readouterr().out
        assert code == 1
        assert "error check-deadline - late (raise with --max-seconds)" in out

    def test_the_max_seconds_argument_is_one_definition(self) -> None:
        parser = argparse.ArgumentParser()
        cli_support.add_max_seconds(parser, 7.0)
        action = next(a for a in parser._actions if "--max-seconds" in a.option_strings)

        assert action.type is cli_support.positive_seconds
        assert parser.parse_args([]).max_seconds == 7.0
        assert parser.parse_args(["--max-seconds", "2"]).max_seconds == 2.0

    def test_neither_cli_repeats_the_report_or_the_argument(self) -> None:
        for mod in (okf_cli, skills_cli):
            source = inspect.getsource(mod)
            assert "raise with --max-seconds" not in source
            assert "type=positive_seconds" not in source
            assert "Counter" not in source

    def test_the_defaults_are_still_the_per_format_constants(self) -> None:
        for mod, default in (
            (okf_cli, OKF_MAX_LOAD_SECONDS),
            (skills_cli, SKILL_MAX_LOAD_SECONDS),
        ):
            parser = mod._build_parser()
            sub = next(
                a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
            )
            action = next(
                a
                for a in sub.choices["validate"]._actions
                if "--max-seconds" in a.option_strings
            )
            assert action.default == default


class TestOkfModuleSurface:
    """F-OK-12: a static ``__all__`` and no private alias only tests need."""

    def test_all_is_static_and_names_real_attributes(self) -> None:
        names = okf_module.__all__

        assert isinstance(names, list | tuple) and len(set(names)) == len(names)
        assert all(hasattr(okf_module, n) for n in names)
        for needed in ("load_bundle", "check_bundle", "strip_unsafe_text", "iso_utc"):
            assert needed in names

    def test_no_private_alias_of_the_frontmatter_seam_is_left(self) -> None:
        assert not hasattr(okf_module, "_Note")
        assert not hasattr(okf_module, "_SURROGATE_RE")
