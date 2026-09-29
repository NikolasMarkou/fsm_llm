"""Tests for fsm_llm.eval.cases and ``fsm-llm-eval run`` (offline, mock LLM).

The fixture FSM moves ``start -> done`` only when Pass 1 extracts ``name``, so
the same case passes with ``extraction_data={"name": "Ada"}`` and fails with an
empty extraction. Both outcomes are asserted: a suite that only ever sees
passing trials cannot tell a working checker from one that checks nothing.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import ClassVar

import pytest

from fsm_llm.eval import (
    ConversationCase,
    EvalConfig,
    EvalConfigError,
    EvalDatasetError,
    Expectations,
    TrialResult,
    check_expectations,
    load_cases,
    pass_rate,
    read_rows,
    run_case_trial,
    run_cases,
    run_dataset,
    wilson_ci,
)
from fsm_llm.eval import __main__ as cli
from fsm_llm.eval import cases as cases_mod
from fsm_llm.eval.cases import CaseReport
from tests.conftest import MockLLM2Interface

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_DATASET = _REPO_ROOT / "evaluation" / "datasets" / "simple_greeting_cases.json"


def _name_fsm() -> dict:
    return {
        "name": "NameBot",
        "description": "Asks for a name, then says goodbye",
        "initial_state": "start",
        "states": {
            "start": {
                "id": "start",
                "description": "Ask for the name",
                "purpose": "Learn the user's name",
                "extraction_instructions": "Extract the user's name as 'name'",
                "response_instructions": "Ask for the user's name",
                "required_context_keys": ["name"],
                "transitions": [
                    {
                        "target_state": "done",
                        "description": "The name is known",
                        "priority": 0,
                        "conditions": [
                            {
                                "description": "name extracted",
                                "requires_context_keys": ["name"],
                                "logic": {"!!": [{"var": "name"}]},
                            }
                        ],
                    }
                ],
            },
            "done": {
                "id": "done",
                "description": "Goodbye",
                "purpose": "Close the conversation",
                "response_instructions": "Say goodbye",
            },
        },
    }


_FULL_EXPECT = {
    "final_state": "done",
    "visited_states": ["start", "done"],
    "context": {"name": "Ada"},
    "context_keys": ["name"],
    "responses_contain": ["HOW CAN I HELP"],
    "ended": True,
}


def _case(**overrides) -> ConversationCase:
    raw = {
        "id": "name_case",
        "fsm": _name_fsm(),
        "turns": ["I am Ada"],
        "expect": _FULL_EXPECT,
    }
    raw.update(overrides)
    return ConversationCase.model_validate(raw)


def _factory(extraction: dict):
    return lambda: MockLLM2Interface(extraction_data=dict(extraction))


def _passing():
    return _factory({"name": "Ada"})


def _failing():
    return _factory({})


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadCases:
    def test_json_list(self, tmp_path: Path):
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        cases, config = load_cases(path)
        assert [c.id for c in cases] == ["name_case"]
        assert config == {}

    def test_json_object_with_embedded_config(self, tmp_path: Path):
        path = _write(
            tmp_path / "d.json",
            {"config": {"trials": 5}, "cases": [_case().model_dump()]},
        )
        cases, config = load_cases(path)
        assert len(cases) == 1
        assert config == {"trials": 5}

    def test_jsonl_skips_blank_lines(self, tmp_path: Path):
        one = json.dumps(_case(id="a").model_dump())
        two = json.dumps(_case(id="b").model_dump())
        path = tmp_path / "d.jsonl"
        path.write_text(f"{one}\n\n   \n{two}\n", encoding="utf-8")
        cases, config = load_cases(path)
        assert [c.id for c in cases] == ["a", "b"]
        assert config == {}

    def test_jsonl_bad_line_names_its_number(self, tmp_path: Path):
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps(_case().model_dump()) + "\n\n{oops\n")
        with pytest.raises(EvalDatasetError, match="line 3"):
            load_cases(path)

    def test_fsm_path_resolves_against_dataset_dir(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        (data_dir / "fsms").mkdir(parents=True)
        _write(data_dir / "fsms" / "bot.json", _name_fsm())
        path = _write(
            data_dir / "d.json", [{**_case().model_dump(), "fsm": "fsms/bot.json"}]
        )
        monkeypatch.chdir(tmp_path)  # CWD is NOT the dataset dir
        cases, _ = load_cases(path)
        assert cases[0].fsm == str((data_dir / "fsms" / "bot.json").resolve())

    def test_missing_fsm_file(self, tmp_path: Path):
        path = _write(tmp_path / "d.json", [{**_case().model_dump(), "fsm": "no.json"}])
        with pytest.raises(EvalDatasetError, match="FSM file not found"):
            load_cases(path)

    def test_duplicate_ids_rejected(self, tmp_path: Path):
        path = _write(tmp_path / "d.json", [_case().model_dump()] * 2)
        with pytest.raises(EvalDatasetError, match="duplicate case id"):
            load_cases(path)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"turns": []},
            {"expect": {}},
            {"expect": {"final_stat": "done"}},
            {"extra_key": 1},
            {"id": ""},
        ],
        ids=["no-turns", "no-checks", "typo-check", "unknown-key", "empty-id"],
    )
    def test_invalid_case_rejected(self, tmp_path: Path, overrides):
        raw = {**_case().model_dump(exclude_none=True), **overrides}
        path = _write(tmp_path / "d.json", [raw])
        with pytest.raises(EvalDatasetError, match="invalid case"):
            load_cases(path)

    def test_zero_cases_rejected(self, tmp_path: Path):
        with pytest.raises(EvalDatasetError, match="no cases"):
            load_cases(_write(tmp_path / "d.json", []))

    def test_bad_embedded_config_rejected(self, tmp_path: Path):
        path = _write(
            tmp_path / "d.json",
            {"config": {"trials": 0}, "cases": [_case().model_dump()]},
        )
        with pytest.raises(EvalConfigError, match="embedded config"):
            load_cases(path)

    def test_unknown_top_level_key_rejected(self, tmp_path: Path):
        path = _write(tmp_path / "d.json", {"cases": [], "extra": 1})
        with pytest.raises(EvalDatasetError, match="unknown top-level"):
            load_cases(path)

    def test_missing_dataset(self, tmp_path: Path):
        with pytest.raises(EvalDatasetError, match="not found"):
            load_cases(tmp_path / "nope.json")

    def test_sample_dataset_loads(self):
        cases, config = load_cases(_SAMPLE_DATASET)
        assert len(cases) >= 2
        assert config == {"trials": 3}
        for case in cases:
            assert isinstance(case.fsm, str) and Path(case.fsm).is_file()


# ---------------------------------------------------------------------------
# Expectation checks (pure)
# ---------------------------------------------------------------------------


def _observed(**overrides) -> TrialResult:
    trial = TrialResult(
        case_id="c",
        trial=1,
        passed=False,
        final_state="done",
        visited_states=["start", "done"],
        responses=["Hello! How can I help you?"],
        context={"name": "Ada", "empty": None},
        ended=True,
    )
    for key, value in overrides.items():
        setattr(trial, key, value)
    return trial


class TestCheckExpectations:
    def test_all_checks_hold(self):
        assert check_expectations(Expectations(**_FULL_EXPECT), _observed()) == []

    @pytest.mark.parametrize(
        ("expect", "fragment"),
        [
            ({"final_state": "start"}, "final_state"),
            ({"visited_states": ["middle"]}, "'middle' not visited"),
            ({"context": {"name": "Bob"}}, "context.name: expected 'Bob'"),
            ({"context": {"age": 3}}, "context.age: missing"),
            ({"context_keys": ["empty"]}, "'empty' missing or null"),
            ({"context_keys": ["absent"]}, "'absent' missing or null"),
            ({"responses_contain": ["goodbye"]}, "'goodbye' not in any response"),
            ({"ended": False}, "ended: expected False"),
        ],
    )
    def test_each_check_can_fail(self, expect, fragment):
        failures = check_expectations(Expectations(**expect), _observed())
        assert len(failures) == 1
        assert fragment in failures[0]

    def test_empty_expectations_rejected(self):
        with pytest.raises(ValueError, match="at least one check"):
            Expectations()


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


class TestRunCaseTrial:
    def test_extraction_passes_every_check(self):
        result = run_case_trial(_case(), EvalConfig(), 1, _passing())
        assert result.error is None
        assert result.failures == []
        assert result.passed is True
        assert result.visited_states == ["start", "done"]
        assert result.context["name"] == "Ada"
        assert result.ended is True

    def test_empty_extraction_fails_state_and_context(self):
        result = run_case_trial(_case(), EvalConfig(), 1, _failing())
        assert result.passed is False
        assert result.error is None
        joined = "\n".join(result.failures)
        assert "final_state: expected 'done', got 'start'" in joined
        assert "context.name: missing" in joined
        assert "ended: expected True, got False" in joined

    def test_fsm_file_path(self, tmp_path: Path):
        fsm_path = _write(tmp_path / "bot.json", _name_fsm())
        result = run_case_trial(_case(fsm=str(fsm_path)), EvalConfig(), 1, _passing())
        assert result.passed is True

    def test_initial_context_is_passed(self):
        case = _case(
            initial_context={"plan": "gold"}, expect={"context": {"plan": "gold"}}
        )
        assert run_case_trial(case, EvalConfig(), 1, _failing()).passed is True

    def test_stops_sending_after_the_conversation_ends(self):
        case = _case(turns=["I am Ada", "one more thing"])
        result = run_case_trial(case, EvalConfig(), 1, _passing())
        assert result.turns_sent == 1
        assert result.passed is True

    def test_raising_factory_is_a_failed_trial(self):
        def boom():
            raise RuntimeError("provider down")

        result = run_case_trial(_case(), EvalConfig(), 2, boom)
        assert result.passed is False
        assert result.error == "RuntimeError: provider down"
        assert result.first_failure() == "error: RuntimeError: provider down"
        assert result.trial == 2

    def test_invalid_inline_fsm_is_a_failed_trial(self):
        result = run_case_trial(_case(fsm={"name": "x"}), EvalConfig(), 1, _passing())
        assert result.passed is False
        assert result.error


# ---------------------------------------------------------------------------
# Whole runs and reports
# ---------------------------------------------------------------------------


class TestRunCases:
    def test_mixed_run_reports_rows_rates_and_ci(self, tmp_path: Path):
        cases = [
            _case(id="passes"),
            _case(id="fails", expect={"final_state": "nowhere"}),
        ]
        config = EvalConfig(trials=3, workers=2)
        report = run_cases(cases, config, tmp_path, llm_interface_factory=_passing())

        rows = read_rows(tmp_path / "rows.jsonl")
        assert len(rows) == len(cases) * config.trials
        assert sorted((r["case_id"], r["trial"]) for r in rows) == sorted(
            (c.id, n) for c in cases for n in (1, 2, 3)
        )

        data = json.loads((tmp_path / "results.json").read_text())
        by_id = {c["id"]: c for c in data["cases"]}
        assert (by_id["passes"]["k"], by_id["passes"]["n"]) == (3, 3)
        assert (by_id["fails"]["k"], by_id["fails"]["n"]) == (0, 3)
        assert by_id["passes"]["wilson_ci"] == list(wilson_ci(3, 3))
        assert by_id["fails"]["wilson_ci"] == list(wilson_ci(0, 3))
        assert "final_state: expected 'nowhere'" in by_id["fails"]["first_failure"]
        assert data["overall"]["k"] == 3 and data["overall"]["n"] == 6
        assert data["overall"]["wilson_ci"] == list(wilson_ci(3, 6))
        assert data["config"]["trials"] == 3
        assert report.overall == data["overall"]

        summary = (tmp_path / "summary.md").read_text()
        assert "| passes | 3/3 | 100.0% |" in summary
        assert "| fails | 0/3 | 0.0% |" in summary
        assert "Overall pass rate" in summary

    def test_failing_extraction_is_reported_as_failure(self, tmp_path: Path):
        report = run_cases(
            [_case()], EvalConfig(trials=2), tmp_path, llm_interface_factory=_failing()
        )
        assert report.overall["k"] == 0
        assert "final_state" in report.cases[0]["first_failure"]
        assert report.cases[0]["failure_counts"]

    def test_raising_factory_completes_the_run(self, tmp_path: Path):
        def boom():
            raise RuntimeError("no model")

        report = run_cases(
            [_case()], EvalConfig(trials=2), tmp_path, llm_interface_factory=boom
        )
        assert report.overall["rate"] == 0.0
        rows = read_rows(tmp_path / "rows.jsonl")
        assert [r["error"] for r in rows] == ["RuntimeError: no model"] * 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def offline_cli(monkeypatch):
    """Route the CLI's run_cases through a mock LLM chosen per test."""

    def use(factory):
        monkeypatch.setattr(
            cli,
            "run_cases",
            functools.partial(run_cases, llm_interface_factory=factory),
        )

    return use


class TestRunCli:
    def test_help_lists_flags(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main_cli(["run", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for flag in (
            "--model",
            "--trials",
            "--workers",
            "--output-dir",
            "--config",
            "--fail-under",
            "--list",
        ):
            assert flag in out

    def test_list_prints_case_ids(self, capsys):
        assert cli.main_cli(["run", str(_SAMPLE_DATASET), "--list"]) == 0
        out = capsys.readouterr().out
        for case in load_cases(_SAMPLE_DATASET)[0]:
            assert case.id in out

    def test_bad_dataset_exits_1(self, tmp_path: Path, capsys):
        path = _write(tmp_path / "d.json", [])
        assert cli.main_cli(["run", str(path)]) == 1
        assert "no cases" in capsys.readouterr().err

    def test_missing_dataset_exits_1(self, tmp_path: Path):
        assert cli.main_cli(["run", str(tmp_path / "nope.json")]) == 1

    def test_fail_under_exits_2_on_failures(self, tmp_path, offline_cli, capsys):
        offline_cli(_failing())
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        out = tmp_path / "out"
        code = cli.main_cli(
            [
                "run",
                str(path),
                "--trials",
                "1",
                "--output-dir",
                str(out),
                "--fail-under",
                "100",
            ]
        )
        assert code == 2
        assert "below --fail-under" in capsys.readouterr().err
        assert (out / "results.json").is_file()
        assert (out / "summary.md").is_file()

    def test_passing_run_exits_0_under_threshold(self, tmp_path, offline_cli):
        offline_cli(_passing())
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        code = cli.main_cli(
            [
                "run",
                str(path),
                "--output-dir",
                str(tmp_path / "o"),
                "--fail-under",
                "100",
            ]
        )
        assert code == 0

    def test_low_score_without_threshold_exits_0(self, tmp_path, offline_cli):
        offline_cli(_failing())
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        assert (
            cli.main_cli(["run", str(path), "--output-dir", str(tmp_path / "o")]) == 0
        )

    def test_default_run_dir_under_output_root(
        self, tmp_path, offline_cli, monkeypatch
    ):
        offline_cli(_passing())
        monkeypatch.chdir(tmp_path)
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        assert cli.main_cli(["run", str(path), "--trials", "1"]) == 0
        run_dirs = list((tmp_path / "evaluation").iterdir())
        assert len(run_dirs) == 1
        assert len(read_rows(run_dirs[0] / "rows.jsonl")) == 1

    def test_settings_precedence(self, tmp_path, offline_cli):
        """defaults (3) < dataset config (2) < --config (4) < --trials (1)."""
        offline_cli(_passing())
        dataset = _write(
            tmp_path / "d.json",
            {"config": {"trials": 2}, "cases": [_case().model_dump()]},
        )
        config_file = _write(tmp_path / "c.json", {"trials": 4})
        runs = {
            "embedded": [],
            "config": ["--config", str(config_file)],
            "flag": ["--config", str(config_file), "--trials", "1"],
        }
        counts = {}
        for name, extra in runs.items():
            out = tmp_path / name
            assert (
                cli.main_cli(["run", str(dataset), "--output-dir", str(out), *extra])
                == 0
            )
            counts[name] = len(read_rows(out / "rows.jsonl"))
        assert counts == {"embedded": 2, "config": 4, "flag": 1}

        plain = _write(tmp_path / "plain.json", [_case().model_dump()])
        assert (
            cli.main_cli(["run", str(plain), "--output-dir", str(tmp_path / "p")]) == 0
        )
        assert len(read_rows(tmp_path / "p" / "rows.jsonl")) == 3


# ---------------------------------------------------------------------------
# Review-iter-1 fixes (D-009)
# ---------------------------------------------------------------------------


class TestUnsentTurns:
    def test_early_end_without_ended_expectation_fails(self):
        """Defect guarded: a case whose FSM ended early passed a final_state
        check although later turns were never sent (review NOTE 10)."""
        case = _case(turns=["I am Ada", "one more"], expect={"final_state": "done"})
        result = run_case_trial(case, EvalConfig(), 1, _passing())
        assert result.turns_sent == 1
        assert result.passed is False
        assert result.failures == [
            "turns: conversation ended with 1 of 2 turns unsent "
            "(declare expect.ended to allow it)"
        ]

    def test_declared_ended_allows_the_early_end(self):
        case = _case(turns=["I am Ada", "one more"], expect={"ended": True})
        assert run_case_trial(case, EvalConfig(), 1, _passing()).passed is True


class TestChecksHook:
    def test_custom_check_failures_are_recorded(self, tmp_path: Path):
        def short_greeting(trial: TrialResult) -> list[str]:
            return [] if len(trial.responses[0]) < 5 else ["greeting too long"]

        report = run_cases(
            [_case()],
            EvalConfig(trials=2),
            tmp_path,
            llm_interface_factory=_passing(),
            checks=[short_greeting],
        )
        assert report.overall["k"] == 0
        assert report.cases[0]["first_failure"] == "greeting too long"

    def test_raising_check_is_a_trial_error(self):
        def boom(_trial: TrialResult) -> list[str]:
            raise ValueError("bad check")

        result = run_case_trial(_case(), EvalConfig(), 1, _passing(), checks=[boom])
        assert result.passed is False
        assert result.error == "ValueError: bad check"


class TestRunDataset:
    def test_one_call_with_layered_settings(self, tmp_path: Path):
        dataset = _write(
            tmp_path / "d.json",
            {"config": {"trials": 4}, "cases": [_case().model_dump()]},
        )
        out = tmp_path / "out"
        report = run_dataset(
            dataset,
            config={"trials": 3},
            llm_interface_factory=_passing(),
            output_dir=str(out),
            trials=2,
        )
        assert report.run_dir == out
        assert (report.overall["k"], report.overall["n"]) == (2, 2)
        assert json.loads((out / "results.json").read_text())["dataset"] == str(dataset)

    def test_config_file_and_model_layers(self, tmp_path: Path):
        dataset = _write(tmp_path / "d.json", [_case().model_dump()])
        cfg = _write(tmp_path / "c.json", {"trials": 1, "model": "file-model"})
        report = run_dataset(
            dataset,
            config=cfg,
            llm_interface_factory=_passing(),
            output_dir=str(tmp_path / "o"),
        )
        assert report.overall["n"] == 1
        assert report.model == "file-model"
        explicit = run_dataset(
            dataset,
            config=EvalConfig(trials=2),
            llm_interface_factory=_passing(),
            output_dir=str(tmp_path / "o2"),
        )
        assert explicit.overall["n"] == 2

    def test_unknown_override_is_a_config_error(self, tmp_path: Path):
        dataset = _write(tmp_path / "d.json", [_case().model_dump()])
        with pytest.raises(EvalConfigError):
            run_dataset(dataset, trails=2)


class _CapturingAPI:
    """Stands in for ``fsm_llm.API``: records the constructor kwargs, then fails."""

    seen: ClassVar[list[dict]] = []

    @classmethod
    def from_definition(cls, _fsm, **kwargs):
        cls.seen.append(kwargs)
        raise RuntimeError("captured")

    from_file = from_definition


class TestLLMSettingsReachTheAPI:
    def test_model_temperature_max_tokens_and_llm_kwargs(self, monkeypatch):
        """Pins the real-model branch (review NOTE 15), which every other
        test bypasses with a factory."""
        monkeypatch.setattr(cases_mod, "API", _CapturingAPI)
        _CapturingAPI.seen = []
        config = EvalConfig(
            model="m-1",
            temperature=0.2,
            max_tokens=64,
            llm_kwargs={"api_base": "http://h", "max_history_size": 2},
        )
        result = run_case_trial(_case(), config, 1)
        assert result.error == "RuntimeError: captured"
        assert _CapturingAPI.seen == [
            {
                "api_base": "http://h",
                "max_history_size": 2,
                "model": "m-1",
                "temperature": 0.2,
                "max_tokens": 64,
            }
        ]

    def test_unset_settings_are_not_passed(self, monkeypatch):
        monkeypatch.setattr(cases_mod, "API", _CapturingAPI)
        _CapturingAPI.seen = []
        run_case_trial(_case(), EvalConfig(model="m-1"), 1)
        assert _CapturingAPI.seen == [{"model": "m-1"}]

    def test_llm_kwargs_values_are_not_written(self, tmp_path: Path):
        config = EvalConfig(trials=1, llm_kwargs={"api_key": "sk-secret-value"})
        run_cases([_case()], config, tmp_path, llm_interface_factory=_passing())
        text = (tmp_path / "results.json").read_text()
        assert "sk-secret-value" not in text
        assert json.loads(text)["config"]["llm_kwargs"] == {"api_key": "<not recorded>"}

    def test_cli_flags_set_temperature_and_max_tokens(self, tmp_path, offline_cli):
        offline_cli(_passing())
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        out = tmp_path / "o"
        argv = ["run", str(path), "--output-dir", str(out), "--trials", "1"]
        assert cli.main_cli([*argv, "--temperature", "0.3", "--max-tokens", "50"]) == 0
        recorded = json.loads((out / "results.json").read_text())["config"]
        assert (recorded["temperature"], recorded["max_tokens"]) == (0.3, 50)


class TestSampleDatasetMeasuresTheModel:
    def test_do_nothing_model_fails_the_extraction_case(self, tmp_path: Path):
        """Defect guarded: the shipped sample passed 100% with a mock that
        extracts nothing, so it measured no model behaviour (review W6)."""
        cases, _ = load_cases(_SAMPLE_DATASET)
        report = run_cases(
            cases,
            EvalConfig(trials=1),
            tmp_path,
            llm_interface_factory=lambda: MockLLM2Interface(),
        )
        rates = {c["id"]: c["k"] for c in report.cases}
        assert rates["name_is_extracted"] == 0
        assert report.overall["k"] < report.overall["n"]

    def test_extracting_model_passes_every_case(self, tmp_path: Path):
        cases, _ = load_cases(_SAMPLE_DATASET)
        report = run_cases(
            cases,
            EvalConfig(trials=1),
            tmp_path,
            llm_interface_factory=_factory({"user_name": "Alex"}),
        )
        assert report.overall["k"] == report.overall["n"] == len(cases)


class TestInputDecoding:
    @pytest.mark.parametrize("name", ["d.json", "d.jsonl"])
    def test_non_utf8_dataset_exits_1_with_path(self, tmp_path, capsys, name):
        path = tmp_path / name
        path.write_bytes(b"\xff\xfe[")
        assert cli.main_cli(["run", str(path)]) == 1
        err = capsys.readouterr().err
        assert "cannot read dataset" in err and str(path) in err


class TestInterrupt:
    def test_ctrl_c_writes_partial_report(self, tmp_path: Path):
        def ctrl_c(*_args):
            raise KeyboardInterrupt

        report = run_cases(
            [_case(id="a"), _case(id="b")],
            EvalConfig(trials=3, workers=1),
            tmp_path,
            llm_interface_factory=_passing(),
            progress=ctrl_c,
        )
        assert report.interrupted is True
        assert report.overall["n"] == 1
        assert len(read_rows(tmp_path / "rows.jsonl")) == 1
        assert json.loads((tmp_path / "results.json").read_text())["interrupted"]
        assert "**Interrupted**: yes" in (tmp_path / "summary.md").read_text()

    def test_cli_exits_130(self, tmp_path, offline_cli, monkeypatch, capsys):
        def ctrl_c(*_args):
            raise KeyboardInterrupt

        offline_cli(_passing())
        monkeypatch.setattr(cli, "_print_trial", ctrl_c)
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        out = tmp_path / "o"
        assert cli.main_cli(["run", str(path), "--output-dir", str(out)]) == 130
        assert "1 of 3 trials" in capsys.readouterr().err
        assert (out / "summary.md").is_file()


class TestFailUnderBoundary:
    @pytest.mark.parametrize(("k", "code"), [(57, 0), (56, 2)])
    def test_57_of_100_meets_57(self, tmp_path, monkeypatch, k, code):
        """Defect guarded: 57/100 is 56.99999... in floats, so
        ``--fail-under 57`` exited 2 while printing 57.0% (review W1)."""

        def fake_run(cases, config, run_dir, **_kwargs):
            return CaseReport(run_dir, "m", [], [], pass_rate(k, 100), 0.0)

        monkeypatch.setattr(cli, "run_cases", fake_run)
        path = _write(tmp_path / "d.json", [_case().model_dump()])
        argv = ["run", str(path), "--output-dir", str(tmp_path / "o")]
        assert cli.main_cli([*argv, "--fail-under", "57"]) == code
