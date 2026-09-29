"""
Conversation-case evaluation: scripted multi-turn conversations with pass/fail
expectations, repeated trials, and Wilson confidence intervals.

A dataset is a JSON list of cases, a JSON object ``{"config": {...}, "cases":
[...]}``, or a ``.jsonl`` file with one case per line. Each case names an FSM
(a path relative to the dataset file, or an inline definition), an optional
``initial_context``, the user ``turns`` to send, and ``expect``: any of
``final_state``, ``visited_states``, ``context``, ``context_keys``,
``responses_contain`` and ``ended``. Every case runs ``config.trials`` times
in-process through :class:`fsm_llm.API`; a trial passes when every declared
expectation (and every ``checks`` callable given in Python) holds, and every
turn was sent unless the case declares ``ended``. The run directory gets ``rows.jsonl`` (one row per trial,
appended as it finishes), ``results.json`` and ``summary.md``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fsm_llm.api import API
from fsm_llm.llm import LLMInterface
from fsm_llm.utilities import redacting_json_default

from ._pool import run_interruptible
from .config import EvalConfig, load_config, merge_config, resolve_model
from .constants import CASES_EVALUATOR_NAME, JSONL_SUFFIX, ROWS_FILENAME
from .exceptions import EvalConfigError, EvalDatasetError
from .records import append_row, git_short_hash, open_run_dir, utc_now, write_json
from .stats import pass_rate

#: Builds a fresh LLM interface for one trial (offline tests, custom providers).
LLMInterfaceFactory = Callable[[], LLMInterface]
#: Called once per finished trial: ``(completed, total, trial)``.
TrialProgressCallback = Callable[[int, int, "TrialResult"], None]
#: A custom check: returns one message per failure for a finished trial.
TrialCheck = Callable[["TrialResult"], list[str]]

_DATASET_KEYS = frozenset({"config", "cases"})


class Expectations(BaseModel):
    """What a trial must show to pass; only declared checks are evaluated.

    - ``final_state``: the state after the last turn sent.
    - ``visited_states``: each listed state was the current state at some
      point (after start or after a turn), in any order.
    - ``context``: each key's value in ``get_data`` equals the given value.
    - ``context_keys``: each key is present in ``get_data`` and not null.
    - ``responses_contain``: each substring appears, case-insensitively, in at
      least one response (the start response included).
    - ``ended``: whether the conversation reached a terminal state.
    """

    model_config = ConfigDict(extra="forbid")

    final_state: str | None = None
    visited_states: list[str] | None = None
    context: dict[str, Any] | None = None
    context_keys: list[str] | None = None
    responses_contain: list[str] | None = None
    ended: bool | None = None

    @model_validator(mode="after")
    def _at_least_one_check(self) -> Expectations:
        # DECISION plan-2026-09-29T061903-581c2634/D-004
        # Do NOT accept an empty `expect` (not even when Python `checks` are
        # passed): a case with no declared check passes every trial and reads
        # as a measured success. See decisions.md D-004.
        if not self.model_dump(exclude_none=True):
            raise ValueError("expect must declare at least one check")
        return self


class ConversationCase(BaseModel):
    """One scripted conversation: the FSM, the user turns, and the expectations.

    After :func:`load_cases`, a string ``fsm`` is an absolute file path.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    fsm: str | dict[str, Any]
    initial_context: dict[str, Any] = Field(default_factory=dict)
    turns: list[str] = Field(min_length=1)
    expect: Expectations
    description: str = ""


@dataclass
class TrialResult:
    """The outcome of one trial of one case (one ``rows.jsonl`` row)."""

    case_id: str
    trial: int
    passed: bool
    failures: list[str] = field(default_factory=list)
    error: str | None = None
    final_state: str | None = None
    visited_states: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    ended: bool = False
    turns_sent: int = 0
    duration: float = 0.0

    def first_failure(self) -> str:
        """The error, else the first failed check, else ``""``."""
        if self.error:
            return f"error: {self.error}"
        return self.failures[0] if self.failures else ""

    def to_row(self) -> dict[str, Any]:
        """JSON-ready row; non-JSON context values become placeholders."""
        row = asdict(self)
        row["duration"] = round(self.duration, 3)
        row["context"] = json.loads(
            json.dumps(self.context, default=redacting_json_default)
        )
        return row


@dataclass
class CaseReport:
    """A finished conversation run: its directory, trials and pass rates."""

    run_dir: Path
    model: str
    trials: list[TrialResult]
    cases: list[dict[str, Any]]
    overall: dict[str, Any]
    wall_time: float
    interrupted: bool = False  # Ctrl-C: trials hold only finished trials


def _case_from(raw: Any, where: str, base_dir: Path) -> ConversationCase:
    """Validate one raw case and resolve its FSM path against ``base_dir``."""
    try:
        case = ConversationCase.model_validate(raw)
    except ValidationError as exc:
        raise EvalDatasetError(f"invalid case ({where}): {exc}") from exc
    if isinstance(case.fsm, str):
        fsm_path = (base_dir / case.fsm).resolve()
        if not fsm_path.is_file():
            raise EvalDatasetError(
                f"case {case.id!r} ({where}): FSM file not found: {fsm_path}",
                details={"case": case.id, "path": str(fsm_path)},
            )
        case = case.model_copy(update={"fsm": str(fsm_path)})
    return case


def _read_json(path: Path) -> tuple[list[Any], dict[str, Any]]:
    """Raw cases and embedded config from a JSON list or ``{config, cases}`` object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalDatasetError(f"dataset {path} is not JSON: {exc}") from exc
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict):
        raise EvalDatasetError(f"dataset {path} must hold a JSON list or object")
    unknown = sorted(set(data) - _DATASET_KEYS)
    if unknown:
        raise EvalDatasetError(f"dataset {path}: unknown top-level keys {unknown}")
    cases, config = data.get("cases"), data.get("config", {})
    if not isinstance(cases, list):
        raise EvalDatasetError(f"dataset {path}: 'cases' must be a list")
    if not isinstance(config, dict):
        raise EvalDatasetError(f"dataset {path}: 'config' must be an object")
    return cases, config


def _read_jsonl(path: Path) -> list[tuple[Any, str]]:
    """Raw cases with their ``line N`` locations; blank lines are skipped."""
    raw = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw.append((json.loads(line), f"line {number}"))
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(
                f"dataset {path} line {number} is not JSON: {exc}"
            ) from exc
    return raw


def load_cases(path: str | Path) -> tuple[list[ConversationCase], dict[str, Any]]:
    """Load a conversation dataset and return ``(cases, embedded_config)``.

    Interface contract (callers: the ``run`` CLI and the Python API):
        - ``.jsonl`` files hold one case per line (blank lines skipped, no
          embedded config); other files hold a JSON list of cases or an
          object ``{"config": {...}, "cases": [...]}``.
        - A string ``fsm`` is resolved against the dataset's directory and must
          exist; the returned case holds the absolute path.
        - ``embedded_config`` is a config layer (validated here) for
          :func:`fsm_llm.eval.config.merge_config`; ``{}`` when absent.
        - Raises ``EvalDatasetError`` for a missing, non-UTF-8 or malformed
          file, an
          invalid case (unknown key, no turns, no expectation), a missing FSM
          file, duplicate ids, or zero cases; ``EvalConfigError`` for a bad
          embedded config.
    """
    dataset = Path(path)
    if not dataset.is_file():
        raise EvalDatasetError(f"dataset file not found: {dataset}")
    config: dict[str, Any] = {}
    try:
        if dataset.suffix == JSONL_SUFFIX:
            located = _read_jsonl(dataset)
        else:
            raw_cases, config = _read_json(dataset)
            located = [(raw, f"case {i}") for i, raw in enumerate(raw_cases)]
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalDatasetError(f"cannot read dataset {dataset}: {exc}") from exc
    try:
        merge_config(config)  # validate the embedded layer before any trial runs
    except EvalConfigError as exc:
        raise EvalConfigError(f"dataset {dataset} embedded config: {exc}") from exc
    base_dir = dataset.resolve().parent
    cases = [_case_from(raw, where, base_dir) for raw, where in located]
    if not cases:
        raise EvalDatasetError(f"dataset {dataset} has no cases")
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise EvalDatasetError(f"dataset {dataset}: duplicate case id {case.id!r}")
        seen.add(case.id)
    return cases, config


def check_expectations(expect: Expectations, trial: TrialResult) -> list[str]:
    """Return one message per failed check in ``expect``; ``[]`` means pass."""
    failures = []
    if expect.final_state is not None and trial.final_state != expect.final_state:
        failures.append(
            f"final_state: expected {expect.final_state!r}, got {trial.final_state!r}"
        )
    for state in expect.visited_states or []:
        if state not in trial.visited_states:
            failures.append(
                f"visited_states: {state!r} not visited ({trial.visited_states})"
            )
    for key, want in (expect.context or {}).items():
        if key not in trial.context:
            failures.append(f"context.{key}: missing (expected {want!r})")
        elif trial.context[key] != want:
            failures.append(
                f"context.{key}: expected {want!r}, got {trial.context[key]!r}"
            )
    for key in expect.context_keys or []:
        if trial.context.get(key) is None:
            failures.append(f"context_keys: {key!r} missing or null")
    lowered = [r.lower() for r in trial.responses]
    for text in expect.responses_contain or []:
        if not any(text.lower() in r for r in lowered):
            failures.append(f"responses_contain: {text!r} not in any response")
    if expect.ended is not None and trial.ended != expect.ended:
        failures.append(f"ended: expected {expect.ended}, got {trial.ended}")
    return failures


def _llm_kwargs(
    config: EvalConfig, llm_interface_factory: LLMInterfaceFactory | None
) -> dict[str, Any]:
    """``API`` keyword arguments for one trial: the factory, else the LLM settings."""
    if llm_interface_factory is not None:
        return {"llm_interface": llm_interface_factory()}
    kwargs: dict[str, Any] = {**config.llm_kwargs, "model": resolve_model(config)}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    return kwargs


def run_case_trial(
    case: ConversationCase,
    config: EvalConfig,
    trial: int,
    llm_interface_factory: LLMInterfaceFactory | None = None,
    *,
    checks: Sequence[TrialCheck] = (),
) -> TrialResult:
    """Run one trial of ``case`` and check its expectations. Never raises.

    Interface contract (callers: :func:`run_cases`, tests):
        - Builds a fresh ``API`` per trial: ``llm_interface=factory()`` when a
          factory is given, else ``model=resolve_model(config)``, the set
          ``temperature``/``max_tokens`` and ``config.llm_kwargs``.
        - Sends the turns in order and stops early once the conversation has
          ended (a terminal state accepts no more turns); ``turns_sent``
          records how many were sent. Unsent turns are a failure unless the
          case declares ``expect.ended``.
        - ``failures`` = the ``expect`` checks, then each of ``checks`` called
          with the finished trial.
        - Any exception (bad FSM, LLM down, a raising factory or check) gives a
          failed trial with ``error`` set; the API is closed in every case.
    """
    result = TrialResult(case_id=case.id, trial=trial, passed=False)
    start = time.monotonic()
    api: API | None = None
    try:
        kwargs = _llm_kwargs(config, llm_interface_factory)
        if isinstance(case.fsm, str):
            api = API.from_file(case.fsm, **kwargs)
        else:
            api = API.from_definition(case.fsm, **kwargs)
        conv_id, greeting = api.start_conversation(dict(case.initial_context))
        result.responses.append(greeting)
        result.visited_states.append(api.get_current_state(conv_id))
        for turn in case.turns:
            if api.has_conversation_ended(conv_id):
                break
            result.responses.append(api.converse(turn, conv_id))
            result.turns_sent += 1
            result.visited_states.append(api.get_current_state(conv_id))
        result.final_state = result.visited_states[-1]
        result.context = api.get_data(conv_id)
        result.ended = api.has_conversation_ended(conv_id)
        result.failures = check_expectations(case.expect, result)
        unsent = len(case.turns) - result.turns_sent
        if unsent and case.expect.ended is None:
            result.failures.append(
                f"turns: conversation ended with {unsent} of {len(case.turns)} "
                "turns unsent (declare expect.ended to allow it)"
            )
        for check in checks:
            result.failures.extend(check(result))
        result.passed = not result.failures
    except Exception as exc:  # one trial's failure is data, never a crashed run
        result.error = f"{type(exc).__name__}: {exc}"
        result.passed = False
    finally:
        if api is not None:
            try:
                api.close()  # ends every active conversation of this API
            except Exception as exc:
                result.error = result.error or f"close failed: {exc}"
                result.passed = False
        result.duration = time.monotonic() - start
    return result


def _summarise(
    cases: Sequence[ConversationCase], trials: list[TrialResult]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-case pass rates (dataset order) and the overall pass rate."""
    by_case: dict[str, list[TrialResult]] = {c.id: [] for c in cases}
    for t in trials:
        by_case[t.case_id].append(t)
    summaries = []
    for case in cases:
        rows = sorted(by_case[case.id], key=lambda t: t.trial)
        failed = [t for t in rows if not t.passed]
        counts: dict[str, int] = {}
        for t in failed:
            for message in t.failures or [f"error: {t.error}"]:
                counts[message] = counts.get(message, 0) + 1
        summaries.append(
            {
                "id": case.id,
                "description": case.description,
                **pass_rate(len(rows) - len(failed), len(rows)),
                "first_failure": failed[0].first_failure() if failed else "",
                "failure_counts": counts,
            }
        )
    passed = sum(1 for t in trials if t.passed)
    return summaries, pass_rate(passed, len(trials))


def run_cases(
    cases: Sequence[ConversationCase],
    config: EvalConfig,
    run_dir: Path,
    *,
    llm_interface_factory: LLMInterfaceFactory | None = None,
    progress: TrialProgressCallback | None = None,
    dataset: str | Path | None = None,
    checks: Sequence[TrialCheck] = (),
) -> CaseReport:
    """Run every case ``config.trials`` times and write the run's reports.

    Interface contract (callers: the ``run`` CLI, :func:`run_dataset`):
        - ``config.workers`` threads run (case, trial) pairs, each through
          :func:`run_case_trial` (with ``checks``); trial numbers start at 1.
        - Each trial's row is appended to ``run_dir/rows.jsonl`` and flushed as
          it finishes, then ``progress`` is called.
        - Ctrl-C (``KeyboardInterrupt``) cancels the trials not yet started and
          returns a partial report (``interrupted=True``, finished trials only).
        - Writes ``results.json`` and ``summary.md`` (:func:`write_case_report`)
          and returns the report; ``dataset`` is recorded there when given.
    """
    # DECISION plan-2026-09-29T061903-581c2634/D-004
    # Trials run on threads in this process. Do NOT move them to a subprocess
    # per trial to get a hard kill of a hung LLM call: that costs a fork per
    # trial and pickling of inline FSMs and factories; hung calls are bounded
    # by litellm's own request timeout instead. See decisions.md D-004.
    model = resolve_model(config)
    pairs = [(case, n) for case in cases for n in range(1, config.trials + 1)]
    rows_path = run_dir / ROWS_FILENAME
    trials: list[TrialResult] = []

    def trial_of(pair: tuple[ConversationCase, int]) -> TrialResult:
        case, n = pair
        return run_case_trial(case, config, n, llm_interface_factory, checks=checks)

    def record(_pair: object, future: Future[TrialResult]) -> None:
        trial = future.result()  # run_case_trial never raises
        trials.append(trial)
        append_row(rows_path, trial.to_row())
        if progress is not None:
            progress(len(trials), len(pairs), trial)

    start = time.monotonic()
    interrupted = run_interruptible(trial_of, pairs, config.workers, record)
    wall_time = time.monotonic() - start
    trials.sort(key=lambda t: (t.case_id, t.trial))
    summaries, overall = _summarise(cases, trials)
    report = CaseReport(
        run_dir, model, trials, summaries, overall, wall_time, interrupted
    )
    write_case_report(report, config, dataset)
    return report


def run_dataset(
    path: str | Path,
    *,
    config: EvalConfig | Mapping[str, Any] | str | Path | None = None,
    llm_interface_factory: LLMInterfaceFactory | None = None,
    checks: Sequence[TrialCheck] = (),
    progress: TrialProgressCallback | None = None,
    **overrides: Any,
) -> CaseReport:
    """Load a dataset, run it, write the reports, and return them: one call.

    Interface contract (the Python counterpart of ``fsm-llm-eval run``):
        - Settings layer like the CLI: defaults < the dataset's embedded
          ``config`` < ``config`` (a JSON file path, a mapping, or an
          ``EvalConfig`` whose explicitly set fields count) < ``overrides``
          (``EvalConfig`` field names, e.g. ``trials=5, model="..."``).
        - The run directory follows the CLI rule (``output_dir``, else a new
          directory under ``output_root``); ``llm_interface_factory``,
          ``checks`` and ``progress`` are passed to :func:`run_cases`.
        - Raises ``EvalDatasetError``/``EvalConfigError`` for a bad dataset or
          setting, ``EvalError`` when no run directory can be created.
    """
    cases, embedded = load_cases(path)
    if isinstance(config, EvalConfig):
        layer: Mapping[str, Any] | None = config.model_dump(exclude_unset=True)
    elif isinstance(config, (str, Path)):
        layer = load_config(config)
    else:
        layer = config
    merged = merge_config(embedded, layer, overrides)
    git_cwd = Path(path).resolve().parent
    run_dir = open_run_dir(
        merged.output_dir, merged.output_root, resolve_model(merged), cwd=git_cwd
    )
    return run_cases(
        cases,
        merged,
        run_dir,
        llm_interface_factory=llm_interface_factory,
        progress=progress,
        dataset=path,
        checks=checks,
    )


def _recorded_config(config: EvalConfig) -> dict[str, Any]:
    """The config for ``results.json``; ``llm_kwargs`` values are never written
    (they can hold an API key or auth headers), only their keys."""
    data = config.model_dump()
    data["llm_kwargs"] = dict.fromkeys(sorted(config.llm_kwargs), "<not recorded>")
    return data


def write_case_report(
    report: CaseReport, config: EvalConfig, dataset: str | Path | None = None
) -> Path:
    """Write ``results.json`` and ``summary.md``; return the summary path."""
    git_cwd = Path(dataset).resolve().parent if dataset is not None else None
    git_hash = git_short_hash(git_cwd)
    overall = report.overall
    lo, hi = overall["wilson_ci"]
    lines = [
        "# Conversation evaluation",
        "",
        f"- **Date**: {utc_now()}",
        f"- **Git commit**: {git_hash}",
        f"- **Model**: {report.model}",
        f"- **Dataset**: {dataset if dataset is not None else '(in-memory)'}",
        f"- **Cases**: {len(report.cases)}",
        f"- **Trials per case**: {config.trials}",
        f"- **Workers**: {config.workers}",
        f"- **Evaluator**: {CASES_EVALUATOR_NAME}",
        *(
            ["- **Interrupted**: yes, partial run (finished trials only)"]
            if report.interrupted
            else []
        ),
        "",
        "## Cases",
        "",
        "| Case | Passed | Rate | Wilson 95% CI | First failure |",
        "|------|--------|------|---------------|---------------|",
    ]
    for c in report.cases:
        c_lo, c_hi = c["wilson_ci"]
        failure = c["first_failure"].replace("|", "\\|")
        lines.append(
            f"| {c['id']} | {c['k']}/{c['n']} | {c['rate'] * 100:.1f}% | "
            f"[{c_lo * 100:.1f}%, {c_hi * 100:.1f}%] | {failure} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- **Overall pass rate**: {overall['k']}/{overall['n']} = "
        f"**{overall['rate'] * 100:.1f}%** (Wilson 95% CI "
        f"[{lo * 100:.1f}%, {hi * 100:.1f}%])",
        f"- **Wall time**: {report.wall_time:.1f}s",
        "",
    ]
    summary_path = report.run_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        report.run_dir / "results.json",
        {
            "date": utc_now(),
            "git_commit": git_hash,
            "model": report.model,
            "dataset": str(dataset) if dataset is not None else None,
            "evaluator": CASES_EVALUATOR_NAME,
            "config": _recorded_config(config),
            "overall": overall,
            "cases": report.cases,
            "wall_time_s": round(report.wall_time, 1),
            "interrupted": report.interrupted,
        },
    )
    return summary_path
