"""`scripts/refresh_pins.py`: the newest litellm and mcp release that has aged and fits the matrix.

Plan d454fd62, step 110 (roadmap row 0.10). The script reads PyPI JSON, picks the highest
release that is stable, not yanked, at least 7 days old (the cooldown that exists because
litellm 1.82.7 and 1.82.8 were malware for hours), admitted by `requires_python` for every
supported Python, in the major of the current pin and inside the declared range, and rewrites
one line of `constraints.txt`. A scheduled workflow runs it (`test_refresh_workflow.py`).

Every test here uses canned JSON built in this file and an injected fetcher: no network. An
autouse fixture makes `urllib.request.urlopen` fail loudly, so a tested path that reached the
network would fail instead of being flaky.

* `scenario_problems(module, tmp_path)` runs the behaviour scenarios against a loaded copy of
  the script and returns the codes of the scenarios that fail; the real script returns the empty
  set. Each of the mutants (`SCRIPT_MUTANTS`) is the real source with one edit, loaded as a
  module, and must return its own code: a check that cannot fail is not a check.
* The specifier evaluator (`parse_specifier`, `admits`, `admits_python`) is a small subset of
  PEP 440 written because `packaging` is not standard library; its table is tested and it is
  compared with `packaging.specifiers` over a grid (the test environment has `packaging`, the
  script does not need it).
* The fixture JSON keeps the shape of `https://pypi.org/pypi/<package>/json`: `releases` maps a
  version to its files, each with `upload_time_iso_8601`, `yanked` and `requires_python`.
"""

from __future__ import annotations

import ast
import contextlib
import io
import itertools
import json
import sys
import types
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.test_fsm_llm_regression.support import load_script_module

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "refresh_pins.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "python-package.yml"

NOW = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-11-14T12:00:00Z"
PYTHONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]
_OK_RP = ">=3.10,<3.16"


def _load_module(source: str, name: str) -> types.ModuleType:
    return load_script_module(source, name, _SCRIPT)


@pytest.fixture(scope="module")
def rp() -> types.ModuleType:
    assert _SCRIPT.is_file(), "scripts/refresh_pins.py is missing"
    module = _load_module(_SCRIPT.read_text(encoding="utf-8"), "_refresh_pins_real")
    yield module
    sys.modules.pop("_refresh_pins_real", None)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a tested path reached the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


# --------------------------------------------------------------------------------------
# Canned data
# --------------------------------------------------------------------------------------

CONSTRAINTS = (
    "# header comment\n"
    "# litellm note: keep this line\n"
    "litellm==1.109.1\n"
    "\n"
    "loguru==0.8.0\n"
    "# the mcp pin\n"
    "mcp==2.2.0  # trailing comment\n"
    "pytest==9.1.1\n"
)
PYPROJECT = (
    "[project]\n"
    'name = "demo"\n'
    'requires-python = ">=3.11"\n'
    "dependencies = [\n"
    '    "litellm>=1.109.0,<2.0",\n'
    '    "pydantic>=2.0,<3",\n'
    "]\n"
    "\n"
    "[project.optional-dependencies]\n"
    'mcp = ["mcp>=2.2,<3"]\n'
)


def _ago(*, days: float = 0, hours: float = 0) -> str:
    moment = NOW - timedelta(days=days, hours=hours)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def release(
    *,
    days: float | None = None,
    hours: float = 0,
    rp: str | None = _OK_RP,
    yanked: bool = False,
    extra_days: tuple[float, ...] = (),
) -> list[dict[str, Any]]:
    """The file list of one release: one file at `days` ago, more at `extra_days`."""
    stamps = [] if days is None else [days, *extra_days]
    return [
        {
            "filename": f"pkg-{i}.whl",
            "upload_time_iso_8601": _ago(days=d, hours=hours if i == 0 else 0),
            "yanked": yanked,
            "requires_python": rp,
        }
        for i, d in enumerate(stamps)
    ]


def document(name: str, releases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"info": {"name": name}, "releases": releases}


def _litellm(**extra: list[dict[str, Any]]) -> dict[str, Any]:
    base = {"1.109.0": release(days=40), "1.109.1": release(days=30)}
    base.update({k.replace("_", "."): v for k, v in extra.items()})
    return document("litellm", base)


def _mcp(**extra: list[dict[str, Any]]) -> dict[str, Any]:
    base = {"2.1.0": release(days=90), "2.2.0": release(days=60)}
    base.update({k.replace("_", "."): v for k, v in extra.items()})
    return document("mcp", base)


class Outcome:
    """The result of one run of `main` on tmp copies of the files."""

    def __init__(
        self, code: int, out: list[str], err: str, constraints: str, pyproject: str
    ):
        self.code = code
        self.out = out
        self.err = err
        self.constraints = constraints
        self.pyproject = pyproject

    @property
    def text(self) -> str:
        return "\n".join(self.out)

    @property
    def changed_line(self) -> str | None:
        lines = [ln for ln in self.out if ln.startswith("changed=")]
        return lines[-1] if lines else None


def run_main(
    module: types.ModuleType,
    tmp: Path,
    docs: dict[str, Any],
    *,
    constraints: str | bytes = CONSTRAINTS,
    pyproject: str = PYPROJECT,
    args: tuple[str, ...] = (),
    pythons: tuple[str, ...] = tuple(PYTHONS),
    now: str = NOW_TEXT,
    packages: tuple[str, ...] | None = None,
    requested: list[str] | None = None,
) -> Outcome:
    tmp.mkdir(parents=True, exist_ok=True)
    cpath, ppath = tmp / "constraints.txt", tmp / "pyproject.toml"
    if isinstance(constraints, bytes):
        cpath.write_bytes(constraints)
    else:
        cpath.write_text(constraints, encoding="utf-8", newline="")
    ppath.write_text(pyproject, encoding="utf-8", newline="")

    def fetch(package: str) -> Any:
        if requested is not None:
            requested.append(package)
        if package not in docs:
            raise module.RefreshError(f"{package}: no canned document")
        return docs[package]

    argv = ["--constraints", str(cpath), "--pyproject", str(ppath), "--now", now]
    if pythons:
        argv += ["--python", *pythons]
    if packages is not None:
        argv += ["--packages", *packages]
    argv += list(args)
    out: list[str] = []
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = module.main(argv, fetch=fetch, stdout=out.append)
    return Outcome(
        code,
        out,
        err.getvalue(),
        cpath.read_bytes().decode("utf-8"),
        ppath.read_bytes().decode("utf-8"),
    )


def _pins(text: str) -> dict[str, str]:
    found = {}
    for line in text.splitlines():
        for name in ("litellm", "mcp"):
            if line.startswith(f"{name}=="):
                found[name] = line.split("==")[1].split()[0]
    return found


# --------------------------------------------------------------------------------------
# The behaviour scenarios (each returns its code when it fails)
# --------------------------------------------------------------------------------------

Scenario = Callable[[types.ModuleType, Path], bool]


def _pin_result(module: types.ModuleType, tmp: Path, docs: dict[str, Any], **kw: Any):
    result = run_main(module, tmp, docs, **kw)
    return result, _pins(result.constraints)


def sc_cooldown_young(module: types.ModuleType, tmp: Path) -> bool:
    # 6 days 23 hours old: skipped; the pin stays.
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=6, hours=23)}),
        "mcp": _mcp(),
    }
    result, pins = _pin_result(module, tmp, docs)
    return result.code == 0 and pins["litellm"] == "1.109.1"


def sc_cooldown_boundary(module: types.ModuleType, tmp: Path) -> bool:
    # exactly 7 days old: chosen.
    docs = {"litellm": _litellm(**{"1_109_2": release(days=7)}), "mcp": _mcp()}
    result, pins = _pin_result(module, tmp, docs)
    return result.code == 0 and pins["litellm"] == "1.109.2"


def sc_cooldown_earliest_file(module: types.ModuleType, tmp: Path) -> bool:
    # the earliest file is 9 days old, a later wheel 1 day old: the release is 9 days old.
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=9, extra_days=(1,))}),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.2"


def sc_major_of_the_pin(module: types.ModuleType, tmp: Path) -> bool:
    # the range was widened by hand to <4, a 3.0.0 exists: still never crosses the major.
    wide = PYPROJECT.replace('"mcp>=2.2,<3"', '"mcp>=2.2,<4"')
    docs = {
        "litellm": _litellm(),
        "mcp": _mcp(**{"2_3_0": release(days=20), "3_0_0": release(days=20)}),
    }
    result, pins = _pin_result(module, tmp, docs, pyproject=wide)
    return pins["mcp"] == "2.3.0" and result.pyproject == wide


def sc_declared_range(module: types.ModuleType, tmp: Path) -> bool:
    # 1.110.0 is excluded by the declared `!=` (how a compromised release stays out).
    spec = PYPROJECT.replace("<2.0", "<2.0,!=1.110.0")
    docs = {
        "litellm": _litellm(**{"1_110_0": release(days=20)}),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs, pyproject=spec)
    return pins["litellm"] == "1.109.1"


def sc_major_two_not_under_lt_two(module: types.ModuleType, tmp: Path) -> bool:
    docs = {
        "litellm": _litellm(**{"2_0_0": release(days=20), "2_1_0": release(days=20)}),
        "mcp": _mcp(),
    }
    result, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.1" and result.pyproject == PYPROJECT


def sc_widen_within_major(module: types.ModuleType, tmp: Path) -> bool:
    narrow = PYPROJECT.replace("<2.0", "<1.115")
    pins_text = CONSTRAINTS.replace("1.109.1", "1.110.0")
    docs = {
        "litellm": _litellm(
            **{
                "1_110_0": release(days=60),
                "1_116_0": release(days=20),
                "2_0_0": release(days=20),
            }
        ),
        "mcp": _mcp(),
    }
    result, pins = _pin_result(
        module, tmp, docs, pyproject=narrow, constraints=pins_text
    )
    return (
        pins["litellm"] == "1.116.0"
        and '"litellm>=1.109.0,<2.0"' in result.pyproject
        and result.pyproject == narrow.replace("<1.115", "<2.0")
    )


def sc_python_ceiling(module: types.ModuleType, tmp: Path) -> bool:
    # the newest release excludes 3.15: skipped, the older one that admits it wins.
    docs = {
        "litellm": _litellm(
            **{
                "1_109_3": release(days=10, rp=">=3.10,<3.15"),
                "1_109_2": release(days=15, rp=">=3.10,<3.16"),
            }
        ),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.2"


def sc_python_floor_patch(module: types.ModuleType, tmp: Path) -> bool:
    # `>=3.11.5` does not admit every 3.11 runner: skipped.
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=15, rp=">=3.11.5")}),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.1"


def sc_python_ceiling_patch(module: types.ModuleType, tmp: Path) -> bool:
    # `<3.15.2` does not admit every 3.15 runner: skipped.
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=15, rp=">=3.10,<3.15.2")}),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.1"


def sc_python_missing(module: types.ModuleType, tmp: Path) -> bool:
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=15, rp=None)}),
        "mcp": _mcp(**{"2_3_0": release(days=15, rp="")}),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins == {"litellm": "1.109.1", "mcp": "2.2.0"}


def sc_python_unknown_operator(module: types.ModuleType, tmp: Path) -> bool:
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=15, rp="^3.10")}),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.1"


def sc_yanked(module: types.ModuleType, tmp: Path) -> bool:
    docs = {
        "litellm": _litellm(
            **{
                "1_109_3": release(days=15, yanked=True),
                "1_109_2": release(days=16),
            }
        ),
        "mcp": _mcp(),
    }
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.2"


def sc_yanked_one_file(module: types.ModuleType, tmp: Path) -> bool:
    files = release(days=15, extra_days=(15,))
    files[1]["yanked"] = True
    docs = {"litellm": _litellm(**{"1_109_3": files}), "mcp": _mcp()}
    _, pins = _pin_result(module, tmp, docs)
    return pins["litellm"] == "1.109.1"


def sc_release_without_files(module: types.ModuleType, tmp: Path) -> bool:
    docs = {"litellm": _litellm(**{"1_109_3": []}), "mcp": _mcp()}
    result, pins = _pin_result(module, tmp, docs)
    return result.code == 0 and pins["litellm"] == "1.109.1"


def sc_prerelease(module: types.ModuleType, tmp: Path) -> bool:
    docs = {
        "litellm": _litellm(
            **{
                "1.110.0rc1": release(days=20),
                "1.110.0.dev3": release(days=20),
                "1.110.0.post1": release(days=20),
                "1.110.0b2": release(days=20),
                "1!2.0": release(days=20),
            }
        ),
        "mcp": _mcp(),
    }
    result, pins = _pin_result(module, tmp, docs)
    return result.code == 0 and pins["litellm"] == "1.109.1"


def sc_no_downgrade(module: types.ModuleType, tmp: Path) -> bool:
    # the pin 1.109.5 was yanked after it was chosen; the newest eligible release is 1.109.2
    # (the rest too young): the pin stays, a refresh never moves it down.
    pinned = CONSTRAINTS.replace("1.109.1", "1.109.5")
    docs = {
        "litellm": document(
            "litellm",
            {
                "1.109.2": release(days=30),
                "1.109.5": release(days=20, yanked=True),
                "1.109.6": release(days=1),
                "1.109.7": release(days=2),
            },
        ),
        "mcp": _mcp(),
    }
    result, pins = _pin_result(module, tmp, docs, constraints=pinned)
    return (
        pins["litellm"] == "1.109.5"
        and result.constraints == pinned
        and result.changed_line == "changed=false"
    )


def sc_no_change(module: types.ModuleType, tmp: Path) -> bool:
    docs = {"litellm": _litellm(), "mcp": _mcp()}
    result = run_main(module, tmp, docs)
    return (
        result.code == 0
        and result.constraints == CONSTRAINTS
        and result.pyproject == PYPROJECT
        and result.changed_line == "changed=false"
        and result.out[-1] == "changed=false"
        and "```diff" not in result.text
    )


def sc_changed_line(module: types.ModuleType, tmp: Path) -> bool:
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=10)}),
        "mcp": _mcp(**{"2_3_0": release(days=10)}),
    }
    result = run_main(module, tmp, docs)
    return (
        result.code == 0
        and result.out[-1] == "changed=true"
        and sum(1 for ln in result.out if ln.startswith("changed=")) == 1
        and _pins(result.constraints) == {"litellm": "1.109.2", "mcp": "2.3.0"}
    )


def sc_other_lines_untouched(module: types.ModuleType, tmp: Path) -> bool:
    # CRLF endings, no final newline, a trailing comment on the mcp line.
    original = CONSTRAINTS.replace("\n", "\r\n").rstrip("\r\n")
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=10)}),
        "mcp": _mcp(**{"2_3_0": release(days=10)}),
    }
    result = run_main(module, tmp, docs, constraints=original)
    expected = original.replace("litellm==1.109.1", "litellm==1.109.2").replace(
        "mcp==2.2.0  # trailing comment", "mcp==2.3.0  # trailing comment"
    )
    return result.constraints == expected and result.pyproject == PYPROJECT


def sc_deterministic(module: types.ModuleType, tmp: Path) -> bool:
    docs_a = {
        "litellm": _litellm(**{"1_109_2": release(days=10)}),
        "mcp": _mcp(**{"2_3_0": release(days=10)}),
    }
    docs_b = {
        "mcp": {
            "releases": dict(reversed(list(docs_a["mcp"]["releases"].items()))),
            "info": docs_a["mcp"]["info"],
        },
        "litellm": {
            "releases": dict(reversed(list(docs_a["litellm"]["releases"].items()))),
            "info": docs_a["litellm"]["info"],
        },
    }
    first = run_main(module, tmp / "a", docs_a, args=("--dry-run",))
    second = run_main(module, tmp / "b", docs_b, args=("--dry-run",))
    third = run_main(
        module, tmp / "c", docs_a, args=("--dry-run",), packages=("mcp", "litellm")
    )
    # the diff names a different tmp file? no: it names `constraints.txt` and `pyproject.toml`.
    return first.out == second.out == third.out and first.code == 0


def sc_dry_run(module: types.ModuleType, tmp: Path) -> bool:
    docs = {"litellm": _litellm(**{"1_109_2": release(days=10)}), "mcp": _mcp()}
    result = run_main(module, tmp, docs, args=("--dry-run",))
    return (
        result.code == 0
        and result.constraints == CONSTRAINTS
        and result.pyproject == PYPROJECT
        and result.changed_line == "changed=true"
        and "-litellm==1.109.1" in result.text
        and "+litellm==1.109.2" in result.text
        and "Dry run" in result.text
    )


def sc_missing_package(module: types.ModuleType, tmp: Path) -> bool:
    without = CONSTRAINTS.replace("mcp==2.2.0  # trailing comment\n", "")
    docs = {"litellm": _litellm(**{"1_109_2": release(days=10)}), "mcp": _mcp()}
    result = run_main(module, tmp, docs, constraints=without)
    return result.code == 1 and result.constraints == without and "mcp" in result.err


def sc_duplicate_pin(module: types.ModuleType, tmp: Path) -> bool:
    twice = CONSTRAINTS + "litellm==1.100.0\n"
    docs = {"litellm": _litellm(), "mcp": _mcp()}
    result = run_main(module, tmp, docs, constraints=twice)
    return result.code == 1 and result.constraints == twice


def sc_malformed_document(module: types.ModuleType, tmp: Path) -> bool:
    # litellm would change; mcp's document is malformed: exit 1 and NOTHING written.
    docs = {
        "litellm": _litellm(**{"1_109_2": release(days=10)}),
        "mcp": {"info": {"name": "mcp"}, "releases": ["not", "a", "mapping"]},
    }
    result = run_main(module, tmp, docs)
    return (
        result.code == 1
        and result.constraints == CONSTRAINTS
        and result.pyproject == PYPROJECT
        and not result.out
    )


def sc_wrong_document(module: types.ModuleType, tmp: Path) -> bool:
    # the document served for `mcp` describes litellm: refuse.
    docs = {"litellm": _litellm(), "mcp": _litellm(**{"1_109_2": release(days=10)})}
    result = run_main(module, tmp, docs)
    return result.code == 1 and result.constraints == CONSTRAINTS


def sc_right_package_fetched(module: types.ModuleType, tmp: Path) -> bool:
    requested: list[str] = []
    docs = {"litellm": _litellm(), "mcp": _mcp()}
    result = run_main(module, tmp, docs, requested=requested)
    return result.code == 0 and sorted(requested) == ["litellm", "mcp"]


def sc_matrix_outside_requires_python(module: types.ModuleType, tmp: Path) -> bool:
    docs = {"litellm": _litellm(), "mcp": _mcp()}
    result = run_main(module, tmp, docs, pythons=("3.10", "3.12"))
    return result.code == 1 and result.constraints == CONSTRAINTS


def sc_failure_exit_code(module: types.ModuleType, tmp: Path) -> bool:
    docs = {"litellm": _litellm()}  # mcp: the fetch fails
    result = run_main(module, tmp, docs)
    return result.code == 1 and result.err.startswith("refresh_pins:")


SCENARIOS: dict[str, Scenario] = {
    "cooldown-young": sc_cooldown_young,
    "cooldown-boundary": sc_cooldown_boundary,
    "cooldown-earliest-file": sc_cooldown_earliest_file,
    "major-of-the-pin": sc_major_of_the_pin,
    "declared-range": sc_declared_range,
    "major-two-not-under-lt-two": sc_major_two_not_under_lt_two,
    "widen-within-major": sc_widen_within_major,
    "python-ceiling": sc_python_ceiling,
    "python-floor-patch": sc_python_floor_patch,
    "python-ceiling-patch": sc_python_ceiling_patch,
    "python-missing": sc_python_missing,
    "python-unknown-operator": sc_python_unknown_operator,
    "yanked": sc_yanked,
    "yanked-one-file": sc_yanked_one_file,
    "release-without-files": sc_release_without_files,
    "prerelease": sc_prerelease,
    "no-downgrade": sc_no_downgrade,
    "no-change": sc_no_change,
    "changed-line": sc_changed_line,
    "other-lines-untouched": sc_other_lines_untouched,
    "deterministic": sc_deterministic,
    "dry-run": sc_dry_run,
    "missing-package": sc_missing_package,
    "duplicate-pin": sc_duplicate_pin,
    "malformed-document": sc_malformed_document,
    "wrong-document": sc_wrong_document,
    "right-package-fetched": sc_right_package_fetched,
    "matrix-outside-requires-python": sc_matrix_outside_requires_python,
    "failure-exit-code": sc_failure_exit_code,
}


def scenario_problems(module: types.ModuleType, tmp_path: Path) -> set[str]:
    """The codes of the scenarios that fail on `module` (a crash counts as a failure)."""
    found: set[str] = set()
    for index, (code, scenario) in enumerate(SCENARIOS.items()):
        try:
            if not scenario(module, tmp_path / f"s{index}"):
                found.add(code)
        except Exception:
            found.add(code)
    return found


# --------------------------------------------------------------------------------------
# The real script
# --------------------------------------------------------------------------------------


class TestTheRealScript:
    def test_every_scenario_passes(self, rp: types.ModuleType, tmp_path: Path):
        assert scenario_problems(rp, tmp_path) == set()

    @pytest.mark.parametrize("code", list(SCENARIOS))
    def test_scenario(self, rp: types.ModuleType, tmp_path: Path, code: str):
        assert SCENARIOS[code](rp, tmp_path), code

    def test_the_cooldown_is_seven_days(self, rp: types.ModuleType):
        assert rp.COOLDOWN == timedelta(days=7)

    def test_the_tracked_packages_are_litellm_and_mcp(self, rp: types.ModuleType):
        assert rp.TRACKED == ("litellm", "mcp")

    def test_a_newer_release_replaces_the_pin_and_the_report_says_so(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        docs = {
            "litellm": _litellm(**{"1_109_2": release(days=10)}),
            "mcp": _mcp(),
        }
        out = run_main(rp, tmp_path, docs)
        assert out.code == 0
        table = [
            ln for ln in out.out if ln.startswith("| litellm") or ln.startswith("| mcp")
        ]
        assert table[0].split("|")[1:7] == [
            " litellm ",
            " 1.109.1 ",
            " 1.109.2 ",
            " 2026-11-04 ",
            " 10 ",
            " newer eligible release ",
        ]
        assert "the pin is the newest eligible release" in table[1]
        assert out.changed_line == "changed=true"

    def test_skipped_newer_releases_are_listed_with_their_reason(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        docs = {
            "litellm": _litellm(
                **{
                    "1_109_2": release(days=3),
                    "1_109_3": release(days=20, yanked=True),
                    "1_109_4": release(days=20, rp=">=3.10,<3.15"),
                }
            ),
            "mcp": _mcp(),
        }
        out = run_main(rp, tmp_path, docs)
        assert "litellm 1.109.2: cooldown (3 days old, needs 7)" in out.text
        assert "litellm 1.109.3: yanked or without files" in out.text
        assert "litellm 1.109.4: Requires-Python >=3.10,<3.15 excludes 3.15" in out.text

    def test_a_pin_outside_its_declared_range_is_refused(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        out = run_main(
            rp,
            tmp_path,
            {"litellm": _litellm(), "mcp": _mcp()},
            constraints=CONSTRAINTS.replace("mcp==2.2.0", "mcp==3.0.0"),
        )
        assert out.code == 1 and "outside the declared range" in out.err

    def test_a_tracked_package_with_no_declared_range_is_refused(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        out = run_main(
            rp,
            tmp_path,
            {"litellm": _litellm(), "mcp": _mcp()},
            pyproject=PYPROJECT.replace('mcp = ["mcp>=2.2,<3"]', "mcp = []"),
        )
        assert out.code == 1 and "no declared range" in out.err

    def test_a_dependency_group_declaration_is_read_too(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        pyproject = PYPROJECT + '\n[dependency-groups]\ndev = ["mcp>=2.2,!=2.3.0,<3"]\n'
        docs = {"litellm": _litellm(), "mcp": _mcp(**{"2_3_0": release(days=10)})}
        out = run_main(rp, tmp_path, docs, pyproject=pyproject)
        assert out.code == 0 and _pins(out.constraints)["mcp"] == "2.2.0"

    def test_the_utc_offset_of_now_is_honoured(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        docs = {"litellm": _litellm(**{"1_109_2": release(days=7)}), "mcp": _mcp()}
        # one hour earlier than the release's seventh day: the release is 6 days 23 hours old.
        out = run_main(rp, tmp_path, docs, now="2026-11-14T11:00:00Z")
        assert _pins(out.constraints)["litellm"] == "1.109.1"
        out = run_main(
            rp, tmp_path / "x", docs, now="2026-11-14T13:00:00+01:00"
        )  # 12:00 UTC
        assert _pins(out.constraints)["litellm"] == "1.109.2"

    def test_a_naive_now_is_read_as_utc(self, rp: types.ModuleType, tmp_path: Path):
        docs = {"litellm": _litellm(**{"1_109_2": release(days=7)}), "mcp": _mcp()}
        out = run_main(rp, tmp_path, docs, now="2026-11-14T12:00:00")
        assert _pins(out.constraints)["litellm"] == "1.109.2"

    def test_the_diff_of_a_real_dry_run_names_both_files_when_the_range_widens(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        narrow = PYPROJECT.replace("<2.0", "<1.115")
        docs = {
            "litellm": _litellm(**{"1_116_0": release(days=20)}),
            "mcp": _mcp(),
        }
        out = run_main(rp, tmp_path, docs, pyproject=narrow, args=("--dry-run",))
        assert "--- a/constraints.txt" in out.text
        assert "--- a/pyproject.toml" in out.text
        assert '+    "litellm>=1.109.0,<2.0",' in out.text
        assert out.pyproject == narrow

    def test_the_writes_are_atomic_files_with_no_leftover(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        docs = {"litellm": _litellm(**{"1_109_2": release(days=10)}), "mcp": _mcp()}
        run_main(rp, tmp_path, docs)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "constraints.txt",
            "pyproject.toml",
        ]


class TestUsageAndFiles:
    def test_usage_errors_exit_two(self, rp: types.ModuleType, tmp_path: Path):
        for argv in (
            ["--no-such-flag"],
            ["--now", "yesterday"],
            ["--python", "three"],
            ["--python", "3.12.1"],
            ["--packages", "bad name"],
            ["--packages", "../evil"],
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                assert rp.main(argv, fetch=lambda p: {}) == 2, argv

    def test_help_exits_zero(self, rp: types.ModuleType):
        with contextlib.redirect_stdout(io.StringIO()):
            assert rp.main(["--help"]) == 0

    def test_a_missing_constraints_file_exits_one(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = rp.main(
                ["--constraints", str(tmp_path / "nope.txt"), "--python", "3.12"],
                fetch=lambda p: {},
            )
        assert code == 1 and "cannot read constraints" in err.getvalue()

    def test_a_workflow_without_a_matrix_exits_one(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        workflow = tmp_path / "wf.yml"
        workflow.write_text("jobs:\n  lint:\n    steps: []\n", encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = rp.main(["--workflow", str(workflow)], fetch=lambda p: {})
        assert code == 1 and "no python-version" in err.getvalue()

    def test_malformed_pyproject_toml_exits_one(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        out = run_main(
            rp,
            tmp_path,
            {"litellm": _litellm(), "mcp": _mcp()},
            pyproject="[project\nname = 1",
        )
        assert out.code == 1 and "not valid TOML" in out.err

    def test_the_default_fetcher_reports_a_network_error_as_exit_one(
        self, rp: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        asked: list[str] = []

        def fail(request: urllib.request.Request, timeout: float = 0) -> None:
            asked.append(request.full_url)
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(urllib.request, "urlopen", fail)
        cpath, ppath = tmp_path / "c.txt", tmp_path / "p.toml"
        cpath.write_text(CONSTRAINTS, encoding="utf-8")
        ppath.write_text(PYPROJECT, encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = rp.main(
                [
                    "--constraints",
                    str(cpath),
                    "--pyproject",
                    str(ppath),
                    *["--python", "3.12", "--now", NOW_TEXT],
                ]
            )
        assert code == 1
        assert asked == ["https://pypi.org/pypi/litellm/json"]
        assert cpath.read_text(encoding="utf-8") == CONSTRAINTS

    def test_the_json_dir_reads_canned_files_and_names_a_missing_one(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        data = tmp_path / "json"
        data.mkdir()
        (data / "litellm.json").write_text(
            json.dumps(_litellm(**{"1_109_2": release(days=10)})), encoding="utf-8"
        )
        cpath, ppath = tmp_path / "c.txt", tmp_path / "p.toml"
        cpath.write_text(CONSTRAINTS, encoding="utf-8")
        ppath.write_text(PYPROJECT, encoding="utf-8")
        argv = ["--constraints", str(cpath), "--pyproject", str(ppath)]
        argv += ["--python", "3.12", "--now", NOW_TEXT, "--json-dir", str(data)]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert rp.main(argv) == 1  # mcp.json is missing
        assert "mcp.json" in err.getvalue()
        assert cpath.read_text(encoding="utf-8") == CONSTRAINTS
        (data / "mcp.json").write_text(json.dumps(_mcp()), encoding="utf-8")
        lines: list[str] = []
        assert rp.main(argv, stdout=lines.append) == 0
        assert lines[-1] == "changed=true"
        assert "litellm==1.109.2" in cpath.read_text(encoding="utf-8")

    def test_malformed_json_text_exits_one_and_writes_nothing(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        data = tmp_path / "json"
        data.mkdir()
        (data / "litellm.json").write_text("{not json", encoding="utf-8")
        (data / "mcp.json").write_text(json.dumps(_mcp()), encoding="utf-8")
        cpath, ppath = tmp_path / "c.txt", tmp_path / "p.toml"
        cpath.write_text(CONSTRAINTS, encoding="utf-8")
        ppath.write_text(PYPROJECT, encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = rp.main(
                [
                    "--constraints",
                    str(cpath),
                    "--pyproject",
                    str(ppath),
                    *["--python", "3.12", "--json-dir", str(data)],
                ]
            )
        assert code == 1
        assert cpath.read_text(encoding="utf-8") == CONSTRAINTS


class TestTheMatrixReader:
    def test_it_reads_the_python_versions_of_the_real_workflow(
        self, rp: types.ModuleType
    ):
        text = _WORKFLOW.read_text(encoding="utf-8")
        assert rp.matrix_pythons(text) == ["3.11", "3.12", "3.13", "3.14", "3.15"]

    def test_the_versions_are_the_matrix_not_the_other_jobs(self, rp: types.ModuleType):
        text = (
            "jobs:\n"
            "  lint:\n"
            "    steps:\n"
            "      - with:\n"
            '          python-version: "3.12"\n'
            "  test:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            '          - {os: a, python-version: "3.11"}\n'
            '          - {os: b, python-version: "3.15"}\n'
            "        # python-version: 3.99\n"
            "    steps:\n"
            "      - with:\n"
            '          python-version: "3.14"\n'
        )
        assert rp.matrix_pythons(text) == ["3.11", "3.15"]

    def test_a_list_form_matrix_is_read(self, rp: types.ModuleType):
        text = 'jobs:\n  t:\n    strategy:\n      matrix:\n        python-version: ["3.10", "3.13"]\n'
        assert rp.matrix_pythons(text) == ["3.10", "3.13"]

    def test_the_versions_sort_numerically(self, rp: types.ModuleType):
        text = (
            "m:\n  matrix:\n    include:\n"
            '      - {python-version: "3.9"}\n      - {python-version: "3.10"}\n'
            '      - {python-version: "3.9"}\n'
        )
        assert rp.matrix_pythons(text) == ["3.9", "3.10"]

    def test_the_matrix_of_the_real_workflow_fits_the_real_requires_python(
        self, rp: types.ModuleType
    ):
        import tomllib

        floor = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["requires-python"]
        for python in rp.matrix_pythons(_WORKFLOW.read_text(encoding="utf-8")):
            assert rp.admits_python(floor, python), python


# --------------------------------------------------------------------------------------
# The specifier evaluator
# --------------------------------------------------------------------------------------

#: (specifier, candidate release, admitted)
_TABLE = [
    (">=3.10,<3.16", "3.15.0", True),
    (">=3.10,<3.16", "3.16.0", False),
    (">=3.10,<3.16", "3.9.9", False),
    (">=3.10", "3.10", True),
    (">=3.10", "3.9.99", False),
    (">3.10", "3.10.0", False),
    (">3.10", "3.10.1", True),
    ("<=3.12", "3.12.0", True),
    ("<=3.12", "3.12.1", False),
    ("<3.15", "3.15.0", False),
    ("<3.15", "3.14.99", True),
    ("==3.12", "3.12.0", True),
    ("==3.12", "3.12.1", False),
    ("==3.12.*", "3.12.7", True),
    ("==3.12.*", "3.13.0", False),
    ("!=3.9.7", "3.9.7", False),
    ("!=3.9.7", "3.9.8", True),
    (">=3.8, !=3.9.*, <4", "3.9.2", False),
    (">=3.8, !=3.9.*, <4", "3.10.0", True),
    ("~=3.10", "3.10.0", True),
    ("~=3.10", "3.99.0", True),
    ("~=3.10", "4.0.0", False),
    ("~=3.10", "3.9.9", False),
    ("~=3.10.2", "3.10.2", True),
    ("~=3.10.2", "3.10.9", True),
    ("~=3.10.2", "3.11.0", False),
    ("~=3.10.2", "3.10.1", False),
    ("", "3.15.0", True),
    (">=3.10,<3.16", "3.10", True),
    (">=3.10.0", "3.10", True),
    ("==3.10.0", "3.10", True),
]

_UNSUPPORTED = [
    "^3.10",
    "===3.12",
    ">=3.10;python_version>'3'",
    ">=3.10rc1",
    "~=3",
    ">=3.*",
    ">=3.10,",
    "latest",
    ">=",
    "<3.16.0.dev0",
    "==3.12.*.*",
    "~=3.10.*",
]


class TestTheSpecifierEvaluator:
    @pytest.mark.parametrize(("spec", "version", "expected"), _TABLE)
    def test_table(self, rp: types.ModuleType, spec: str, version: str, expected: bool):
        candidate = tuple(int(p) for p in version.split("."))
        assert rp.admits(rp.parse_specifier(spec), candidate) is expected

    @pytest.mark.parametrize("spec", _UNSUPPORTED)
    def test_unsupported_forms_raise(self, rp: types.ModuleType, spec: str):
        with pytest.raises(ValueError):
            rp.parse_specifier(spec)

    @pytest.mark.parametrize("spec", _UNSUPPORTED)
    def test_unsupported_forms_admit_no_python(self, rp: types.ModuleType, spec: str):
        assert rp.admits_python(spec, "3.12") is False

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_a_missing_requires_python_admits_nothing(
        self, rp: types.ModuleType, empty
    ):
        assert rp.admits_python(empty, "3.12") is False

    def test_a_python_must_be_a_two_part_version(self, rp: types.ModuleType):
        assert rp.admits_python(">=3.10", "3.12.1") is False
        assert rp.admits_python(">=3.10", "3") is False
        assert rp.admits_python(">=3.10", "three") is False

    def test_python_admission_tests_both_ends_of_a_minor_series(
        self, rp: types.ModuleType
    ):
        assert rp.admits_python(">=3.10,<3.16", "3.15") is True
        assert rp.admits_python(">=3.10,<3.15", "3.15") is False
        assert rp.admits_python(">=3.11.5", "3.11") is False
        assert rp.admits_python("<3.15.2", "3.15") is False
        assert rp.admits_python("!=3.15.1", "3.15") is True

    def test_it_agrees_with_packaging_over_a_grid(self, rp: types.ModuleType):
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        operators = [">=", ">", "<=", "<", "==", "!=", "~="]
        versions = ["3.9", "3.10", "3.10.1", "3.11", "3.12.4", "3.15", "3.16", "4.0"]
        candidates = [
            "3.9.0",
            "3.10.0",
            "3.10.1",
            "3.11.5",
            "3.12.4",
            "3.15.2",
            "3.16.0",
        ]
        checked = 0
        for op, version in itertools.product(operators, versions):
            spec = f"{op}{version}"
            if op == "~=" and "." not in version:
                continue
            for candidate in candidates:
                expected = Version(candidate) in SpecifierSet(spec)
                got = rp.admits(
                    rp.parse_specifier(spec),
                    tuple(int(p) for p in candidate.split(".")),
                )
                assert got is expected, (spec, candidate)
                checked += 1
        for spec in (">=3.10,<3.16", ">=3.8,!=3.9.*,<4", "==3.12.*", "!=3.11.*,>=3.10"):
            for candidate in candidates:
                expected = Version(candidate) in SpecifierSet(spec)
                got = rp.admits(
                    rp.parse_specifier(spec),
                    tuple(int(p) for p in candidate.split(".")),
                )
                assert got is expected, (spec, candidate)
                checked += 1
        assert checked > 300

    def test_release_parsing_is_stable_releases_only(self, rp: types.ModuleType):
        assert rp.parse_release("2.3.0") == (2, 3, 0)
        assert rp.parse_release("2.3") == (2, 3)
        for text in (
            "2",
            "2.3.0rc1",
            "2.3.0.dev1",
            "2.3.0.post1",
            "2.3.0b2",
            "1!2.3.0",
            "2.3.0+local",
            "2.3.0.1.2",
            "v2.3.0",
            "",
        ):
            assert rp.parse_release(text) is None, text

    def test_names_normalise_like_pep_503(self, rp: types.ModuleType):
        assert rp.normalize("Foo_Bar.baz") == "foo-bar-baz"


# --------------------------------------------------------------------------------------
# The script is standard library only and reads no network on the tested paths
# --------------------------------------------------------------------------------------


class TestStandardLibraryOnly:
    def test_it_imports_only_the_standard_library(self):
        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "no relative import"
                names.add((node.module or "").split(".")[0])
        assert names, "no imports found"
        assert names <= set(sys.stdlib_module_names) | {"__future__"}, sorted(
            names - set(sys.stdlib_module_names)
        )

    def test_the_only_network_call_is_in_the_default_fetcher(self):
        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        callers: set[str] = set()
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr == "urlopen":
                    callers.add(func.name)
        assert callers == {"fetch_pypi"}

    def test_the_url_is_https_pypi_json(self, rp: types.ModuleType):
        assert rp.PYPI_URL == "https://pypi.org/pypi/{package}/json"

    def test_importing_the_script_reaches_no_network(self):
        # `_no_network` makes urlopen raise: loading the module again must still work.
        assert _load_module(_SCRIPT.read_text(encoding="utf-8"), "_refresh_pins_again")
        sys.modules.pop("_refresh_pins_again", None)

    def test_it_is_a_main_guarded_script(self):
        text = _SCRIPT.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env python")
        assert 'if __name__ == "__main__":' in text


# --------------------------------------------------------------------------------------
# Against the real files (scratch copies only; the repository files are never written)
# --------------------------------------------------------------------------------------


class TestTheRealFiles:
    def test_the_real_pins_and_declared_ranges_are_found(self, rp: types.ModuleType):
        import tomllib

        constraints = (_ROOT / "constraints.txt").read_text(encoding="utf-8")
        pyproject = tomllib.loads(
            (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        for package in rp.TRACKED:
            _, version, _ = rp.read_pin(constraints, package)
            declared = rp.declarations(pyproject, package)
            assert declared, package
            assert all(rp.admits(d.clauses, version) for d in declared), package

    def test_a_dry_run_over_scratch_copies_changes_nothing_on_disk(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        constraints = (_ROOT / "constraints.txt").read_text(encoding="utf-8")
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        docs = {
            "litellm": _litellm(**{"1_109_9": release(days=30)}),
            "mcp": _mcp(**{"2_9_0": release(days=30)}),
        }
        out = run_main(
            rp,
            tmp_path,
            docs,
            constraints=constraints,
            pyproject=pyproject,
            args=("--dry-run",),
        )
        assert out.code == 0 and out.changed_line == "changed=true"
        assert out.constraints == constraints and out.pyproject == pyproject

    def test_a_real_run_on_scratch_copies_touches_only_the_two_pin_lines(
        self, rp: types.ModuleType, tmp_path: Path
    ):
        constraints = (_ROOT / "constraints.txt").read_text(encoding="utf-8")
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        _, _, old_litellm = rp.read_pin(constraints, "litellm")
        _, _, old_mcp = rp.read_pin(constraints, "mcp")
        docs = {
            "litellm": _litellm(**{"1_109_9": release(days=30)}),
            "mcp": _mcp(**{"2_9_0": release(days=30)}),
        }
        out = run_main(rp, tmp_path, docs, constraints=constraints, pyproject=pyproject)
        assert out.code == 0
        before, after = constraints.splitlines(), out.constraints.splitlines()
        assert len(before) == len(after)
        changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
        assert sorted(changed) == sorted(
            [
                (f"litellm=={old_litellm}", "litellm==1.109.9"),
                (f"mcp=={old_mcp}", "mcp==2.9.0"),
            ]
        )
        assert out.pyproject == pyproject
        assert (_ROOT / "constraints.txt").read_text(encoding="utf-8") == constraints


# --------------------------------------------------------------------------------------
# Mutants of the script: the real source with one edit each; every scenario must catch its own
# --------------------------------------------------------------------------------------

#: (label, expected scenario code, exact old text, new text)
SCRIPT_MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "cooldown boundary is exclusive",
        "cooldown-boundary",
        "if age < COOLDOWN:",
        "if age <= COOLDOWN:",
    ),
    (
        "cooldown is six days",
        "cooldown-young",
        "COOLDOWN = timedelta(days=7)",
        "COOLDOWN = timedelta(days=6)",
    ),
    (
        "cooldown is eight days",
        "cooldown-boundary",
        "COOLDOWN = timedelta(days=7)",
        "COOLDOWN = timedelta(days=8)",
    ),
    (
        "cooldown off by one day in the comparison",
        "cooldown-boundary",
        "if age < COOLDOWN:",
        "if age < COOLDOWN + timedelta(days=1):",
    ),
    (
        "cooldown measured from the latest file",
        "cooldown-earliest-file",
        "uploaded = min(known) if known",
        "uploaded = max(known) if known",
    ),
    (
        "major of the pin ignored",
        "major-of-the-pin",
        '    if release.version[0] != major:\n        return f"another major',
        '    if False:\n        return f"another major',
    ),
    (
        "declared range ignored",
        "declared-range",
        '    if not range_decision(declared, release, major)[0]:\n        return "outside',
        '    if False:\n        return "outside',
    ),
    (
        "major and range both ignored",
        "major-two-not-under-lt-two",
        '    if release.version[0] != major:\n        return f"another major ({release.version[0]}, the pin is {major})"\n'
        '    if not range_decision(declared, release, major)[0]:\n        return "outside the declared range"\n',
        "",
    ),
    (
        "widening never happens",
        "widen-within-major",
        "        widened = widen_upper_bound(item.spec, major)\n",
        "        widened = None\n",
    ),
    (
        "widening jumps two majors",
        "widen-within-major",
        'new = f"<{major + 1}" + ".0" * (len(old) - 1)',
        'new = f"<{major + 2}" + ".0" * (len(old) - 1)',
    ),
    (
        "the specifier ignores the ceiling",
        "python-ceiling",
        '        "<": cmp < 0,',
        '        "<": True,',
    ),
    (
        "only the first patch is tested",
        "python-ceiling-patch",
        "return admits(clauses, (*base, 0)) and admits(clauses, (*base, _PATCH_HIGH))",
        "return admits(clauses, (*base, 0))",
    ),
    (
        "only the last patch is tested",
        "python-floor-patch",
        "return admits(clauses, (*base, 0)) and admits(clauses, (*base, _PATCH_HIGH))",
        "return admits(clauses, (*base, _PATCH_HIGH))",
    ),
    (
        "only the first python is tested",
        "python-ceiling",
        "    for python in pythons:\n        if not admits_python(release.requires_python, python):",
        "    for python in pythons[:1]:\n        if not admits_python(release.requires_python, python):",
    ),
    (
        "a missing requires_python is admitted",
        "python-missing",
        "    if not requires_python or not requires_python.strip():\n        return False",
        "    if not requires_python or not requires_python.strip():\n        return True",
    ),
    (
        "an unknown operator fails open",
        "python-unknown-operator",
        "    except ValueError:\n        return False\n    base = parse_release(python)",
        "    except ValueError:\n        return True\n    base = parse_release(python)",
    ),
    (
        "a yanked release is allowed",
        "yanked",
        'yanked=(not file_list) or any(f.get("yanked") for f in file_list),',
        "yanked=False,",
    ),
    (
        "only a fully yanked release is excluded",
        "yanked-one-file",
        'yanked=(not file_list) or any(f.get("yanked") for f in file_list),',
        'yanked=(not file_list) or all(f.get("yanked") for f in file_list),',
    ),
    (
        "pre-releases are allowed",
        "prerelease",
        '_STABLE = re.compile(r"^\\d+(?:\\.\\d+){1,3}$")',
        '_STABLE = re.compile(r"^\\d+(?:\\.\\d+){1,3}")',
    ),
    (
        "downgrade allowed",
        "no-downgrade",
        "if best is not None and _compare(best.version, pin) > 0:",
        "if best is not None and _compare(best.version, pin) != 0:",
    ),
    (
        "the pin is rewritten whatever the result",
        "no-change",
        "    changed = new_constraints != constraints or new_pyproject != pyproject_text",
        "    changed = True",
    ),
    (
        "the changed line is always false",
        "changed-line",
        "stdout(f\"changed={'true' if changed else 'false'}\")",
        'stdout("changed=false")',
    ),
    (
        "other lines are rewritten",
        "other-lines-untouched",
        '    return "".join(lines)\n\n\ndef widen_upper_bound',
        '    return "\\n".join(ln.rstrip() for ln in lines) + "\\n"\n\n\ndef widen_upper_bound',
    ),
    (
        "output order depends on the argument order",
        "deterministic",
        "for package in sorted(set(packages)):",
        "for package in packages:",
    ),
    (
        "the releases keep the document order",
        "deterministic",
        "    return sorted(found, key=lambda r: r.version)",
        "    return found",
    ),
    (
        "a file is written before every package is read",
        "malformed-document",
        "            new_constraints = rewrite_pin(new_constraints, package, choice.new)\n",
        "            new_constraints = rewrite_pin(new_constraints, package, choice.new)\n"
        "            _atomic_write(constraints_path, new_constraints)\n",
    ),
    (
        "dry run writes",
        "dry-run",
        "        if not dry_run:",
        "        if True:",
    ),
    (
        "the wrong package is fetched",
        "right-package-fetched",
        "choice = choose(package, fetch(package),",
        "choice = choose(package, fetch(packages[0]),",
    ),
    (
        "the name of the document is not checked",
        "wrong-document",
        'if normalize(info["name"]) != normalize(package):',
        "if False:",
    ),
    (
        "a failure exits zero",
        "failure-exit-code",
        '        print(f"refresh_pins: {exc}", file=sys.stderr)\n        return 1',
        '        print(f"refresh_pins: {exc}", file=sys.stderr)\n        return 0',
    ),
    (
        "a duplicate pin line is accepted",
        "duplicate-pin",
        "    if len(hits) != 1:",
        "    if len(hits) < 1:",
    ),
    (
        "a missing pin line is accepted",
        "missing-package",
        "    if len(hits) != 1:",
        "    if len(hits) > 1:",
    ),
    (
        "the matrix is not checked against requires-python",
        "matrix-outside-requires-python",
        "        if not admits_python(floor, python):\n            raise RefreshError(",
        "        if False:\n            raise RefreshError(",
    ),
]


def _mutated(old: str, new: str) -> str:
    source = _SCRIPT.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"the mutation anchor is not unique: {old[:60]!r}"
    return source.replace(old, new)


class TestScriptMutantsAreKilled:
    def test_there_are_at_least_twelve_mutants(self):
        assert len(SCRIPT_MUTANTS) >= 12

    def test_every_scenario_has_a_mutant(self):
        # `release-without-files` is guarded twice (the yanked rule and the unreadable-time
        # rule), so no single edit changes its outcome; the scenario still guards the pair.
        covered = {code for _, code, _, _ in SCRIPT_MUTANTS}
        assert set(SCENARIOS) - covered == {"release-without-files"}

    @pytest.mark.parametrize(
        ("label", "code", "old", "new"),
        SCRIPT_MUTANTS,
        ids=[m[0] for m in SCRIPT_MUTANTS],
    )
    def test_mutant(self, tmp_path: Path, label: str, code: str, old: str, new: str):
        module = _load_module(_mutated(old, new), "_refresh_pins_mutant")
        try:
            assert code in scenario_problems(module, tmp_path), label
        finally:
            sys.modules.pop("_refresh_pins_mutant", None)
