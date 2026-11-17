"""The three guards of ``tests/conftest.py`` that keep a Windows or 3.11 CI leg alive.

* ``compact_param_id`` / ``pytest_make_parametrize_id``: a 100,000 character parameter
  gets a short test id (pytest copies the id into ``PYTEST_CURRENT_TEST`` and Windows
  refuses an environment variable of more than 32,767 characters, so those tests errored
  in setup and never ran).
* ``pytest_collection_modifyitems``: a node id of more than ``MAX_NODEID_CHARS`` fails
  collection, so a new huge parameter is found on Linux.
* ``name_refused`` / ``write_if_creatable`` / ``skip_if_backslash_separates``: the
  hostile file name tests write an ESC or a newline name only where the file system
  takes it (NTFS refuses both) and skip the backslash test where ``\\`` separates.
* ``huge_operand_explanation`` / ``guard_assertrepr_compare`` (installed by
  ``pytest_configure`` over pytest's ``util.assertrepr_compare``; the
  ``pytest_assertrepr_compare`` hook itself cannot stop pytest's own implementation, which
  runs whatever a conftest returns): a failed ``==`` on huge operands is explained without
  ``difflib.ndiff`` (cubic on Python 3.11 to 3.13: one failed assert held the Windows
  3.11 leg for 35 minutes).
* ``EveryReadLater`` (``tests/clocks.py``): the stand-in clock of the tests that need a
  deadline spent, an idle sweep or a timestamp that moved, whatever the platform tick
  (Windows before Python 3.13 reads one value for 15.6 ms).
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest as guards
from tests.conftest import EveryReadLater, cpu_timer, posix_only

_REPO = Path(__file__).resolve().parents[2]


class TestCompactParamId:
    def test_a_short_value_keeps_the_default_id(self) -> None:
        assert guards.compact_param_id("abc", "x") is None
        assert guards.compact_param_id(b"abc", "x") is None
        assert (
            guards.compact_param_id("a" * guards.MAX_PARAM_VALUE_ID_CHARS, "x") is None
        )

    @pytest.mark.parametrize("value", [None, 5, 1.5, ["x" * 5_000], {"a": "x" * 5_000}])
    def test_only_text_and_bytes_are_compacted(self, value: object) -> None:
        assert guards.compact_param_id(value, "x") is None

    def test_a_long_value_gets_a_short_stable_id(self) -> None:
        value = "x" * (guards.MAX_PARAM_VALUE_ID_CHARS + 1)

        first = guards.compact_param_id(value, "bad")
        again = guards.compact_param_id(value, "bad")

        assert first == again
        assert first is not None and first.startswith("bad-str1001-")
        assert len(first) < 40

    def test_two_values_of_one_length_differ_by_the_digest(self) -> None:
        a = guards.compact_param_id("a" * 5_000, "v")
        b = guards.compact_param_id("b" * 5_000, "v")

        assert a != b

    def test_bytes_and_text_of_the_same_content_are_told_apart(self) -> None:
        assert guards.compact_param_id(b"[" * 5_000, "v") != guards.compact_param_id(
            "[" * 5_000, "v"
        )
        assert guards.compact_param_id(b"[" * 5_000, "v").startswith("v-bytes5000-")

    def test_a_lone_surrogate_does_not_raise(self) -> None:
        assert guards.compact_param_id("\ud800" * 5_000, "v") is not None

    def test_the_hook_is_the_helper(self) -> None:
        assert guards.pytest_make_parametrize_id(None, "x" * 5_000, "bad") == (
            guards.compact_param_id("x" * 5_000, "bad")
        )
        assert guards.pytest_make_parametrize_id(None, "x", "bad") is None

    @pytest.mark.parametrize("huge", ["x" * 100_000, b"[" * 100_000, '{"a":[' * 20_000])
    def test_the_real_node_id_of_a_huge_parameter_is_short(
        self, huge: object, request: pytest.FixtureRequest
    ) -> None:
        # the id of this very test: on the parent it carried the whole value
        assert len(request.node.nodeid) < 300, request.node.nodeid[:200]


class TestNodeIdLimit:
    @staticmethod
    def _items(*lengths: int) -> list[SimpleNamespace]:
        return [SimpleNamespace(nodeid="t.py::t[" + "x" * n + "]") for n in lengths]

    def test_ordinary_ids_pass(self) -> None:
        guards.pytest_collection_modifyitems(None, self._items(10, 1_000, 3_900))

    def test_an_id_over_the_limit_fails_collection_with_the_remedy(self) -> None:
        items = self._items(10, guards.MAX_NODEID_CHARS)

        with pytest.raises(pytest.UsageError, match="PYTEST_CURRENT_TEST") as caught:
            guards.pytest_collection_modifyitems(None, items)

        assert "1 test id(s)" in str(caught.value)
        assert "ids=" in str(caught.value)

    def test_the_limit_is_far_below_the_windows_variable_cap(self) -> None:
        assert guards.MAX_NODEID_CHARS < 32_767 // 4


def _lines(count: int, offset: int, tail: str) -> str:
    return "".join(f"line {i + offset:06d}{tail}\n" for i in range(count))


class TestAssertReprGuard:
    def test_two_huge_multiline_strings_are_explained_fast(self) -> None:
        left = _lines(5_000, 0, "")
        right = _lines(7_000, 2_000, " ")

        with cpu_timer() as timer:
            explanation = guards.huge_operand_explanation("==", left, right)

        assert explanation is not None
        assert timer.elapsed < 2
        text = "\n".join(explanation)
        assert "str of length" in text and "full diff skipped" in text
        assert "first difference at index 7" in text  # 'line 000000' vs 'line 002000'

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("a" * 10_001, "a" * 10_001 + "b"),  # characters
            ("l\n" * 400, "l\n" * 300),  # line pairs (401 x 301 over 100,000)
            (b"x" * 10_001, b"y"),  # bytes
            (list(range(1_001)), list(range(1_000))),
            (tuple(range(1_001)), (1,)),
            ({i: i for i in range(1_001)}, {}),
            (set(range(1_001)), {1}),
        ],
    )
    def test_each_kind_of_huge_operand_is_caught(
        self, left: object, right: object
    ) -> None:
        assert guards.huge_operand_explanation("==", left, right)

    @pytest.mark.parametrize(
        ("op", "left", "right"),
        [
            ("==", "small", "smaller"),
            ("==", "a\n" * 200, "b\n" * 200),  # 40,401 line pairs: pytest's own diff
            ("==", "a\n" * 300, "b\n" * 300),  # 90,601 line pairs, still under
            ("==", _lines(201, 0, ""), _lines(201, 1, "")),  # 201 lines, 895 chars
            ("==", "a\n" * 5_000, "b"),  # one-sided: ndiff has no pairs to compare
            ("==", [1] * 1_000, [2] * 1_000),
            ("==", 1, 2),
            ("!=", "x" * 20_000, "x" * 20_000),
            ("in", "x", "y" * 20_000),
        ],
    )
    def test_small_operands_and_other_operators_keep_pytest_default(
        self, op: str, left: object, right: object
    ) -> None:
        assert guards.huge_operand_explanation(op, left, right) is None

    def test_the_dict_and_set_reports_count_one_sided_keys(self) -> None:
        text = "\n".join(
            guards.huge_operand_explanation("==", {i: i for i in range(1_001)}, {0: 0})
        )

        assert "1000 keys or members on one side only" in text

    def test_equal_prefix_reports_the_length_of_the_shorter_side(self) -> None:
        text = "\n".join(
            guards.huge_operand_explanation("==", "a" * 10_001, "a" * 10_001 + "b")
        )

        assert "first difference at index 10001" in text

    def test_the_live_pytest_uses_the_guard(self) -> None:
        from _pytest.assertion import util

        assert hasattr(util.assertrepr_compare, "__wrapped__")

    def test_the_wrapper_answers_huge_keyword_calls_and_delegates_the_rest(
        self,
    ) -> None:
        seen: list[dict] = []

        def original(*args, **kwargs):
            seen.append(kwargs)
            return iter(["original"])

        guarded = guards.guard_assertrepr_compare(original)
        big = "a" * 20_000

        huge = list(guarded(op="==", left=big, right="b", verbose=1))
        small = list(guarded(op="==", left="a", right="b", verbose=1))
        other_op = list(guarded(op="!=", left=big, right=big, verbose=1))
        positional = list(guarded("==", big, "b"))  # another pytest: fail open

        assert "full diff skipped" in "\n".join(huge)
        assert small == other_op == positional == ["original"]
        assert len(seen) == 3 and guarded.__wrapped__ is original

    def test_the_wrapper_answers_a_huge_operand_with_a_list(self) -> None:
        """pytest 9.0.x calls ``len()`` on the result (``truncate_if_required``) and 9.1.x
        wraps it in ``list()``: a list serves both, an iterator breaks 9.0.x (review of
        step 11.1: ``TypeError: object of type 'list_iterator' has no len()``)."""
        guarded = guards.guard_assertrepr_compare(lambda *a, **k: iter(["original"]))
        big = "a" * 20_000

        by_90_keywords = guarded(config=object(), op="==", left=big, right="b")
        by_91_keywords = guarded(
            op="==", left=big, right="b", verbose=1, highlighter=None
        )

        for result in (by_90_keywords, by_91_keywords):
            assert isinstance(result, list)
            assert len(result) >= 2  # what 9.0.x does first
            assert list(result) == result  # what 9.1.x does
            assert "full diff skipped" in "\n".join(result)

    def test_a_call_with_the_operands_by_position_reaches_the_original(self) -> None:
        calls: list[tuple] = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return ["original"]

        guarded = guards.guard_assertrepr_compare(original)

        assert guarded("==", "a" * 20_000, "b") == ["original"]
        assert calls == [(("==", "a" * 20_000, "b"), {})]

    def test_a_failing_assert_of_huge_texts_finishes_in_the_real_pytest(
        self, tmp_path: Path
    ) -> None:
        """End to end: a scratch project whose conftest imports the hook. Without it the
        3.11 to 3.13 ``difflib`` never returns (the 90 s timeout is the hang guard)."""
        (tmp_path / "conftest.py").write_text(
            "from tests.conftest import pytest_configure  # noqa: F401\n",
            encoding="utf-8",
        )
        (tmp_path / "test_big.py").write_text(
            textwrap.dedent(
                """
                def _lines(count, offset, tail):
                    return "".join(f"line {i + offset:06d}{tail}\\n" for i in range(count))


                def test_big():
                    assert _lines(5000, 0, "") == _lines(7000, 2000, " ")
                """
            ),
            encoding="utf-8",
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_REPO), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )

        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "-v",
                "--tb=short",
                str(tmp_path / "test_big.py"),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
            check=False,
        )

        assert done.returncode == 1, done.stdout[-1500:] + done.stderr[-500:]
        assert "full diff skipped" in done.stdout
        assert "1 failed" in done.stdout


class TestHostileNameHelpers:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (OSError(errno.EINVAL, "Invalid argument"), True),
            (OSError(errno.ENOENT, "missing parent"), False),
            (PermissionError(errno.EACCES, "denied"), False),
            (OSError(errno.ENOSPC, "full"), False),
        ],
    )
    def test_only_a_refused_name_is_a_name_error(
        self, error: OSError, expected: bool
    ) -> None:
        assert guards.name_refused(error) is expected

    @pytest.mark.parametrize("winerror", [123, 161, 206, 267])
    def test_the_windows_name_errors_count(self, winerror: int) -> None:
        error = OSError(errno.ENOENT, "WinError")  # WinError 123 maps to ENOENT
        error.winerror = winerror  # type: ignore[attr-defined]

        assert guards.name_refused(error) is True

    def test_another_windows_error_does_not(self) -> None:
        error = OSError(errno.EACCES, "WinError 5")
        error.winerror = 5  # type: ignore[attr-defined]

        assert guards.name_refused(error) is False

    @posix_only("an ESC or a newline in a file name needs a POSIX file system")
    def test_a_posix_file_system_takes_the_hostile_names(self, tmp_path: Path) -> None:
        """Non-vacuity: where the names exist the helper writes them, so the skill and
        OKF tests exercise the real control characters on Linux."""
        names = ["a\x1b[31mb.txt", "e\nf", "c\u202ed.txt", "sub/x\x1b.txt"]

        assert [guards.write_if_creatable(tmp_path / n) for n in names] == [True] * 4
        assert (tmp_path / "e\nf").read_bytes() == b"x"
        assert (tmp_path / "sub" / "x\x1b.txt").exists()

    def test_a_refused_name_is_false_and_nothing_else_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(self: Path, data: bytes) -> int:
            raise OSError(errno.EINVAL, "Invalid argument")

        monkeypatch.setattr(Path, "write_bytes", refuse)
        assert guards.write_if_creatable(tmp_path / "bad") is False

        def deny(self: Path, data: bytes) -> int:
            raise PermissionError(errno.EACCES, "denied")

        monkeypatch.setattr(Path, "write_bytes", deny)
        with pytest.raises(PermissionError):
            guards.write_if_creatable(tmp_path / "other")

    @posix_only("the backslash probe is about a file system that treats it as a name")
    def test_the_backslash_probe_does_not_skip_where_a_backslash_is_a_name(
        self, tmp_path: Path
    ) -> None:
        guards.skip_if_backslash_separates(tmp_path)

        assert list(tmp_path.iterdir()) == []  # the probe folder is removed

    @staticmethod
    def _write_as_windows_would(path: Path, content: bytes = b"x") -> bool:
        """``write_if_creatable`` on a file system where a backslash separates.

        Splits the NAME, not the Path: on Windows pathlib has already done that split
        (``path.parent`` is the folder ``a`` there and ``a\\b.txt`` is already
        ``a/b.txt``), on Linux the backslash is still inside the final component.
        """
        target = Path(str(path).replace("\\", "/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return True

    @pytest.mark.parametrize("shape", ["backslash-in-name", "split-by-pathlib"])
    def test_the_windows_stand_in_takes_a_path_in_either_shape(
        self, tmp_path: Path, shape: str
    ) -> None:
        """CI run 44028115098: the first stand-in built `<probe>/a/b.txt` from
        `path.parent`, which on Windows is already `<probe>/a`, and failed with
        WinError 3. Both shapes of the path must end in one folder `a` with one file."""
        probe = tmp_path / "probe"
        probe.mkdir()
        path = (
            probe / "a\\b.txt"
            if shape == "backslash-in-name"
            else probe / "a" / "b.txt"
        )

        assert self._write_as_windows_would(path) is True

        assert sorted(p.relative_to(probe).as_posix() for p in probe.rglob("*")) == [
            "a",
            "a/b.txt",
        ]

    def test_the_backslash_probe_skips_where_it_separates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(guards, "write_if_creatable", self._write_as_windows_would)

        with pytest.raises(pytest.skip.Exception, match="path separator"):
            guards.skip_if_backslash_separates(tmp_path)

        assert list(tmp_path.iterdir()) == []


class TestEveryReadLater:
    """The shared stand-in clock: no two reads are ever equal (CI runs 44028115098 and
    44038473796: a real clock read twice in one 15.6 ms tick gave the same value)."""

    def test_every_read_is_later_than_the_one_before(self) -> None:
        clock = EveryReadLater()

        reads = [clock.monotonic(), clock.time(), clock.monotonic(), clock.time()]

        assert reads == sorted(set(reads))  # strictly increasing: no repeats
        assert reads == [1001.0, 1002.0, 1003.0, 1004.0]  # one second per read

    def test_start_and_step_are_the_arguments(self) -> None:
        clock = EveryReadLater(start=5.0, step=0.25)

        assert (clock.monotonic(), clock.time()) == (5.25, 5.5)

    def test_the_two_instances_do_not_share_a_counter(self) -> None:
        first, second = EveryReadLater(), EveryReadLater()
        first.monotonic()

        assert second.monotonic() == 1001.0

    def test_any_other_attribute_is_the_real_time_modules(self) -> None:
        clock = EveryReadLater()

        assert clock.sleep is time.sleep
        assert clock.perf_counter is time.perf_counter
        with pytest.raises(AttributeError):
            clock.no_such_clock  # noqa: B018  (the attribute read is the test)
