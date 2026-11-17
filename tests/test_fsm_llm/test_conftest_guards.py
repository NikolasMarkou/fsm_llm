"""The three guards of ``tests/conftest.py`` that keep a Windows or 3.11 CI leg alive.

* ``compact_param_id`` / ``pytest_make_parametrize_id``: a 100,000 character parameter
  gets a short test id (pytest copies the id into ``PYTEST_CURRENT_TEST`` and Windows
  refuses an environment variable of more than 32,767 characters, so those tests errored
  in setup and never ran).
* ``pytest_collection_modifyitems``: a node id of more than ``MAX_NODEID_CHARS`` fails
  collection, so a new huge parameter is found on Linux.
* ``huge_operand_explanation`` / ``guard_assertrepr_compare`` (installed by
  ``pytest_configure`` over pytest's ``util.assertrepr_compare``; the
  ``pytest_assertrepr_compare`` hook itself cannot stop pytest's own implementation, which
  runs whatever a conftest returns): a failed ``==`` on huge operands is explained without
  ``difflib.ndiff`` (cubic on Python 3.11 to 3.13: one failed assert held the Windows
  3.11 leg for 35 minutes).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest as guards
from tests.conftest import cpu_timer

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
            ("l\n" * 201, "l\n" * 200),  # lines
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
            ("==", "a\n" * 200, "b\n" * 200),  # at the line limit: pytest's own diff
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
