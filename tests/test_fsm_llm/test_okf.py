"""Tests for the Open Knowledge Format (OKF) support.

This file grows with the OKF plan; the first class guards the one
path-confinement helper that the OKF loader and writer share with the harness
workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
