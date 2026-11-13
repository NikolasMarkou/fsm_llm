"""The public API snapshot `docs/api.txt` equals what the code declares now.

`scripts/api_snapshot.py` prints every public name (the static `__all__` of
the eight packages and four core modules, plus `schema_registry`, `codes` and
`flow_analysis`) with its signature. A removed, renamed or retyped public
parameter changes that output, so this test fails until the snapshot is
regenerated (`make api-snapshot`) and reviewed in the same commit.

The generator runs in a child process: a test session may hold monkeypatched
modules, and the snapshot must be the same for any caller.
"""

import difflib
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")  # fsm_llm.monitor needs it, like test_packaging

_REPO_ROOT = pathlib.Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "api_snapshot.py"
_SNAPSHOT = _REPO_ROOT / "docs" / "api.txt"
_HINT = "run `make api-snapshot` and review `git diff docs/api.txt`"


def _generate() -> str:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"api_snapshot.py exited {proc.returncode}: " + (proc.stderr[-2000:])
    )
    return proc.stdout


class TestApiSnapshot:
    def test_snapshot_file_exists(self):
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"

    def test_snapshot_matches_the_code(self):
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"
        expected = _SNAPSHOT.read_text(encoding="utf-8")
        actual = _generate()
        if actual == expected:
            return
        diff = list(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                "docs/api.txt",
                "generated",
                lineterm="",
                n=0,
            )
        )
        shown = "\n".join(line[:300] for line in diff[:40])
        pytest.fail(
            f"the public API differs from docs/api.txt ({len(diff)} diff lines, "
            f"first 40):\n{shown}\n{_HINT}"
        )

    def test_snapshot_shape(self):
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"
        text = _SNAPSHOT.read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n")
        assert "__version__" not in text
        for header in (
            "# fsm_llm",
            "# fsm_llm.schema_registry",
            "# fsm_llm.codes",
            "# fsm_llm.flow_analysis",
            "# fsm_llm.synthesis",
        ):
            assert header in text.splitlines(), header

    def test_snapshot_names_no_private_base(self):
        """A mixin is a part of its class: its name is never printed as a base."""
        assert _SNAPSHOT.is_file(), f"{_SNAPSHOT} is missing: {_HINT}"
        offenders = [
            line
            for line in _SNAPSHOT.read_text(encoding="utf-8").splitlines()
            if " class(" in line
            and any(
                base.strip().startswith("_")
                for base in line.split(" class(", 1)[1].rstrip(")").split(",")
            )
        ]
        assert offenders == []


def _load_script():
    """Load the script as a private module object: nothing enters `sys.modules`."""
    spec = importlib.util.spec_from_file_location("_api_snapshot_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make(name, bases=(), module="fakepkg_a.sub.api", **members):
    """A class that claims to live in *module*; no module object is created."""
    return type(name, tuple(bases), {"__module__": module, **members})


def _def_names(lines):
    """The member names printed under a class line, in order."""
    found = []
    for line in lines[1:]:
        match = re.match(r"  (?:static |classmethod )?def (\w+)|  property (\w+)", line)
        if match:
            found.append(match.group(1) or match.group(2))
    return found


class TestMroAwareClassLines:
    """`cls_lines` lists what a class inherits from its private or Mixin bases.

    Classes are built with `type()` and a claimed `__module__`; nothing is
    imported or registered, so nothing can leak between tests.
    """

    snap = _load_script()

    def _lines(self, cls):
        return self.snap.cls_lines("fakepkg_a.sub", cls.__name__, cls)

    def test_a_private_mixin_contributes_its_methods_and_loses_its_name(self):
        mixin = _make(
            "_PrivateMixin",
            helper=lambda self, n: n,
            prop=property(lambda self: 1),
        )
        mixin.helper.__annotations__ = {"n": int, "return": int}
        cls = _make("Public", [mixin])
        lines = self._lines(cls)
        assert lines[0] == "fakepkg_a.sub.Public class"
        assert "  def helper(self, n: int) -> int" in lines
        assert "  property prop" in lines

    def test_only_public_bases_are_named(self):
        mixin = _make("_PrivateMixin", helper=lambda self: None)
        base = _make("Base", module="fakepkg_b.other")
        cls = _make("Public", [mixin, base], module="fakepkg_a.sub.api")
        assert self._lines(cls)[0] == "fakepkg_a.sub.Public class(Base)"

    def test_a_private_base_of_a_private_base_is_looked_through(self):
        root = _make("Root", module="fakepkg_b.other")
        inner = _make("_Inner", [root], inner_method=lambda self: None)
        outer = _make("_Outer", [inner], outer_method=lambda self: None)
        cls = _make("Public", [outer])
        lines = self._lines(cls)
        assert lines[0] == "fakepkg_a.sub.Public class(Root)"
        assert _def_names(lines) == ["inner_method", "outer_method"]

    def test_an_override_wins_and_the_member_prints_once(self):
        mixin = _make("_PrivateMixin", run=lambda self, a: a, other=lambda self: 0)
        cls = _make("Public", [mixin], run=lambda self, a, b: (a, b))
        lines = self._lines(cls)
        assert [x for x in lines if x.startswith("  def run")] == [
            "  def run(self, a, b)"
        ]
        assert _def_names(lines) == ["other", "run"]

    def test_two_private_mixins_both_contribute_in_sorted_order(self):
        first = _make("_First", zeta=lambda self: 0, shared=lambda self: "first")
        second = _make("_Second", alpha=lambda self: 0, shared=lambda self, x: x)
        cls = _make("Public", [first, second])
        lines = self._lines(cls)
        assert lines[0] == "fakepkg_a.sub.Public class"
        # MRO order: `_First` precedes `_Second`, so its `shared` is the real one.
        assert _def_names(lines) == ["alpha", "shared", "zeta"]
        assert "  def shared(self)" in lines

    def test_a_public_base_that_shadows_a_mixin_member_keeps_it_unlisted(self):
        mixin = _make("_PrivateMixin", clash=lambda self: 0, kept=lambda self: 0)
        base = _make("Base", module="fakepkg_b.other", clash=lambda self, x: x)
        cls = _make("Public", [base, mixin])
        # `Base.clash` is what the instance really gets, and `Base` has its own
        # snapshot entry: the class must not print the mixin's `clash`.
        assert _def_names(self._lines(cls)) == ["kept"]

    def test_the_output_does_not_depend_on_the_order_members_were_defined(self):
        one = _make("_M", b=lambda self: 0, a=lambda self: 0)
        two = _make("_M", a=lambda self: 0, b=lambda self: 0)
        assert self._lines(_make("P", [one])) == self._lines(_make("P", [two]))

    def test_inherited_members_of_a_classmethod_and_a_staticmethod_keep_their_kind(
        self,
    ):
        mixin = _make(
            "_M",
            make=classmethod(lambda cls: cls),
            util=staticmethod(lambda x: x),
        )
        lines = self._lines(_make("P", [mixin]))
        assert "  classmethod def make(cls)" in lines
        assert "  static def util(x)" in lines


class TestSamePackageMixinRule:
    """A `<Group>Mixin` of the same package is a part of the class, not a base.

    A later layout split cuts an oversized class into mixins that live next to
    it (`fsm_llm.engine.api.<group>`); the snapshot must not change when it
    does. "Same package" means the same first two dotted parts of
    `__module__`. A public base of the same package that is not a Mixin
    (`TimestampedModel`, `BasePromptConfig`) stays a named base, as today.
    """

    snap = _load_script()

    def _lines(self, cls):
        return self.snap.cls_lines("pkg", cls.__name__, cls)

    def test_the_package_is_the_first_two_dotted_parts(self):
        package = self.snap.package_of
        assert package("fakepkg_a.sub.api.conversation") == "fakepkg_a.sub"
        assert package("fakepkg_a.sub") == "fakepkg_a.sub"
        assert package("fakepkg_a") == "fakepkg_a"

    def test_a_mixin_in_a_submodule_of_the_same_package_is_part_of_the_class(self):
        mixin = _make(
            "ConversationMixin",
            module="fakepkg_a.sub.api.conversation",
            converse=lambda self, message: message,
        )
        cls = _make("API", [mixin], module="fakepkg_a.sub.api")
        lines = self._lines(cls)
        assert lines[0] == "pkg.API class"
        assert "  def converse(self, message)" in lines

    def test_a_mixin_of_another_package_is_a_declared_base_and_stays_named(self):
        mixin = _make("ForeignMixin", module="fakepkg_b.other", extra=lambda self: None)
        cls = _make("API", [mixin], module="fakepkg_a.sub.api")
        lines = self._lines(cls)
        assert lines[0] == "pkg.API class(ForeignMixin)"
        assert _def_names(lines) == []

    def test_a_public_same_package_base_that_is_not_a_mixin_stays_named(self):
        base = _make("TimestampedModel", module="fakepkg_a.sub.definitions")
        base.age = property(lambda self: 0)
        cls = _make("Step", [base], module="fakepkg_a.sub.definitions")
        lines = self._lines(cls)
        assert lines[0] == "pkg.Step class(TimestampedModel)"
        assert "  property age" not in lines

    def test_moving_a_mixin_between_modules_of_one_package_changes_nothing(self):
        before = _make("GroupMixin", module="fakepkg_a.sub.api", go=lambda self: 0)
        after = _make("GroupMixin", module="fakepkg_a.sub.api.group", go=lambda self: 0)
        assert self._lines(
            _make("API", [before], module="fakepkg_a.sub.api")
        ) == self._lines(_make("API", [after], module="fakepkg_a.sub.api"))
