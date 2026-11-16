"""The commands the owner runs by hand are written down correctly (final REFLECT, plan d454fd62).

Plan step 121.1 (completion fix of review concerns 1, 8 and 10). `CONTRIBUTING.md` "Releasing" holds
the steps that only the repository owner can take: the GitHub environment `pypi` with its required
reviewer and its tag restriction, the branch protection of `main`, the way to re-run a release.
No workflow ever exercises them, and a slip is silent until the first release (the first draft of
step 4 sent the booleans of `deployment_branch_policy` with `gh api -f`, which makes them the
strings `"true"` and `"false"`, and set no reviewer although `release.yml` relies on one). So the
text is read here as data:

* `owner_command_problems(text)` is a pure function of the CONTRIBUTING text and returns a set of
  problem codes. The real text must return the empty set (`TestTheRealOwnerSteps`); each mutant
  of `MUTANTS` is the real text with one edit and must return its own code
  (`TestMutantsAreKilled`): a check that cannot fail is not a check.
* The JSON body of the environment `PUT` is taken out of the text and parsed: its booleans are real
  booleans, exactly one of the two branch-policy switches is true (the API answers 422 when both
  agree), the reviewer is a `User` with an integer id.
* `no_string_boolean_fields(text)` is the repository-wide rule: no `-f`/`--raw-field` with a value
  `true` or `false` in the owner documents (CONTRIBUTING, SECURITY, README, the docs folder).
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CONTRIBUTING = _ROOT / "CONTRIBUTING.md"

#: A raw (string) field whose value is a JSON literal: `gh api` sends it as the string "true".
_STRING_BOOLEAN = re.compile(
    r"""(?:^|\s)(?:-f|--raw-field)\s+(['"]?)[\w\[\]]+=(?:true|false|null)\1(?=\s|$)""",
    re.MULTILINE,
)
_BODY_LINE = re.compile(r"^\s*body='(?P<body>\{.*\})'\s*$", re.MULTILINE)
_OWNER_ID = re.compile(r"""'"\$owner_id"'""")


def no_string_boolean_fields(text: str) -> list[str]:
    """The `-f key=true|false|null` fields of `text` (each would be a string on the wire)."""
    return [m.group(0).strip() for m in _STRING_BOOLEAN.finditer(text)]


def _releasing(text: str) -> str:
    start = text.index("## Releasing")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def put_body(text: str) -> dict | None:
    """The JSON of the environment `PUT`, with `$owner_id` replaced by 1; None when unreadable."""
    m = _BODY_LINE.search(_releasing(text))
    if m is None:
        return None
    try:
        return json.loads(_OWNER_ID.sub("1", m.group("body")))
    except ValueError:
        return None


def owner_command_problems(text: str) -> set[str]:
    """Problem codes of the owner steps in a CONTRIBUTING text; empty when they are right."""
    rel = _releasing(text)
    problems: set[str] = set()

    if no_string_boolean_fields(rel):
        problems.add("string-boolean-field")

    body = put_body(text)
    if body is None:
        problems.add("put-body-unreadable")
    else:
        policy = body.get("deployment_branch_policy")
        if not isinstance(policy, dict):
            problems.add("policy-missing")
        else:
            flags = (
                policy.get("protected_branches"),
                policy.get("custom_branch_policies"),
            )
            if not all(isinstance(f, bool) for f in flags):
                problems.add("policy-not-boolean")
            elif flags != (False, True):
                problems.add("policy-not-tags")
        reviewers = body.get("reviewers")
        if (
            not isinstance(reviewers, list)
            or len(reviewers) != 1
            or reviewers[0].get("type") != "User"
            or not isinstance(reviewers[0].get("id"), int)
        ):
            problems.add("reviewer-missing")
        if body.get("prevent_self_review") is not False:
            problems.add("self-review-not-allowed")
    if "-X PUT repos/NikolasMarkou/fsm_llm/environments/pypi --input -" not in rel:
        problems.add("put-command")
    if "gh api users/NikolasMarkou --jq .id" not in rel:
        problems.add("owner-id-lookup")

    if not re.search(
        r"-X POST repos/NikolasMarkou/fsm_llm/environments/pypi/deployment-branch-policies"
        r"\s*\\\n\s*-f name='v\*' -f type=tag",
        rel,
    ):
        problems.add("tag-policy")
    if (
        "gh api repos/NikolasMarkou/fsm_llm/environments/pypi "
        "--jq '{protection_rules, deployment_branch_policy}'"
    ) not in rel:
        problems.add("verify-command")
    if not re.search(r"Settings, Environments, `pypi`.*Required reviewers", rel):
        problems.add("web-ui-path")
    if "Prevent self-review" not in rel:
        problems.add("self-review-note")

    if "require a pull request before merging" not in rel:
        problems.add("pull-request-rule")
    if not re.search(r"`guard`.*runs on pull requests only", rel):
        problems.add("guard-reason")

    if (
        '"Re-run failed jobs" only' not in rel
        or '"Re-run all jobs" rebuilds' not in rel
    ):
        problems.add("rerun-guidance")
    if "`mike delete <version> --push`" not in rel or "mike alias --push" not in rel:
        problems.add("docs-deploy-sentence")
    return problems


def _real() -> str:
    return _CONTRIBUTING.read_text(encoding="utf-8")


class TestTheRealOwnerSteps:
    def test_the_real_text_has_no_problem(self):
        assert owner_command_problems(_real()) == set()

    def test_the_put_body_is_what_the_environments_api_accepts(self):
        body = put_body(_real())
        assert body is not None
        assert set(body) == {
            "prevent_self_review",
            "reviewers",
            "deployment_branch_policy",
        }
        assert body["deployment_branch_policy"] == {
            "protected_branches": False,
            "custom_branch_policies": True,
        }
        assert body["reviewers"] == [{"type": "User", "id": 1}]

    def test_the_commands_name_the_repository_and_the_environment(self):
        rel = _releasing(_real())
        assert rel.count("repos/NikolasMarkou/fsm_llm/environments/pypi") >= 4
        assert "pypi" in rel and "v*" in rel

    @pytest.mark.parametrize(
        "path",
        [
            "CONTRIBUTING.md",
            "SECURITY.md",
            "README.md",
            "AGENTS.md",
            "docs/README.md",
        ],
    )
    def test_no_owner_document_sends_a_boolean_as_a_string(self, path):
        text = (_ROOT / path).read_text(encoding="utf-8")
        assert no_string_boolean_fields(text) == []

    def test_no_markdown_under_docs_sends_a_boolean_as_a_string(self):
        found = {
            p.relative_to(_ROOT).as_posix(): no_string_boolean_fields(
                p.read_text(encoding="utf-8")
            )
            for p in sorted((_ROOT / "docs").rglob("*.md"))
        }
        assert {k: v for k, v in found.items() if v} == {}

    def test_the_environment_payload_is_written_in_one_tracked_file(self):
        """The `deployment_branch_policy` payload appears in CONTRIBUTING.md and nowhere else
        among the tracked text files (the plan text lives in the ignored plan folder)."""
        tracked = (
            subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=_ROOT,
                capture_output=True,
                check=True,
            )
            .stdout.decode("utf-8")
            .split("\0")
        )
        hits = []
        for rel in tracked:
            path = _ROOT / rel
            if (
                not rel
                or path.suffix not in {".md", ".yml", ".yaml", ".toml", ".txt", ".py"}
                or rel == "tests/test_fsm_llm_regression/test_owner_commands.py"
                or not path.is_file()
            ):
                continue
            if "deployment_branch_policy" in path.read_text(
                encoding="utf-8", errors="replace"
            ):
                hits.append(rel)
        assert hits == ["CONTRIBUTING.md"]


class TestTheFlagsOfTheScanner:
    @pytest.mark.parametrize(
        "line",
        [
            "gh api -X PUT x -f 'deployment_branch_policy[protected_branches]=false'",
            "gh api -X PUT x -f deployment_branch_policy[custom_branch_policies]=true",
            "gh api x --raw-field enabled=true",
            "gh api x -f key=null",
            'gh api x -f "a[b]=false"',
        ],
    )
    def test_a_string_boolean_field_is_found(self, line):
        assert no_string_boolean_fields(line)

    @pytest.mark.parametrize(
        "line",
        [
            "gh api -X POST x -f name='v*' -f type=tag",
            "gh api -X PUT x -F 'deployment_branch_policy[protected_branches]=false'",
            "gh api x -f note='this is true'",
            "gh api x -f flag=truely",
        ],
    )
    def test_a_correct_field_is_not_found(self, line):
        assert no_string_boolean_fields(line) == []


def _swap(old: str, new: str) -> Callable[[str], str]:
    def edit(text: str) -> str:
        assert text.count(old) >= 1, old
        return text.replace(old, new, 1)

    return edit


MUTANTS: dict[str, tuple[Callable[[str], str], str]] = {
    "boolean-as-raw-field": (
        _swap(
            "-f name='v*' -f type=tag",
            "-f name='v*' -f type=tag -f 'deployment_branch_policy[protected_branches]=false'",
        ),
        "string-boolean-field",
    ),
    "booleans-as-strings-in-json": (
        _swap('"protected_branches":false', '"protected_branches":"false"'),
        "policy-not-boolean",
    ),
    "both-switches-true": (
        _swap('"protected_branches":false', '"protected_branches":true'),
        "policy-not-tags",
    ),
    "policy-dropped": (
        _swap(
            ',"deployment_branch_policy":{"protected_branches":false,'
            '"custom_branch_policies":true}',
            "",
        ),
        "policy-missing",
    ),
    "reviewer-dropped": (
        _swap('"reviewers":[{"type":"User","id":\'"$owner_id"\'}],', ""),
        "reviewer-missing",
    ),
    "reviewer-is-a-team": (_swap('"type":"User"', '"type":"Team"'), "reviewer-missing"),
    "reviewer-id-is-a-string": (
        _swap('"id":\'"$owner_id"\'', '"id":"\'"$owner_id"\'"'),
        "reviewer-missing",
    ),
    "self-review-blocked": (
        _swap('"prevent_self_review":false', '"prevent_self_review":true'),
        "self-review-not-allowed",
    ),
    "put-without-input": (
        _swap('--input - <<< "$body"', '-f body="$body"'),
        "put-command",
    ),
    "owner-id-lookup-dropped": (
        _swap("gh api users/NikolasMarkou --jq .id", "echo 1"),
        "owner-id-lookup",
    ),
    "tag-policy-wrong-type": (_swap("-f type=tag", "-f type=branch"), "tag-policy"),
    "tag-policy-wrong-pattern": (_swap("-f name='v*'", "-f name='main'"), "tag-policy"),
    "verify-command-dropped": (
        _swap("--jq '{protection_rules, deployment_branch_policy}'", ""),
        "verify-command",
    ),
    "web-ui-path-dropped": (
        _swap("Settings, Environments, `pypi`", "Settings"),
        "web-ui-path",
    ),
    "self-review-note-dropped": (
        _swap('"Prevent self-review"', '"the self-review switch"'),
        "self-review-note",
    ),
    "pull-request-rule-dropped": (
        _swap("require a pull request before merging", "require the checks"),
        "pull-request-rule",
    ),
    "guard-reason-dropped": (
        _swap("runs on pull requests only", "runs in CI"),
        "guard-reason",
    ),
    "rerun-all-allowed": (
        _swap('"Re-run failed jobs" only', '"Re-run all jobs"'),
        "rerun-guidance",
    ),
    "docs-deletion-dropped": (
        _swap("`mike delete <version> --push`", "`mike deploy`"),
        "docs-deploy-sentence",
    ),
}


class TestMutantsAreKilled:
    @pytest.mark.parametrize("name", sorted(MUTANTS))
    def test_each_mutant_returns_its_own_code(self, name):
        edit, code = MUTANTS[name]
        assert code in owner_command_problems(edit(_real())), name

    def test_there_is_a_mutant_for_every_problem_code(self):
        wanted = {code for _, code in MUTANTS.values()}
        produced = set()
        # every code the function can return appears among the mutants' expected codes
        source = Path(__file__).read_text(encoding="utf-8")
        for m in re.finditer(r"problems\.add\(\"([a-z-]+)\"\)", source):
            produced.add(m.group(1))
        assert produced - wanted == {"put-body-unreadable"}, produced - wanted

    def test_an_unreadable_body_is_a_problem(self):
        text = _swap("body='{", "body='[")(_real())
        assert "put-body-unreadable" in owner_command_problems(text)
