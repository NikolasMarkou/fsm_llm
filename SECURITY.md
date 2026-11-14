# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities to nikolasmarkou@gmail.com.

## Supply Chain Security

- `pyproject.toml` bounds every declared dependency on both sides (a floor and an upper bound at the next major, or `<1` for 0.x packages), and `tests/test_packaging.py` fails when a range has no upper bound or does not contain its pin.
- The compromised litellm releases 1.82.7 and 1.82.8 are excluded by the `litellm>=1.109.0` floor. No `!=` specifiers are used.
- CI runs `.pth` file auditing on every build (`scripts/audit_pth.py`, in the `lint` job and on every leg of the test matrix)
- CI audits the installed environment, transitive dependencies included, with `pip-audit --skip-editable` in the job `supply-chain` on every push and pull request, and a known vulnerability in any installed package fails the build. A canary step in the same job runs `pip-audit` over `tests/fixtures/canary-vulnerable-pins.txt` (pins of old releases with published advisories, never installed) and passes only when the audit exits 1 with a non-empty report, so a broken audit, a changed exit code or an unreachable advisory source turns the build red instead of passing silently. `pip-audit` is pinned in `constraints.txt` and listed in the `dev` extra.
- Every GitHub Action in `.github/workflows` is pinned to a full commit SHA with its release tag in a trailing comment, and a checkout keeps no credentials except in the two jobs that push (the docs deploy and the pin refresh). `tests/test_fsm_llm_regression/test_supply_chain.py` reads every workflow file and fails on a floating tag, on `pull_request_target`, on a workflow with no read-only top-level token and on a write scope outside its allowlist. The allowlist today is four grants on three jobs: `contents: write` on the docs deploy job (it pushes the site on a version tag), `pull-requests: write` on the dependency-review job (its failure comment), and `contents: write` plus `pull-requests: write` on the `refresh` job of `refresh-pins.yml` (it pushes a branch and opens a pull request).
- A scheduled workflow, `.github/workflows/refresh-pins.yml` (weekly and on demand), proposes newer litellm and mcp pins through `scripts/refresh_pins.py`. A release is eligible only after a 7-day cooldown (the litellm 1.82.7 and 1.82.8 releases were malware for hours, and a bad release is usually yanked within days), and only when it is stable, not yanked, admits every Python of the test matrix and stays in the major of the current pin; the script never downgrades and never crosses a major. When a pin moves, the same job installs it and runs the whole non-slow suite, mypy, the `.pth` audit and `pip-audit` BEFORE it pushes a branch and opens a pull request (a pull request opened with the default token does not trigger the CI workflow, so the gate runs inside the job). The pull request is reviewed and merged by the owner; nothing is merged automatically.
- Dependency review (`actions/dependency-review-action`) runs on pull requests and fails a pull request that adds a dependency with a high or critical advisory.
- The workflows are linted with `actionlint` and `zizmor` in the `lint` job; both are pinned in `constraints.txt`, and `zizmor` runs offline (no token is named in any workflow).
- `constraints.txt` pins every direct dependency and the build backend to one exact version for dev/CI reproducibility (for example `litellm==1.109.1`)

### Checking Your Environment

```bash
# Audit for malicious .pth files
make audit

# Verify installed litellm version
pip show litellm | grep Version
```

### litellm Incident (March 2026)

litellm versions 1.82.7 and 1.82.8 contained credential-stealing malware injected via
`.pth` file. See CHANGELOG.md for details and remediation steps.
