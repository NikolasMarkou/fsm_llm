# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities to nikolasmarkou@gmail.com.

## Supply Chain Security

- `pyproject.toml` bounds every declared dependency on both sides (a floor and an upper bound at the next major, or `<1` for 0.x packages), and `tests/test_packaging.py` fails when a range has no upper bound or does not contain its pin.
- The compromised litellm releases 1.82.7 and 1.82.8 are excluded by the `litellm>=1.109.0` floor. No `!=` specifiers are used.
- CI runs `.pth` file auditing on every build (`scripts/audit_pth.py`, in the `lint` job and on every leg of the test matrix)
- CI audits the installed environment, transitive dependencies included, with `pip-audit --skip-editable` in the job `supply-chain` on every push and pull request, and a known vulnerability in any installed package fails the build. A canary step in the same job runs `pip-audit` over `tests/fixtures/canary-vulnerable-pins.txt` (pins of old releases with published advisories, never installed) and passes only when the audit exits 1 with a non-empty report, so a broken audit, a changed exit code or an unreachable advisory source turns the build red instead of passing silently. `pip-audit` is pinned in `constraints.txt` and listed in the `dev` extra.
- Every GitHub Action in `.github/workflows` is pinned to a full commit SHA with its release tag in a trailing comment, and the checkouts keep no credentials. `tests/test_fsm_llm_regression/test_supply_chain.py` reads every workflow file and fails on a floating tag, on `pull_request_target`, on a workflow with no read-only top-level token and on a write scope outside its allowlist. The allowlist today is two grants: `contents: write` on the docs deploy job (it pushes the site on a version tag) and `pull-requests: write` on the dependency-review job (its failure comment).
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
