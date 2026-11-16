# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities to nikolasmarkou@gmail.com.

## Supply Chain Security

- `pyproject.toml` bounds every dependency it declares (the core dependencies, every extra and the `docs` group) on both sides: a floor and an upper bound, normally the next major or `<1` for 0.x packages (a few ranges span more than one major, for example `tenacity>=8.0,<10` and `pytest>=7.4.0,<10`). `tests/test_packaging.py` fails when such a range has no upper bound or does not contain its exact pin in `constraints.txt`. The one exception is the build backend `setuptools>=77`, which has a floor only; its exact version is pinned in `constraints.txt` and the release build runs under those pins.
- The compromised litellm releases 1.82.7 and 1.82.8 are excluded by the `litellm>=1.109.0` floor. No `!=` specifiers are used.
- CI runs `.pth` file auditing on every build (`scripts/audit_pth.py`, in the `lint` job and on every leg of the test matrix)
- CI audits the installed environment, transitive dependencies included, with `pip-audit --skip-editable` in the job `supply-chain` on every push and pull request, and a known vulnerability in any installed package fails the build. A canary step in the same job runs `pip-audit` over `tests/fixtures/canary-vulnerable-pins.txt` (pins of old releases with published advisories, never installed) and passes only when the audit exits 1 with a non-empty report, so a broken audit, a changed exit code or an unreachable advisory source turns the build red instead of passing silently. `pip-audit` is pinned in `constraints.txt` and listed in the `dev` extra.
- Every GitHub Action in `.github/workflows` is pinned to a full commit SHA with its release tag in a trailing comment, and a checkout keeps no credentials except in the two jobs that push (the docs `deploy` and the pin refresh `propose`). `tests/test_fsm_llm_regression/test_supply_chain.py` reads every workflow file and fails on a floating tag, on `pull_request_target`, on a workflow with no read-only top-level token and on a write scope outside its allowlist. The allowlist today is six grants on five jobs: `contents: write` on the docs deploy job (it pushes the site on a version tag), `pull-requests: write` on the dependency-review job (its failure comment), `contents: write` plus `pull-requests: write` on the `propose` job of `refresh-pins.yml` (it pushes a branch and opens a pull request), `id-token: write` on the `publish` job of `release.yml` (the OIDC token that PyPI trusted publishing exchanges for a short-lived upload token) and `contents: write` on its `github-release` job (it creates the GitHub release of the pushed tag). A second rule keeps third-party code away from those tokens: no job that holds a write scope installs a package or runs project code (an installer, an interpreter, a test runner, a project script or a `setup-*` action in such a job fails the same test). There is one named waiver, the docs `deploy` job: it installs the docs tools to run `mike`, and they are pinned exactly in `constraints.txt`, are never moved by the weekly pin refresh (which changes `litellm` and `mcp` only) and change only in a reviewed commit, so no new third-party release reaches that job unreviewed; the waiver and its reason are pinned in the test.
- A scheduled workflow, `.github/workflows/refresh-pins.yml` (weekly and on demand), proposes newer litellm and mcp pins through `scripts/refresh_pins.py`. A release is eligible only after a 7-day cooldown (the litellm 1.82.7 and 1.82.8 releases were malware for hours, and a bad release is usually yanked within days), and only when it is stable, not yanked, admits every Python of the test matrix and stays in the major of the current pin; the script never downgrades and never crosses a major. When a pin moves, the `test` job installs it and runs the whole non-slow suite, mypy, the `.pth` audit and `pip-audit`, and only after it passed does the `propose` job push a branch and open a pull request (a pull request opened with the default token does not trigger the CI workflow, so the gate runs in the workflow itself). The split is the point: the `test` job executes a brand-new third-party release, so it has the read-only token and no credentials, and it uploads the three files of the proposal (`constraints.txt`, `pyproject.toml` and the script's report) as an artifact before it installs anything; the `propose` job holds the write token, installs nothing and runs no project code, and a guard step refuses anything but those three regular files and any change beyond the litellm and mcp pin and range lines. The pull request is reviewed and merged by the owner; nothing is merged automatically.
- Releases are made by one workflow, `.github/workflows/release.yml`, which starts on a pushed tag `v*` and on nothing else. The package is built once (sdist and wheel, with the build backend under the pins of `constraints.txt`), checked with `twine check --strict`, the tag against `src/fsm_llm/__version__.py` (final releases only) and the release notes against `CHANGELOG.md`, and the same files are then uploaded to PyPI by trusted publishing: the `publish` job holds the only `id-token: write`, runs in the GitHub environment `pypi` (a required reviewer must approve the deployment), is a plain hosted job with no container and no reusable workflow, and publishes PEP 740 attestations with the files. There is no long-lived PyPI credential anywhere (no API token, no password, no secret), and `twine upload` is never used: the workflow tests fail on either. The workflow also writes an SBOM (CycloneDX 1.6 JSON, `cyclonedx-py environment` over a fresh environment with the wheel installed with every extra; `cyclonedx-bom` is pinned in `constraints.txt`) and attaches it, with the distributions and the notes, to the GitHub release; the SBOM is not attested and is not on PyPI.
- Dependency review (`actions/dependency-review-action`) runs on pull requests and fails a pull request that adds a dependency with a high or critical advisory.
- The workflows are linted with `actionlint` and `zizmor` in the `lint` job; both are pinned in `constraints.txt`, and `zizmor` runs offline (its online audits need a token that the `lint` job does not name).
- `constraints.txt` pins every direct dependency and the build backend to one exact version for dev/CI reproducibility (for example `litellm==1.109.1`)

### Checking Your Environment

```bash
# Audit for malicious .pth files
make audit

# Verify you have this project: the distribution is llm-fsm (fsm-llm on PyPI is another project)
pip show llm-fsm | grep Version

# Verify installed litellm version
pip show litellm | grep Version
```

### litellm Incident (March 2026)

litellm versions 1.82.7 and 1.82.8 contained credential-stealing malware injected via
`.pth` file. See CHANGELOG.md for details and remediation steps.
