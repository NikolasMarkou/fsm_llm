# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities to nikolasmarkou@gmail.com.

## Supply Chain Security

- `pyproject.toml` bounds every declared dependency on both sides (a floor and an upper bound at the next major, or `<1` for 0.x packages), and `tests/test_packaging.py` fails when a range has no upper bound or does not contain its pin.
- The compromised litellm releases 1.82.7 and 1.82.8 are excluded by the `litellm>=1.109.0` floor. No `!=` specifiers are used.
- CI runs `.pth` file auditing on every build
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
