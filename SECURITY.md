# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities to nikolasmarkou@gmail.com.

## Supply Chain Security

- `pyproject.toml` sets a lower bound on every dependency. Only `litellm` also has an upper bound (`<2.0`); the others have none.
- The compromised litellm releases 1.82.7 and 1.82.8 are excluded by the `litellm>=1.83.0` floor. No `!=` specifiers are used.
- CI runs `.pth` file auditing on every build
- `constraints.txt` pins exact dependency versions for dev/CI reproducibility (for example `litellm==1.102.1`)

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
