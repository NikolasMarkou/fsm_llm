# Security Policy

## Reporting Vulnerabilities

Report security vulnerabilities to nikolasmarkou@gmail.com.

## Supply Chain Security

- All dependencies are constrained in `pyproject.toml` with upper bounds
- Known-compromised versions are explicitly excluded (`!=` specifiers)
- CI runs `.pth` file auditing on every build
- `constraints.txt` pins exact dependency versions for dev/CI reproducibility

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
