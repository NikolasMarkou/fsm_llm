"""
Security Audit -- ReactAgent with Penetration Testing Tools
============================================================

Demonstrates a ReAct agent conducting a comprehensive security
audit across network scanning, vulnerability assessment, log
analysis, patch verification, and report generation.

Run:
    export LLM_MODEL=ollama_chat/qwen3.5:4b
    python examples/agents/security_audit/run.py
"""

import os
from typing import Annotated

from fsm_llm.stdlib.agents import AgentConfig, ReactAgent, ToolRegistry, tool


@tool
def scan_network(
    target: Annotated[str, "Target IP range or hostname to scan"],
    scan_type: Annotated[str, "Scan type: quick, full, stealth, service"],
) -> str:
    """Perform network reconnaissance and service discovery."""
    st = scan_type.lower()

    if "quick" in st or "service" in st:
        return (
            f"NETWORK SCAN RESULTS -- {target}\n"
            "Scan type: Service discovery\n"
            "Hosts discovered: 47 active (of 254 in range)\n"
            "Open ports by frequency:\n"
            "  443/tcp (HTTPS):  42 hosts -- TLS 1.2 (8), TLS 1.3 (34)\n"
            "  80/tcp  (HTTP):   31 hosts -- 12 redirecting to HTTPS, 19 serving HTTP\n"
            "  22/tcp  (SSH):    28 hosts -- OpenSSH 8.9 (15), OpenSSH 9.x (13)\n"
            "  3306/tcp (MySQL): 4 hosts  -- EXPOSED TO NETWORK (CRITICAL)\n"
            "  6379/tcp (Redis): 2 hosts  -- No authentication detected (CRITICAL)\n"
            "  8080/tcp (Alt):   8 hosts  -- Tomcat 9.x (3), Nginx (5)\n"
            "OS fingerprinting: Ubuntu 22.04 (60%), CentOS 8 (25%), Windows Server (15%)\n"
            "WARNING: 19 hosts serving unencrypted HTTP without redirect"
        )
    elif "full" in st:
        return (
            f"FULL PORT SCAN -- {target}\n"
            "Duration: 12 minutes 34 seconds\n"
            "Total ports scanned: 65,535 per host (47 hosts)\n"
            "Additional findings beyond standard ports:\n"
            "  5432/tcp (PostgreSQL): 3 hosts -- v14.2, v15.1, v15.4\n"
            "  9200/tcp (Elasticsearch): 2 hosts -- No auth, cluster exposed\n"
            "  27017/tcp (MongoDB): 1 host -- Auth disabled (CRITICAL)\n"
            "  11211/tcp (Memcached): 3 hosts -- UDP reflection risk\n"
            "  2049/tcp (NFS): 1 host -- Exports visible to subnet\n"
            "Firewall analysis: Ingress filtering present but egress unrestricted\n"
            "DNS zone transfer: Allowed on ns1 (information disclosure)"
        )
    return (
        f"STEALTH SCAN -- {target}\n"
        "SYN scan completed with rate limiting (100 pps)\n"
        "No IDS alerts triggered during scan\n"
        "Key findings: 47 hosts, 156 open services, 6 critical exposures"
    )


@tool
def check_vulnerabilities(
    service: Annotated[str, "Service name and version to check (e.g. 'OpenSSH 8.9')"],
    scope: Annotated[str, "Check scope: cve, config, all"],
) -> str:
    """Check known vulnerabilities (CVEs) and misconfigurations for a service."""
    s = service.lower()

    if "ssh" in s:
        return (
            "VULNERABILITY ASSESSMENT -- OpenSSH\n"
            "CVE-2024-6387 (regreSSHion): CRITICAL (CVSS 8.1)\n"
            "  Affected: OpenSSH 8.5p1 through 9.7p1 on glibc-based Linux\n"
            "  Impact: Unauthenticated remote code execution as root\n"
            "  Status: 15 of 28 SSH hosts potentially vulnerable\n"
            "  Fix: Upgrade to OpenSSH 9.8p1 or set LoginGraceTime to 0\n\n"
            "Configuration issues:\n"
            "  - Password authentication enabled on 8 hosts (should use keys only)\n"
            "  - Root login permitted on 3 hosts\n"
            "  - Weak ciphers (3des-cbc) allowed on 5 hosts"
        )
    elif "mysql" in s or "database" in s:
        return (
            "VULNERABILITY ASSESSMENT -- MySQL\n"
            "CVE-2024-21047: HIGH (CVSS 7.2) -- Privilege escalation via optimizer\n"
            "CVE-2024-21062: MEDIUM (CVSS 4.9) -- Server crash via DML\n\n"
            "Configuration issues:\n"
            "  - 4 MySQL instances exposed to network (should be localhost only)\n"
            "  - Default 'root' account accessible from any host on 2 instances\n"
            "  - SSL/TLS not enforced for client connections\n"
            "  - Binary logging disabled (no point-in-time recovery possible)\n"
            "  - No query audit logging configured"
        )
    elif "redis" in s:
        return (
            "VULNERABILITY ASSESSMENT -- Redis\n"
            "CVE-2024-31228: HIGH (CVSS 7.5) -- DoS via crafted patterns\n"
            "CVE-2024-31449: CRITICAL (CVSS 8.8) -- RCE via Lua scripting\n\n"
            "Configuration issues:\n"
            "  - NO AUTHENTICATION on 2 instances (CRITICAL)\n"
            "  - Protected mode disabled\n"
            "  - Bound to 0.0.0.0 (all interfaces)\n"
            "  - CONFIG command not renamed/disabled\n"
            "  - Persistence (RDB/AOF) not configured -- data loss risk"
        )
    elif "elastic" in s:
        return (
            "VULNERABILITY ASSESSMENT -- Elasticsearch\n"
            "CVE-2024-23450: MEDIUM (CVSS 6.5) -- DoS via malformed request\n\n"
            "Configuration issues:\n"
            "  - X-Pack security not enabled (no authentication)\n"
            "  - Cluster accessible without TLS\n"
            "  - _all indices API exposed (data enumeration)\n"
            "  - Snapshot repository writable (data exfiltration risk)"
        )
    elif "tls" in s or "ssl" in s or "https" in s:
        return (
            "TLS/SSL ASSESSMENT\n"
            "Certificate issues:\n"
            "  - 3 hosts with expired certificates\n"
            "  - 5 hosts using self-signed certificates\n"
            "  - 2 hosts with certificates expiring within 30 days\n"
            "Protocol issues:\n"
            "  - 8 hosts still accepting TLS 1.2 (recommend TLS 1.3 only)\n"
            "  - 0 hosts accepting TLS 1.0/1.1 (good)\n"
            "Cipher suite: All hosts support AEAD ciphers, no weak ciphers detected"
        )
    return f"Vulnerability check for {service}: No critical CVEs found in NVD database."


@tool
def analyze_logs(
    log_source: Annotated[str, "Log source: auth, firewall, application, access"],
    time_range: Annotated[str, "Time range to analyze (e.g. 24h, 7d, 30d)"],
) -> str:
    """Analyze security logs for suspicious activity and anomalies."""
    src = log_source.lower()

    if "auth" in src:
        return (
            "AUTHENTICATION LOG ANALYSIS (last 7 days)\n"
            "Total auth events: 142,856\n"
            "Failed login attempts: 12,847 (8.99%)\n"
            "Brute-force patterns detected:\n"
            "  - 8,200 attempts from 45.155.205.0/24 (known botnet) targeting SSH\n"
            "  - 2,100 attempts against admin panels from Tor exit nodes\n"
            "  - 450 credential stuffing attempts against API endpoints\n"
            "Account lockouts: 23 (12 resolved, 11 pending review)\n"
            "Impossible travel: 3 accounts accessed from 2+ countries within 1 hour\n"
            "Privilege escalation: 2 sudo events from unexpected users (INVESTIGATE)\n"
            "Recommendation: Implement fail2ban, enforce MFA on all admin accounts"
        )
    elif "firewall" in src:
        return (
            "FIREWALL LOG ANALYSIS (last 7 days)\n"
            "Total events: 2.3M\n"
            "Blocked connections: 890,000 (38.7%)\n"
            "Top blocked sources:\n"
            "  1. 45.155.205.0/24: 340K blocks (SSH brute-force)\n"
            "  2. 185.220.101.0/24: 120K blocks (web scanning)\n"
            "  3. 89.248.167.0/24: 85K blocks (port scanning)\n"
            "Egress anomalies:\n"
            "  - 3 hosts with DNS over HTTPS to non-standard resolvers\n"
            "  - 1 host with 4.2GB outbound to unknown IP (DATA EXFILTRATION RISK)\n"
            "  - Unrestricted egress on ports 80/443 -- recommend whitelist approach"
        )
    elif "application" in src or "app" in src:
        return (
            "APPLICATION LOG ANALYSIS (last 7 days)\n"
            "Total events: 567,000\n"
            "Error rate: 2.3% (within threshold)\n"
            "Security-relevant events:\n"
            "  - SQL injection attempts: 145 (all blocked by WAF)\n"
            "  - XSS attempts: 89 (all blocked)\n"
            "  - Path traversal attempts: 34 (2 bypassed WAF -- INVESTIGATE)\n"
            "  - API rate limit violations: 1,200\n"
            "  - JWT token forgery attempts: 12\n"
            "Sensitive data in logs: 3 instances of PII logged in debug mode (GDPR risk)"
        )
    return (
        f"Log analysis for {log_source} ({time_range}): "
        "Standard activity patterns observed, no critical anomalies."
    )


@tool
def verify_patches(
    system: Annotated[str, "System or software to check patch status"],
) -> str:
    """Verify current patch levels and identify missing security updates."""
    s = system.lower()

    if "linux" in s or "ubuntu" in s or "os" in s:
        return (
            "PATCH STATUS -- Linux Servers (47 hosts)\n"
            "Fully patched: 31 hosts (66%)\n"
            "Missing critical patches: 16 hosts (34%)\n"
            "Critical missing patches:\n"
            "  - CVE-2024-6387 (OpenSSH): 15 hosts -- RCE, patch available\n"
            "  - CVE-2024-1086 (nf_tables): 8 hosts -- local privilege escalation\n"
            "  - CVE-2024-0646 (ktls): 4 hosts -- kernel memory corruption\n"
            "Automatic updates: Enabled on 22 hosts, disabled on 25\n"
            "Average patch lag: 18 days (target: <7 days)\n"
            "EOL systems: 3 hosts running Ubuntu 18.04 (unsupported since April 2023)"
        )
    elif "web" in s or "nginx" in s or "apache" in s:
        return (
            "PATCH STATUS -- Web Servers\n"
            "Nginx: 5 hosts, all on 1.24.x (current)\n"
            "Tomcat: 3 hosts, 1 on 9.0.82 (outdated -- update to 9.0.87)\n"
            "Apache: 0 hosts (not in use)\n"
            "WAF rules: Last updated 12 days ago (refresh recommended)"
        )
    return f"Patch status for {system}: Current versions installed, no critical updates pending."


@tool
def generate_report(
    findings: Annotated[str, "Summary of all audit findings"],
    severity: Annotated[
        str, "Overall severity assessment: critical, high, medium, low"
    ],
) -> str:
    """Generate a structured security audit report with remediation priorities."""
    return (
        "SECURITY AUDIT REPORT -- EXECUTIVE SUMMARY\n"
        "=" * 50 + "\n"
        f"Overall Risk Rating: {severity.upper()}\n\n"
        "CRITICAL FINDINGS (Immediate Action Required):\n"
        "  C1. Redis instances with no authentication -- RCE risk\n"
        "  C2. MySQL databases exposed to network -- data breach risk\n"
        "  C3. OpenSSH CVE-2024-6387 on 15 hosts -- remote root compromise\n\n"
        "HIGH FINDINGS (Action Within 72 Hours):\n"
        "  H1. Elasticsearch cluster without security -- data exfiltration\n"
        "  H2. Suspicious 4.2GB egress from production server\n"
        "  H3. 3 EOL Ubuntu 18.04 systems -- no security patches available\n\n"
        "MEDIUM FINDINGS (Action Within 30 Days):\n"
        "  M1. 19 services on unencrypted HTTP\n"
        "  M2. SSH password auth and root login on multiple hosts\n"
        "  M3. WAF path traversal bypass detected (2 instances)\n"
        "  M4. PII in application debug logs\n\n"
        "REMEDIATION COST ESTIMATE: $45,000-65,000\n"
        "TIMELINE: Critical fixes 1 week, full remediation 6 weeks\n"
        "NEXT AUDIT: Recommended in 90 days post-remediation"
    )


def main() -> None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key and "ollama" not in model.lower():
        print("Set OPENAI_API_KEY or use: export LLM_MODEL=ollama_chat/qwen3.5:4b")
        return

    # Register tools
    registry = ToolRegistry()
    registry.register(scan_network._tool_definition)
    registry.register(check_vulnerabilities._tool_definition)
    registry.register(analyze_logs._tool_definition)
    registry.register(verify_patches._tool_definition)
    registry.register(generate_report._tool_definition)

    config = AgentConfig(model=model, max_iterations=18, temperature=0.5)
    agent = ReactAgent(tools=registry, config=config)

    task = (
        "Conduct a comprehensive security audit of our production infrastructure "
        "at 10.0.1.0/24. This is an annual penetration test and compliance review "
        "for our SOC 2 Type II certification renewal, scheduled for Q2 2026. "
        "The scope covers all production systems including web servers, databases, "
        "caching layers, message queues, and supporting infrastructure. Our "
        "environment serves 2.3 million monthly active users and processes "
        "approximately $45M in annual transaction volume through PCI-DSS scoped "
        "payment systems. The previous audit (Q2 2025) identified 4 medium findings "
        "that were reportedly remediated -- verify closure of those items as well.\n\n"
        "PHASE 1 -- RECONNAISSANCE:\n"
        "Perform a service discovery scan of the entire 10.0.1.0/24 subnet. "
        "Identify all active hosts, open ports, and running services. Flag any "
        "unexpected services, shadow IT assets, or unauthorized network listeners. "
        "Pay special attention to database ports (3306, 5432, 6379, 27017, 9200) "
        "that should not be exposed beyond the application tier. Document the "
        "operating system distribution and identify any end-of-life systems that "
        "represent an unpatched attack surface.\n\n"
        "PHASE 2 -- VULNERABILITY ASSESSMENT:\n"
        "For each critical service discovered, check for known CVEs and "
        "misconfigurations. Priority targets: (a) SSH servers -- check for "
        "regreSSHion CVE-2024-6387 allowing unauthenticated RCE as root, "
        "(b) database services (MySQL, Redis, Elasticsearch) -- verify "
        "authentication and network exposure, (c) TLS/SSL configurations "
        "across all HTTPS endpoints.\n\n"
        "PHASE 3 -- LOG ANALYSIS:\n"
        "Analyze authentication logs for the past 7 days looking for brute-force "
        "patterns, credential stuffing campaigns, and impossible-travel anomalies "
        "that could indicate compromised accounts. Review firewall logs for egress "
        "anomalies and potential data exfiltration indicators, particularly any "
        "large outbound transfers to unknown destinations or DNS tunneling patterns. "
        "Check application logs for injection attempts and WAF bypass incidents.\n\n"
        "PHASE 4 -- PATCH COMPLIANCE:\n"
        "Verify patch levels across all Linux servers against the latest security "
        "advisories. Identify any end-of-life operating systems or software that "
        "no longer receives security updates. Assess the average patch lag across "
        "the fleet and compare against our 7-day SLA target for critical patches "
        "and 30-day SLA for non-critical patches. Check web server and application "
        "framework versions for known vulnerabilities.\n\n"
        "PHASE 5 -- REPORTING:\n"
        "Generate a consolidated security audit report with findings categorized "
        "by severity (Critical, High, Medium, Low) following CVSS v3.1 scoring. "
        "Include an executive summary for the CISO briefing, detailed technical "
        "findings for the engineering team, a prioritized remediation roadmap with "
        "cost estimates, and a recommended timeline distinguishing immediate actions "
        "(24-72 hours) from short-term (30 days) and strategic improvements (90 days)."
    )

    print("=" * 60)
    print("Security Audit -- ReactAgent")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tools: {', '.join(registry.tool_names)}")
    print(f"Task: {task[:80]}...")
    print("-" * 60)

    try:
        result = agent.run(task)

        print(f"\nAudit Report:\n{result.answer}")
        print(f"\nSuccess: {result.success}")
        print(f"Iterations: {result.iterations_used}")
        print(f"Tools used: {result.tools_used}")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = {
        "answer_present": result.answer is not None and len(str(result.answer)) > 10,
        "iterations_ok": result.iterations_used >= 1,
        "completed": result.iterations_used < config.max_iterations,
        "tools_called": len(result.tools_used) > 0,
    }
    extracted = 0
    for key, passed in checks.items():
        status = "EXTRACTED" if passed else "MISSING"
        if passed:
            extracted += 1
        print(f"  {key:25s}: {passed!s:40s} [{status}]")
    print(
        f"\nExtraction rate: {extracted}/{len(checks)} ({100 * extracted / len(checks):.0f}%)"
    )


if __name__ == "__main__":
    main()
