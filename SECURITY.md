\# Security Policy



\## Supported versions

Security fixes are provided for:

\- `main` branch (current)

\- the most recent tagged release (when releases exist)



\## Reporting a vulnerability

Please \*\*do not\*\* open a public GitHub Issue for security reports.



Instead, use one of these:

\- GitHub: \*\*Security → Advisories → Report a vulnerability\*\* (preferred)

\- Email: security@whispr.dev



Include:

\- a clear description of the issue

\- reproduction steps (or PoC) if possible

\- affected component(s): `dailystonks/engine` and/or `dailystonks-delivery`

\- potential impact (data exposure, auth bypass, RCE, etc.)



We aim to respond within \*\*72 hours\*\*.



\## Scope

In-scope examples:

\- authentication/authorization issues (e.g., admin token handling)

\- injection issues (HTML/template injection, header injection, etc.)

\- secrets handling (leaking tokens/keys, unsafe defaults)

\- database vulnerabilities (SQL injection, unsafe migrations, privilege issues)

\- insecure dependency usage in the delivery service and engine



Out-of-scope examples:

\- “trading strategy” correctness (this repo is software, not financial advice)

\- issues requiring compromised local machines or already-stolen credentials

\- social engineering



\## Coordinated disclosure

If you report responsibly, we’ll work with you on a coordinated timeline for disclosure once a fix is available.

