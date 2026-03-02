# Contributing to daily-stonks-clean

Thanks for taking the time to contribute! This repository is intentionally **clean and minimal** so it stays easy to run, test, and automate.

## Repo layout (important)
This repo is a small “monorepo” containing two primary components:

- `dailystonks/engine/`  
  Engine logic + config used to build report content.

- `dailystonks-delivery/`  
  Delivery service (API + queue + runner) that persists delivery runs and sends output.

When changing behavior, consider whether the change belongs in the engine, delivery, or both.

## How to contribute
- Fix a bug
- Improve tests (unit or integration)
- Improve reliability of the runner (idempotence, safe fallbacks)
- Improve docs / developer experience

For any non-trivial change, please open an Issue first so we can align on approach.

## Development setup

### Prereqs
- Python (the repo currently targets Python 3.10+)
- Docker Desktop (only required for integration tests)

### Install (recommended)
From the delivery directory:

```bash
cd dailystonks-delivery
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
pip install -e ../dailystonks/engine

Running tests
Unit tests (fast, default)
cd dailystonks-delivery
python -m pytest -q
Integration tests (Docker Postgres, opt-in)
cd dailystonks-delivery
python -m pytest -q -m integration tests/integration -rs

Notes:
Integration tests spin up a temporary Postgres container and set DATABASE_URL automatically.
SMTP is patched so no real email is sent.
The harness will create minimal fixture data if required.
Engine path override (only if your layout is unusual)

If the integration harness can’t auto-detect the engine, set:

PowerShell:
$env:DAILYSTONKS_ENGINE="D:\code\daily-stonks\dailystonks\engine"

bash:
export DAILYSTONKS_ENGINE="/path/to/daily-stonks/dailystonks/engine"

Pull requests
Use a short, descriptive title.
Include the “why” in the description.
Prefer small PRs that are easy to review.
Add/adjust tests when changing behavior.
If you changed runner/db/config/admin/email flows, run integration tests locally.

Commit style (recommended)
Use clear messages; conventional commits are welcome but not required.

Examples:
fix: handle invalid unsubscribe token safely
test: add integration coverage for runner fallback banner
refactor: simplify engine config discovery

Security issues

Please do not open public issues for security vulnerabilities. See SECURITY.md.