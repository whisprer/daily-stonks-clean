[README.md]

<p align="center">
  <a href="https://github.com/whisprer/daily-stonks-clean">
    <img src="https://img.shields.io/github/stars/whisprer/daily-stonks-clean?style=for-the-badge" alt="GitHub stars" />
  </a>
  <a href="https://github.com/whisprer/daily-stonks-clean/issues">
    <img src="https://img.shields.io/github/issues/whisprer/daily-stonks-clean?style=for-the-badge" alt="GitHub issues" />
  </a>
  <a href="https://github.com/whisprer/daily-stonks-clean/fork">
    <img src="https://img.shields.io/github/forks/whisprer/daily-stonks-clean?style=for-the-badge" alt="GitHub forks" />
  </a>
</p>

# daily-stonks-clean

[![tests](https://github.com/whisprer/daily-stonks-clean/actions/workflows/tests.yml/badge.svg)](https://github.com/whisprer/daily-stonks-clean/actions/workflows/tests.yml)

A clean, minimal, CI-friendly repository for **DailyStonks** consisting of:

- `dailystonks/engine/` — report/content generation (“engine”) + configuration
- `dailystonks-delivery/` — delivery service (API + queue + runner) + tests

This repo is intentionally trimmed to the operational minimum so it stays easy to run, test, and automate.

---

## Repository layout

```text
daily-stonks-clean/
  dailystonks/
    engine/
    data/
  dailystonks-delivery/
    app/
    tests/
  .github/workflows/
Development setup
Prereqs

Python 3.10+ recommended

Docker Desktop (only needed for integration tests)

Install (recommended)
cd dailystonks-delivery

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
pip install -e ../dailystonks/engine
Running tests (unit + integration)

This repo uses pytest with a clear split:

Unit tests: fast, run by default

Integration tests: slower, require Docker (Postgres), run only when explicitly requested

Quick start (unit tests)

From dailystonks-delivery/:

python -m pytest -q

That runs unit tests only.

How tests are organized

tests/ → unit tests (default)

tests/integration/ → integration tests (opt-in)

pytest.ini excludes integration by default via:

addopts = -q -m "not integration"

marker integration is declared

Running integration tests (Docker Postgres)

Integration tests spin up a temporary postgres:16-alpine container on a random host port, set DATABASE_URL, and then run the app/runner against it.

Run (from dailystonks-delivery/):

python -m pytest -q -m integration tests/integration -rs

Notes:

You must have Docker running

SMTP is blocked/monkeypatched during integration runs (no real emails go out)

The harness sets up minimal fixture data (e.g. tiny sp500_constituents.csv) so the engine can render

Environment variables
DAILYSTONKS_ENGINE (optional override)

The integration harness tries to auto-detect the engine directory from the repo layout.
If you keep dailystonks-delivery + engine in a non-standard layout, set:

PowerShell:

$env:DAILYSTONKS_ENGINE="D:\code\daily-stonks\dailystonks\engine"

bash:

export DAILYSTONKS_ENGINE="/path/to/daily-stonks/dailystonks/engine"

Then rerun integration tests.

DATABASE_URL

Integration tests set this automatically to the temporary Docker Postgres instance.

Useful pytest flags

Show full trace on failures:

python -m pytest -q --maxfail=1 -vv

Run a single test file:

python -m pytest -q tests/test_admin_auth.py

Run tests matching a substring:

python -m pytest -q -k unsubscribe

See skip/xfail reasons / extra summary:

python -m pytest -q -rs
CI (GitHub Actions)

This repo is set up to:

run unit tests on pushes and pull requests

run integration tests on a schedule and/or manually (depending on workflow config)

Check the Actions tab in GitHub for recent runs.

Troubleshooting
"Could not locate engine package" / config not found

Set DAILYSTONKS_ENGINE to the folder that contains the dailystonks/ package (the engine root), e.g.:

$env:DAILYSTONKS_ENGINE="D:\code\daily-stonks\dailystonks\engine"
Docker errors / Postgres container won’t start

Ensure Docker Desktop is running

Ensure you can run:

docker ps

If ports are weird, restart Docker (the harness uses random ports, so collisions are rare)

psycopg2 import/connect issues

Integration tests require psycopg2 (or psycopg2-binary) installed in the venv used for testing.

If unit tests pass (python -m pytest -q) and integration tests pass
(python -m pytest -q -m integration tests/integration -rs), you’re in a “safe to ship” state.