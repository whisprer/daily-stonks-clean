\# Testing (Unit + Integration)



This repo uses \*\*pytest\*\* with a clear split:



\- \*\*Unit tests\*\*: fast, run by default

\- \*\*Integration tests\*\*: slower, require Docker (Postgres), run only when explicitly requested



\## Quick start



Activate your venv, then:



```bash

python -m pytest -q



That runs unit tests only.



How tests are organized



tests/ → unit tests (default)



tests/integration/ → integration tests (opt-in)



pytest.ini excludes integration by default via:



addopts = -q -m "not integration"



marker integration is declared



Running integration tests (Docker Postgres)



Integration tests spin up a temporary postgres:16-alpine container on a random host port, set DATABASE\_URL, and then run the app/runner against it.



Run:



python -m pytest -q -m integration tests/integration -rs



Notes:



You must have Docker running



SMTP is blocked/monkeypatched during integration runs (no real emails go out)



The harness sets up minimal fixture data (e.g. tiny sp500\_constituents.csv) so the engine can render



Environment variables

DAILYSTONKS\_ENGINE (optional override)



The integration harness tries to auto-detect the engine directory from the repo layout.

If you keep delivery + engine in a non-standard layout, set:



PowerShell:



$env:DAILYSTONKS\_ENGINE="D:\\code\\daily-stonks\\dailystonks\\engine"



bash:



export DAILYSTONKS\_ENGINE="/path/to/daily-stonks/dailystonks/engine"



Then rerun integration tests.



DATABASE\_URL



Integration tests set this automatically to the temporary Docker Postgres instance.



Useful pytest flags



Show full trace on failures:



python -m pytest -q --maxfail=1 -vv



Run a single test file:



python -m pytest -q tests/test\_admin\_auth.py



Run tests matching a substring:



python -m pytest -q -k unsubscribe



See skip/xfail reasons / extra summary:



python -m pytest -q -rs

Troubleshooting

"Could not locate engine package" / config not found



Set DAILYSTONKS\_ENGINE to the folder that contains the dailystonks/ package (the engine root), e.g.:



$env:DAILYSTONKS\_ENGINE="D:\\code\\daily-stonks\\dailystonks\\engine"

Docker errors / Postgres container won’t start



Ensure Docker Desktop is running



Ensure you can run:



docker ps



If ports are weird, restart Docker (the harness uses random ports, so collisions are rare)



psycopg2 import/connect issues



Integration tests require psycopg2 (or psycopg2-binary) installed in the venv used for testing.



If unit tests pass (python -m pytest -q) and integration tests pass

(python -m pytest -q -m integration tests/integration -rs), you’re in a “safe to ship” state.





---

