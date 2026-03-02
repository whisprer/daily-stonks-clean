import os
import sys
import time
import socket
import subprocess
import importlib
from pathlib import Path
from typing import Iterator, Tuple

import pytest
from fastapi.testclient import TestClient

import psycopg2


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True).strip()


def _have_docker() -> bool:
    try:
        _docker("version")
        _docker("ps")
        return True
    except Exception:
        return False


def _wait_host_connect(port: int, timeout_s: int = 60) -> None:
    dsn = f"host=127.0.0.1 port={port} dbname=dailystonks_test user=postgres password=postgres"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(dsn)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
                return
            finally:
                conn.close()
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(
        f"Postgres not reachable from host on 127.0.0.1:{port} within {timeout_s}s. Last error: {last_err}"
    )


def _start_postgres_container() -> Tuple[str, int]:
    port = _find_free_port()
    name = f"ds-test-pg-{port}"

    cid = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        "POSTGRES_USER=postgres",
        "-e",
        "POSTGRES_PASSWORD=postgres",
        "-e",
        "POSTGRES_DB=dailystonks_test",
        "-p",
        f"{port}:5432",
        "postgres:16-alpine",
    )

    deadline = time.time() + 90
    while time.time() < deadline:
        # If it died, fail loud with logs
        status = "unknown"
        try:
            status = _docker("inspect", "-f", "{{.State.Status}}", cid)
        except Exception:
            pass
        if status != "running":
            logs = "(no logs)"
            try:
                logs = _docker("logs", "--tail=200", cid)
            except Exception:
                pass
            raise RuntimeError(f"Postgres container exited early (status={status}). Logs:\n{logs}")

        try:
            out = _docker("exec", cid, "pg_isready", "-U", "postgres", "-d", "dailystonks_test")
            if "accepting connections" in out:
                _wait_host_connect(port, timeout_s=60)
                return cid, port
        except Exception:
            pass

        time.sleep(0.5)

    logs = "(no logs)"
    try:
        logs = _docker("logs", "--tail=200", cid)
    except Exception:
        pass
    raise RuntimeError(f"Postgres container did not become ready in time. Logs:\n{logs}")


def _stop_container(cid: str) -> None:
    try:
        _docker("rm", "-f", cid)
    except Exception:
        pass


def _ensure_engine_on_syspath() -> None:
    env_root = os.environ.get("DAILYSTONKS_ENGINE")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    # Auto-detect from repo layout:
    # ...\dailystonks-delivery\tests\integration\conftest.py -> parents[2] = ...\dailystonks-delivery
    here = Path(__file__).resolve()
    delivery_root = here.parents[2]
    candidates.append(delivery_root.parent / "dailystonks" / "engine")

    for root in candidates:
        root = root.resolve()
        if (root / "dailystonks").is_dir():
            sys.path.insert(0, str(root))
            return

    raise RuntimeError(
        "Could not locate engine package. Set DAILYSTONKS_ENGINE to the folder containing `dailystonks/`.\n"
        r'Example: $env:DAILYSTONKS_ENGINE="D:\code\daily-stonks\dailystonks\engine"'
    )


def _purge_app_modules() -> None:
    """
    Critical: prevents stale DB engines/sessions created with old DATABASE_URL.
    """
    purge_prefixes = (
        "app.config",
        "app.db",
        "app.models",
        "app.delivery.runner",
        "app.main",
        "scripts.init_db",
        "scripts.run_due_deliveries",
    )
    for k in list(sys.modules.keys()):
        if k in purge_prefixes:
            sys.modules.pop(k, None)


def _import_app_fresh():
    """
    Import app after env vars are set and modules are purged.
    """
    import app.config
    import app.db
    import app.models
    import app.delivery.runner
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.db)
    importlib.reload(app.models)
    importlib.reload(app.delivery.runner)
    importlib.reload(app.main)

    return app.main


@pytest.fixture(scope="session")
def admin_token() -> str:
    return "test-admin-token"


@pytest.fixture(scope="session")
def postgres_db_url() -> Iterator[str]:
    if not _have_docker():
        pytest.skip("Docker not available (docker CLI not working).")

    cid, port = _start_postgres_container()
    try:
        yield f"postgresql+psycopg2://postgres:postgres@127.0.0.1:{port}/dailystonks_test"
    finally:
        _stop_container(cid)


@pytest.fixture()
def integration_client(monkeypatch, postgres_db_url, admin_token):
    _ensure_engine_on_syspath()

    # Set env BEFORE importing app modules
    monkeypatch.setenv("DS_INTEGRATION_DB", "1")
    monkeypatch.setenv("ADMIN_TOKEN", admin_token)
    monkeypatch.setenv("DATABASE_URL", postgres_db_url)

    # Helpful debug (leave it in until stable)
    print("INTEGRATION DATABASE_URL =", os.environ.get("DATABASE_URL"))


    from pathlib import Path
    
    # ... inside integration_client(monkeypatch, postgres_db_url, admin_token):
    
    engine_root = os.environ.get("DAILYSTONKS_ENGINE")
    if not engine_root:
        # fallback if you didn't set env: use the same path you used in _ensure_engine_on_syspath()
        here = Path(__file__).resolve()
        delivery_root = here.parents[2]
        engine_root = str((delivery_root.parent / "dailystonks" / "engine").resolve())
    
    cfg_dir = Path(engine_root).resolve() / "config"
    tiers_yaml = cfg_dir / "tiers.yaml"
    if not tiers_yaml.is_file():
        raise RuntimeError(f"tiers.yaml not found where expected: {tiers_yaml}")
    
    monkeypatch.chdir(str(cfg_dir))
    print("INTEGRATION CWD =", os.getcwd())
    
    from pathlib import Path

    # Ensure data files exist for engine cards (minimal fixture data)
    engine_root = Path(engine_root).resolve()  # this is your DAILYSTONKS_ENGINE folder (...\dailystonks\engine)
    mono_dailystonks_root = engine_root.parent  # ...\dailystonks
    data_dir = mono_dailystonks_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sp500_csv = data_dir / "sp500_constituents.csv"
    if not sp500_csv.is_file():
        sp500_csv.write_text("Symbol\nAAPL\nMSFT\nSPY\n", encoding="utf-8")

    print("INTEGRATION sp500_constituents.csv =", str(sp500_csv))
    print("INTEGRATION tiers.yaml =", str(tiers_yaml))
    
    
    # Prevent real SMTP sending
    import smtplib


    class _DummySMTP:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, *a, **kw):
            return None

        def login(self, *a, **kw):
            return None

        def sendmail(self, *a, **kw):
            return {}

        def send_message(self, *a, **kw):
            return {}

        def quit(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(smtplib, "SMTP", _DummySMTP, raising=True)
    if hasattr(smtplib, "SMTP_SSL"):
        monkeypatch.setattr(smtplib, "SMTP_SSL", _DummySMTP, raising=True)

    # Purge stale imports, then import app fresh
    _purge_app_modules()
    m = _import_app_fresh()

    # Create schema
    try:
        import scripts.init_db as init_db
        if hasattr(init_db, "main"):
            init_db.main()
        elif hasattr(init_db, "init_db"):
            init_db.init_db()
    except Exception:
        import app.db as dbmod
        import app.models as models
        base = getattr(models, "Base", None) or getattr(dbmod, "Base", None)
        engine = getattr(dbmod, "engine", None)
        if base is None or engine is None:
            raise
        base.metadata.create_all(bind=engine)

    return TestClient(m.app)