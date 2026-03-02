import os
import sys
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _ensure_engine_on_syspath() -> str:
    # Your engine root is the folder that CONTAINS the "dailystonks/" package folder.
    env_root = os.environ.get("DAILYSTONKS_ENGINE")
    if env_root:
        root = Path(env_root).resolve()
        if (root / "dailystonks").is_dir():
            sys.path.insert(0, str(root))
            return str(root)

    # Auto-detect: D:\code\daily-stonks\dailystonks-delivery\tests\conftest.py
    # -> D:\code\daily-stonks\dailystonks\engine
    here = Path(__file__).resolve()
    mono_root = here.parents[2]  # D:\code\daily-stonks
    root = (mono_root / "dailystonks" / "engine").resolve()
    if (root / "dailystonks").is_dir():
        sys.path.insert(0, str(root))
        return str(root)

    raise RuntimeError(
        "Could not locate engine package. Set DAILYSTONKS_ENGINE to the folder that contains `dailystonks/`.\n"
        r'Example: $env:DAILYSTONKS_ENGINE="D:\code\daily-stonks\dailystonks\engine"'
    )


ENGINE_ROOT = _ensure_engine_on_syspath()


class _FakeQuery:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None

    def all(self):
        return []


class _FakeSession:
    """
    Minimal Session replacement so API endpoints can run without touching a real DB.
    """
    def query(self, *args, **kwargs):
        return _FakeQuery()

    def add(self, *args, **kwargs):
        return None

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None

    # In case code uses: `with SessionLocal() as s:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _reload_app():
    """
    Reload modules so env vars and patches apply predictably per test.
    """
    import app.config
    import app.db
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.db)
    importlib.reload(app.main)
    return app.main


@pytest.fixture(scope="session")
def admin_token() -> str:
    return "test-admin-token"


@pytest.fixture()
def client(monkeypatch, admin_token):
    # For unit tests: never touch real infra
    monkeypatch.setenv("ADMIN_TOKEN", admin_token)

    # IMPORTANT: force a harmless URL even if your machine has DATABASE_URL set.
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")

    m = _reload_app()

    # Unit-test mode: patch SessionLocal so endpoints don't hit DB at all.
    # If later you want integration DB tests, set DS_INTEGRATION_DB=1 in env and skip this patch.
    if os.environ.get("DS_INTEGRATION_DB", "").strip() != "1":
        if hasattr(m, "SessionLocal"):
            m.SessionLocal = lambda: _FakeSession()

        # In case main imports SessionLocal from app.db and references it there:
        try:
            import app.db as _db
            _db.SessionLocal = lambda: _FakeSession()
        except Exception:
            pass

    return TestClient(m.app)