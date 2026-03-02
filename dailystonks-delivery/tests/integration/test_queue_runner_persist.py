import json
from pathlib import Path
import pytest


def _iter_text_fields(obj):
    for k, v in vars(obj).items():
        if k.startswith("_"):
            continue
        if isinstance(v, str) and v.strip():
            yield k, v
        elif isinstance(v, (dict, list)):
            yield k, json.dumps(v, default=str)


def _iter_html_paths(obj):
    for k, v in vars(obj).items():
        if not isinstance(v, str):
            continue
        # rough heuristic: absolute-ish path and html suffix
        if (":\\" in v or v.startswith("/") or v.startswith("\\")) and v.lower().endswith((".html", ".htm")):
            p = Path(v)
            if p.is_file():
                yield k, p


def _find_support_evidence(obj):
    needles = ("Support note:", "Integration test run")
    # 1) search string/json fields
    for k, txt in _iter_text_fields(obj):
        if any(n in txt for n in needles):
            return f"field:{k}", txt

    # 2) search any HTML file paths referenced by the record
    for k, p in _iter_html_paths(obj):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(n in content for n in needles):
            return f"file:{k}", str(p)

    return None, None


@pytest.mark.integration
def test_queue_runner_persists_delivery_run(integration_client):
    client = integration_client
    admin = {"x-admin-token": "test-admin-token"}
    email = "it_user@whispr.dev"

    # 1) Create user
    r = client.post(
        "/admin/create-user",
        headers=admin,
        data={"email": email, "tier": "pro"},
    )
    assert r.status_code in (200, 201), r.text

    # 2) Queue a run (regen=1 so banner should be injected)
    r = client.post(
        "/admin/queue-user-now",
        headers=admin,
        data={"email": email, "regen": "1", "note": "Integration test run"},
    )
    assert r.status_code == 200, r.text

    # 3) Direct runner call
    import app.delivery.runner as runner
    assert hasattr(runner, "run_due_deliveries"), "runner.run_due_deliveries not found"
    runner.run_due_deliveries(limit=50)

    # 4) Assert persisted run exists
    import app.db as dbmod
    import app.models as models

    SessionLocal = getattr(dbmod, "SessionLocal", None)
    assert SessionLocal is not None, "app.db.SessionLocal not found"

    DeliveryRun = getattr(models, "DeliveryRun", None)
    if DeliveryRun is None:
        for _, obj in vars(models).items():
            if isinstance(obj, type) and hasattr(obj, "__tablename__"):
                tn = str(getattr(obj, "__tablename__", "")).lower()
                if "delivery" in tn and "run" in tn:
                    DeliveryRun = obj
                    break
    assert DeliveryRun is not None, "Could not find DeliveryRun model in app.models"

    with SessionLocal() as s:
        runs = s.query(DeliveryRun).all()
        assert runs, "No delivery runs persisted"
        last = runs[-1]

        where, evidence = _find_support_evidence(last)

        if evidence is None:
            # dump available fields to make the next tweak trivial
            fields = [(k, type(v).__name__) for k, v in vars(last).items() if not k.startswith("_")]
            raise AssertionError(
                "DeliveryRun persisted but no Support note / Integration note found in any field or HTML path.\n"
                f"Available fields: {fields}"
            )

        # success
        assert evidence  # non-empty