import pytest


@pytest.mark.integration
def test_unsubscribe_disables_schedule_and_runner_skips(integration_client):
    client = integration_client
    admin = {"x-admin-token": "test-admin-token"}
    email = "unsub_user@whispr.dev"

    # 1) Create user + schedule/target
    r = client.post(
        "/admin/create-user",
        headers=admin,
        data={"email": email, "tier": "pro"},
    )
    assert r.status_code in (200, 201), r.text

    # 2) Fetch target + schedule IDs from DB
    import app.db as dbmod
    import app.models as models

    SessionLocal = dbmod.SessionLocal
    with SessionLocal() as s:
        user = s.query(models.User).filter(models.User.email == email.lower()).first()
        assert user is not None
        target = (
            s.query(models.DeliveryTarget)
            .filter(models.DeliveryTarget.user_id == user.id, models.DeliveryTarget.is_primary == True)  # noqa: E712
            .first()
        )
        assert target is not None
        sched = (
            s.query(models.DeliverySchedule)
            .filter(models.DeliverySchedule.user_id == user.id, models.DeliverySchedule.target_id == target.id)
            .first()
        )
        assert sched is not None
        schedule_id = sched.id
        target_id = target.id

    # 3) Unsubscribe via signed token
    import app.main as mainmod
    from app.security import make_signed_token

    token = make_signed_token(mainmod.settings, {"target_id": str(target_id)})

    r = client.get(f"/unsubscribe/{token}")
    assert r.status_code == 200, r.text
    assert "Unsubscribed" in r.text

    # 4) Queue a run now (admin endpoint doesn't gate unsub, runner does)
    r = client.post(
        "/admin/queue-user-now",
        headers=admin,
        data={"email": email, "regen": "1", "note": "Unsub integration"},
    )
    assert r.status_code == 200, r.text

    # 5) Run runner
    import app.delivery.runner as runner

    runner.run_due_deliveries(limit=50)

    # 6) Assert schedule is disabled and no DeliveryRun exists for this schedule
    with SessionLocal() as s:
        sched2 = s.query(models.DeliverySchedule).filter(models.DeliverySchedule.id == schedule_id).first()
        assert sched2 is not None
        assert sched2.enabled is False

        t2 = s.query(models.DeliveryTarget).filter(models.DeliveryTarget.id == target_id).first()
        assert t2 is not None
        assert t2.unsubscribed_at is not None

        runs = s.query(models.DeliveryRun).filter(models.DeliveryRun.schedule_id == schedule_id).all()
        assert runs == []
