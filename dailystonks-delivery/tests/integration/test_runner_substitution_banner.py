import pytest


@pytest.mark.integration
def test_runner_persists_and_includes_substitution_banner_when_card_fails(integration_client, monkeypatch):
    client = integration_client
    admin = {"x-admin-token": "test-admin-token"}
    email = "sub_banner_user@whispr.dev"

    # Keep the engine deterministic + offline (no yfinance).
    monkeypatch.setenv("REPORT_OFFLINE_SYNTH", "1")

    # Ensure engine adapter cache reflects env change.
    import app.engine_adapter as eng
    if hasattr(eng, "_market_and_sp500"):
        eng._market_and_sp500.cache_clear()

    # Create user
    r = client.post(
        "/admin/create-user",
        headers=admin,
        data={"email": email, "tier": "pro"},
    )
    assert r.status_code in (200, 201), r.text

    # Fetch schedule_id
    import app.db as dbmod
    import app.models as models

    with dbmod.SessionLocal() as s:
        user = s.query(models.User).filter(models.User.email == email.lower()).first()
        assert user is not None
        sched = (
            s.query(models.DeliverySchedule)
            .filter(models.DeliverySchedule.user_id == user.id, models.DeliverySchedule.enabled == True)  # noqa: E712
            .first()
        )
        assert sched is not None
        schedule_id = sched.id

    # Queue run with forced include of a key that has a fallback.
    key = "anomaly.sigma_intraday_alerts"
    r = client.post(
        "/admin/queue-user-now",
        headers=admin,
        data={
            "email": email,
            "regen": "1",
            "force_include": key,
            "note": "Integration substitution test",
        },
    )
    assert r.status_code == 200, r.text

    # Patch the card to raise so engine_adapter triggers the fallback + substitution banner.
    import dailystonks.core.registry as reg
    from dailystonks.core.models import CardSpec

    orig = reg.CARD_REGISTRY[key]

    def _boom(_ctx):
        raise RuntimeError("boom")

    reg.CARD_REGISTRY[key] = CardSpec(
        key=orig.key,
        title=orig.title,
        category=orig.category,
        min_tier=orig.min_tier,
        cost=orig.cost,
        heavy=orig.heavy,
        fn=_boom,
        slots=orig.slots,
        tags=orig.tags,
    )

    try:
        import app.delivery.runner as runner

        runner.run_due_deliveries(limit=50)
    finally:
        reg.CARD_REGISTRY[key] = orig

    # Assert run exists and HTML includes substitution banner.
    with dbmod.SessionLocal() as s:
        run = (
            s.query(models.DeliveryRun)
            .filter(models.DeliveryRun.schedule_id == schedule_id)
            .order_by(models.DeliveryRun.run_at.desc())
            .first()
        )
        assert run is not None, "Expected a DeliveryRun row to be persisted"
        assert run.report_html, "Expected report_html to be persisted"

        html = run.report_html
        assert "temporarily unavailable" in html.lower()
        assert "substitut" in html.lower()
        assert "anomaly.sigma_intraday_alerts" in html
        assert "price.candles_basic" in html
