def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    # your /health returns plain text "ok" in your setup
    assert r.text.strip().lower() in {"ok", "healthy"}