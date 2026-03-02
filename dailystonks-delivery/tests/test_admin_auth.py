def test_admin_queue_user_now_missing_token_401(client):
    r = client.post(
        "/admin/queue-user-now",
        data={"email": "nope@example.com", "regen": "0"},
    )
    assert r.status_code == 401


def test_admin_queue_user_now_invalid_token_401(client):
    r = client.post(
        "/admin/queue-user-now",
        headers={"x-admin-token": "wrong"},
        data={"email": "nope@example.com", "regen": "0"},
    )
    assert r.status_code == 401


def test_admin_create_user_missing_token_401(client):
    r = client.post(
        "/admin/create-user",
        data={"email": "test@example.com", "tier": "FREE"},
    )
    assert r.status_code == 401


def test_admin_create_user_invalid_token_401(client):
    r = client.post(
        "/admin/create-user",
        headers={"x-admin-token": "wrong"},
        data={"email": "test@example.com", "tier": "FREE"},
    )
    assert r.status_code == 401
