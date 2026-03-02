def test_queue_user_now_rejects_boolean_string(client, admin_token):
    r = client.post(
        "/admin/queue-user-now",
        headers={"x-admin-token": admin_token},
        data={
            "email": "test1@whispr.dev",
            "regen": "true",
            "note": "test",
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body

def test_queue_user_now_accepts_int(client, admin_token):
    r = client.post(
        "/admin/queue-user-now",
        headers={"x-admin-token": admin_token},
        data={
            "email": "test1@whispr.dev",
            "regen": "1",
            "note": "test",
        },
    )
    # This may be 200 even if it can't find the user/schedule depending on your implementation.
    # If your endpoint requires DB, we can mark it integration later.
    assert r.status_code in (200, 202, 404)