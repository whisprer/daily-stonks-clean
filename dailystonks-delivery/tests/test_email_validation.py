import uuid


class _CaptureQuery:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None

    def all(self):
        return []


class _CaptureSession:
    """Capture objects added by endpoints, without a real DB."""

    def __init__(self):
        self.added = []

    def query(self, *args, **kwargs):
        return _CaptureQuery()

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        # Ensure ORM-like IDs exist so endpoints that stringify IDs behave.
        for obj in self.added:
            if hasattr(obj, "id") and getattr(obj, "id") is None:
                try:
                    setattr(obj, "id", uuid.uuid4())
                except Exception:
                    pass

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


def test_admin_create_user_rejects_bad_email(client, admin_token):
    r = client.post(
        "/admin/create-user",
        headers={"x-admin-token": admin_token},
        data={"email": "not-an-email", "tier": "FREE"},
    )
    assert r.status_code == 422


def test_admin_create_user_normalizes_email_lowercase(client, admin_token, monkeypatch):
    # Patch SessionLocal so we can inspect what the endpoint tried to persist.
    import app.db as dbmod

    cap = _CaptureSession()
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: cap, raising=True)

    r = client.post(
        "/admin/create-user",
        headers={"x-admin-token": admin_token},
        data={"email": "TeSt@Example.COM", "tier": "pro"},
    )
    assert r.status_code in (200, 201), r.text

    # Find the created User object in captured session.added
    created_users = [
        o
        for o in cap.added
        if hasattr(o, "tier") and hasattr(o, "preference_token") and hasattr(o, "email")
    ]
    assert created_users, f"Expected a User-like object to be added, got: {[type(o).__name__ for o in cap.added]}"

    u = created_users[0]
    assert u.email == "test@example.com"
