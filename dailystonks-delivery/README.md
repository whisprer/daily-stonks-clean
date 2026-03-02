# DailyStonks Delivery Service (cards → preferences → scheduled email)

This repo is a **minimal, working** delivery layer that:
- exposes a card catalog
- hosts a simple preferences page (checkbox selection + schedule time)
- runs scheduled deliveries via a tiny runner you can call from cron
- accepts PayPal webhooks to upgrade/downgrade tiers (optional signature verification)

## Quick start (local)

```bash
# 1) Start Postgres
docker compose up -d

# 2) Create venv + deps
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 3) Configure env
cp .env.example .env
# edit .env (SECRET_KEY, SMTP, PUBLIC_BASE_URL, etc.)

# 4) Create tables
python scripts/init_db.py

# 5) Run API
uvicorn app.main:app --reload --port 8000
```

### Create a test user

```bash
# Using SECRET_KEY or ADMIN_TOKEN as x-admin-token:
curl -X POST http://localhost:8000/admin/create-user \\
  -H "x-admin-token: <ADMIN_TOKEN_OR_SECRET_KEY>" \\
  -F "email=you@example.com" \\
  -F "tier=PRO"
```

Open the returned `prefs_link`, click **Send test email now** (requires SMTP).

## Scheduled delivery

Set a cron job (every 5 minutes is fine):

```cron
*/5 * * * * cd /path/to/repo && /path/to/repo/.venv/bin/python scripts/run_due_deliveries.py >> logs/delivery.log 2>&1
```

## Plug in your real cards and data

- Replace `app/cards/registry.py` to load **your existing card modules**.
- Replace `app/data/provider.py` so `load_payload(asof_date)` returns the bundle your cards expect.
- Ensure your site hosts charts under `${PUBLIC_BASE_URL}/charts/YYYY-MM-DD/...`.

## PayPal tiers

Set `PAYPAL_PLAN_TIER_MAP` in `.env` to map your PayPal plan IDs to tiers.
If you set `PAYPAL_WEBHOOK_ID`, `PAYPAL_CLIENT_ID`, and `PAYPAL_CLIENT_SECRET`,
the webhook receiver will attempt to verify signatures.

Endpoint: `POST /webhooks/paypal`
