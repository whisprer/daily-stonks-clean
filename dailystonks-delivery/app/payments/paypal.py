from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx

from ..config import Settings

def _env_json(name: str, default: str = "{}") -> Dict[str, Any]:
    raw = os.getenv(name, default)
    try:
        return json.loads(raw)
    except Exception:
        return {}

def extract_customer_email(event: Dict[str, Any]) -> Optional[str]:
    r = event.get("resource") or {}
    # Common locations for emails in PayPal events:
    for path in [
        ("subscriber", "email_address"),
        ("payer", "email_address"),
        ("payer", "payer_info", "email"),
    ]:
        cur: Any = r
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and "@" in cur:
            return cur.strip().lower()
    # Some events may put email at top-level.
    email = event.get("email") or event.get("payer_email")
    if isinstance(email, str) and "@" in email:
        return email.strip().lower()
    return None

def extract_plan_id(event: Dict[str, Any]) -> Optional[str]:
    r = event.get("resource") or {}
    for k in ["plan_id", "billing_plan_id", "planId"]:
        v = r.get(k)
        if isinstance(v, str) and v:
            return v
    return None

def map_tier_from_plan_id(plan_id: Optional[str]) -> Optional[str]:
    if not plan_id:
        return None
    mapping = _env_json("PAYPAL_PLAN_TIER_MAP", "{}")  # {"P-XXXX":"PRO", ...}
    tier = mapping.get(plan_id)
    if isinstance(tier, str) and tier:
        return tier.upper()
    return None

def should_deactivate(event_type: str) -> bool:
    et = (event_type or "").upper()
    return any(x in et for x in ["CANCEL", "SUSPEND", "EXPIRE", "TERMINAT"])

async def get_access_token(settings: Settings) -> str:
    if not settings.paypal_client_id or not settings.paypal_client_secret:
        raise RuntimeError("Missing PAYPAL_CLIENT_ID/SECRET")
    auth = (settings.paypal_client_id, settings.paypal_client_secret)
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{settings.paypal_api_base.rstrip('/')}/v1/oauth2/token",
            data={"grant_type": "client_credentials"},
            auth=auth,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

async def verify_webhook_signature(settings: Settings, headers: Dict[str, str], event: Dict[str, Any]) -> bool:
    # If webhook verification isn't configured, accept (development mode).
    if not settings.paypal_webhook_id or not settings.paypal_client_id or not settings.paypal_client_secret:
        return True

    # PayPal sends a set of headers we forward to verify endpoint.
    # Header names can vary in casing. We normalize to lowercase before calling.
    h = {k.lower(): v for k, v in headers.items()}
    required = ["paypal-transmission-id", "paypal-transmission-time", "paypal-cert-url", "paypal-auth-algo", "paypal-transmission-sig"]
    if any(k not in h for k in required):
        return False

    token = await get_access_token(settings)
    payload = {
        "auth_algo": h["paypal-auth-algo"],
        "cert_url": h["paypal-cert-url"],
        "transmission_id": h["paypal-transmission-id"],
        "transmission_sig": h["paypal-transmission-sig"],
        "transmission_time": h["paypal-transmission-time"],
        "webhook_id": settings.paypal_webhook_id,
        "webhook_event": event,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{settings.paypal_api_base.rstrip('/')}/v1/notifications/verify-webhook-signature",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        status = (data.get("verification_status") or "").upper()
        return status == "SUCCESS"
