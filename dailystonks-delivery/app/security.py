from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, Dict

from .config import Settings

def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")

def _unb64url(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _sign(settings: Settings, payload_b64: str) -> str:
    mac = hmac.new(settings.secret_key.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(mac)

def make_signed_token(settings: Settings, payload: Dict[str, Any]) -> str:
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url(payload_b)
    sig = _sign(settings, payload_b64)
    return f"{payload_b64}.{sig}"

def verify_signed_token(settings: Settings, token: str) -> Dict[str, Any]:
    try:
        payload_b64, sig = token.split(".", 1)
    except ValueError:
        raise ValueError("bad token")

    expected = _sign(settings, payload_b64)
    if not hmac.compare_digest(expected, sig):
        raise ValueError("bad signature")

    payload = json.loads(_unb64url(payload_b64).decode("utf-8"))
    return payload
