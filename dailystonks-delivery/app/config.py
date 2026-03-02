from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default

@dataclass(frozen=True)
class Settings:
    database_url: str
    public_base_url: str
    smtp_host: str
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    smtp_use_tls: bool
    mail_from: str
    secret_key: str
    paypal_webhook_id: str | None
    paypal_client_id: str | None
    paypal_client_secret: str | None
    paypal_api_base: str
    tier_policy_path: Path

def get_settings() -> Settings:
    # Defaults assume docker-compose from this repo.
    return Settings(
        database_url=_env("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/dailystonks") or "",
        public_base_url=_env("PUBLIC_BASE_URL", "http://localhost:8000") or "",
        smtp_host=_env("SMTP_HOST", "localhost") or "",
        smtp_port=int(_env("SMTP_PORT", "25") or "25"),
        smtp_username=_env("SMTP_USERNAME"),
        smtp_password=_env("SMTP_PASSWORD"),
        smtp_use_tls=(_env("SMTP_USE_TLS", "0") == "1"),
        mail_from=_env("MAIL_FROM", "DailyStonks <no-reply@dailystonks.org>") or "",
        secret_key=_env("SECRET_KEY", "dev-secret-change-me") or "",
        paypal_webhook_id=_env("PAYPAL_WEBHOOK_ID"),
        paypal_client_id=_env("PAYPAL_CLIENT_ID"),
        paypal_client_secret=_env("PAYPAL_CLIENT_SECRET"),
        paypal_api_base=_env("PAYPAL_API_BASE", "https://api-m.paypal.com") or "",
        tier_policy_path=Path(_env("TIER_POLICY_PATH", str(Path(__file__).resolve().parent.parent / "config" / "tiers.json")) or ""),
    )
