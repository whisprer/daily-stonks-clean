from __future__ import annotations

import re
import html as _html
import logging
import os
import random
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

from sqlalchemy.exc import SQLAlchemyError

from ..config import get_settings
from ..db import SessionLocal
from ..models import User, DeliverySchedule, DeliveryTarget, CardSelection, DeliveryRun
from ..engine_adapter import (
    choose_keys_for_user,
    run_html,
    normalize_tier,
    can_customize,
)
from .mailer import send_email

log = logging.getLogger("dailystonks.runner")


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def _parse_list_csv(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def compute_next_run(timezone: str, send_time_hhmm: str, now_utc: datetime | None = None) -> datetime:
    """
    Compute next scheduled run time (UTC naive) for a local HH:MM in a timezone.

    Notes:
    - DB stores timestamps without timezone; we treat them as UTC naive consistently.
    - Daylight Savings is handled by ZoneInfo conversions.
    """
    now_utc = now_utc or datetime.utcnow()

    try:
        tz = ZoneInfo((timezone or "").strip() or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")

    # Parse HH:MM
    try:
        hh_s, mm_s = (send_time_hhmm or "").split(":")
        hh_i = int(hh_s)
        mm_i = int(mm_s)
        hh_i = max(0, min(23, hh_i))
        mm_i = max(0, min(59, mm_i))
    except Exception:
        hh_i, mm_i = 6, 30  # safe default

    now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    candidate_local = datetime.combine(now_local.date(), time(hh_i, mm_i), tzinfo=tz)

    if candidate_local <= now_local:
        candidate_local = candidate_local + timedelta(days=1)

    return candidate_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def _as_of_date_for_schedule(now_utc: datetime, timezone: str) -> tuple[datetime, datetime.date]:
    """Return (now_local, as_of_date) for a schedule timezone."""
    try:
        tz = ZoneInfo((timezone or "").strip() or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    return now_local, now_local.date()


_SUPPORT_BANNER_MARKER = 'data-ds-support-banner="1"'
_SUPPORT_BANNER_RE = re.compile(
    r'<div[^>]*data-ds-support-banner="1"[^>]*>.*?</div>',
    re.DOTALL | re.IGNORECASE,
)


def inject_support_banner(html: str, note: str, support_ref: str) -> str:
    """
    Insert (or replace) a support banner into an HTML report.

    - Pure function: no DB, no globals, no side effects.
    - Idempotent: if banner already exists, replace it (keeps exactly one).
    - Safe: escapes note/ref to avoid accidental HTML injection.
    """
    html = html or ""
    note_esc = _html.escape((note or "").strip())
    ref_esc = _html.escape((support_ref or "").strip())

    if not note_esc:
        note_esc = "support rerun"

    banner = (
        f'<div {_SUPPORT_BANNER_MARKER} '
        'style="padding:10px 12px;border:1px solid #335;background:#0f1720;'
        'border-radius:12px;margin:12px 0;color:#cfe3ff;font-size:13px;">'
        f"<b>Support note:</b> {note_esc} "
        f'<span style="opacity:.8">(ref: <code>{ref_esc}</code>)</span>'
        "</div>"
    )

    # Replace existing banner if present (ensures exactly one banner)
    if _SUPPORT_BANNER_RE.search(html):
        return _SUPPORT_BANNER_RE.sub(banner, html, count=1)

    # Otherwise insert after </header> if present; else prepend.
    if "</header>" in html:
        return html.replace("</header>", "</header>" + banner, 1)

    return banner + html


def run_due_deliveries(limit: int = 50) -> int:
    """
    Process up to `limit` schedules due for delivery.

    Guarantees:
    - Never crashes the whole runner for one bad schedule.
    - Writes a DeliveryRun row for each attempt.
    - Stores report_html/chosen_keys/subject/to_email for support resend.
    - Applies one-time overrides (regen/exclude/include/note) and clears them after use.
    """
    settings = get_settings()
    now_utc = datetime.utcnow()
    processed = 0

    s = SessionLocal()
    try:
        q = (
            s.query(DeliverySchedule)
            .filter(DeliverySchedule.enabled == True)  # noqa: E712
            .filter(DeliverySchedule.next_run_at <= now_utc)
            .order_by(DeliverySchedule.next_run_at.asc())
        )

        # Postgres runner safety: avoid double-sending across multiple runners
        try:
            q = q.with_for_update(skip_locked=True)
        except Exception:
            pass

        due = q.limit(int(limit)).all()

        for sched in due:
            try:
                user = s.query(User).filter(User.id == sched.user_id).first()
                target = s.query(DeliveryTarget).filter(DeliveryTarget.id == sched.target_id).first()
                if not user or not target:
                    sched.enabled = False
                    s.add(sched)
                    s.commit()
                    continue

                # Skip unsubscribed targets
                if getattr(target, "unsubscribed_at", None) is not None:
                    sched.enabled = False
                    s.add(sched)
                    s.commit()
                    continue

                tier = normalize_tier(getattr(user, "tier", "free"))
                now_local, as_of = _as_of_date_for_schedule(now_utc, getattr(sched, "timezone", "UTC"))

                selections = (
                    s.query(CardSelection)
                    .filter(CardSelection.user_id == user.id, CardSelection.enabled == True)  # noqa: E712
                    .order_by(CardSelection.position.asc(), CardSelection.created_at.asc())
                    .all()
                )
                stored_keys = [sel.card_id for sel in selections]

                # One-shot overrides (support features)
                regen = bool(getattr(sched, "override_regen", False))
                exclude_raw = (getattr(sched, "override_exclude", None) or "").strip()
                include_raw = (getattr(sched, "override_include", None) or "").strip()
                note = (getattr(sched, "override_note", None) or "").strip()

                exclude_keys = set(_parse_list_csv(exclude_raw))
                include_keys_in_order = _parse_list_csv(include_raw)

                # Seed: keep stable if configured; regen forces a fresh selection
                seed_base = int((_env("REPORT_SEED", "0") or "0").strip() or "0")
                seed = seed_base if not regen else random.randint(0, 2**31 - 1)

                base_selected = [] if regen else stored_keys
                chosen = choose_keys_for_user(as_of=as_of, tier=tier, selected_keys=base_selected, seed=seed)

                # Apply exclude/include (preserve order)
                if exclude_keys:
                    chosen = [k for k in chosen if k not in exclude_keys]

                if include_keys_in_order:
                    for k in include_keys_in_order:
                        if k and (k not in exclude_keys) and (k not in chosen):
                            chosen.append(k)

                # Re-enforce budgets for customizable tiers while prioritizing forced includes.
                # (For non-custom tiers, choose_keys_for_user ignores selected_keys, so don't call it.)
                if can_customize(tier) and chosen:
                    ordered: list[str] = []
                    for k in include_keys_in_order:
                        if k and (k not in exclude_keys) and (k not in ordered):
                            ordered.append(k)
                    for k in chosen:
                        if k and (k not in exclude_keys) and (k not in ordered):
                            ordered.append(k)
                    chosen = choose_keys_for_user(as_of=as_of, tier=tier, selected_keys=ordered, seed=seed_base)

                # Report inputs
                tickers = _parse_list_csv(_env("DEFAULT_TICKERS", "SPY"))
                start = _env("REPORT_START", "2020-01-01").strip() or "2020-01-01"
                end = _env("REPORT_END", "").strip() or None
                interval = _env("REPORT_INTERVAL", "1d").strip() or "1d"
                universe = _env("REPORT_UNIVERSE", "sp500").strip() or "sp500"
                max_universe = int(_env("REPORT_MAX_UNIVERSE", "50") or "50")

                support_ref = f"sch={str(sched.id)[:8]} run={now_utc.isoformat()}Z"

                html = run_html(
                    as_of=as_of,
                    tier=tier,
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    universe=universe,
                    max_universe=max_universe,
                    chosen_keys=chosen,
                    support_ref=support_ref,
                )

                # Support banner injection for overrides (regen/exclude/include/note)
                if regen or exclude_keys or include_keys_in_order or note:
                    parts: list[str] = []
                    if note:
                        parts.append(note)
                    if regen:
                        parts.append("regenerated safe set")
                    if exclude_keys:
                        parts.append("excluded: " + ", ".join(sorted(exclude_keys)))
                    if include_keys_in_order:
                        parts.append("force include: " + ", ".join(include_keys_in_order))
                    msg = "; ".join([p for p in parts if p]).strip() or "support rerun"

                # Support banner injection for overrides (regen/exclude/include/note)
                if regen or exclude_keys or include_keys_in_order or note:
                    parts: list[str] = []
                    if note:
                        parts.append(note)
                    if regen:
                        parts.append("regenerated safe set")
                    if exclude_keys:
                        parts.append("excluded: " + ", ".join(sorted(exclude_keys)))
                    if include_keys_in_order:
                        parts.append("force include: " + ", ".join(include_keys_in_order))
                    msg = "; ".join([p for p in parts if p]).strip() or "support rerun"
                
                    html = inject_support_banner(html, msg, support_ref)

                subject = _env("REPORT_SUBJECT", "").strip() or f"DailyStonks {tier.upper()} — {as_of.isoformat()}"

                run = DeliveryRun(
                    schedule_id=sched.id,
                    run_at=now_utc,
                    status="PENDING",
                )
                run.report_html = html
                run.chosen_keys = ",".join(chosen)
                run.subject = subject
                run.to_email = target.email

                s.add(run)
                s.flush()

                try:
                    unsub_url = _env("UNSUB_BASE_URL", "").strip()
                    msgid = send_email(settings, target.email, subject, html, unsub_url)
                    run.status = "SENT"
                    run.message_id = msgid
                    run.error = None
                except Exception as e:
                    run.status = "FAILED"
                    run.error = repr(e)

                # schedule next run + clear overrides (one-shot)
                sched.next_run_at = compute_next_run(sched.timezone, sched.send_time_hhmm, now_utc=now_utc)

                sched.override_regen = False
                sched.override_exclude = None
                sched.override_include = None
                sched.override_note = None

                s.add(sched)
                s.add(run)
                s.commit()
                processed += 1

            except SQLAlchemyError:
                s.rollback()
                log.exception("DB error while processing schedule id=%s", getattr(sched, "id", None))
            except Exception:
                s.rollback()
                log.exception("Schedule processing failed id=%s", getattr(sched, "id", None))

        return processed

    finally:
        s.close()
