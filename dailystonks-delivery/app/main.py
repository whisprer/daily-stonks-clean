from __future__ import annotations

import json
import os
import secrets
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Form, Header
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import EmailStr

from .config import get_settings
from .db import session_scope
from .models import (
    User,
    DeliveryTarget,
    CardSelection,
    DeliverySchedule,
    DeliveryRun,
    PaymentEvent,
    Entitlement,
)
from .security import verify_signed_token
from .payments.paypal import (
    extract_customer_email,
    extract_plan_id,
    map_tier_from_plan_id,
    should_deactivate,
    verify_webhook_signature,
)
from .delivery.runner import compute_next_run
from .engine_adapter import (
    list_cards as engine_list_cards,
    choose_keys_for_user,
    normalize_tier,
    can_customize,
    tier_limits,
)

settings = get_settings()
app = FastAPI(title="DailyStonks Delivery Service")


def _admin_token() -> str:
    # Prefer settings attr, but always allow env override
    for attr in ("admin_token", "ADMIN_TOKEN", "adminToken", "admintoken"):
        if hasattr(settings, attr):
            v = getattr(settings, attr)
            if v:
                return str(v).strip()
    return str(os.getenv("ADMIN_TOKEN") or "").strip()


def _require_admin(request: Request) -> None:
    admin = (request.headers.get("x-admin-token") or "").strip()
    expected = _admin_token()
    if not expected or admin != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _get_user_by_pref_token(s, token: str) -> User:
    u = s.query(User).filter(User.preference_token == token, User.is_active == True).first()  # noqa: E712
    if not u:
        raise HTTPException(status_code=404, detail="user not found")
    return u


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


@app.get("/cards")
def list_cards():
    return {"cards": engine_list_cards()}


@app.get("/prefs/{token}", response_class=HTMLResponse)
def prefs_page(token: str):
    """
    Preferences UI.

    FREE/BASIC: fixed card set (engine selection) — checkboxes disabled.
    PRO/BLACK: customizable — checkboxes enabled, selection stored as engine card keys.
    """
    from zoneinfo import ZoneInfo

    cards = engine_list_cards()

    with session_scope() as s:
        user = _get_user_by_pref_token(s, token)

        # Ensure primary target + default schedule exists
        sched = (
            s.query(DeliverySchedule)
            .filter(DeliverySchedule.user_id == user.id)
            .order_by(DeliverySchedule.created_at.asc())
            .first()
        )
        if not sched:
            target = (
                s.query(DeliveryTarget)
                .filter(DeliveryTarget.user_id == user.id, DeliveryTarget.is_primary == True)  # noqa: E712
                .first()
            )
            if not target:
                target = DeliveryTarget(user_id=user.id, email=user.email, label="primary", is_primary=True)
                s.add(target)
                s.flush()

            sched = DeliverySchedule(
                user_id=user.id,
                target_id=target.id,
                timezone="Europe/London",
                send_time_hhmm="06:30",
            )
            sched.next_run_at = compute_next_run(sched.timezone, sched.send_time_hhmm, now_utc=datetime.utcnow())
            s.add(sched)
            s.flush()

        tier = normalize_tier(user.tier)

        # Determine "as_of" date in the user's timezone
        tz = ZoneInfo(sched.timezone or "Europe/London")
        now_local = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
        as_of = now_local.date()

        # Stored selections (engine card keys)
        selections = (
            s.query(CardSelection)
            .filter(CardSelection.user_id == user.id, CardSelection.enabled == True)  # noqa: E712
            .order_by(CardSelection.position.asc(), CardSelection.created_at.asc())
            .all()
        )
        stored_keys = [sel.card_id for sel in selections]

        # Engine-resolved keys for this user/day
        seed = int(os.getenv("REPORT_SEED") or "0")
        resolved_keys = choose_keys_for_user(as_of=as_of, tier=tier, selected_keys=stored_keys, seed=seed)

        # Checkbox state:
        # - If customizable and user has stored selections -> show stored selections checked
        # - If customizable but none stored -> show resolved defaults checked
        # - If fixed tier -> show resolved defaults checked (but disabled)
        show_checked = stored_keys if (can_customize(tier) and stored_keys) else resolved_keys

        max_cards, max_cost, heavy_max = tier_limits(tier)
        is_custom = can_customize(tier)

        # Tier gating helper from engine
        from dailystonks.core.selector import tier_allows as _tier_allows

        rows = []
        for c in cards:
            key = c["key"]
            title = c.get("title") or key
            cat = c.get("category") or "general"
            min_tier = (c.get("min_tier") or "free").strip().lower()
            cost = c.get("cost")
            heavy = c.get("heavy")
            slots = ",".join(c.get("slots") or [])
            tags = ",".join(c.get("tags") or [])

            allowed = _tier_allows(tier, min_tier)
            checked = "checked" if key in show_checked else ""
            disabled = "" if (is_custom and allowed) else "disabled"
            opacity = "1" if allowed else "0.45"

            rows.append(
                f"""
            <label style='display:block;padding:8px 0;opacity:{opacity};'>
              <input type='checkbox' name='card_id' value='{key}' {checked} {disabled}/>
              <b>{title}</b>
              <span style='color:#666;font-size:12px;'>({key})</span><br/>
              <span style='color:#444;font-size:13px;'>
                <b>{cat}</b> · min: <b>{min_tier}</b> · cost: <b>{cost}</b> · heavy: <b>{heavy}</b>
                {(' · slots: <b>'+slots+'</b>') if slots else ''}{(' · tags: <b>'+tags+'</b>') if tags else ''}
              </span>
            </label>
            """
            )

        if not is_custom:
            note = "<p><b>Note:</b> Your tier uses a fixed set of cards. Upgrade to PRO or BLACK to customize.</p>"
        else:
            note = (
                f"<p style='color:#666;font-size:13px;margin:0 0 10px 0;'>"
                f"Budgets: max_cards={max_cards}, max_cost={max_cost}, heavy_max={heavy_max}. "
                f"Your saved list will be auto-trimmed to fit.</p>"
            )

        html = f"""<!doctype html>
<html><head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width,initial-scale=1'/>
  <title>DailyStonks Preferences</title>
</head>
<body style='font-family:Arial,Helvetica,sans-serif;background:#f7f7f7;margin:0;'>
<div style='max-width:920px;margin:0 auto;padding:18px;'>
  <h1 style='margin:0 0 6px 0;'>DailyStonks Preferences</h1>
  <div style='color:#666;margin-bottom:18px;'>
    Account: <b>{user.email}</b> · Tier: <b>{tier}</b> · As-of: <b>{as_of.isoformat()}</b>
  </div>

  {note}

  <form method='post' action='/prefs/{token}/save'
        style='background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:14px;'>
    <h2 style='margin:0 0 10px 0;font-size:18px;'>Cards</h2>
    {''.join(rows)}

    <hr style='border:none;border-top:1px solid #eee;margin:14px 0;'/>

    <h2 style='margin:0 0 10px 0;font-size:18px;'>Schedule</h2>
    <label>Timezone
      <input name='timezone' value='{sched.timezone}' style='width:240px;padding:6px;margin-left:8px;'/>
    </label><br/>
    <label>Send time (HH:MM)
      <input name='send_time_hhmm' value='{sched.send_time_hhmm}' style='width:90px;padding:6px;margin-left:8px;'/>
    </label>

    <div style='margin-top:12px;'>
      <button type='submit' style='padding:10px 14px;border-radius:10px;border:1px solid #111;background:#111;color:#fff;'>
        Save
      </button>
      <a href='/prefs/{token}' style='margin-left:10px;color:#333;'>Refresh</a>
    </div>
  </form>

  <form method='post' action='/prefs/{token}/send-test' style='margin-top:14px;'>
    <button type='submit' style='padding:10px 14px;border-radius:10px;border:1px solid #555;background:#fff;'>
      Send test email now
    </button>
  </form>
</div>
</body></html>"""
        return HTMLResponse(html)


@app.post("/prefs/{token}/save")
def prefs_save(
    token: str,
    timezone: str = Form(...),
    send_time_hhmm: str = Form(...),
    card_id: list[str] = Form(default=[]),
):
    """
    Save schedule always. Save card selection only for PRO/BLACK.
    Stored card IDs are engine card keys.
    """
    from zoneinfo import ZoneInfo
    from dailystonks.core.selector import tier_allows as _tier_allows

    with session_scope() as s:
        user = _get_user_by_pref_token(s, token)
        tier = normalize_tier(user.tier)

        # Ensure schedule exists
        sched = (
            s.query(DeliverySchedule)
            .filter(DeliverySchedule.user_id == user.id)
            .order_by(DeliverySchedule.created_at.asc())
            .first()
        )
        if not sched:
            target = (
                s.query(DeliveryTarget)
                .filter(DeliveryTarget.user_id == user.id, DeliveryTarget.is_primary == True)  # noqa: E712
                .first()
            )
            if not target:
                target = DeliveryTarget(user_id=user.id, email=user.email, label="primary", is_primary=True)
                s.add(target)
                s.flush()
            sched = DeliverySchedule(user_id=user.id, target_id=target.id)
            s.add(sched)
            s.flush()

        sched.timezone = timezone.strip()
        sched.send_time_hhmm = send_time_hhmm.strip()
        sched.next_run_at = compute_next_run(sched.timezone, sched.send_time_hhmm, now_utc=datetime.utcnow())
        s.add(sched)

        # Only customizable tiers persist selections
        if can_customize(tier):
            tz = ZoneInfo(sched.timezone or "Europe/London")
            now_local = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
            as_of = now_local.date()

            cards = engine_list_cards()
            min_map = {c["key"]: (c.get("min_tier") or "free") for c in cards}

            submitted: list[str] = []
            seen: set[str] = set()
            for k in card_id:
                k = (k or "").strip()
                if not k or k in seen:
                    continue
                mt = (min_map.get(k) or "").strip().lower()
                if not mt:
                    continue
                if not _tier_allows(tier, mt):
                    continue
                submitted.append(k)
                seen.add(k)

            # If nothing came through the form (e.g. all checkboxes disabled),
            # do NOT wipe existing selections.
            if not submitted:
                return RedirectResponse(url=f"/prefs/{token}", status_code=303)

            seed = int(os.getenv("REPORT_SEED") or "0")
            chosen = choose_keys_for_user(as_of=as_of, tier=tier, selected_keys=submitted, seed=seed)

            s.query(CardSelection).filter(CardSelection.user_id == user.id).delete()
            for i, k in enumerate(chosen):
                s.add(CardSelection(user_id=user.id, card_id=k, enabled=True, position=i))

        return RedirectResponse(url=f"/prefs/{token}", status_code=303)


@app.post("/prefs/{token}/send-test")
def send_test(token: str):
    """Queue a run now (runner timer will pick it up)."""
    try:
        with session_scope() as s:
            user = _get_user_by_pref_token(s, token)
            sched = s.query(DeliverySchedule).filter(DeliverySchedule.user_id == user.id).first()
            if not sched:
                raise HTTPException(status_code=400, detail="no schedule")
            sched.next_run_at = datetime.utcnow()
            s.add(sched)

        return RedirectResponse(url=f"/prefs/{token}", status_code=303)

    except Exception as e:
        return HTMLResponse(
            f"<h1>Send test failed</h1><pre>{type(e).__name__}: {e}</pre>",
            status_code=500,
        )


@app.get("/unsubscribe/{token}", response_class=HTMLResponse)
def unsubscribe(token: str):
    with session_scope() as s:
        payload = verify_signed_token(settings, token)
        target_id = payload.get("target_id")
        if not target_id:
            raise HTTPException(status_code=400, detail="bad token")
        target = s.query(DeliveryTarget).filter(DeliveryTarget.id == target_id).first()
        if not target:
            raise HTTPException(status_code=404, detail="target not found")
        target.unsubscribed_at = datetime.utcnow()
        s.add(target)

    return HTMLResponse(
        """<!doctype html><html><body style="font-family:Arial,Helvetica,sans-serif;">
<h1>Unsubscribed</h1><p>You will no longer receive DailyStonks emails to this address.</p></body></html>"""
    )


@app.post("/webhooks/paypal")
async def paypal_webhook(request: Request):
    event = await request.json()
    ok = await verify_webhook_signature(settings, dict(request.headers), event)
    if not ok:
        raise HTTPException(status_code=400, detail="invalid signature")

    event_id = str(event.get("id") or "")
    event_type = str(event.get("event_type") or "")
    if not event_id or not event_type:
        raise HTTPException(status_code=400, detail="missing id/event_type")

    raw = json.dumps(event, separators=(",", ":"), ensure_ascii=False)

    with session_scope() as s:
        existing = s.query(PaymentEvent).filter(PaymentEvent.event_id == event_id).first()
        if existing:
            return {"ok": True}

        pe = PaymentEvent(event_id=event_id, event_type=event_type, raw_json=raw)
        s.add(pe)
        s.flush()

        email = extract_customer_email(event)
        plan_id = extract_plan_id(event)
        new_tier = map_tier_from_plan_id(plan_id)

        if email:
            user = s.query(User).filter(User.email == email).first()
            if not user:
                user = User(email=email, tier="FREE", is_active=True, preference_token=secrets.token_urlsafe(24))
                s.add(user)
                s.flush()

                target = DeliveryTarget(user_id=user.id, email=email, label="primary", is_primary=True)
                s.add(target)
                s.flush()

                sched = DeliverySchedule(
                    user_id=user.id,
                    target_id=target.id,
                    timezone="Europe/London",
                    send_time_hhmm="06:30",
                )
                sched.next_run_at = compute_next_run(sched.timezone, sched.send_time_hhmm, now_utc=datetime.utcnow())
                s.add(sched)

            if should_deactivate(event_type):
                user.tier = "FREE"
                s.add(Entitlement(user_id=user.id, tier=user.tier, status="CANCELLED"))
            else:
                if new_tier:
                    user.tier = new_tier
                    s.add(Entitlement(user_id=user.id, tier=new_tier, status="ACTIVE"))

            s.add(user)

        pe.processed_at = datetime.utcnow()
        s.add(pe)

    return {"ok": True}


@app.post("/admin/create-user")
def admin_create_user(request: Request, email: EmailStr = Form(...), tier: str = Form("FREE")):
    _require_admin(request)
    with session_scope() as s:
        u = s.query(User).filter(User.email == str(email).lower()).first()
        if not u:
            u = User(
                email=str(email).lower(),
                tier=tier.upper(),
                is_active=True,
                preference_token=secrets.token_urlsafe(24),
            )
            s.add(u)
            s.flush()

            t = DeliveryTarget(user_id=u.id, email=u.email, label="primary", is_primary=True)
            s.add(t)
            s.flush()

            sched = DeliverySchedule(user_id=u.id, target_id=t.id, timezone="Europe/London", send_time_hhmm="06:30")
            sched.next_run_at = compute_next_run(sched.timezone, sched.send_time_hhmm, now_utc=datetime.utcnow())
            s.add(sched)

        link = f"{settings.public_base_url.rstrip('/')}/prefs/{u.preference_token}"
        return {"user_id": str(u.id), "prefs_link": link}


@app.post("/admin/queue-user-now")
def admin_queue_user_now(
    email: str = Form(...),
    regen: int = Form(0),
    exclude: str = Form(""),
    force_include: str = Form(""),
    note: str = Form(""),
    x_admin_token: str = Header(default=""),
):
    if x_admin_token.strip() != _admin_token():
        raise HTTPException(status_code=401, detail="unauthorized")

    from datetime import timedelta

    with session_scope() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="user not found")

        sched = (
            s.query(DeliverySchedule)
            .filter(DeliverySchedule.user_id == user.id, DeliverySchedule.enabled == True)  # noqa: E712
            .order_by(DeliverySchedule.created_at.asc())
            .first()
        )
        if not sched:
            raise HTTPException(status_code=404, detail="schedule not found")

        sched.override_regen = bool(regen)
        sched.override_exclude = (exclude or "").strip() or None
        sched.override_include = (force_include or "").strip() or None
        sched.override_note = (note or "").strip() or None

        sched.next_run_at = datetime.utcnow() - timedelta(minutes=1)
        s.add(sched)

    return {
        "queued": True,
        "email": email,
        "regen": bool(regen),
        "exclude": exclude,
        "force_include": force_include,
        "note": note,
    }


@app.post("/admin/resend-last")
def admin_resend_last(
    email: str = Form(...),
    x_admin_token: str = Header(default=""),
):
    if x_admin_token.strip() != _admin_token():
        raise HTTPException(status_code=401, detail="unauthorized")

    from .delivery.mailer import send_email

    with session_scope() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="user not found")

        sched = (
            s.query(DeliverySchedule)
            .filter(DeliverySchedule.user_id == user.id)
            .order_by(DeliverySchedule.created_at.asc())
            .first()
        )
        if not sched:
            raise HTTPException(status_code=404, detail="schedule not found")

        run = (
            s.query(DeliveryRun)
            .filter(DeliveryRun.schedule_id == sched.id, DeliveryRun.status == "SENT")
            .order_by(DeliveryRun.run_at.desc())
            .first()
        )
        if not run or not run.report_html:
            raise HTTPException(status_code=404, detail="no prior successful run stored")

        subject = run.subject or "DailyStonks (resend)"
        unsub_url = ""  # already embedded in html footer by runner
        msgid = send_email(settings, run.to_email or email, subject, run.report_html, unsub_url)

        return {"resent": True, "email": email, "message_id": msgid, "run_at": str(run.run_at)}


@app.get("/engine-cards")
def engine_cards():
    return {"cards": engine_list_cards()}
