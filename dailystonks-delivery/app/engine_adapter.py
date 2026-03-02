from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt
import os
import yaml
import logging
import html as _html

log = logging.getLogger('dailystonks.delivery')


# Ensure all cards register
import dailystonks.cards  # noqa: F401

from dailystonks.core.models import CardContext, CardResult
from dailystonks.core.registry import CARD_REGISTRY, CardSpec
from dailystonks.core.selector import select_cards, tier_allows
from dailystonks.data.marketdata import MarketData
from dailystonks.data.sp500 import SP500Universe
from dailystonks.render.html import render_report_html


# Card substitution map for "zero visible errors" in customer emails.
# If a card fails, we transparently run a safe alternative and add a tiny note.
FALLBACK_MAP = {
    "anomaly.sigma_intraday_alerts": "price.candles_basic",
}
TIERS = ("free", "basic", "pro", "black")

def normalize_tier(t: str) -> str:
    s = (t or "").strip().lower()
    # tolerate older naming
    aliases = {
        "FREE": "free", "BASIC": "basic", "PRO": "pro", "BLACK": "black",
        "gold": "black", "custom": "black",
    }
    return aliases.get(s, s) if s in aliases else (s if s in TIERS else "free")

@lru_cache(maxsize=1)
def engine_root() -> Path:
    import dailystonks as pkg
    # .../engine/dailystonks/__init__.py -> .../engine
    return Path(pkg.__file__).resolve().parents[1].parent

@lru_cache(maxsize=1)
def slot_map() -> Dict[str, dict]:
    root = engine_root()
    p = Path(os.getenv("ENGINE_CONFIG_DIR","")) / "slots.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def tiers_cfg() -> Dict[str, dict]:
    root = engine_root()
    p = Path(os.getenv("ENGINE_CONFIG_DIR","")) / "tiers.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def can_customize(tier: str) -> bool:
    # You can change this later (env override)
    env = (os.getenv("CAN_CUSTOMIZE_TIERS") or "").strip()
    if env:
        allowed = {x.strip().lower() for x in env.split(",") if x.strip()}
        return normalize_tier(tier) in allowed
    return normalize_tier(tier) in ("pro", "black")

def tier_limits(tier: str) -> Tuple[int, int, int]:
    t = normalize_tier(tier)
    cfg = tiers_cfg().get(t, {})
    lim = cfg.get("limits", {}) or {}
    return int(lim.get("max_cards", 12)), int(lim.get("max_cost", 999)), int(lim.get("heavy_max", 999))

def list_cards() -> List[Dict[str, Any]]:
    # lightweight meta for UI
    out: List[Dict[str, Any]] = []
    for k, spec in CARD_REGISTRY.items():
        out.append({
            "key": spec.key,
            "title": spec.title,
            "category": spec.category,
            "min_tier": spec.min_tier,
            "cost": spec.cost,
            "heavy": spec.heavy,
            "slots": list(spec.slots),
            "tags": list(spec.tags),
        })
    out.sort(key=lambda x: (x["category"], x["key"]))
    return out

def _budget_filter(tier: str, keys_in_order: List[str]) -> List[str]:
    t = normalize_tier(tier)
    max_cards, max_cost, heavy_max = tier_limits(t)
    chosen: List[str] = []
    cost = 0
    heavy = 0

    for key in keys_in_order:
        spec: Optional[CardSpec] = CARD_REGISTRY.get(key)
        if not spec:
            continue
        if not tier_allows(t, spec.min_tier):
            continue

        if len(chosen) >= max_cards:
            break
        if cost + int(spec.cost) > max_cost:
            continue
        if spec.heavy and (heavy + 1) > heavy_max:
            continue

        chosen.append(key)
        cost += int(spec.cost)
        heavy += 1 if spec.heavy else 0

    return chosen

def choose_keys_for_user(
    *,
    as_of: dt.date,
    tier: str,
    selected_keys: List[str],
    seed: int,
    overrides: Dict[str, str] | None = None,
) -> List[str]:
    t = normalize_tier(tier)
    overrides = overrides or {}

    # If user can customize and has a selection list, respect it (with budgets).
    if can_customize(t) and selected_keys:
        return _budget_filter(t, selected_keys)

    # Otherwise use engine slot/tier selection (defaults + deterministic rotation).
    cfg_t = tiers_cfg()[t]
    return select_cards(
        as_of=as_of,
        tier=t,
        slot_map=slot_map(),
        tier_cfg=cfg_t,
        overrides=overrides,
        seed=seed,
    )

@lru_cache(maxsize=1)
def _market_and_sp500() -> Tuple[MarketData, SP500Universe, str]:
    root = engine_root()
    cache_dir = str(root / ".cache")
    sp_csv = root / "data" / "sp500_constituents.csv"
    sp500 = SP500Universe(csv_path=str(sp_csv))
    offline = (os.getenv("REPORT_OFFLINE_SYNTH") or "0") == "1"
    market = MarketData(cache_dir=cache_dir, offline_synth=offline)
    return market, sp500, cache_dir


def _is_intraday(interval: str) -> bool:
    i = (interval or "").strip().lower()
    return any(i.endswith(x) for x in ("m","h")) and i not in ("1d","1wk","1mo")

def _clamp_intraday_start(start: str, interval: str, as_of: dt.date) -> tuple[str, str | None]:
    # Conservative caps to avoid Yahoo failures
    limits = {"1h": 720, "30m": 120, "15m": 60, "5m": 30, "2m": 7, "1m": 7}
    if not _is_intraday(interval):
        return start, None
    lim_days = limits.get(interval.strip().lower(), 30)
    try:
        start_dt = dt.date.fromisoformat(start)
    except Exception:
        return start, f"Bad start '{start}'"
    min_dt = as_of - dt.timedelta(days=lim_days)
    if start_dt < min_dt:
        return min_dt.isoformat(), f"Intraday '{interval}' limited; clamped start to {min_dt.isoformat()} (was {start})."
    return start, None

def run_html(
    *,
    as_of: dt.date,
    tier: str,
    tickers: List[str],
    start: str,
    end: Optional[str],
    interval: str,
    universe: str,
    max_universe: int,
    chosen_keys: List[str],
    support_ref: str | None = None,) -> str:
    t = normalize_tier(tier)
    market, sp500, cache_dir = _market_and_sp500()

    start2, clamp_note = _clamp_intraday_start(start, interval, as_of)
    if clamp_note:
        log.warning(clamp_note)
        start = start2

    ctx = CardContext(
        as_of=as_of,
        start=start,
        end=end,
        interval=interval,
        tier=t,
        universe=universe,
        max_universe=max_universe,
        tickers=tickers,
        market=market,
        sp500=sp500,
        cache_dir=cache_dir,
        signals={},
    )

    results: List[CardResult] = []
    substitutions: List[str] = []

    for key in chosen_keys:
        spec = CARD_REGISTRY.get(key)
        if not spec:
            continue

        fb_key = FALLBACK_MAP.get(key)

        try:
            res = spec.fn(ctx)

            # If the card returns warnings (data unavailable etc.) and we have a fallback,
            # substitute to enforce "zero visible errors".
            warn_list = getattr(res, "warnings", None) or []
            warn_txt = " ".join([str(w) for w in warn_list]).strip()

            if fb_key and warn_txt:
                fb_spec = CARD_REGISTRY.get(fb_key)
                if fb_spec:
                    try:
                        fb_res = fb_spec.fn(ctx)
                        substitutions.append((key, fb_key, warn_txt[:160] if "warn_txt" in locals() else ""))
                        results.append(fb_res)
                        continue
                    except Exception:
                        log.exception("Fallback card failed after warning-substitution: %s -> %s", key, fb_key)
                        # If fallback also fails, drop silently
                        continue

            results.append(res)
            continue

        except Exception:
            # Exception-based substitution path
            if fb_key:
                fb_spec = CARD_REGISTRY.get(fb_key)
                if fb_spec:
                    try:
                        fb_res = fb_spec.fn(ctx)
                        substitutions.append((key, fb_key, warn_txt[:160] if "warn_txt" in locals() else ""))
                        results.append(fb_res)
                        continue
                    except Exception:
                        log.exception("Fallback card also failed: %s -> %s", key, fb_key)
                        continue

            log.exception("Card failed (dropped): %s", key)
            continue

    # Build HTML once, then optionally inject substitution banner
    html = render_report_html(as_of=as_of, tier=t, tickers=tickers, results=results)

    if substitutions:
        def _esc(x: str) -> str:
            return (x or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        lines = []
        for item in substitutions:
            # item can be a tuple (src, dst, reason) or a string (backwards compat)
            if isinstance(item, tuple) and len(item) == 3:
                src, dst, reason = item
                lines.append(
                    f"<div style=\"margin:6px 0;\"><code>{_esc(src)}</code> → <code>{_esc(dst)}</code>"
                    f"<div style=\"opacity:.9; margin-top:2px;\">Reason: {_esc(str(reason))}</div></div>"
                )
            else:
                lines.append(f"<div style=\"margin:6px 0;\">{_esc(str(item))}</div>")

        ref_html = ""
        if support_ref:
            ref_html = f"<div style=\"opacity:.9;margin-top:6px;\"><b>Ref:</b> <code>{_esc(support_ref)}</code></div>"

        note = (
            "<div style=\"padding:10px 12px;border:1px solid #445;background:#10131a;"
            "border-radius:12px;margin:12px 0;color:#cfd7ff;font-size:13px;\">"
            "<b>Note:</b> Some cards were temporarily unavailable and were substituted:"
            + "".join(lines)
            + ref_html
            + "</div>"
        )
        if "</header>" in html:
            html = html.replace("</header>", "</header>" + note)
        else:
            html = note + html
    return html



