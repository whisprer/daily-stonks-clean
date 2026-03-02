from __future__ import annotations
from ._util import escape
from ..base import CardMeta, RenderContext

class Sp500SummaryCard:
    def meta(self) -> CardMeta:
        return CardMeta(
            id="sp500_summary",
            title="S&P 500 Summary",
            description="Quick snapshot: close, daily change, trend.",
            category="index",
            min_tier="FREE",
            default_enabled=True,
            default_position=10,
        )

    def render_html(self, ctx: RenderContext) -> str:
        spx = ctx.data("spx", {})
        close = escape(str(spx.get("close", "n/a")))
        chg = escape(str(spx.get("chg", "n/a")))
        pct = escape(str(spx.get("pct", "n/a")))
        img = ctx.asset_url(f"charts/{ctx.asof_date.isoformat()}/spx.png")
        return f"""
        <section style="padding:14px;border:1px solid #e5e5e5;border-radius:12px;margin:12px 0;">
          <h2 style="margin:0 0 6px 0;font-size:18px;">S&P 500</h2>
          <div style="font-size:14px;margin-bottom:10px;">
            <b>Close:</b> {close} &nbsp; <b>Δ:</b> {chg} ({pct})
          </div>
          <img src="{img}" alt="S&P 500 chart" style="width:100%;max-width:640px;border-radius:10px;" />
        </section>
        """
