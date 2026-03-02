from __future__ import annotations
from ._util import escape
from ..base import CardMeta, RenderContext

class TopMoversCard:
    def meta(self) -> CardMeta:
        return CardMeta(
            id="top_movers",
            title="Top Movers",
            description="Biggest gainers and losers.",
            category="market",
            min_tier="FREE",
            default_enabled=True,
            default_position=20,
        )

    def render_html(self, ctx: RenderContext) -> str:
        movers = ctx.data("top_movers", [])
        rows = []
        for m in movers[:10]:
            sym = escape(str(m.get("symbol","")))
            pct = escape(str(m.get("pct","")))
            rows.append(f"<tr><td style='padding:4px 8px;'><b>{sym}</b></td><td style='padding:4px 8px;text-align:right;'>{pct}</td></tr>")
        table = "<table style='width:100%;border-collapse:collapse;'>" + "".join(rows) + "</table>"
        return f"""
        <section style="padding:14px;border:1px solid #e5e5e5;border-radius:12px;margin:12px 0;">
          <h2 style="margin:0 0 6px 0;font-size:18px;">Top Movers</h2>
          {table}
        </section>
        """
