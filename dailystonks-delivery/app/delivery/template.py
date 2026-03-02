from __future__ import annotations
from datetime import date

def wrap_email(subject: str, body_html: str, asof: date, unsubscribe_url: str) -> str:
    # Conservative HTML for email clients.
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>{subject}</title>
  </head>
  <body style="font-family:Arial,Helvetica,sans-serif;background:#f7f7f7;margin:0;padding:0;">
    <div style="max-width:720px;margin:0 auto;padding:18px;">
      <header style="padding:10px 0 16px 0;">
        <div style="font-size:22px;font-weight:700;">DailyStonks</div>
        <div style="color:#666;font-size:13px;">Market briefing for {asof.isoformat()}</div>
      </header>
      <main>
        {body_html}
      </main>
      <footer style="margin-top:18px;color:#777;font-size:12px;line-height:1.4;">
        <p style="margin:0 0 6px 0;">You are receiving this because you subscribed to DailyStonks.</p>
        <p style="margin:0;"><a href="{unsubscribe_url}">Unsubscribe</a></p>
      </footer>
    </div>
  </body>
</html>"""
