from __future__ import annotations
import html

def escape(s: str) -> str:
    return html.escape(s, quote=True)
