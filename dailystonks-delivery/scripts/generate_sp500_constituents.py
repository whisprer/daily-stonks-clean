#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import urllib.request
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = Path("/opt/dailystonks.org/dailystonks/data/sp500_constituents.csv")

def _normalize_symbol(sym: str) -> str:
    sym = (sym or "").strip().upper()
    return sym.replace(".", "-")  # yfinance style

def fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Connection": "close",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    html = fetch_html(WIKI_URL)

    # parse tables from the HTML string (no direct urlopen inside pandas)
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found in fetched HTML.")

    df = tables[0].copy()
    if "Symbol" not in df.columns:
        raise RuntimeError(f"Unexpected format. Columns: {list(df.columns)}")

    df["Symbol"] = df["Symbol"].astype(str).map(_normalize_symbol)

    keep = [c for c in [
        "Symbol",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "Headquarters Location",
        "Date added",
        "CIK",
        "Founded",
    ] if c in df.columns]

    df = df[keep].dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"]).sort_values("Symbol")
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}")

if __name__ == "__main__":
    main()
