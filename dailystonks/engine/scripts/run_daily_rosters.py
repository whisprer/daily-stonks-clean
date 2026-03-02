from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd: Path, env: dict):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd), env=env)

def main():
    ap = argparse.ArgumentParser(description="Generate tiered reports using rosters into out/daily/DATE and out/latest.")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--start", default="2005-01-01", help="history start date for reports")
    ap.add_argument("--max-universe", type=int, default=350)
    ap.add_argument("--tickers", default="SPY,QQQ,AAPL,MSFT,NVDA", help="comma list")
    ap.add_argument("--out-root", default="out", help="output root folder")
    ap.add_argument("--tiers", default="free,basic,pro,gold,black", help="comma list")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = root / args.out_root
    day = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    seed = day.isoformat()

    # Map tiers -> roster (None means "no roster", just run normal selection)
    tier_roster = {
        "free": None,
        "basic": None,
        "pro": "pro_core_daily",
        "gold": "gold_macro_focus",
        "black": "black_terminal_daily",
    }

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    tickers = args.tickers

    daily_dir = out_root / "daily" / seed
    latest_dir = out_root / "latest"
    daily_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    py = sys.executable

    for tier in tiers:
        roster = tier_roster.get(tier)
        out_html = daily_dir / f"report_{tier}_{seed}.html"
        out_plan = daily_dir / f"report_{tier}_{seed}.plan.json"

        cmd = [py, "-m", "dailystonks",
               "--tier", tier,
               "--out", str(out_html),
               "--start", args.start,
               "--max-universe", str(args.max_universe),
               "--tickers", tickers,
               "--seed", seed,
               "--dump-plan", str(out_plan)]

        if roster:
            cmd += ["--roster", roster]

        run(cmd, cwd=root, env=env)

        # copy “latest” aliases (for mailer / web)
        latest_html = latest_dir / f"report_{tier}.html"
        latest_manifest = latest_dir / f"report_{tier}.html.manifest.json"
        latest_plan = latest_dir / f"report_{tier}.plan.json"

        shutil.copyfile(out_html, latest_html)

        man = Path(str(out_html) + ".manifest.json")
        if man.exists():
            shutil.copyfile(man, latest_manifest)

        if out_plan.exists():
            shutil.copyfile(out_plan, latest_plan)

    print("OK ->", daily_dir)
    print("OK ->", latest_dir)

if __name__ == "__main__":
    main()