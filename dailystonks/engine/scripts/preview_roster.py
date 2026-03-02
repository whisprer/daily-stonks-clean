
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

def _find_engine_root() -> Path:
    p = Path(__file__).resolve().parent
    # walk upwards until we find dailystonks/__main__.py
    while True:
        if (p / "dailystonks" / "__main__.py").exists():
            return p
        if p.parent == p:
            raise RuntimeError("Could not locate engine root (missing dailystonks/__main__.py)")
        p = p.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="black")
    ap.add_argument("--roster", default="black_terminal_daily")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--start", default=None)
    args = ap.parse_args()

    root = _find_engine_root()
    sys.path.insert(0, str(root))

    from dailystonks.core.rosters import resolve_roster

    start = dt.date.fromisoformat(args.start) if args.start else dt.date.today()
    rows = []
    for i in range(args.days):
        d = start + dt.timedelta(days=i)
        seed = d.isoformat()
        overrides, plan = resolve_roster(args.roster, args.tier, seed=seed, strict=False)
        applied = plan.get("applied", {})
        flat = {slot: v.get("key") for slot, v in applied.items()}
        rows.append({"date": seed, "applied": flat})

    print(json.dumps({"tier": args.tier, "roster": args.roster, "days": args.days, "preview": rows}, indent=2))

if __name__ == "__main__":
    main()
