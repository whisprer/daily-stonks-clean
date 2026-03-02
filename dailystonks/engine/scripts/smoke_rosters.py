from __future__ import annotations
import subprocess
from pathlib import Path
import sys
import datetime as dt

def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "out" / "smoke_rosters"
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    cases = [
        ("pro",  "pro_core_daily"),
        ("gold", "gold_macro_focus"),
        ("black","black_terminal_daily"),
    ]

    py = sys.executable
    for tier, roster in cases:
        html = out / f"{tier}_{roster}_{today}.html"
        plan = out / f"{tier}_{roster}_{today}.plan.json"
        run([py, "-m", "dailystonks",
             "--tier", tier,
             "--roster", roster,
             "--seed", today,
             "--dump-plan", str(plan),
             "--out", str(html)])
        man = Path(str(html) + ".manifest.json")
        if not man.exists():
            raise SystemExit(f"Missing manifest: {man}")

    print("OK: smoke rosters complete:", out)

if __name__ == "__main__":
    main()
