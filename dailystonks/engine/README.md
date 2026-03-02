# DailyStonks Modular Report Engine (prototype)

This repo is a practical, modular "card + slots" report engine for DailyStonks mailouts.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# (Option A) install editable
pip install -e .

# (Option B) run without install
# Windows PowerShell: $env:PYTHONPATH = (Get-Location)
# Linux/macOS: export PYTHONPATH=.

# Generate a report (HTML) for a small S&P500 subset (fast)
python scripts/run_report.py --tier black --out out/report.html --universe sp500 --max-universe 60 --tickers SPY,QQQ,AAPL,MSFT,NVDA
```

## Concepts

- **Card**: a module that produces charts/tables + metrics + bullet narrative.
- **Slots**: stable layout positions in the email/terminal.
- **Tier presets**: Free/Basic/Pro/Black choose which slots exist and default cards.
- **Rotation-ready**: selection engine supports locked vs rotating slots (policy-driven).

## Data

S&P500 constituent list is bundled in `data/sp500_constituents.csv` (open dataset derived from Wikipedia).

Market data is downloaded via `yfinance` and cached to Parquet under `.cache/`.

## Notes

This is a prototype focused on correctness and modularity.
