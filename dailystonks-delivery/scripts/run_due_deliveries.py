from __future__ import annotations
from app.delivery.runner import run_due_deliveries

def main() -> None:
    n = run_due_deliveries(limit=50)
    print(f"Processed {n} schedule(s).")

if __name__ == "__main__":
    main()
