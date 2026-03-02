from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict

@dataclass
class DataProvider:
    data_dir: Path

    def load_payload(self, asof: date) -> Dict[str, Any]:
        # Contract: your existing pipeline should write a daily JSON bundle the cards can use.
        # Example path: {data_dir}/bundles/2026-02-27.json
        p = self.data_dir / "bundles" / f"{asof.isoformat()}.json"
        if not p.exists():
            # Empty payload is fine; cards should degrade gracefully.
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

def get_provider() -> DataProvider:
    base = Path(os.getenv("DAILYSTONKS_DATA_DIR", "./sample_data"))
    return DataProvider(base)
