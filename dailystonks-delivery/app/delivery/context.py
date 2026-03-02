from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

@dataclass
class DeliveryContext:
    asof_date: date
    public_base_url: str
    payload: Dict[str, Any]

    def asset_url(self, path: str) -> str:
        base = self.public_base_url.rstrip("/")
        path2 = path.lstrip("/")
        return f"{base}/{path2}"

    def data(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)
