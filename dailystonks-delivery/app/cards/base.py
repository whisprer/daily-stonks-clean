from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Any

@dataclass(frozen=True)
class CardMeta:
    id: str
    title: str
    description: str
    category: str = "general"
    min_tier: str = "FREE"
    default_enabled: bool = True
    default_position: int = 100

class RenderContext(Protocol):
    # Provide whatever your existing data pipeline exposes.
    asof_date: date
    public_base_url: str

    def asset_url(self, path: str) -> str: ...
    def data(self, key: str, default: Any = None) -> Any: ...

class Card(Protocol):
    def meta(self) -> CardMeta: ...
    def render_html(self, ctx: RenderContext) -> str: ...
