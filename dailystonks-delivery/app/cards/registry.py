from __future__ import annotations

from typing import Dict, List
from .base import Card, CardMeta

class CardRegistry:
    def __init__(self) -> None:
        self._cards: Dict[str, Card] = {}

    def register(self, card: Card) -> None:
        meta = card.meta()
        if meta.id in self._cards:
            raise ValueError(f"Duplicate card id: {meta.id}")
        self._cards[meta.id] = card

    def get(self, card_id: str) -> Card:
        return self._cards[card_id]

    def list_meta(self) -> List[CardMeta]:
        return [c.meta() for c in self._cards.values()]

    def ids(self) -> List[str]:
        return list(self._cards.keys())

def registry_from_builtin() -> CardRegistry:
    from .builtin.sp500_summary import Sp500SummaryCard
    from .builtin.top_movers import TopMoversCard

    reg = CardRegistry()
    reg.register(Sp500SummaryCard())
    reg.register(TopMoversCard())
    return reg
