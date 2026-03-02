from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from .config import Settings, get_settings
from .cards.base import CardMeta

@dataclass(frozen=True)
class TierPolicy:
    can_customize: bool
    max_cards: int
    allowed_cards: List[str]  # list of ids or ["*"]

def load_tier_policies(settings: Settings | None = None) -> Dict[str, TierPolicy]:
    settings = settings or get_settings()
    data = json.loads(Path(settings.tier_policy_path).read_text(encoding="utf-8"))
    tiers = {}
    for tier, cfg in data.get("tiers", {}).items():
        tiers[tier] = TierPolicy(
            can_customize=bool(cfg.get("can_customize", False)),
            max_cards=int(cfg.get("max_cards", 6)),
            allowed_cards=list(cfg.get("allowed_cards", [])),
        )
    return tiers

def _allowed_set(policy: TierPolicy, registry_ids: Set[str]) -> Set[str]:
    if "*" in policy.allowed_cards:
        return set(registry_ids)
    return set([c for c in policy.allowed_cards if c in registry_ids])

def resolve_cards_for_user(
    user_tier: str,
    policy: TierPolicy,
    registry_meta: List[CardMeta],
    selected_card_ids: List[str],
) -> List[str]:
    # Return ordered list of card_ids to include in the email.
    registry_ids = set([m.id for m in registry_meta])
    allowed = _allowed_set(policy, registry_ids)

    if not policy.can_customize:
        # Use defaults from meta, filtered by allowed.
        metas = [m for m in registry_meta if m.id in allowed and m.default_enabled]
        metas.sort(key=lambda m: (m.default_position, m.id))
        return [m.id for m in metas[:policy.max_cards]]

    # Customizable: use user's selection order, restricted by allowed and max_cards.
    seen = set()
    ordered = []
    for cid in selected_card_ids:
        if cid in allowed and cid not in seen:
            ordered.append(cid)
            seen.add(cid)
        if len(ordered) >= policy.max_cards:
            break

    # If user selected nothing, fall back to defaults.
    if not ordered:
        metas = [m for m in registry_meta if m.id in allowed and m.default_enabled]
        metas.sort(key=lambda m: (m.default_position, m.id))
        ordered = [m.id for m in metas[:policy.max_cards]]

    return ordered
