from pathlib import Path


def test_delivery_tier_policy_json_loads_and_sane():
    from app.tier import load_tier_policies

    policies = load_tier_policies()
    # Delivery service policy file is uppercase tier keys.
    for key in ("FREE", "BASIC", "PRO", "BLACK"):
        assert key in policies

    for tier, pol in policies.items():
        assert pol.max_cards > 0, f"{tier} max_cards must be > 0"
        assert isinstance(pol.allowed_cards, list)
        assert pol.allowed_cards, f"{tier} allowed_cards must not be empty"


def test_engine_tiers_yaml_loads_and_has_limits(monkeypatch):
    # This tests the engine-side tiers.yaml that the delivery runner uses.
    here = Path(__file__).resolve()
    delivery_root = here.parents[1]  # .../dailystonks-delivery
    engine_root = (delivery_root.parent / "dailystonks" / "engine").resolve()
    engine_cfg = engine_root / "config"
    assert (engine_cfg / "tiers.yaml").is_file(), "engine/config/tiers.yaml missing"
    assert (engine_cfg / "slots.yaml").is_file(), "engine/config/slots.yaml missing"

    monkeypatch.setenv("ENGINE_CONFIG_DIR", str(engine_cfg))

    from app import engine_adapter

    # Clear caches so env var takes effect.
    engine_adapter.tiers_cfg.cache_clear()
    engine_adapter.slot_map.cache_clear()

    tiers = engine_adapter.tiers_cfg()
    slots = engine_adapter.slot_map()

    for t in ("free", "basic", "pro", "black"):
        assert t in tiers, f"missing tier '{t}' in tiers.yaml"
        lim = (tiers[t] or {}).get("limits", {}) or {}
        assert int(lim.get("max_cards", 0)) > 0
        assert int(lim.get("max_cost", 0)) > 0
        assert int(lim.get("heavy_max", 0)) >= 0

    assert isinstance(slots, dict) and slots, "slots.yaml did not parse to a non-empty dict"
