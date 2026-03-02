from app import tier

def test_known_tiers_exist():
    # adjust names to your enum/constants
    assert hasattr(tier, "Tier")

def test_tier_names():
    names = {t.value for t in tier.Tier}
    # include the tiers you actually support
    assert {"free", "basic", "pro", "black"}.issubset(names)