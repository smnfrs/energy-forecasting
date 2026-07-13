"""Tests for config/availability.py — availability rules parse correctly."""

from energy_forecasting.config.availability import AVAILABILITY_RULES, AvailabilityRule


def test_availability_rules_not_empty():
    assert len(AVAILABILITY_RULES) > 0


def test_all_rules_are_availability_rule_instances():
    for rule in AVAILABILITY_RULES:
        assert isinstance(rule, AvailabilityRule)


def test_offset_days_are_non_positive():
    for rule in AVAILABILITY_RULES:
        assert rule.max_offset_days <= 0, (
            f"Rule '{rule.pattern}' has positive offset {rule.max_offset_days}"
        )


def test_cutoff_hour_in_valid_range():
    for rule in AVAILABILITY_RULES:
        if rule.cutoff_hour is not None:
            assert 0 <= rule.cutoff_hour < 24, (
                f"Rule '{rule.pattern}' has invalid cutoff_hour {rule.cutoff_hour}"
            )


def test_all_rules_have_reason():
    for rule in AVAILABILITY_RULES:
        assert rule.reason, f"Rule '{rule.pattern}' has empty reason"


def test_patterns_cover_key_categories():
    patterns = {r.pattern for r in AVAILABILITY_RULES}
    # Forecasts
    assert "forecast_*" in patterns
    # Prices
    assert "price" in patterns
    assert "price_*" in patterns
    # Generation
    assert "gen_*" in patterns
    # Load
    assert "load" in patterns
    # Commodities
    assert "carbon" in patterns
    assert "ttf" in patterns
    assert "brent" in patterns
