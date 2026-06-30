"""Tests for config/smard.py — sanity checks on SMARD config."""

from energy_forecasting.config.smard import (
    KNOWN_MISSING,
    NATIONAL_REGIONS,
    TSO_FILTER_KEYS,
    TSO_REGIONS,
    TSO_SUFFIXES,
)


def test_known_missing_filter_ids_exist():
    """All KNOWN_MISSING filter IDs must be in TSO_FILTER_KEYS."""
    for fid, region in KNOWN_MISSING:
        assert fid in TSO_FILTER_KEYS, f"Filter {fid} not in TSO_FILTER_KEYS"


def test_known_missing_regions_exist():
    """All KNOWN_MISSING regions must be in TSO_REGIONS."""
    for fid, region in KNOWN_MISSING:
        assert region in TSO_REGIONS, f"Region {region} not in TSO_REGIONS"


def test_tso_suffixes_match_regions():
    """All TSO_SUFFIXES keys must match TSO_REGIONS."""
    assert set(TSO_SUFFIXES.keys()) == set(TSO_REGIONS)


def test_national_and_tso_no_overlap():
    """NATIONAL_REGIONS and TSO_REGIONS must not overlap."""
    assert set(NATIONAL_REGIONS).isdisjoint(set(TSO_REGIONS))


def test_tso_suffixes_are_unique():
    """Each TSO suffix must be unique."""
    suffixes = list(TSO_SUFFIXES.values())
    assert len(suffixes) == len(set(suffixes))


def test_tso_filter_keys_nonempty():
    """TSO_FILTER_KEYS should have at least 10 entries."""
    assert len(TSO_FILTER_KEYS) >= 10
