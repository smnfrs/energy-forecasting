"""Tests for config/columns.py — short name registry and SMARD mappings."""

from energy_forecasting.config.columns import (
    CROSS_BORDER_DE_AT_LU,
    CROSS_BORDER_DE_LU,
    EXCLUDED_KEYS,
    INSTALLED_CAPACITY_KEYS,
    REVERSE_SHORT_NAMES,
    SCHEDULED_COMMERCIAL_KEYS,
    SHORT_NAMES,
    SMARD_COLUMN_NAMES,
    SMARD_FILTER_KEYS,
    clean_column_name,
)


def test_short_names_no_duplicate_values():
    values = list(SHORT_NAMES.values())
    assert len(values) == len(set(values)), "SHORT_NAMES has duplicate column name values"


def test_short_names_no_duplicate_keys():
    keys = list(SHORT_NAMES.keys())
    assert len(keys) == len(set(keys)), "SHORT_NAMES has duplicate keys"


def test_reverse_short_names_round_trip():
    for short, col in SHORT_NAMES.items():
        assert REVERSE_SHORT_NAMES[col] == short


def test_clean_column_name_basic():
    assert clean_column_name("Stromerzeugung: Braunkohle") == "stromerzeugung_braunkohle"


def test_clean_column_name_slash():
    result = clean_column_name("Marktpreis: Deutschland/Luxemburg")
    assert result == "marktpreis_deutschland_luxemburg"


def test_clean_column_name_parentheses():
    result = clean_column_name("Stromverbrauch: Gesamt (Netzlast)")
    assert result == "stromverbrauch_gesamt_(netzlast)"


def test_smard_filter_keys_produces_valid_column_names():
    for key, desc in SMARD_FILTER_KEYS.items():
        col = SMARD_COLUMN_NAMES[key]
        assert col == clean_column_name(desc)
        # Column names should be lowercase with no spaces
        assert col == col.lower()
        assert " " not in col


def test_smard_filter_keys_count():
    # EP had ~40 entries in filter_dict (excluding cross-border)
    assert len(SMARD_FILTER_KEYS) >= 40


def test_cross_border_de_lu_has_imports_and_exports():
    descs = list(CROSS_BORDER_DE_LU.values())
    assert any("imports" in d for d in descs)
    assert any("exports" in d for d in descs)


def test_cross_border_de_at_lu_has_imports_and_exports():
    descs = list(CROSS_BORDER_DE_AT_LU.values())
    assert any("imports" in d for d in descs)
    assert any("exports" in d for d in descs)


def test_excluded_keys_is_union():
    assert EXCLUDED_KEYS == INSTALLED_CAPACITY_KEYS | SCHEDULED_COMMERCIAL_KEYS


def test_excluded_keys_all_in_smard_or_scheduled():
    # All installed capacity keys should be in SMARD_FILTER_KEYS
    for key in INSTALLED_CAPACITY_KEYS:
        assert key in SMARD_FILTER_KEYS, f"Capacity key {key} not in SMARD_FILTER_KEYS"
