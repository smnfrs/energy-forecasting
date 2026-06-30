"""Sanity checks for config/merge.py constants."""

from energy_forecasting.config.columns import SMARD_COLUMN_NAMES
from energy_forecasting.config.merge import (
    BIDDING_AREA_SPLIT,
    PRICE_POST_SPLIT,
    PRICE_PRE_SPLIT,
    TSO_TO_NATIONAL,
)
from energy_forecasting.config.smard import TSO_FILTER_KEYS


def test_bidding_area_split_is_utc():
    assert BIDDING_AREA_SPLIT.tzinfo is not None
    assert str(BIDDING_AREA_SPLIT.tzinfo) == "UTC"


def test_price_columns_exist_in_smard_column_names():
    smard_cols = set(SMARD_COLUMN_NAMES.values())
    assert PRICE_POST_SPLIT in smard_cols
    assert PRICE_PRE_SPLIT in smard_cols


def test_tso_to_national_keys_match_tso_filter_keys():
    tso_values = set(TSO_FILTER_KEYS.values())
    for key in TSO_TO_NATIONAL:
        assert key in tso_values, f"{key} not in TSO_FILTER_KEYS values"


def test_tso_to_national_values_in_smard_column_names():
    smard_cols = set(SMARD_COLUMN_NAMES.values())
    for val in TSO_TO_NATIONAL.values():
        assert val in smard_cols, f"{val} not in SMARD_COLUMN_NAMES"
