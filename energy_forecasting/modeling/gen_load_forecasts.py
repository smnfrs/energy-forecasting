"""Timezone alignment helpers for gen/load forecast artifacts."""

from __future__ import annotations

import pandas as pd


def _align_tz(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Bring a source series's index timezone into agreement with the target.

    Merged price data is stored tz-naive and represents Europe/Berlin local
    delivery time after DST normalisation. Stage 5b forecast artifacts are
    usually tz-aware UTC. Reindexing across that mismatch silently returns
    NaN, so convert aware artifacts to Europe/Berlin delivery time and strip
    the timezone before lookup.
    """
    src_tz = series.index.tz
    dst_tz = target_index.tz
    if src_tz is None and dst_tz is None:
        return series
    if src_tz is not None and dst_tz is None:
        local = series.tz_convert("Europe/Berlin").tz_localize(None)
        if local.index.has_duplicates:
            local = local.groupby(level=0).mean()
        return local
    if src_tz is None and dst_tz is not None:
        local = series.tz_localize(
            "Europe/Berlin",
            ambiguous="infer",
            nonexistent="shift_forward",
        )
        return local.tz_convert(dst_tz)
    return series.tz_convert(dst_tz)
