"""Suffix DSL parser for feature engineering.

Parses feature strings like ``price_d7_d1_std`` or ``gen_wind_on_ewma_24_d1_h10``
into structured ``FeatureSpec`` dataclasses. See docs/stage4_feature_engineering.md
section 4.3 for the full grammar specification.

Grammar (informal BNF)::

    feature      := interaction | simple
    interaction  := simple "__x__" simple
    simple       := short_name suffix?
    suffix       := ewma_suffix | lag_suffix | agg_suffix | fourier_suffix | daily_agg_suffix
    lag_suffix   := "_h" INT
    agg_suffix   := "_d" INT ("_d" INT)? ("_eh" INT)? ("_h" INT "_h" INT)? ("_" STAT)?
    ewma_suffix  := "_ewma_" INT ("_d" INT ("_h" INT)?)?
    fourier_suffix := "_fourier_" INT "_" INT
    daily_agg_suffix := "_daily_" STAT

    STAT := "avg" | "std" | "min" | "max" | "sum" | "range" | "share"
    INT  := [0-9]+
"""

import difflib
import re
from dataclasses import dataclass

from energy_forecasting.config.columns import SHORT_NAMES

# Valid stat types for aggregation
VALID_STATS = frozenset({"avg", "std", "min", "max", "sum", "range", "share"})


# ── Parser output types ──────────────────────────────────────────


@dataclass(frozen=True)
class HourlyLag:
    hours: int


@dataclass(frozen=True)
class Aggregation:
    start_day: int  # negative, e.g. -7
    end_day: int  # negative, e.g. -1
    stat: str = "avg"  # "avg", "std", "min", "max", "sum", "range"
    end_hour: int | None = None  # truncate final day at this hour
    hour_start: int | None = None  # filter all days to [hour_start, hour_end)
    hour_end: int | None = None


@dataclass(frozen=True)
class EWMA:
    span: int
    cutoff_day: int | None = None
    cutoff_hour: int | None = None


@dataclass(frozen=True)
class Fourier:
    period: int
    order: int


@dataclass(frozen=True)
class DailyAggregate:
    stat: str  # "sum", "avg", "max", "min", "std", "share"


@dataclass(frozen=True)
class FeatureSpec:
    base: str  # short name
    raw_col: str  # resolved column from SHORT_NAMES
    lag: HourlyLag | None = None
    agg: Aggregation | None = None
    ewma: EWMA | None = None
    fourier: Fourier | None = None
    daily_agg: DailyAggregate | None = None


@dataclass(frozen=True)
class InteractionSpec:
    left: FeatureSpec
    right: FeatureSpec


# ── Short name resolution ────────────────────────────────────────

# Sort by length descending so longest prefix matches first
_SORTED_SHORT_NAMES = sorted(SHORT_NAMES.keys(), key=len, reverse=True)


def _resolve_short_name(feature_str: str) -> tuple[str, str, str]:
    """Resolve the short name prefix from a feature string.

    Returns (short_name, raw_col, remaining_suffix).
    """
    for name in _SORTED_SHORT_NAMES:
        if feature_str == name or feature_str.startswith(name + "_"):
            suffix = feature_str[len(name) :]
            return name, SHORT_NAMES[name], suffix

    # No match — suggest close names
    close = difflib.get_close_matches(feature_str.split("_")[0], SHORT_NAMES.keys(), n=3)
    suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"Unknown short name in feature string {feature_str!r}.{suggestion}")


# ── Suffix patterns ──────────────────────────────────────────────

# EWMA: _ewma_SPAN or _ewma_SPAN_dDAY or _ewma_SPAN_dDAY_hHOUR
_EWMA_RE = re.compile(r"^_ewma_(\d+)(?:_d(\d+)(?:_h(\d+))?)?$")

# Fourier: _fourier_PERIOD_ORDER
_FOURIER_RE = re.compile(r"^_fourier_(\d+)_(\d+)$")

# Daily aggregate: _daily_STAT
_DAILY_AGG_RE = re.compile(r"^_daily_(" + "|".join(VALID_STATS) + r")$")

# Hourly lag: _hN (but NOT _h that's part of aggregation hour filter)
_HOURLY_LAG_RE = re.compile(r"^_h(\d+)$")

# Aggregation: _dX or _dX_dY with optional _ehH or _hA_hB and optional _STAT
# We parse this in stages rather than one mega-regex
_AGG_START_RE = re.compile(r"^_d(\d+)")


def _parse_suffix(suffix: str) -> dict:
    """Parse suffix into kwargs for FeatureSpec construction."""
    if not suffix:
        return {}

    # Try EWMA
    m = _EWMA_RE.match(suffix)
    if m:
        span = int(m.group(1))
        cutoff_day = -int(m.group(2)) if m.group(2) else None
        cutoff_hour = int(m.group(3)) if m.group(3) else None
        return {"ewma": EWMA(span=span, cutoff_day=cutoff_day, cutoff_hour=cutoff_hour)}

    # Try Fourier
    m = _FOURIER_RE.match(suffix)
    if m:
        return {"fourier": Fourier(period=int(m.group(1)), order=int(m.group(2)))}

    # Try daily aggregate
    m = _DAILY_AGG_RE.match(suffix)
    if m:
        return {"daily_agg": DailyAggregate(stat=m.group(1))}

    # Try hourly lag
    m = _HOURLY_LAG_RE.match(suffix)
    if m:
        return {"lag": HourlyLag(hours=int(m.group(1)))}

    # Try aggregation
    m = _AGG_START_RE.match(suffix)
    if m:
        return {"agg": _parse_aggregation(suffix)}

    raise ValueError(f"Invalid suffix {suffix!r}")


def _parse_aggregation(suffix: str) -> Aggregation:
    """Parse an aggregation suffix like _d7, _d7_d1, _d7_d1_std, _d7_d1_eh8, etc."""
    remaining = suffix
    stat = "avg"
    end_hour = None
    hour_start = None
    hour_end = None

    # Parse first _dX
    m = re.match(r"^_d(\d+)", remaining)
    if not m:
        raise ValueError(f"Expected _dX in aggregation suffix {suffix!r}")
    start_day_val = int(m.group(1))
    remaining = remaining[m.end() :]

    # Parse optional second _dY
    end_day_val = start_day_val  # default: single day
    m = re.match(r"^_d(\d+)", remaining)
    if m:
        end_day_val = int(m.group(1))
        remaining = remaining[m.end() :]

    # Parse optional _ehH (end-hour)
    m = re.match(r"^_eh(\d+)", remaining)
    if m:
        end_hour = int(m.group(1))
        remaining = remaining[m.end() :]

    # Parse optional _hA_hB (hour filter)
    m = re.match(r"^_h(\d+)_h(\d+)", remaining)
    if m:
        if end_hour is not None:
            raise ValueError(
                f"Cannot combine _eh and _h_h in {suffix!r}: "
                "_eh truncates the final day, _h_h filters all days"
            )
        hour_start = int(m.group(1))
        hour_end = int(m.group(2))
        remaining = remaining[m.end() :]

    # Parse optional stat
    if remaining:
        m = re.match(r"^_(" + "|".join(VALID_STATS) + r")$", remaining)
        if m:
            stat = m.group(1)
            remaining = remaining[m.end() :]

    if remaining:
        raise ValueError(
            f"Unexpected trailing text in aggregation suffix {suffix!r}: {remaining!r}"
        )

    return Aggregation(
        start_day=-start_day_val,
        end_day=-end_day_val,
        stat=stat,
        end_hour=end_hour,
        hour_start=hour_start,
        hour_end=hour_end,
    )


# ── Public API ────────────────────────────────────────────────────


def parse_feature(feature_str: str) -> FeatureSpec | InteractionSpec:
    """Parse a feature string into a FeatureSpec or InteractionSpec.

    Examples::

        >>> parse_feature("price_d7_d1_std")
        FeatureSpec(base='price', raw_col='target_price',
                    agg=Aggregation(start_day=-7, end_day=-1, stat='std'))

        >>> parse_feature("gen_solar_h48__x__day_index")
        InteractionSpec(left=FeatureSpec(...), right=FeatureSpec(...))

    Raises ValueError for unknown short names or invalid suffixes.
    """
    # Check for interaction
    if "__x__" in feature_str:
        parts = feature_str.split("__x__")
        if len(parts) != 2:
            raise ValueError(f"Invalid interaction format: {feature_str!r}")
        left = parse_feature(parts[0])
        right = parse_feature(parts[1])
        if isinstance(left, InteractionSpec) or isinstance(right, InteractionSpec):
            raise ValueError("Nested interactions not supported")
        return InteractionSpec(left=left, right=right)

    base, raw_col, suffix = _resolve_short_name(feature_str)
    kwargs = _parse_suffix(suffix)
    return FeatureSpec(base=base, raw_col=raw_col, **kwargs)
