## Stage 3: Data Cleaning & Merging

**Goal:** Clean merged dataset produced from raw data, ready for feature engineering. `make process` runs end-to-end and produces `data/processed/merged.parquet` (national, for price models) and `data/processed/tso/*.parquet` (per-TSO, for gen/load models).

**Source material:**
- EP: `src/data/processing.py` (merge pipeline: `merge_datasets()`, `create_unified_target()`, `merge_commodities()`, `run_merge_pipeline()`), `src/features/transforms.py` (`handle_missing_values()`, `interpolate_small_gaps()`), `src/features/ts_transforms.py` (`normalize_dst()`), `src/config/processing.py` (regime dates, column lists)
- EMA: `data_modules/utils.py` (`fix_broken_periodicity_with_interpolation()`, `handle_nans_with_interpolation()`, `validate_dataframe()`), `data_modules/data_loaders.py` (`impute_smard_nans()`)
- Already implemented (stages 1-2): `config/cleaning.py` (12-rule cleaning orchestrator), `data/processing.py` (all cleaning helpers), `data/commodities.py` (`reconstruct_ttf()`, `merge_carbon()`), `data/io.py` (`load_parquet()`, `save_parquet()`)

---

### 3.1 Pre-requisite: Umlaut Fix in `clean_column_name()`

**Problem:** `clean_column_name("Marktpreis: Österreich")` produces `marktpreis_österreich` (Unicode `ö`), but all config references (`config/columns.py`, `config/cleaning.py`) use `marktpreis_oesterreich` (ASCII `oe`). This causes cleaning rules to silently skip these columns.

**Fix in `energy_forecasting/config/columns.py`:**

```python
def clean_column_name(description: str) -> str:
    """Convert German SMARD description to snake_case column name.

    >>> clean_column_name("Stromerzeugung: Braunkohle")
    'stromerzeugung_braunkohle'
    >>> clean_column_name("Marktpreis: Österreich")
    'marktpreis_oesterreich'
    """
    text = description.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace(":", "_")
    # Transliterate German umlauts to ASCII
    text = text.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
    text = text.replace("Ö", "Oe").replace("Ä", "Ae").replace("Ü", "Ue")
    text = re.sub(r'[<>"|?*]', "", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text.lower()
```

Only 2 SMARD columns contain umlauts (both with `Österreich`), so the impact is narrow. After this change, `SMARD_COLUMN_NAMES` (derived from `clean_column_name()`) matches all existing config references.

---

### 3.2 Config: Merge Constants

**`energy_forecasting/config/merge.py`** (new file) — Named constants for the merge pipeline. Ported from EP's `src/config/processing.py`.

```python
"""Merge pipeline configuration constants.

Key dates for bidding-area and pricing-regime transitions,
column name references for target price creation, and
imputation thresholds.
"""

import pandas as pd

# ── Regime-change timestamps (UTC) ────────────────────────────────
# Germany-Austria-Luxembourg → Germany-Luxembourg bidding area split.
# Exact boundary: 2018-10-01 00:00 CET = 2018-09-30 22:00 UTC.
BIDDING_AREA_SPLIT = pd.Timestamp("2018-09-30T22:00:00", tz="UTC")

# EPEX SPOT resolution changes from hourly to 15-minute intervals.
QUARTER_HOURLY_START = pd.Timestamp("2025-10-01", tz="UTC")

# ── Column names for unified target price ─────────────────────────
# Must match output of clean_column_name() applied to SMARD descriptions.
# Filter 4169: "Marktpreis: Deutschland/Luxemburg"
PRICE_POST_SPLIT = "marktpreis_deutschland_luxemburg"
# Filter 251: "Marktpreis: Deutschland/Österreich/Luxemburg"
# (After umlaut transliteration: Ö → Oe → oe)
PRICE_PRE_SPLIT = "marktpreis_deutschland_oesterreich_luxemburg"

# ── Periodicity enforcement ───────────────────────────────────────
# Maximum consecutive missing hourly timestamps before raising an error.
# Small gaps (≤3) are transient collection failures; larger gaps indicate
# structural data issues that need investigation.
MAX_MISSING_CONSECUTIVE = 3

# ── Imputation gap thresholds ─────────────────────────────────────
# Three tiers of gap handling, applied in order:
#   1. Small (≤SMALL_GAP_MAX): cubic spline interpolation (existing in clean())
#   2. Medium (SMALL_GAP_MAX < gap ≤ MEDIUM_GAP_MAX): same-hour-of-day averaging
#   3. Large (> MEDIUM_GAP_MAX): rejected — logged as structural issue
SMALL_GAP_MAX = 5    # hours
MEDIUM_GAP_MAX = 48  # hours

# Window (days) for same-hour-of-day imputation: mean of same hour
# from ±IMPUTE_WINDOW_DAYS surrounding the gap.
IMPUTE_WINDOW_DAYS = 14

# ── SMARD physical limit warnings ────────────────────────────────
# Pre-cleaning diagnostic: values outside these bounds are logged as
# warnings (not clipped or removed). Bounds are in MW except prices
# in EUR/MWh. Uses fnmatch patterns for column matching.
SMARD_WARN_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "stromerzeugung_*":         (0,      100_000),
    "stromverbrauch_*":         (20_000, 100_000),
    "prognostiziert*":          (0,      120_000),
    "target_price":             (-500,   1_000),
    "cross-border_flows_*":     (None,   50_000),
}

# ── TSO aggregation mapping ──────────────────────────────────────
# Maps per-TSO filter key names to the equivalent national column name.
# Used for cross-validation (stage 3) and potential aggregation (future).
TSO_TO_NATIONAL: dict[str, str] = {
    "wind_offshore":  "stromerzeugung_wind_offshore",
    "wind_onshore":   "stromerzeugung_wind_onshore",
    "solar":          "stromerzeugung_photovoltaik",
    "load":           "stromverbrauch_gesamt_(netzlast)",
    "biomass":        "stromerzeugung_biomasse",
    "gas":            "stromerzeugung_erdgas",
    "hard_coal":      "stromerzeugung_steinkohle",
    "lignite":        "stromerzeugung_braunkohle",
    "pumped_storage": "stromerzeugung_pumpspeicher",
    "hydro":          "stromerzeugung_wasserkraft",
    "other_conv":     "stromerzeugung_sonstige_konventionelle",
    "other_renew":    "stromerzeugung_sonstige_erneuerbare",
}
```

**What's new vs EP/EMA:** EP had `BIDDING_AREA_SPLIT` and `QUARTER_HOURLY_START` in `config/processing.py` alongside column lists. EMA had no equivalent constants. The imputation thresholds and SMARD warning bounds are new — EP used fixed `max_gap=5` hardcoded in the interpolation call; EMA used `max_gap=48` with no tiered strategy. The `TSO_TO_NATIONAL` mapping enables cross-validation between national and per-TSO data, which neither repo did.

---

### 3.3 Merge Pipeline

**`energy_forecasting/data/merge.py`** (new file, ~300 lines) — The core merge pipeline. Orchestrates loading raw Parquet files, combining national SMARD eras, creating the unified target price, merging commodities, enforcing periodicity, imputing gaps, running the cleaning rules, normalizing DST, and validating the output.

#### 3.3.1 Physical bounds warnings

```python
def warn_physical_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> None:
    """Log warnings for SMARD values outside expected physical ranges.

    Pre-cleaning diagnostic — does NOT modify the data. Bounds use
    fnmatch patterns for column matching, same as clip_bounds().
    Called before cleaning to surface data quality issues.

    Ported from: new (neither EP nor EMA had pre-cleaning diagnostics).

    Args:
        df: DataFrame to check.
        bounds: Pattern→(min, max) dict. Defaults to SMARD_WARN_BOUNDS.
    """
    ...
```

Iterates `bounds` patterns via `_match_columns()` (already in `data/processing.py`). For each matched column, counts values below min or above max and logs a warning with the count, column name, and observed extremes. No data modification.

#### 3.3.2 Periodicity enforcement

```python
def enforce_periodicity(
    df: pd.DataFrame,
    max_gap: int = MAX_MISSING_CONSECUTIVE,
) -> pd.DataFrame:
    """Detect and fill small gaps in the hourly timestamp index.

    1. Build expected hourly DatetimeIndex from df.index.min() to .max().
    2. Find missing timestamps via set difference.
    3. Group consecutive missing into runs. Raise ValueError if any run > max_gap.
    4. Reindex to the full hourly range.
    5. Interpolate new rows using method='time' (linear, time-aware).

    Ported from: EMA's fix_broken_periodicity_with_interpolation()
    (data_modules/utils.py:44). Logic is identical; parameter made configurable.

    Uses linear (method='time') interpolation — not cubic spline — because
    this fills structural timestamp gaps (1-3 missing hours), not value gaps.
    The distinction: periodicity enforcement adds missing index rows;
    value imputation (later) fills NaN values in existing rows.

    Args:
        df: DataFrame with UTC DatetimeIndex (possibly with small gaps).
        max_gap: Maximum allowed consecutive missing hours.

    Returns:
        DataFrame with complete hourly index, gap rows interpolated.

    Raises:
        ValueError: If any consecutive gap exceeds max_gap.
    """
    ...
```

EMA's original raises on >3 consecutive. We make this configurable via `MAX_MISSING_CONSECUTIVE` from config. The grouping logic uses `diff() != Timedelta(hours=1)` cumsum trick from EMA.

#### 3.3.3 Medium-gap imputation (same-hour-of-day)

```python
def impute_medium_gaps(
    df: pd.DataFrame,
    small_gap_max: int = SMALL_GAP_MAX,
    medium_gap_max: int = MEDIUM_GAP_MAX,
    window_days: int = IMPUTE_WINDOW_DAYS,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Fill medium NaN gaps (6-48h) using same-hour-of-day averaging.

    For each contiguous NaN run between small_gap_max and medium_gap_max
    hours, fills each position with the mean of the same UTC hour from
    the surrounding ±window_days window.

    This respects the daily load/generation profile — a missing Tuesday
    at 14:00 is filled with the average of 14:00 from the surrounding
    Tuesdays and adjacent days, not by interpolating between 13:00 and
    15:00 values (which would lose the profile shape).

    Gaps > medium_gap_max are not filled (logged as structural issues).
    Gaps ≤ small_gap_max are left for interpolate_gaps() in clean().

    Ported from: new. EMA used column-mean imputation (crude);
    EP only interpolated gaps ≤5h. This fills the middle ground.

    Args:
        df: DataFrame with DatetimeIndex.
        small_gap_max: Gaps at or below this size are skipped (handled later by clean()).
        medium_gap_max: Gaps above this size are skipped (structural).
        window_days: Days before/after to average from.
        exclude: Column names/patterns to skip (e.g., regime indicators).

    Returns:
        DataFrame with medium gaps filled.
    """
    ...
```

Implementation approach:
1. For each numeric column (excluding `exclude` list):
2. Identify NaN runs using the cumsum gap-grouping pattern (same as `interpolate_gaps()`).
3. Filter runs where `small_gap_max < size <= medium_gap_max`.
4. For each NaN position in a qualifying run, extract the hour-of-day (`ts.hour`).
5. Compute the mean of that hour from `±window_days` non-NaN values.
6. Fill the NaN with that mean.

For gaps > `medium_gap_max`, log a warning with the column, start timestamp, and gap length.

This function lives in `data/merge.py` rather than `data/processing.py` because it's specific to the merge pipeline's tiered imputation strategy, not a general-purpose cleaning helper.

#### 3.3.4 National SMARD merge

```python
def merge_national_smard(
    df_de_lu: pd.DataFrame,
    df_de_at_lu: pd.DataFrame,
    cutoff: pd.Timestamp = BIDDING_AREA_SPLIT,
) -> pd.DataFrame:
    """Split and concatenate the two national SMARD datasets at the cutoff.

    Takes DE-AT-LU rows before cutoff and DE-LU rows from cutoff onwards.
    Uses outer concat to preserve all columns from both eras — columns
    unique to one era (e.g., Austria cross-border flows only in DE-AT-LU)
    will be NaN in the other.

    Ported from: EP's merge_datasets() (src/data/processing.py:277).
    Logic is identical; parameter names clarified.

    Args:
        df_de_lu: Post-split national SMARD data (loaded from DE-LU.parquet).
        df_de_at_lu: Pre-split national SMARD data (from DE-AT-LU.parquet).
        cutoff: Timestamp where DE-LU data starts being used.

    Returns:
        Merged DataFrame sorted by index, with union of all columns.
    """
    ...
```

Straightforward: `df_de_at_lu[index < cutoff]` + `df_de_lu[index >= cutoff]`, outer concat, sort.

#### 3.3.5 Unified target price

```python
def create_unified_target(
    df: pd.DataFrame,
    ec_fallback: pd.Series | None = None,
) -> pd.DataFrame:
    """Create target_price from post-split and pre-split price columns.

    Priority chain:
    1. Post-split price (DE-LU, filter 4169) — primary for Oct 2018+
    2. Pre-split price (DE-AT-LU, filter 251) — primary for pre-Oct 2018
    3. Energy Charts fallback — fills remaining gaps (SMARD publication lag)

    Uses combine_first() so post-split takes priority where both exist.

    Ported from: EP's create_unified_target() (src/data/processing.py:367).
    Column names updated to match clean_column_name() output (with umlaut fix).

    Args:
        df: Merged DataFrame with both price columns.
        ec_fallback: Optional Energy Charts day-ahead price series for gap-filling.

    Returns:
        DataFrame with target_price column added.
    """
    ...
```

Key implementation detail: uses `df[PRICE_POST_SPLIT].combine_first(df[PRICE_PRE_SPLIT])` which means post-split takes priority where it exists, falls back to pre-split where post-split is NaN.

#### 3.3.6 Energy Charts extension

```python
def extend_with_energy_charts(
    df: pd.DataFrame,
    ec_fallback: pd.Series | None,
) -> pd.DataFrame:
    """Append Energy Charts rows beyond the last SMARD timestamp.

    When SMARD hasn't published data for recent hours but EC already has
    (e.g., today's day-ahead prices from the D-1 noon auction), this
    appends new rows with target_price from EC and NaN for all other
    columns.

    EC data post-Oct 2025 may be quarter-hourly (15-min). Values within
    each hour are averaged to produce the hourly price:
        ec_hourly = ec_beyond.resample("h").mean()

    Ported from: EP's _extend_with_energy_charts() (src/data/processing.py:322).
    Input is now Parquet-based (stage 2 saves EC as Parquet, not CSV).

    Args:
        df: Main dataset with datetime index.
        ec_fallback: Energy Charts price series, or None.

    Returns:
        DataFrame, possibly extended with EC-only rows.
    """
    ...
```

No-op when `ec_fallback` is None or when EC has no data beyond the last SMARD timestamp.

#### 3.3.7 Regime indicators

```python
def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary regime indicator columns.

    - regime_de_at_lu: 1 if index < BIDDING_AREA_SPLIT, else 0
      Marks the old DE-AT-LU bidding area era (pre-Oct 2018).

    - regime_quarter_hourly: 1 if index >= QUARTER_HOURLY_START, else 0
      Marks the new quarter-hourly pricing era (Oct 2025+).

    These are used as model features (regime dummies) to handle
    structural breaks in the target variable.

    Ported from: EP's add_regime_dummies() (src/data/processing.py:407).
    Renamed for clarity (these are indicators, not dummies in the
    statistical sense since there's only one column per regime).

    Args:
        df: DataFrame with UTC DatetimeIndex.

    Returns:
        DataFrame with two integer regime columns added.
    """
    ...
```

#### 3.3.8 Commodity daily construction

```python
def build_commodity_daily(raw_dir: Path) -> pd.DataFrame:
    """Build a daily commodity price DataFrame from raw Parquet files.

    Calls the existing reconstruction/merge functions from data/commodities.py:
    1. reconstruct_ttf(raw_dir) — FRED gap-fill for Dec 2014-Oct 2017
    2. merge_carbon(raw_dir) — ICAP + CO2.L dual-source with bias correction
    3. load brent.parquet directly

    Joins all into a single DataFrame with daily DatetimeIndex (UTC)
    and columns: ttf_eur_per_mwh, carbon_eur_per_ton,
    carbon_realtime_eur_per_ton, brent_usd_per_barrel.

    New function — EP did this as a separate processing step that saved
    to an interim file. We do it in-memory since the data is small
    (~4000 rows x 4 columns) and only needed during merge.

    Args:
        raw_dir: Path to data/raw/commodities/.

    Returns:
        Daily DataFrame with commodity price columns.
    """
    ...
```

Calls `reconstruct_ttf(raw_dir)` and `merge_carbon(raw_dir)` (already implemented in `data/commodities.py`, stage 2), loads `brent.parquet`, joins on the date index.

#### 3.3.9 Commodity merge (daily → hourly)

```python
def merge_commodities(
    df: pd.DataFrame,
    commodity_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Join daily commodity prices onto the hourly dataset via forward-fill.

    Each hourly row gets its calendar day's commodity prices. Alignment:
    1. Normalize hourly timestamps to UTC midnight → get the calendar date
    2. Reindex the daily commodity DataFrame to these dates with ffill
    3. Restore the original hourly index
    4. Left-join onto the main dataset

    Forward-fill means weekends/holidays get Friday's closing price.

    Ported from: EP's merge_commodities() (src/data/processing.py:434).
    Takes a pre-built daily DataFrame instead of a file path.

    Args:
        df: Hourly main dataset.
        commodity_daily: Daily commodity DataFrame from build_commodity_daily().

    Returns:
        DataFrame with commodity columns added.
    """
    ...
```

#### 3.3.10 DST normalization

```python
def normalize_dst(
    df: pd.DataFrame,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Normalize DST transitions to ensure exactly 24 hours per local day.

    The German electricity market operates on local time (CET/CEST).
    Delivery hours are 0-23 in Europe/Berlin. SMARD data is in UTC,
    so DST transitions create 23-hour or 25-hour local days.

    Spring-forward (last Sunday in March, 23 local hours):
    - CET skips 02:00 → 03:00. The UTC hour that maps to the missing
      local hour is identified.
    - A new row is interpolated: numeric columns = (prev + next) / 2,
      non-numeric columns = copy from previous hour.

    Fall-back (last Sunday in October, 25 local hours):
    - CET repeats 02:00. The two UTC hours that map to the same local
      hour are identified.
    - Numeric columns are averaged, non-numeric keep first value.
    - One row is removed to produce exactly 24 local hours.

    After normalization, the index is converted to Europe/Berlin
    (local time). This is correct because:
    - Delivery hours are local-time concepts (hour 10 = 10:00-11:00 CET/CEST)
    - Feature engineering (lags, same-hour, daily aggregations) operates on
      delivery hours, not UTC hours
    - The market structure imposes local time on the problem — the hours
      being predicted shift in UTC terms twice a year

    Ported from: EP's normalize_dst() (src/features/ts_transforms.py:324).
    Logic is identical. EP's output was also local time.

    Args:
        df: DataFrame with UTC DatetimeIndex.
        timezone: Target timezone for delivery hours.

    Returns:
        DataFrame with Europe/Berlin DatetimeIndex, exactly 24 hours/day.
    """
    ...
```

Implementation follows EP's approach exactly:
1. `df.index = df.index.tz_convert(timezone)`
2. Count hours per day via `pd.Series(df.index.date).value_counts()`
3. Spring days (23h): find missing hour, interpolate new row, insert
4. Fall days (25h): find duplicate hour, average numeric, keep first non-numeric, remove duplicate
5. Return with local-time index

#### 3.3.11 NaN validation gate

```python
def validate_no_nans(
    df: pd.DataFrame,
    critical_cols: list[str] | None = None,
) -> None:
    """Post-cleaning validation: log NaN summary and warn on critical columns.

    Does NOT modify the data. Provides a final data quality report
    after all cleaning and imputation steps have run. Distinguishes:
    - Critical columns (target_price, load, major generation): warn if NaN
    - Other columns: info-level log of remaining NaN counts
    - Trailing NaN from SMARD publication delay: expected, not alarming

    Ported from: EMA's NaN gate concept (validate_dataframe() in
    data_modules/utils.py:80). EMA raised errors; we log warnings
    because some trailing NaN is expected.

    Args:
        df: Cleaned DataFrame.
        critical_cols: Column names where NaN is unexpected after cleaning.
            Defaults to target_price, load, and major generation columns.
    """
    ...
```

Default critical columns: `["target_price", "stromverbrauch_gesamt_(netzlast)", "stromerzeugung_wind_onshore", "stromerzeugung_photovoltaik"]`.

#### 3.3.12 Cross-validation (national vs per-TSO)

```python
def cross_validate_national_vs_tso(
    national_path: Path,
    tso_dir: Path,
    mapping: dict[str, str] | None = None,
) -> None:
    """Compare national generation/load totals against sum-of-TSO.

    Loads the merged national dataset and all cleaned per-TSO files,
    sums per-TSO columns, and compares against the equivalent national
    columns. Logs discrepancies as warnings.

    This is a data quality check, not a correction. Discrepancies may
    indicate download timing differences, SMARD reporting gaps, or
    rounding. Significant persistent discrepancies warrant investigation.

    New function — neither EP nor EMA cross-validated national vs per-TSO.

    Args:
        national_path: Path to processed/merged.parquet.
        tso_dir: Path to processed/tso/ containing cleaned per-TSO files.
        mapping: TSO column base name → national column name.
            Defaults to TSO_TO_NATIONAL from config/merge.py.
    """
    ...
```

For each entry in `TSO_TO_NATIONAL`:
1. Load the national column from `merged.parquet`
2. For each TSO, load the corresponding column (e.g., `wind_onshore_50hz`)
3. Sum across TSOs (skipping KNOWN_MISSING combinations which are genuinely zero)
4. Compute MAPE between national and aggregated, and the max absolute difference
5. Log the comparison results

#### 3.3.13 Per-TSO cleaning

```python
def clean_tso_data(
    tso_input_dir: Path,
    tso_output_dir: Path,
) -> None:
    """Clean all per-TSO SMARD Parquet files.

    Applies the same cleaning framework used for national data:
    1. enforce_periodicity() — fill small timestamp gaps
    2. impute_medium_gaps() — same-hour-of-day for 6-48h gaps
    3. interpolate_gaps(method="cubicspline", max_gap=5) — small value gaps

    Replaces EMA's impute_smard_nans() (column-mean imputation), which
    was recognised as overly blunt. The tiered imputation strategy
    respects daily load/generation profiles.

    Args:
        tso_input_dir: Path to data/raw/smard/tso/ (stage 2 output).
        tso_output_dir: Path to data/processed/tso/ (stage 3 output).
    """
    ...
```

Iterates `TSO_REGIONS`, loads each `{TSO}.parquet`, applies the three cleaning steps, saves to output dir.

#### 3.3.14 Pipeline orchestrator

```python
def run_merge_pipeline(
    smard_dir: Path | None = None,
    commodities_dir: Path | None = None,
    ec_dir: Path | None = None,
    output_path: Path | None = None,
    tso_output_dir: Path | None = None,
) -> pd.DataFrame:
    """Execute the full merge pipeline.

    Steps:
     1. Load DE-LU and DE-AT-LU Parquet files
     2. warn_physical_bounds() — pre-cleaning diagnostics
     3. enforce_periodicity() on each dataset
     4. merge_national_smard() at BIDDING_AREA_SPLIT
     5. Load EC fallback → create_unified_target()
     6. extend_with_energy_charts()
     7. add_regime_indicators()
     8. build_commodity_daily() → merge_commodities()
     9. impute_medium_gaps() — fill 6-48h gaps with same-hour-of-day
    10. clean() from config/cleaning.py — structural fills + small gaps
    11. normalize_dst() → Europe/Berlin local time
    12. validate_no_nans() — post-cleaning NaN gate
    13. save_parquet() → data/processed/merged.parquet

    Additionally cleans per-TSO data and cross-validates:
    14. clean_tso_data() → data/processed/tso/*.parquet
    15. cross_validate_national_vs_tso()

    All path arguments default to config constants but are overridable
    for testing.

    Ported from: EP's run_merge_pipeline() (src/data/processing.py:483).
    Differences: no CSV combine step (stage 2 writes Parquet directly),
    adds per-TSO cleaning, adds tiered imputation, adds cross-validation.

    Args:
        smard_dir: Raw SMARD directory. Defaults to config.SMARD_DIR.
        commodities_dir: Raw commodities directory. Defaults to config.COMMODITIES_DIR.
        ec_dir: Energy Charts directory. Defaults to config.ENERGY_CHARTS_DIR.
        output_path: Merged output path. Defaults to PROCESSED_DATA_DIR / "merged.parquet".
        tso_output_dir: Cleaned per-TSO output. Defaults to PROCESSED_DATA_DIR / "tso".

    Returns:
        The final merged DataFrame (also saved to disk).
    """
    ...
```

---

### 3.4 Imputation Strategy

The tiered imputation strategy replaces both EP's single-tier approach (cubicspline, max_gap=5) and EMA's column-mean imputation:

```
Gap size          Method                              Applied in
─────────────────────────────────────────────────────────────────────
≤3 hours          Periodicity: time-based linear      enforce_periodicity()
                  interpolation (fills missing index
                  rows, not NaN values)

≤5 hours          Cubic spline interpolation           clean() → interpolate_gaps()
                  (fills NaN values in existing rows)  [already implemented, stage 1]

6-48 hours        Same-hour-of-day averaging:          impute_medium_gaps()
                  mean of same hour from ±14 days.     [new, stage 3]
                  Respects daily profile shape.

>48 hours         Rejected — logged as structural      impute_medium_gaps() logs warning
                  issue. Not imputed.

Weather data      Linear interpolation (method='time') enforce_periodicity()
(per-TSO only)    for timestamp gaps ≤3h.             [not cubicspline — safer for
                  No value imputation needed.           physical quantities]
```

**Excluded from imputation:** `regime_de_at_lu`, `regime_quarter_hourly`, `target_price`. These are either binary indicators or the prediction target — imputing them would be incorrect.

**Rationale for same-hour-of-day:** Energy data has strong diurnal patterns. A missing Tuesday at 14:00 should look like other recent 14:00 values, not like an interpolation between 13:00 and 15:00. Cubic spline over a 24-hour gap would produce wild oscillations through the daily profile; same-hour averaging preserves the shape.

---

### 3.5 DST Handling

The German electricity market operates on CET/CEST (Europe/Berlin). Delivery hours 0-23 are local-time concepts — "hour 10" means 10:00-11:00 CET in winter and 10:00-11:00 CEST in summer. In UTC terms, hour 10 is 09:00 UTC in winter but 08:00 UTC in summer.

`normalize_dst()` converts the UTC-indexed merged dataset to local time, ensuring exactly 24 rows per local day:

**Spring-forward (last Sunday in March):**
- Local time skips 02:00 → 03:00. SMARD has 23 rows for this local day.
- The missing local hour is detected and a new row is interpolated: numeric values = average of adjacent hours, non-numeric = copy from previous.

**Fall-back (last Sunday in October):**
- Local time repeats 02:00. SMARD has 25 rows for this local day.
- The duplicate local hour is detected and the two rows are averaged (numeric) or deduplicated (non-numeric).

**Output timezone:** Europe/Berlin. This is deliberate — all downstream feature engineering (lags, same-hour-yesterday, daily aggregations) operates on delivery hours. Keeping local time simplifies stage 4.

Ported from EP's `normalize_dst()`. EP also output local time.

---

### 3.6 CLI and Makefile

**`energy_forecasting/cli.py`** — Add a `process` command:

```python
@app.command("process")
def process(
    output: Path = typer.Option(None, help="Output path for merged.parquet"),
):
    """Clean and merge raw data into processed/merged.parquet + processed/tso/."""
    from energy_forecasting.data.merge import run_merge_pipeline

    run_merge_pipeline(output_path=output)
```

This is a top-level command (alongside `download` and `update`), matching the Makefile target structure.

**`Makefile`** — Activate the `process` target:

```makefile
.PHONY: help install lint format test mlflow serve clean data update data-smard data-weather data-commodities process

process:  ## Clean and merge -> processed/merged.parquet + processed/tso/
	energy-forecasting process
```

---

### 3.7 Output Layout

```
data/processed/
├── merged.parquet                 # National dataset for price models
│   ├── Index: DatetimeIndex (Europe/Berlin, hourly, 24 rows/day)
│   ├── ~87,000 rows (2015-01-01 to present)
│   └── Columns:
│       ├── target_price                              # Unified target
│       ├── stromerzeugung_* (12 generation sources)  # Actuals
│       ├── prognostizierte_* (6 forecast columns)    # Published forecasts
│       ├── stromverbrauch_* (load, residual)         # Demand
│       ├── marktpreis_* (14 neighbour prices)        # Cross-border prices
│       ├── cross-border_flows_* (~20 import/export)  # Physical flows
│       ├── ttf_eur_per_mwh                           # Commodities
│       ├── brent_usd_per_barrel
│       ├── carbon_eur_per_ton
│       ├── carbon_realtime_eur_per_ton
│       ├── regime_de_at_lu                           # Binary: 1 pre-2018
│       └── regime_quarter_hourly                     # Binary: 1 post-2025-10
│
└── tso/
    ├── 50Hertz.parquet            # Per-TSO cleaned data for gen/load models
    ├── Amprion.parquet            #   Index: DatetimeIndex (UTC, hourly)
    ├── TenneT.parquet             #   Columns: wind_offshore_50hz, wind_onshore_50hz,
    ├── TransnetBW.parquet         #     solar_50hz, load_50hz, biomass_50hz, etc.
    └── Creos.parquet              #   (minus KNOWN_MISSING combinations)
```

**Why local time for merged.parquet but UTC for per-TSO:**
- merged.parquet feeds the price model which predicts for local delivery hours
- Per-TSO feeds gen/load models which will have their own feature engineering (stage 4) that may make different timezone choices
- Per-TSO weather data (stage 2) is already in UTC; keeping per-TSO SMARD in UTC avoids mixed-timezone joins in stage 4

---

### 3.8 Milestone & Tests

**Tests** (`tests/`):

- **`test_merge.py`** — Unit tests for all merge pipeline functions:

  - `test_enforce_periodicity_no_gaps()` — complete hourly index passes through unchanged
  - `test_enforce_periodicity_fills_small_gap()` — 2-hour gap is filled by time-based interpolation
  - `test_enforce_periodicity_rejects_large_gap()` — 4+ hour gap raises ValueError
  - `test_impute_medium_gaps_fills_day_gap()` — 24h gap filled with correct hour-of-day values
  - `test_impute_medium_gaps_skips_small_gaps()` — gaps ≤5h are not touched (left for clean())
  - `test_impute_medium_gaps_skips_large_gaps()` — gaps >48h are not touched (logged only)
  - `test_impute_medium_gaps_excludes_columns()` — regime indicators not imputed
  - `test_merge_national_smard_splits_at_cutoff()` — pre-cutoff rows from DE-AT-LU, post from DE-LU
  - `test_merge_national_smard_outer_columns()` — columns unique to one era appear with NaN in the other
  - `test_create_unified_target_post_priority()` — post-split price takes priority
  - `test_create_unified_target_falls_back_to_pre()` — NaN in post-split filled from pre-split
  - `test_create_unified_target_ec_fallback()` — remaining NaN filled from Energy Charts
  - `test_extend_appends_beyond_smard()` — EC rows beyond SMARD's last timestamp are appended
  - `test_extend_resamples_quarter_hourly()` — 15-min EC data averaged to hourly
  - `test_extend_noop_when_no_ec()` — returns unchanged when ec_fallback is None
  - `test_add_regime_indicators_de_at_lu()` — 1 before split, 0 after
  - `test_add_regime_indicators_quarter_hourly()` — 0 before 2025-10-01, 1 after
  - `test_merge_commodities_forward_fills()` — daily prices forward-filled to hourly
  - `test_merge_commodities_aligns_utc_midnight()` — each hour gets its calendar day's price
  - `test_normalize_dst_spring_forward()` — 23h day gets interpolated missing hour
  - `test_normalize_dst_fall_back()` — 25h day gets averaged duplicate hour
  - `test_normalize_dst_output_is_local_time()` — index timezone is Europe/Berlin
  - `test_normalize_dst_normal_day()` — 24h days pass through unchanged
  - `test_validate_no_nans_warns_critical()` — warns when critical columns have NaN
  - `test_validate_no_nans_passes_clean()` — no warnings when data is clean
  - `test_warn_physical_bounds_logs_violations()` — out-of-range values trigger log warnings
  - `test_run_merge_pipeline_smoke(tmp_path)` — synthetic Parquets → valid output with expected columns, continuous index, correct regimes

- **`test_config_merge.py`** — Sanity checks on merge configuration:
  - `test_bidding_area_split_is_utc()` — timestamp is timezone-aware UTC
  - `test_price_columns_exist_in_smard_column_names()` — PRICE_POST_SPLIT and PRICE_PRE_SPLIT are valid SMARD column names
  - `test_tso_to_national_keys_match_tso_filter_keys()` — all TSO_TO_NATIONAL keys are in TSO_FILTER_KEYS values
  - `test_tso_to_national_values_in_smard_column_names()` — all national column names are valid

- **Existing tests updated:**
  - `test_cleaning_rules.py` — add a test that `clean_column_name("Marktpreis: Österreich")` returns `"marktpreis_oesterreich"` (verifying umlaut fix)

**Milestone checklist:**

- [ ] `clean_column_name()` transliterates umlauts (existing tests still pass)
- [ ] `make process` runs end-to-end with real data
- [ ] `data/processed/merged.parquet` exists with:
  - [ ] Continuous hourly Europe/Berlin index (24 rows per day, no gaps)
  - [ ] `target_price` column with no NaN (except possibly trailing hours)
  - [ ] Regime indicators correct: `regime_de_at_lu=1` before 2018-09-30 22:00 UTC
  - [ ] Commodity columns present with forward-filled daily values
  - [ ] No unexpected NaN in generation/load columns after cleaning
- [ ] `data/processed/tso/*.parquet` exist with:
  - [ ] Continuous hourly UTC index
  - [ ] No NaN after imputation
  - [ ] Column names match TSO suffix convention (e.g., `wind_onshore_50hz`)
- [ ] Cross-validation log shows national vs per-TSO totals within reasonable tolerance
- [ ] Physical bounds warnings logged for any suspicious values
- [ ] `make test` passes all stage 3 tests
- [ ] `make lint` passes

**Stage-gate verification** (from master plan):
- merged.parquet `target_price` matches EP's `merged_dataset_hourly.parquet` target_price within floating-point tolerance (date range may differ slightly due to timing)
- Generation and load columns match EP's values in the overlapping date range
- Commodity columns match EP's `commodity_prices_daily.parquet` after forward-fill alignment

---

### Implementation Notes

**Estimated complexity by component:**
- `config/merge.py` — straightforward constants (~50 lines)
- `config/columns.py` umlaut fix — 3 lines added to `clean_column_name()`
- `data/merge.py` pipeline functions — **main work** (~300 lines). Most functions are near-direct ports from EP. The new functions (`impute_medium_gaps`, `warn_physical_bounds`, `validate_no_nans`, `clean_tso_data`, `cross_validate_national_vs_tso`) are moderate complexity.
- `cli.py` — 5-line addition
- `Makefile` — 2-line change
- `tests/test_merge.py` — ~27 tests with synthetic fixtures (~300 lines)

**What's already done (no work needed):**
- All cleaning helpers in `data/processing.py` (10 functions, 186 lines)
- Cleaning orchestrator in `config/cleaning.py` (136 lines)
- `reconstruct_ttf()` and `merge_carbon()` in `data/commodities.py`
- `load_parquet()` and `save_parquet()` in `data/io.py`
- 13 existing cleaning rule tests in `test_cleaning_rules.py`

**Potential blockers:**
- If no data has been downloaded yet (stage 2 not run), `make process` will fail gracefully with a "file not found" message. The pipeline requires stage 2 outputs.
- DST normalization inserts/removes rows, changing the DataFrame length. If any downstream code assumes a fixed row count, this will break. (Not expected to be an issue — feature engineering handles variable-length DataFrames.)
- The same-hour-of-day imputation for medium gaps may produce poor results at series boundaries (start of dataset, where the ±14-day window is truncated). The function should fall back to using whatever days are available.

---
