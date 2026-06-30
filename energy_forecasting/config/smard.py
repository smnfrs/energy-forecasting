"""SMARD API region, TSO, and filter key configuration.

Column name mappings (filter key -> column name) are in config/columns.py.
This module handles the API-specific configuration: which regions exist,
which filter keys are valid per region, and known-missing combinations.
"""

# ── Resolution map ──────────────────────────────────────────────────
# SMARD API resolution string -> expected records per day
RESOLUTION_PERIODS: dict[str, int] = {
    "quarterhour": 96,
    "hour": 24,
}

# ── National region codes ───────────────────────────────────────────
# Used in SMARD API URLs. EP downloads from these.
NATIONAL_REGIONS: list[str] = [
    "DE-LU",  # Current bidding zone (Oct 2018+)
    "DE-AT-LU",  # Historical (pre-Oct 2018)
]

# ── TSO region codes ────────────────────────────────────────────────
# Per-TSO SMARD data. EMA downloads from these for gen/load models.
TSO_REGIONS: list[str] = [
    "50Hertz",
    "Amprion",
    "TenneT",
    "TransnetBW",
    "Creos",
]

# TSO suffix for column names (from EMA's eu_locations)
TSO_SUFFIXES: dict[str, str] = {
    "50Hertz": "_50hz",
    "Amprion": "_ampr",
    "TenneT": "_tenn",
    "TransnetBW": "_tran",
    "Creos": "_lu",
}

# ── Per-TSO filter keys ─────────────────────────────────────────────
# These are the generation/load filters available at TSO level.
# Ported from EMA's collect_data_smard_v2.py.
TSO_FILTER_KEYS: dict[int, str] = {
    1225: "wind_offshore",
    4067: "wind_onshore",
    4068: "solar",
    410: "load",
    4066: "biomass",
    4071: "gas",
    4069: "hard_coal",
    1223: "lignite",
    4070: "pumped_storage",
    1226: "hydro",
    1227: "other_conv",
    1228: "other_renew",
}

# ── Known missing combinations ──────────────────────────────────────
# (filter_id, region) pairs that return 404 from the API.
# Ported from EMA's KNOWN_MISSING.
KNOWN_MISSING: set[tuple[int, str]] = {
    # No offshore wind in landlocked/partial TSOs
    (1225, "Amprion"),
    (1225, "TransnetBW"),
    (1225, "Creos"),
    # No lignite in some TSOs
    (1223, "TransnetBW"),
    (1223, "Creos"),
    # No hard coal in Creos
    (4069, "Creos"),
    # No pumped storage in Creos
    (4070, "Creos"),
    # No other renewables in Creos
    (1228, "Creos"),
}

# ── SMARD API base URL ──────────────────────────────────────────────
SMARD_API_BASE = "https://smard.api.proxy.bund.dev/app/chart_data"

# ── Default download parameters ─────────────────────────────────────
DEFAULT_RESOLUTION = "hour"
DEFAULT_REDUNDANCY_DAYS = 14  # Overlap for incremental updates
TSO_REDUNDANCY_HOURS = 72  # EMA's overlap for per-TSO updates
BOOTSTRAP_DAYS = 45  # Days for bootstrapping new keys
MAX_SMARD_WORKERS = 8  # ThreadPoolExecutor parallelism for SMARD API calls
HTTP_TIMEOUT_SECONDS = 30  # Default timeout for SMARD API requests
