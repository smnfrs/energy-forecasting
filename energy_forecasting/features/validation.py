"""Leakage validation for feature specifications.

Checks that every feature's suffix implies a lag that respects
the data source's publication delay (from config/availability.py).

Works on parsed FeatureSpec objects — no pipeline introspection needed.
"""

import fnmatch
from dataclasses import dataclass

from energy_forecasting.config.availability import AVAILABILITY_RULES, AvailabilityRule
from energy_forecasting.features.parser import (
    FeatureSpec,
    InteractionSpec,
    parse_feature,
)


@dataclass
class ValidationError:
    feature_str: str
    rule: AvailabilityRule
    reason: str


def _match_rule(short_name: str) -> AvailabilityRule | None:
    """Find the availability rule that matches a short name."""
    for rule in AVAILABILITY_RULES:
        if fnmatch.fnmatch(short_name, rule.pattern):
            return rule
    return None


def _check_spec(spec: FeatureSpec) -> list[str]:
    """Validate a single FeatureSpec against availability rules.

    Returns list of error messages (empty if valid).
    """
    errors: list[str] = []
    rule = _match_rule(spec.base)

    if rule is None:
        errors.append(
            f"No availability rule matches '{spec.base}'. "
            f"Add a rule to config/availability.py or this feature cannot be validated."
        )
        return errors

    offset = rule.max_offset_days  # 0 = today, -1 = yesterday, -2 = two days ago
    cutoff_hour = rule.cutoff_hour  # hour by which data is published

    # Deterministic features (offset=0) — no lag needed
    if offset == 0:
        return errors

    # --- Bare name (no suffix) → always leaks for non-deterministic data ---
    has_suffix = (
        spec.lag is not None
        or spec.agg is not None
        or spec.ewma is not None
        or spec.fourier is not None
        or spec.daily_agg is not None
    )
    if not has_suffix:
        errors.append(
            f"Bare column name '{spec.base}' for non-deterministic data "
            f"(offset={offset}). Must have explicit lag/agg suffix."
        )
        return errors

    # --- Hourly lag ---
    if spec.lag is not None:
        min_lag_hours = -offset * 24
        if cutoff_hour is not None:
            # Data is only available up to cutoff_hour on the offset day.
            # For prediction at hour H on day D, data from D+offset is available
            # only up to cutoff_hour. At H=0, need at least |offset|*24 hours of lag.
            # At H=23, need |offset|*24 - 23 hours. But we must be safe for ALL hours,
            # so minimum lag is (|offset|-1)*24 + cutoff_hour for the worst case (H=cutoff_hour).
            # Simpler: lag must be >= (|offset|-1)*24 + (24 - cutoff_hour)
            # Actually: for any hour H, data at time (D+offset, cutoff_hour) is
            # H_lag hours back where H_lag = (D - (D+offset))*24 + (H - cutoff_hour)
            # = |offset|*24 + (H - cutoff_hour). Minimum when H=0:
            # lag_available = |offset|*24 - cutoff_hour.
            # We need spec.lag.hours >= that minimum.
            min_lag_hours = -offset * 24 - cutoff_hour
        if spec.lag.hours < min_lag_hours:
            errors.append(
                f"Hourly lag h{spec.lag.hours} insufficient for '{spec.base}' "
                f"(min={min_lag_hours}h, offset={offset}, cutoff={cutoff_hour})"
            )

    # --- Aggregation ---
    if spec.agg is not None:
        agg = spec.agg
        # end_day must respect offset (most recent day in window)
        # agg.end_day is negative (e.g. -1 = yesterday)
        if agg.end_day > offset:
            errors.append(
                f"Aggregation end_day={agg.end_day} too recent for '{spec.base}' "
                f"(max_offset={offset})"
            )
        # If end_day equals offset and there's a cutoff_hour, check end_hour
        if agg.end_day == offset and cutoff_hour is not None:
            if agg.end_hour is not None and agg.end_hour > cutoff_hour:
                errors.append(
                    f"Aggregation end_hour={agg.end_hour} exceeds cutoff_hour={cutoff_hour} "
                    f"for '{spec.base}' on day offset={offset}"
                )
            elif agg.end_hour is None:
                # Using full day at offset, but data only available up to cutoff_hour
                errors.append(
                    f"Aggregation uses full day at end_day={agg.end_day} but '{spec.base}' "
                    f"is only available up to hour {cutoff_hour} (need _eh{cutoff_hour} or earlier end_day)"
                )

    # --- EWMA ---
    if spec.ewma is not None:
        ewma = spec.ewma
        if ewma.cutoff_day is None:
            errors.append(
                f"EWMA without cutoff_day for '{spec.base}' (offset={offset}). "
                f"Must specify information boundary."
            )
        elif ewma.cutoff_day > offset:
            errors.append(
                f"EWMA cutoff_day={ewma.cutoff_day} too recent for '{spec.base}' "
                f"(max_offset={offset})"
            )
        elif ewma.cutoff_day == offset and cutoff_hour is not None:
            if ewma.cutoff_hour is not None and ewma.cutoff_hour > cutoff_hour:
                errors.append(
                    f"EWMA cutoff_hour={ewma.cutoff_hour} exceeds availability "
                    f"cutoff_hour={cutoff_hour} for '{spec.base}'"
                )

    # --- Daily aggregate (current-day broadcast) ---
    if spec.daily_agg is not None:
        # _daily_stat computes stat over all hours of the current day.
        # Only valid for data available today (offset=0).
        if offset < 0:
            errors.append(
                f"Daily aggregate on '{spec.base}' uses current-day data but "
                f"offset={offset} (data not available today)"
            )

    return errors


def validate_features(feature_strings: list[str]) -> list[ValidationError]:
    """Validate all features in a feature list against availability rules.

    Returns list of ValidationErrors. Empty list means all features are valid.
    """
    errors: list[ValidationError] = []

    for feat_str in feature_strings:
        spec = parse_feature(feat_str)

        if isinstance(spec, InteractionSpec):
            # Validate both sides
            for sub_spec, label in [(spec.left, "left"), (spec.right, "right")]:
                sub_errors = _check_spec(sub_spec)
                rule = _match_rule(sub_spec.base)
                for msg in sub_errors:
                    errors.append(
                        ValidationError(
                            feature_str=feat_str,
                            rule=rule,
                            reason=f"[{label}] {msg}",
                        )
                    )
        else:
            spec_errors = _check_spec(spec)
            rule = _match_rule(spec.base)
            for msg in spec_errors:
                errors.append(ValidationError(feature_str=feat_str, rule=rule, reason=msg))

    return errors
