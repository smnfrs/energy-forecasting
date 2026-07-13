"""MLflow helpers: TrackedRun context manager, audit, compare, archive.

TrackedRun validates required tags at __enter__ — fails BEFORE any training compute.
Auto-sets tags that can be derived from arguments.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from loguru import logger

from energy_forecasting.config import MLFLOW_TRACKING_URI
from energy_forecasting.config.modeling import EXPERIMENTS

# Tags set automatically — never provided manually
_AUTO_TAGS = {
    "dataset_name",
    "cv_mode",
    "cv_folds",
    "holdout_days",
    "n_features",
    "n_train_rows",
    "model_class",
}

# Tags that must be provided by the caller, with allowed values.
# Other ad-hoc tags are allowed but these must be present and valid.
_REQUIRED_TAGS = {
    "stage": {"feature_selection", "model_training", "production"},
    "feature_version": None,  # any non-empty string
    "feature_contract": None,  # comparability boundary for feature/data contracts
}


def ensure_mlflow_tracking() -> None:
    """Use the repo-local MLflow backend consistently."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class TrackedRun:
    """Context manager for MLflow runs with tag validation.

    Validates required tags at __enter__ — fails BEFORE any training compute.
    Auto-sets tags that can be derived from arguments (dataset_name, etc.).

    Usage::

        with TrackedRun("price/model_training", stage="model_training",
                        feature_version="shap_top40",
                        feature_contract="forecast_v1") as run:
            # ... training code ...
            mlflow.log_metrics({"mae": 9.93})
    """

    def __init__(self, experiment: str, *, dataset_name: str | None = None, **tags: str):
        self.experiment = experiment
        self.dataset_name = dataset_name
        self.user_tags = tags
        self._run = None

    def __enter__(self):
        ensure_mlflow_tracking()

        # Validate experiment name is in registry (strict — catches typos)
        if self.experiment not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{self.experiment}'. "
                f"Must be one of: {sorted(EXPERIMENTS.keys())}"
            )
        experiment_path = EXPERIMENTS[self.experiment]

        # Validate required tags are present
        missing = set(_REQUIRED_TAGS) - set(self.user_tags)
        if missing:
            raise ValueError(
                f"TrackedRun missing required tags: {missing}. "
                f"Required: {sorted(_REQUIRED_TAGS.keys())}"
            )

        # Validate required tag values
        for tag, allowed in _REQUIRED_TAGS.items():
            value = self.user_tags[tag]
            if not value or not str(value).strip():
                raise ValueError(f"Tag '{tag}' must be a non-empty string")
            if allowed is not None and value not in allowed:
                raise ValueError(
                    f"Tag '{tag}' has invalid value '{value}'. Allowed: {sorted(allowed)}"
                )

        # Set or create experiment
        mlflow.set_experiment(experiment_path)

        # Build tag dict
        tags = dict(self.user_tags)
        if self.dataset_name:
            tags["dataset_name"] = self.dataset_name

        # Start run
        self._run = mlflow.start_run(tags=tags)
        logger.info(
            f"Started MLflow run {self._run.info.run_id} in experiment '{experiment_path}'"
        )
        return self._run

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._run is not None:
            if exc_type is not None:
                mlflow.set_tag("run_status", "failed")
                mlflow.set_tag("error", str(exc_val)[:250])
            mlflow.end_run()
        return False  # don't suppress exceptions

    def set_auto_tags(
        self,
        *,
        cv_mode: str | None = None,
        cv_folds: int | None = None,
        holdout_days: int | None = None,
        n_features: int | None = None,
        n_train_rows: int | None = None,
        model_class: str | None = None,
    ):
        """Set auto-derived tags after run is started."""
        auto = {
            "cv_mode": cv_mode,
            "cv_folds": str(cv_folds) if cv_folds is not None else None,
            "holdout_days": str(holdout_days) if holdout_days is not None else None,
            "n_features": str(n_features) if n_features is not None else None,
            "n_train_rows": str(n_train_rows) if n_train_rows is not None else None,
            "model_class": model_class,
        }
        for k, v in auto.items():
            if v is not None:
                mlflow.set_tag(k, v)


def get_best_run(
    experiment_name: str,
    metric: str = "mae",
    stage: str | None = None,
    target: str | None = None,
    region: str | None = None,
    exclude_archived: bool = True,
) -> dict | None:
    """Return the best run from an experiment, optionally filtered by tags.

    Returns dict with keys: run_id, metrics, params, tags.
    Returns None if no matching runs found.
    """
    ensure_mlflow_tracking()
    experiment_path = EXPERIMENTS.get(experiment_name, experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment is None:
        return None

    filter_parts = []
    if stage:
        filter_parts.append(f"tags.stage = '{stage}'")
    if target:
        filter_parts.append(f"tags.target = '{target}'")
    if region:
        filter_parts.append(f"tags.region = '{region}'")
    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    # Note: MLflow's `tags.archived != 'true'` filter excludes runs that don't
    # have the `archived` tag at all, so we filter in Python after the search.
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=[f"metrics.{metric} ASC"],
    )
    if runs.empty:
        return None

    if exclude_archived and "tags.archived" in runs.columns:
        runs = runs[runs["tags.archived"] != "true"]
    if runs.empty:
        return None

    row = runs.iloc[0]
    return {
        "run_id": row["run_id"],
        "metrics": {
            k.removeprefix("metrics."): v
            for k, v in row.items()
            if k.startswith("metrics.") and pd.notna(v)
        },
        "params": {
            k.removeprefix("params."): v
            for k, v in row.items()
            if k.startswith("params.") and pd.notna(v)
        },
        "tags": {
            k.removeprefix("tags."): v
            for k, v in row.items()
            if k.startswith("tags.") and pd.notna(v)
        },
    }


def compare_models(
    experiment_name: str,
    metric: str = "mae",
    exclude_archived: bool = True,
) -> pd.DataFrame:
    """Side-by-side comparison of model types within an experiment.

    Returns DataFrame with columns: model_class, best_{metric}, mean_{metric},
    std_{metric}, n_runs.
    """
    ensure_mlflow_tracking()
    experiment_path = EXPERIMENTS.get(experiment_name, experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment is None:
        return pd.DataFrame()

    # Note: MLflow's `tags.archived != 'true'` filter excludes runs that don't
    # have the `archived` tag at all, so we filter in Python after the search.
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        return pd.DataFrame()

    if exclude_archived and "tags.archived" in runs.columns:
        runs = runs[runs["tags.archived"] != "true"]
    if runs.empty:
        return pd.DataFrame()

    metric_col = f"metrics.{metric}"
    model_col = "tags.model_class"
    if metric_col not in runs.columns or model_col not in runs.columns:
        return pd.DataFrame()

    grouped = runs.groupby(model_col)[metric_col].agg(["min", "mean", "std", "count"])
    grouped.columns = [f"best_{metric}", f"mean_{metric}", f"std_{metric}", "n_runs"]
    grouped.index.name = "model_class"
    return grouped.sort_values(f"best_{metric}").reset_index()


def audit_experiment(experiment_name: str) -> pd.DataFrame:
    """Flag runs with missing tags or inconsistent holdout/CV/features.

    Returns DataFrame of flagged runs with reasons. Empty = all consistent.
    """
    ensure_mlflow_tracking()
    experiment_path = EXPERIMENTS.get(experiment_name, experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment is None:
        return pd.DataFrame(columns=["run_id", "issue"])

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        return pd.DataFrame(columns=["run_id", "issue"])

    issues = []
    for tag in _REQUIRED_TAGS:
        col = f"tags.{tag}"
        if col in runs.columns:
            missing = runs[runs[col].isna()]
            for run_id in missing["run_id"]:
                issues.append({"run_id": run_id, "issue": f"missing tag: {tag}"})

    # Check consistency of holdout_days and cv_folds within the experiment
    for tag in ["holdout_days", "cv_folds"]:
        col = f"tags.{tag}"
        if col in runs.columns:
            values = runs[col].dropna().unique()
            if len(values) > 1:
                for run_id in runs["run_id"]:
                    issues.append(
                        {
                            "run_id": run_id,
                            "issue": f"inconsistent {tag}: {sorted(values)}",
                        }
                    )

    return pd.DataFrame(issues)


def archive_runs(run_ids: list[str], reason: str = "superseded"):
    """Tag runs as archived with a reason."""
    ensure_mlflow_tracking()
    client = mlflow.MlflowClient()
    for run_id in run_ids:
        client.set_tag(run_id, "archived", "true")
        client.set_tag(run_id, "archive_reason", reason)
        logger.info(f"Archived run {run_id}: {reason}")
