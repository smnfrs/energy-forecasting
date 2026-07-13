# MLflow Conventions

Operational guidelines for MLflow usage in the energy-forecasting project. Referenced from the master plan (Stage 5.1).

**Source:** Distilled from `docs/archive/EP_EMA_merge_plan.md` Section 5 and `docs/archive/merge_decisions.md` decisions #2 and #5.

---

## Core Rule

**Runs within an experiment must be comparable** — same holdout period, same CV strategy, same features (unless features are the thing being tested). This is decision #2 from the merge plan.

---

## Experiment Structure

| Experiment | Purpose | What varies | What's held constant |
|---|---|---|---|
| `price/feature_selection` | Test different feature lists | Feature set | Model (e.g., LightGBM defaults), holdout, CV |
| `price/model_training` | Joint preprocessing + hyperparam search | Hyperparameters + preprocessing (scaler, target transform, sample weights) | Feature set, holdout, CV |
| `price/production` | Final trained models for blending | Nothing — these are production | Everything fixed |
| `generation/wind_onshore` | Per-target EMA experiments | Features + model jointly | CV strategy |
| `generation/wind_offshore` | (same) | | |
| `generation/solar` | (same) | | |
| `generation/load` | (same) | | |

**Note on feature sets in `price/model_training`:** Run separate tuning studies per feature set so runs remain comparable within each study. Tag with `feature_version` to distinguish. Each model type may end up preferring a different feature set.

---

## When to Create a New Experiment vs a New Run

- **New run:** changing model hyperparameters, trying a different model class, adjusting preprocessing (scaler, target transform, sample weights). These are jointly searchable within the same experiment.
- **New experiment:** changing the feature set, changing the CV strategy, changing the holdout period, switching from daily to hourly architecture. These are structural changes that make runs non-comparable.

---

## Required Tags

Implement `TrackedRun` context manager that validates required tags before closing:

```python
mlflow.set_tags({
    "stage": "feature_selection",       # or "hyperparam_tuning", "production"
    "feature_version": "v5",            # which feature list was used
    "feature_contract": "forecast_v1",  # data/feature contract epoch
    "holdout_days": "90",
    "cv_folds": "5",
    "cv_mode": "expanding",             # or "sliding"
    "target_transform": "log_shift",
})
```

Filter in UI: `tags.stage = "production"` to see all production runs, `tags.feature_version = "v5"` for all v5 runs, and `tags.feature_contract = "forecast_v1"` for the post-forecast-fix epoch.

---

## Model Naming Convention

`{category}_{feature_version}` for model registry names (e.g., `lgbm_v5`, `xgboost_v5`). Groups related versions and makes the registry navigable.

---

## Run Lifecycle

1. **Active** — current experiment, results being evaluated.
2. **Archived** — superseded by newer experiments, tagged `archived=true` with a reason. Still queryable but excluded from default views.
3. **Deleted** — clearly broken runs (crashed, wrong data, bugs). Delete rather than archive.

---

## Dataset Tracking

Datasets are **not** runs or experiments. They are tracked via MLflow's built-in dataset API and appear in each run's Datasets tab.

**How it works:**
1. `prepare_dataset()` computes features and saves to `data/processed/datasets/{name}.parquet`. No MLflow involvement at this point.
2. Inside `train_model()`, the training code registers the dataset and logs it as an input:
   ```python
   dataset = mlflow.data.from_pandas(X, source=dataset_path, name=dataset_name)
   mlflow.log_input(dataset, context="training")
   ```
3. MLflow computes a content hash automatically — if two runs use identical data, MLflow knows.
4. The dataset appears in the run's **Datasets tab** in the UI with schema, source path, and hash.

**What NOT to do** (mistakes from EP):
- Don't track datasets as MLflow models — wrong abstraction.
- Don't create a `datasets` experiment with dataset-as-runs — datasets aren't experiments.
- Don't track datasets via ad-hoc tags — use the built-in API.

---

## Helper Functions

```python
def audit_experiment(experiment_name: str) -> pd.DataFrame:
    """Flag runs with missing tags, inconsistent holdout/CV/features.

    Checks: do all runs have the same holdout_days? Same cv_folds?
    Same feature_version (in non-feature-selection experiments)?
    Flags outliers for review. Returns DataFrame of flagged runs with reasons.
    """

def archive_runs(run_ids: list[str], reason: str = "superseded"):
    """Tag runs as archived with a reason. Excludes from default queries
    without deleting data. Reason is stored in tag 'archive_reason'."""

def get_best_run(experiment_name: str, metric: str = "mae",
                 stage: str = None, exclude_archived: bool = True) -> dict:
    """Return the best run, optionally filtered by stage tag.
    Excludes archived runs by default."""

def compare_feature_sets(experiment_name: str) -> pd.DataFrame:
    """Compare all feature-set experiments.
    Columns: feature_version, n_features, mae, rmse, r2, n_runs.
    Only meaningful for feature_selection experiments."""

def compare_models(experiment_name: str, metric: str = "mae") -> pd.DataFrame:
    """Side-by-side comparison of model types within an experiment.
    Shows per-model-class best/mean/std of the target metric."""

def cleanup_orphaned_artifacts():
    """Find MLflow artifacts (models, datasets) not referenced by any
    active or archived run. List for manual review before deletion."""

def export_experiment_summary(experiment_name: str, path: str):
    """Export a self-contained summary of an experiment to markdown.
    Includes: purpose, best run, all runs table, tags, notes.
    Useful for documenting completed experiment rounds."""
```

`audit_experiment` is the most important — run it before any selection step to catch inconsistencies.

---

## Workflow for a Typical Experiment Round

1. Create experiment with descriptive name.
2. Run experiments, tagging each run consistently.
3. Run `audit_experiment()` to check consistency.
4. Run `compare_feature_sets()` or `compare_models()` to pick the winner.
5. Export summary with `export_experiment_summary()`.
6. Archive superseded runs with `archive_runs()`.
7. Promote winner to the next stage.
