# Energy Forecasting

Merging two repos — `energy_prices` (EP) and `energy_market_analysis` (EMA) — into a single codebase for day-ahead electricity price and generation/load forecasting for the German energy market.

## Plans

- **`docs/master_plan.md`** — the single source of truth. High-level stages 1-8 with milestones and evaluation templates filled in after each stage.
- **`docs/stage{N}_*.md`** — detailed implementation plan for each stage. Written before implementation begins, lives alongside the code as permanent documentation. Do NOT use Claude Code's plan mode — write plans directly into `docs/`.
- **`docs/source_repo_guide.md`** — where to find things in the two source repos (`~/projects/energy_prices/` and `~/projects/energy_market_analysis/`). Use this when porting code.
- **`docs/mlflow_conventions.md`** — MLflow experiment structure, tagging rules, helper function specs.
- **`docs/archive/`** — earlier plans and analysis that produced the master plan. Reference only.

## Planning workflow

Stage plans go in `docs/stage{N}_*.md`, not in Claude Code's ephemeral plan mode. This keeps plans:
- Visible in the repo and version-controlled
- Readable across sessions without needing to recover context
- Reviewable by the user outside of Claude Code

When starting a new stage: read the master plan section, explore source repos, write the detailed plan to `docs/`, get user approval, then implement.

## Status

Stages 1-4 complete. Stage 5a (Training Infrastructure) complete. **Stage 5b (Gen/Load Models) complete (2026-05-04)** — full 70-trial training, 48 base + 16 ensembles, 0 failures. **Phase A extension (2026-05-07)** retrained at expanded EMA-aligned FE search spaces and bumped `GEN_LOAD_HISTORICAL_FOLDS` 40 → 218, giving ~4.18 years of leak-free historical_forecasts (2022-01-15 → 2026-03-27, 36,788 rows per file) saved per (target, region) plus DE_NATIONAL aggregates. Training order enforced by CLI: wind/solar → load (uses wind/solar actuals) → gen_load_diff (uses all). **Stage 5c (Price Models) complete (2026-06-30)** — production ensemble holdout MAE **11.148** (5 models: LGBM 34.6%, XGB×2 38.5%, CatBoost 18.4%, Ridge 8.4%). LGBM root-caused to `num_leaves=31` cap + MAE objective zero-hessian; fixed by scaling `num_leaves = 2^max_depth − 1`. Diversity experiment confirmed LGBMQuantile and Huber earn zero ensemble weight — both removed from pipeline. All 8 review items closed (see `docs/stage5c_status_2026-06-06.md`). Pre-requisite `_eh10` → `_eh7` fix applied 2026-04-10. Backlog of model improvements lives in `docs/roadmap.md`. **Stage 5d (Feature Selection Analysis) not started.** **Stage 6 (Inference, API & CI/CD) complete (2026-06-30)** — full inference pipeline (gen/load wave-by-wave → price → conformal PI), stateless FastAPI (7 endpoints), daily GitHub Actions workflow (08:00 UTC), monthly price retrain workflow, `deploy/` CLI sub-group. 490/495 tests pass (5 pre-existing). **Stage 7 (Dashboard) complete (2026-06-30)** — static HTML/JS dashboard served from `deploy/` with 6 chart panels (price forecast + PI, national gen/load, history, per-source bars, error bands), DE/EN toggle, monitoring page. Pipeline bug fixes landed alongside Stage 7: Makefile conda prefix, SMARD API URL (smard.api.proxy.bund.dev → www.smard.de), NaN gate in merge.py, broken `build_merged_dataset` import, Ridge pre-scale scaler export+inference, solar night-hour lag-contamination clamp (elevation mask), SOLAR_NIGHT_HOURS CET/CEST correction. First clean end-to-end run 2026-06-30: price mean 131 EUR/MWh, all 7 JSON outputs written. Next: Stage 8 (Monitoring & Alerting) or Stage 5d (Feature Selection Analysis).

Future training runs can use `--parallel N` (default 1 = sequential) for ~2-3x speedup on the tower (16 cores). Long-running training must be launched detached per `~/.claude/CLAUDE.md` § Long-Running Processes.
