# Energy Forecasting

Merging two repos — `energy_prices` (EP) and `energy_market_analysis` (EMA) — into a single codebase for day-ahead electricity price and generation/load forecasting for the German energy market.

## Domain gotchas

**`gen_load_diff` = total_generation − total_load** (`modeling/gen_load.py`), i.e. the **net balance = net exports + grid losses**. It is a SMALL, ± quantity (recently ≈ −2.6 GW, often a slight net import), *not* "everything apart from renewables". Confirmed by `forecast_inputs.py`: `forecast_gen_total = forecast_load + gen_load_diff`. This has caused repeated bugs — people assume it's the ~24 GW residual. It is not:
- **residual load** = load − renewables ≈ 24 GW (large, positive)
- **non-renewable generation** = total_gen − renewables ≈ 21 GW (large, positive)
- **gen_load_diff** = gen − load ≈ −2.6 GW (small, ±)

On the dashboard, the "Other Generation" band is **total_gen − renewables** — actual = summed non-renewable generation types from the TSO parquets; forecast = `(load + gen_load_diff) − renewables`. `renewables + other ≈ load` but off by the net balance (never exactly equal). Do NOT stack raw `gen_load_diff` as the other-generation band.

## Git

Write commit messages the way a person would: a plain, capitalised sentence describing what the commit does, e.g. "Created new story dashboard site" or "Template the forecast-story narrative and retire the LLM path". Do **not** use Conventional Commits prefixes (`fix(...)`, `feat(...)`, `ci(...)`, etc.) or lower-case type codes. One clear sentence is enough; add a short body only when the change genuinely needs explaining. Never add `Co-Authored-By` trailers.

## Plans

- **`dev-notes/master_plan.md`** — the single source of truth. High-level stages 1-8 with milestones and evaluation templates filled in after each stage.
- **`dev-notes/stage{N}_*.md`** — detailed implementation plan for each stage. Written before implementation begins, lives alongside the code as permanent documentation. Do NOT use Claude Code's plan mode — write plans directly into `dev-notes/`.
- **`dev-notes/source_repo_guide.md`** — where to find things in the two source repos (`~/projects/energy_prices/` and `~/projects/energy_market_analysis/`). Use this when porting code.
- **`dev-notes/mlflow_conventions.md`** — MLflow experiment structure, tagging rules, helper function specs.
- **`dev-notes/archive/`** — earlier plans and analysis that produced the master plan. Reference only.

## Planning workflow

Stage plans go in `dev-notes/stage{N}_*.md`, not in Claude Code's ephemeral plan mode. This keeps plans:
- Visible in the repo and version-controlled
- Readable across sessions without needing to recover context
- Reviewable by the user outside of Claude Code

When starting a new stage: read the master plan section, explore source repos, write the detailed plan to `dev-notes/`, get user approval, then implement.

## Status

Stages 1-4 complete. Stage 5a (Training Infrastructure) complete. **Stage 5b (Gen/Load Models) complete (2026-05-04)** — full 70-trial training, 48 base + 16 ensembles, 0 failures. **Phase A extension (2026-05-07)** retrained at expanded EMA-aligned FE search spaces and bumped `GEN_LOAD_HISTORICAL_FOLDS` 40 → 218, giving ~4.18 years of leak-free historical_forecasts (2022-01-15 → 2026-03-27, 36,788 rows per file) saved per (target, region) plus DE_NATIONAL aggregates. Training order enforced by CLI: wind/solar → load (uses wind/solar actuals) → gen_load_diff (uses all). **Stage 5c (Price Models) complete (2026-06-30; ensemble construction reworked to EP-faithful 2026-07-20)** — production ensemble follows EP verbatim: a category floor (best-MAE + best-RMSE per model family → up to 8 members, 2 per family) with inverse-MAE weights fit on the recent 90-day holdout; base models refit from fresh `merged` each retrain. **No method bakeoff in production** — the SLSQP/stacker bakeoff was reverted 2026-07-20 as a divergence from EP and now lives only in `scripts/ensemble_method_comparison.py` (diagnostic). The current MAE is a shifting-regime number that lives in `models/ensemble_config.json` (in-sample recent-holdout, EP's optimistic `blend_mae` convention) and the live-forward monitoring log — deliberately **not** pinned here. LGBM root-caused to `num_leaves=31` cap + MAE objective zero-hessian; fixed by scaling `num_leaves = 2^max_depth − 1`. All 8 review items closed (see `dev-notes/stage5c_status_2026-06-06.md`); EP-fidelity rework in `dev-notes/bug-fixes/ep_fidelity_reproduction_plan.md`. Pre-requisite `_eh10` → `_eh7` fix applied 2026-04-10. Backlog of model improvements lives in `dev-notes/roadmap.md`. **Stage 5d (Feature Selection Analysis) not started.** **Stage 6 (Inference, API & CI/CD) complete (2026-06-30)** — full inference pipeline (gen/load wave-by-wave → price → conformal PI), stateless FastAPI (7 endpoints), daily GitHub Actions workflow (08:00 UTC), monthly price retrain workflow, `deploy/` CLI sub-group. 490/495 tests pass (5 pre-existing). **Stage 7 (Dashboard) complete (2026-07-01)** — static HTML/JS dashboard served from `deploy/` with 6 chart panels (price forecast + PI, national gen/load, history, per-source bars, error bands), DE/EN toggle, monitoring page. All 9 review issues from §7.14 resolved: CI history preservation (cache), per-TSO gen/load JSON files, gen/load actuals overlay, gen/load accuracy chart, retrain history logging, `errors_summary` ordering fix, API_DATA_BASE override, Playwright package.json, status wording corrected. 512 tests pass. Pipeline bug fixes landed alongside Stage 7: Makefile conda prefix, SMARD API URL (smard.api.proxy.bund.dev → www.smard.de), NaN gate in merge.py, broken `build_merged_dataset` import, Ridge pre-scale scaler export+inference, solar night-hour lag-contamination clamp (elevation mask), SOLAR_NIGHT_HOURS CET/CEST correction. First clean end-to-end run 2026-06-30: price mean 131 EUR/MWh, all 7 JSON outputs written. **Stage 9 (Public Storytelling Site) built 2026-07-07, committed and deployed 2026-07-09** — scrollytelling history site at `deploy/story/index.html`. **Stage 10 (AI Narrative Forecast Story) implemented, committed and deployed 2026-07-09** — rolling-365-day yearly recap facts + Groq narrative (weekly), real per-instance SHAP price-driver attribution wired into `run_price_inference` (verified against production models to ~1.7e-5 EUR/MWh), daily forecast-driver Groq narrative, new page `deploy/story/forecast/index.html` cross-linked with Stage 9. 535 tests pass (34 new). `GROQ_API_KEY` added as a GitHub Actions secret. `deploy/story/` (both stages' data + inlined HTML) is git-tracked and kept fresh by the new weekly `.github/workflows/story_data.yml`, which regenerates and commits it straight to `master`; the daily Pages deploy picks up the latest commit with no separate restore step. Repo/Pages visibility stays public (explicit decision — not worth private-repo friction for a solo project yet). See `dev-notes/stage10_ai_narrative_forecast_story.md`. Next: Stage 8 (Monitoring & Alerting) or Stage 5d (Feature Selection Analysis).

Future training runs can use `--parallel N` (default 1 = sequential) for ~2-3x speedup on the tower (16 cores). Long-running training must be launched detached per `~/.claude/CLAUDE.md` § Long-Running Processes.
