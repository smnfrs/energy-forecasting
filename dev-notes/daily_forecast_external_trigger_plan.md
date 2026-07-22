# Daily Forecast External Trigger Plan

Date: 2026-07-22

## Decision

Start with Option 1: Station dispatches `daily_forecast.yml` when it wakes at
`06:55 Europe/Berlin`, and the GitHub workflow waits until `09:00 Europe/Berlin`
before collecting data.

This avoids relying on GitHub's scheduled-event delivery as the primary trigger,
while avoiding the more complex RTC handoff needed to wake Station again near
09:00.

## Current Station Wake

Observed on `station`:

- Timer: `/etc/systemd/system/daily-plan.timer`
- Schedule: `06:55` local time daily
- Service: `/etc/systemd/system/daily-plan.service`
- Script: `/home/smnfrs/projects/second-brain/_reference/scripts/daily-plan-station.sh`
- Current behavior: the script arms tomorrow's RTC wake first, runs the daily
  plan, then auto-shuts down only when uptime is under 10 minutes and there are
  no interactive `smnfrs` sessions.

Station's `gh` CLI is authenticated as `smnfrs` and has the `workflow` scope.

## Implemented Shape

Energy repo helper:

- `scripts/ops/trigger_daily_forecast.sh`

The helper:

1. Waits briefly for GitHub CLI/API reachability.
2. Checks recent `daily_forecast.yml` runs.
3. Skips dispatch if today's workflow already has a queued, running, or
   successful run.
4. Dispatches the workflow on `master` with:

   ```bash
   gh workflow run daily_forecast.yml \
     --repo smnfrs/energy-forecasting \
     --ref master \
     -f wait_until_berlin_0900=true
   ```

Station integration:

- `_reference/scripts/daily-plan-station.sh` in `second-brain` now calls the
  energy helper immediately after its network wait.
- The daily-plan job continues even if the energy forecast dispatch fails.

GitHub workflow changes:

- `workflow_dispatch` has a `wait_until_berlin_0900` boolean input.
- Station dispatches with the input enabled.
- Ordinary manual dispatches default to no wait.
- A first `preflight` job prevents scheduled backup runs from duplicating a
  queued, running, or successful daily run from the same UTC date.
- The `collect-data` job waits until `09:00 Europe/Berlin` when triggered by
  Station or by the scheduled backup.

## Backup Schedule

For now, keep GitHub's scheduled backup in `daily_forecast.yml` at `07:39 UTC`.

Behavior:

- In summer, this is `09:39 CEST`; if Station already dispatched, preflight
  skips the backup.
- In winter, this is `08:39 CET`; if Station did not dispatch, the backup waits
  until `09:00 CET` before collecting data.

Once the Station path has run successfully unattended for a few mornings, the
backup can be moved later if desired.

## Tradeoff

This is intentionally the simplest reliable first step. It ties up a GitHub
runner while waiting from roughly `06:55` to `09:00 Europe/Berlin`, but it avoids
changing the RTC wake chain.

If that wait becomes annoying or if runner availability becomes a problem, the
next step is Option 2: daily-plan wakes Station at 06:55, Station arms a same-day
RTC wake shortly before 09:00, shuts down, then the energy trigger dispatches at
09:00 and re-arms tomorrow's 06:55 wake.

## Optional Standalone Timer

Standalone systemd reference files are included for a later setup where Station
is expected to stay awake or has a separate wake path:

- `deploy/systemd/energy-forecast-trigger.service`
- `deploy/systemd/energy-forecast-trigger.timer`

They were not installed during initial rollout because installing system units
requires sudo authentication in an interactive terminal.
