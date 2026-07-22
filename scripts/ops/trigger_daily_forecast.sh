#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-smnfrs/energy-forecasting}"
WORKFLOW="${WORKFLOW:-daily_forecast.yml}"
REF="${REF:-master}"
WAIT_INPUT="${WAIT_INPUT:-true}"
LOOKBACK_LIMIT="${LOOKBACK_LIMIT:-20}"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

wait_for_github() {
  for _ in $(seq 1 30); do
    if gh api rate_limit >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  log "ERROR: GitHub API is not reachable via gh"
  return 1
}

today_has_active_or_successful_run() {
  local today_utc
  local found
  today_utc="$(date -u +%F)"
  found="no"

  while IFS=$'\t' read -r status conclusion url; do
    case "$status:$conclusion" in
      queued:*|in_progress:*|waiting:*|requested:*|pending:*|completed:success)
        log "Daily forecast already handled today: status=${status} conclusion=${conclusion:-none} url=${url}"
        found="yes"
        ;;
    esac
  done < <(gh run list \
    --repo "$REPO" \
    --workflow "$WORKFLOW" \
    --limit "$LOOKBACK_LIMIT" \
    --json createdAt,status,conclusion,url \
    --jq ".[] | select(.createdAt | startswith(\"${today_utc}\")) | [.status, (.conclusion // \"\"), .url] | @tsv")

  [ "$found" = "yes" ]
}

main() {
  log "Checking GitHub connectivity"
  wait_for_github

  if today_has_active_or_successful_run; then
    exit 0
  fi

  log "Dispatching ${WORKFLOW} on ${REF} with wait_until_berlin_0900=${WAIT_INPUT}"
  if [ "${DRY_RUN:-false}" = "true" ]; then
    log "DRY_RUN=true; not dispatching workflow"
    exit 0
  fi

  gh workflow run "$WORKFLOW" \
    --repo "$REPO" \
    --ref "$REF" \
    -f "wait_until_berlin_0900=${WAIT_INPUT}"

  sleep 5
  gh run list \
    --repo "$REPO" \
    --workflow "$WORKFLOW" \
    --limit 1 \
    --json databaseId,status,createdAt,url \
    --jq '.[] | "Dispatched run: id=\(.databaseId) status=\(.status) createdAt=\(.createdAt) url=\(.url)"' \
    || true
}

main "$@"
