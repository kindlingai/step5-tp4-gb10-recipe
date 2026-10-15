#!/usr/bin/env bash
# Boot-only host page-cache workaround: once every 10 seconds for 15 minutes.
set -euo pipefail
DRY_RUN=0
case ${1:-} in
  --dry-run) DRY_RUN=1 ;;
  '') ;;
  *) echo 'Usage: drop-page-cache.sh [--dry-run]' >&2; exit 2 ;;
esac
if (( ! DRY_RUN )) && [[ ! -w /proc/sys/vm/drop_caches ]]; then
  echo 'Page-cache drop requires elevated host privileges' >&2
  exit 2
fi
trap 'exit 130' INT
trap 'exit 143' TERM
for ((elapsed=0; elapsed<900; elapsed+=10)); do
  if (( DRY_RUN )); then
    printf 't=%ss: echo 1 > /proc/sys/vm/drop_caches\n' "$elapsed"
  else
    echo 1 > /proc/sys/vm/drop_caches
    printf '%s dropped clean page cache\n' "$(date -u +%FT%TZ)"
    sleep 10
  fi
done
