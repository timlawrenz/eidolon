#!/usr/bin/env bash
# Launcher for the audited daily enrich sweep. Detached so no runner-side
# timeout or pty teardown can SIGTERM it mid-run: the 2026-10-28 occurrence
# died at 3 s with ENRICH_EXIT=143 (SIGTERM) and
# "bash: [..] tcsetattr: Inappropriate ioctl for device", i.e. the sweep was
# killed by the launcher's terminal teardown before it ever reached the
# projection loop. setsid + nohup removes that failure mode.
#
# Usage: bash run_enrich.sh YYYYMMDD
set -o pipefail

DAY="${1:?usage: run_enrich.sh YYYYMMDD}"
ROOT=/home/tim/source/activity/eidolon
AUDIT="$ROOT/docs/assets/lda-reprojection-audit"
LOG="$AUDIT/enrich-$DAY.log"
DATASET=/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1

cd "$ROOT" || exit 1
{
  echo "=== enrich launch $(date) (git $(git rev-parse --short HEAD), LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-<empty>}') ==="
  START=$(date +%s)
  .venv/bin/python -m tools.hegre_dataset enrich \
      --dataset "$DATASET" --status approved --skip-stratum
  RC=$?
  END=$(date +%s)
  echo "ENRICH_EXIT=$RC WALL=$((END-START))s END=$(date)"
} > "$LOG" 2>&1

echo "$LOG"
