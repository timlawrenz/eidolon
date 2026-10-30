#!/usr/bin/env bash
# Post-run audit checks for the daily LDA-reprojection no-op sweep.
# Read-only, NFS-heavy (combined ~1.1M stat calls) — exceeds the 420 s
# foreground window, so it is launched as a tracked background job.
# Usage: bash postchecks.sh YYYYMMDD
set -o pipefail
DAY="${1:?usage: postchecks.sh YYYYMMDD}"
ROOT=/home/tim/source/activity/eidolon
AUDIT="$ROOT/docs/assets/lda-reprojection-audit"
V1=/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1

{
  echo "=== verify_writes $DAY ==="
  cd "$V1" || exit 1
  echo "=== newermt TODAY 00:00 ==="
  find lda -name '*.npy' -newermt "$(date +%Y-%m-%d) 00:00" | wc -l
  echo "=== newermt YESTERDAY 00:00 ==="
  find lda -name '*.npy' -newermt "$(date -d yesterday +%Y-%m-%d) 00:00" | wc -l
  echo "=== mmin -1440 (last 24h) ==="
  find lda -name '*.npy' -mmin -1440 | wc -l
  echo "=== newest in tree ==="
  find lda -name '*.npy' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -1
  echo "=== counts ==="
  for d in auraface lda; do printf "%s: " "$d"; find "$d" -name '*.npy' | wc -l; done
  echo "VERIFY_DONE"

  echo "=== gapcheck $DAY ==="
  echo -n "newermt 2026-10-20: "; find lda -name '*.npy' -newermt '2026-10-20 00:00' | wc -l
  echo -n "newermt 2026-10-29: "; find lda -name '*.npy' -newermt '2026-10-29 00:00' | wc -l
  echo "GAPCHECK_DONE"
} > "$AUDIT/verify_writes-$DAY.log" 2>&1

cd "$ROOT" || exit 1
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" .venv/bin/python "$AUDIT/probe_scan.py" \
  > "$AUDIT/probe_scan-$DAY.log" 2>&1
echo "PROBE_EXIT=$?" >> "$AUDIT/probe_scan-$DAY.log"
echo "POSTCHECKS_DONE"
