#!/bin/bash
# Post-run write verification for the daily LDA-reprojection no-op check.
# Read-only, NFS-heavy: proves the "0 files written" line is a true zero by
# scanning v1/lda for any .npy newer than the run, and re-counts the tree.
# Usage: bash verify_writes.sh > verify_writes-$(date +%Y%m%d).log 2>&1
set -u
V1=/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1
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
