#!/usr/bin/env bash
# CI test gate.
#
# Runs the in-scope test selection (shared tools + tool unit tests) minus the
# files recorded in ci/known-failing.txt. The exclusion list is printed every
# run so the debt is loud, not hidden.
#
# Out of scope by design (needs a GPU / NAS / heavy extras, so cannot run on a
# neutral runner): tools/hegre_dataset/tests + experiments/*/tests that import
# torch or read /mnt/nas-ai-models, and experiments/geometry_pca/tests (torch +
# NAS fixtures). Those are exercised locally, not here.
set -euo pipefail
cd "$(dirname "$0")/.."

ignores=()
while IFS= read -r line; do
  line="${line%%#*}"                       # strip trailing comments
  line="$(echo "$line" | xargs)"           # trim
  [ -z "$line" ] && continue
  ignores+=("--ignore=$line")
done < ci/known-failing.txt

echo "::group::Excluded from this gate (known-failing — see ci/known-failing.txt, issue #3)"
grep -vE '^\s*(#|$)' ci/known-failing.txt
echo "::endgroup::"

python -m pytest tests/tools tools/hegre_dataset/tests -q "${ignores[@]}"
