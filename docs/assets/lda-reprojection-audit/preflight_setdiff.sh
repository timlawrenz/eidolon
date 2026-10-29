#!/usr/bin/env bash
# Pre-flight whole-tree set-difference between auraface/ and lda/.
#
# Compares tree-RELATIVE paths (see the 2026-10-19 method note in README.md):
# feeding `comm` the raw `find` output puts every line in both difference sets
# because the auraface/ vs lda/ prefix differs on every path, and comm silently
# returns the full count instead of erroring. Sort with LC_ALL=C or comm emits
# "file 1 is not in sorted order" and produces the same full-count artifact.
#
# No `set -u`: the runner invokes bash -lic which sources rvm, and `set -u`
# aborts with "rvm_bash_nounset: unbound variable" (see 2026-10-27 note 2).
set -o pipefail

DS="${1:-/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1}"

echo "=== preflight whole-tree set-difference (tree-relative, LC_ALL=C) $(date) ==="
echo "auraface count: $(find "$DS/auraface" -name '*.npy' | wc -l)"
echo "lda count:      $(find "$DS/lda" -name '*.npy' | wc -l)"

AF_LIST=$(mktemp)
LDA_LIST=$(mktemp)
( cd "$DS/auraface" && find . -name '*.npy' | LC_ALL=C sort > "$AF_LIST" )
( cd "$DS/lda"      && find . -name '*.npy' | LC_ALL=C sort > "$LDA_LIST" )

echo "in auraface not lda (comm -23): $(comm -23 "$AF_LIST" "$LDA_LIST" | wc -l)"
echo "in lda not auraface (comm -13): $(comm -13 "$AF_LIST" "$LDA_LIST" | wc -l)"

rm -f "$AF_LIST" "$LDA_LIST"
echo "PREFLIGHT_DONE"
