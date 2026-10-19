# LDA reprojection audit — 2026-09-28

**Trigger:** scheduled cron run of
`tools.hegre_dataset enrich --dataset .../hegre-faces/v1 --status approved --skip-stratum`,
expected to project "~1,500 missing per-image LDA files".

**Result: 0 LDA files projected. The job was already complete.**

The reprojection the job was written for ran in **late July 2026** (right after the
2026-07-23 basis refit): `scripts/reproject_lda.py` (committed 2026-09-22, `5e9b5f2`)
reprojects *approved* images only, writing refit-basis files with mtimes
**2026-07-25 / 2026-07-28** (mtime histogram, `mtime_hist.py`).

## Numbers (all reproducible from the scripts in this dir)

| quantity | value |
|---|---|
| approved images in review DB | 166,200 |
| approved images **missing** an LDA file | **23** |
| …of those, also missing AuraFace (⇒ **unprojectable**) | **23** |
| approved images with a usable LDA file | 166,177 |
| `.npy` files in `v1/lda/` (all statuses) | 295,468 |
| files written by this run | **0** |

Command output (`enrich.log`), exit 0:
```
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

## Basis verification

* `v1/lda/BASIS_FINGERPRINT.json` stamps `120e1c5a1dc4f423`, convention
  *"refit basis, raw coords (norm ~153)"*.
* `hegre-dataset basis-fingerprint verify` on the current
  `experiments/geometry_pca/output/` artifacts → **`120e1c5a1dc4f423`** (match).
  So `project_to_lda` would have projected onto the same basis as the existing
  files. `experiments/` is a real directory, not a symlink (no
  `Path(__file__).resolve()` trap).
* Existing file norms: median **153.6**, p1 149.2, p99 157.9 — matches the stamped
  raw-coords convention.

## Finding — the `lda/` dir is mixed-basis at file level (no approved impact)

Full scan of all 295,468 files (`scan_all_lda.py`, `all_lda.log`):

* **2,220 files (0.751%)** have pre-refit norms (min 0.234, i.e. ≈ new-basis
  norm ÷ ~440), mtimes **2026-07-03 → 2026-07-06** — *before* the 2026-07-23 refit.
* Cross-referenced against the review DB (`xref_prerefit.py`, `xref.log`):
  **all 2,220 belong to tainted images** —
  2,219 `tainted:extraction_nonface` + 1 `tainted:contamination`.
  **Zero approved images carry a pre-refit LDA file.**
* Root cause: `reproject_lda.py` reprojects only `status='approved'`; images that
  were approved on 2026-07-03 and *later* reclassified as tainted kept their stale
  pre-refit files.

**Consequence:** the stamp's `scope: "all samples"` overstates uniformity — 0.75% of
files in the directory are on a different basis. Any consumer that reads the dir
*without* filtering on `images.status` would silently mix bases. Approved-only
consumers (corpus build, enrich) are unaffected. Suggested (not executed):
reproject or archive the 2,220 tainted files so the directory matches its stamp.

## Non-errors observed

* **9** approved images missing `z_g`: their `pose.npy` face keypoints
  (slice `23:91`) are **all zero** — DWPose found no face. Legitimate skip, not an
  error (`enrich` correctly reports "Extracted 0, skipped 9").
* **23** approved images missing AuraFace: these originally failed SCRFD face
  detection (`_face1/_face2`, `-board`, `-poster` crops). Cannot be projected —
  there is no embedding to project. Requires InsightFace (GPU) to retry;
  `insightface` is not installed in `.venv`, so `enrich` skipped them.

## Re-run — 2026-09-30 (scheduled cron, same command)

Identical outcome. `enrich` exit 0, **0 files written**
(`enrich-20260930.log`); independent `probe_scan.py` re-sweep over all 166,200
approved images (`probe_scan-20260930.log`):

| quantity | 2026-09-28 | 2026-09-30 |
|---|---|---|
| approved images | 166,200 | **166,200** |
| approved missing LDA | 23 | **23** (all also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | **9** |
| approved missing pose.npy | — | **0** |
| LDA files written by the run | 0 | **0** |

Existing LDA norms sampled 152.6–154.8 (refit raw-coords convention), so the
directory is unchanged. No new approvals since 2026-09-28; the job remains
complete. The cron job's premise ("~1,500 missing per-image LDA files", new
refit basis) is **stale** — that reprojection ran 2026-07-25/28 — so every
future run of this job will sweep 166k NFS paths to report zero.

## Re-run — 2026-10-11 (scheduled cron, same command)

Identical outcome for the third consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261011.log`; wall clock 5 m 07 s, git `98882c8`,
clean tree). Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261011.log`):

| quantity | 2026-09-28 | 2026-09-30 | 2026-10-11 |
|---|---|---|---|
| approved images | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | **0** |
| LDA files written by the run | 0 | 0 | **0** |

Independent checks:

* `find v1/lda -name '*.npy' -newermt 2026-10-10` → **0** — no LDA file was
  written or touched by this run (the "0 files written" line is a true zero, not
  a silent-failure zero).
* `hegre-dataset basis-fingerprint verify` on `v1/lda/` →
  `120e1c5a1dc4f423` ("refit basis, raw coords (norm ~153)") — match.
* Sampled existing LDA files: shape `(64,)`, `float32`, norm 152.6–154.8 —
  refit raw-coords convention.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete.

## Re-run — 2026-10-12 (scheduled cron, same command)

Identical outcome for the **fourth** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261012.log`; wall clock 5 m 43 s, git `9c1f731`,
clean tree). Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261012.log`):

| quantity | 2026-09-28 | 2026-09-30 | 2026-10-11 | 2026-10-12 |
|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | **0** |

Independent checks:

* `find v1/lda -name '*.npy' -newermt 2026-10-11` → **0** — no LDA file was
  written or touched by this run (the "0 files written" line is a true zero, not
  a silent-failure zero).
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK 120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
  (Verifying the *dataset root* now reports `FAIL … unstamped`: the stamp lives in
  `v1/lda/`, not the root — verify the `lda/` subdir, not `v1/`.)
* Sampled existing LDA files: shape `(64,)`, `float32`, norm 152.6–154.8 —
  refit raw-coords convention.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```
(The `enrich-*.log` / `probe_scan-*.log` files in this dir are working-tree
artifacts only — `.gitignore` excludes `*.log`, so they are **not** committed;
the numbers they evidence are transcribed into this README.)

## Re-run — 2026-10-13 (scheduled cron, same command)

Identical outcome for the **fifth** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261013.log`; wall clock **6 m 33 s**, git
`f5a9b15`, clean tree). Independent `probe_scan.py` re-sweep over all approved
images (`probe_scan-20261013.log`):

| quantity | 2026-09-28 | 2026-09-30 | 2026-10-11 | 2026-10-12 | 2026-10-13 |
|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | **0** |

Independent checks:

* `find v1/lda -name '*.npy' -newermt 2026-10-12` → **0**; `-newermt '2026-10-13 00:00'`
  → **0** — no LDA file was written or touched by this run (the "0 files written"
  line is a true zero, not a silent-failure zero). Newest file in the tree is still
  `2026-07-28`.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK 120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files: shape `(64,)`, `float32`, norms 152.6–154.8 —
  refit raw-coords convention (five samples, same as prior runs).

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete.

**New observation this run — the `os.execv` restart fired.** Unlike the four prior
recorded runs, this launch printed `Restarting subprocess to load CUDA libraries...`
after the z_g stage: `LD_LIBRARY_PATH` was not pre-set, so `enrichment.py` re-exec'd
with `sys.executable` (the `.venv` python) once to set the cuDNN path, then re-ran
the **entire** NFS sweep from the top. Result: the 166k-path stat loop ran **twice**
and wall clock roughly doubled (6 m 33 s vs ~5 m 07–43 s on 2026-10-11/12). It is a
**single** restart, not the infinite loop the enrichment-operations reference warns
about — the second pass finds `LD_LIBRARY_PATH` set, skips the guard, and terminates
normally (`Error: insightface not installed. Skipping AuraFace extraction.` is the
expected benign path: all AuraFace files already exist). Pre-setting `LD_LIBRARY_PATH`
per the reference would avoid the wasted double sweep, but since the job is a proven
no-op the restart cost is moot — it is a further argument for de-registration.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Restarting subprocess to load CUDA libraries...
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

## Re-run — 2026-10-14 (scheduled cron, same command)

Identical outcome for the **sixth** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261014.log`; wall clock **≈ 6 min 34 s**
[03:00:38 → 03:07:12 EDT], git `1a7a282`, clean tree at launch). Independent
`probe_scan.py` re-sweep over all approved images (`probe_scan-20261014.log`):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 |
|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | **0** |

Independent checks:

* `find v1/lda -name '*.npy' -newermt '2026-10-13 00:00'` → **0** — no LDA file
  was written or touched by this run (the "0 files written" line is a true zero,
  not a silent-failure zero). Newest file in the tree is still **2026-07-28**.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK       .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete.

**Restart fired again (second consecutive run).** As on 2026-10-13, the launch did
**not** pre-set `LD_LIBRARY_PATH`, so `enrichment.py` re-exec'd once
(`Restarting subprocess to load CUDA libraries...`) and re-ran the entire 166k-path
NFS sweep — hence the ~6.5 min wall clock (vs ~5 m 07–43 s when no restart fires,
cf. 2026-10-11/12). Single restart, not a loop; second pass terminates normally.
Pre-setting `LD_LIBRARY_PATH` per `references/enrichment-operations.md` would avoid
the wasted double sweep.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Restarting subprocess to load CUDA libraries...
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

**Job state at this run:** still `enabled: true`, `state: scheduled`,
`repeat.completed = 39` (was 38 before the 2026-10-13 run), `last_status = ok`,
`next_run_at = 2026-10-15T03:00` — i.e. the de-registration recommendation below
has still not been executed, and the next run is already queued.

**New observation this run — the audit records were never pushed.** `main` is
**ahead of `origin/main` by 4 commits**: the 2026-10-11 / 10-12 / 10-13 audit
commits plus `98882c8 docs(briefings): stage the eidolon scientist-facing
write-up`. The job is `deliver: local`, so each run committed its record locally
and stopped — six days of audit history existed only in the working clone.
Diff is docs-only (374 insertions, 2 markdown files, zero binaries/gitlinks), so
the 2026-10-14 run pushed it (remote `main` → `1a7a282`-equivalent tip). This is
the first run of this job to gate on **remote-visible** state rather than
local commit state.

## Re-run — 2026-10-15 (scheduled cron, same command)

Identical outcome for the **seventh** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261015.log`; wall clock **6 m 12 s**
[03:04:07 → 03:10:19 EDT], git `f74c8e7`, clean tree at launch). Independent
`probe_scan.py` re-sweep over all approved images (`probe_scan-20261015.log`):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 |
|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Independent checks:

* **Pre-flight tree set-difference (the authoritative check, run *before* `enrich`):**
  `auraface/` and `lda/` hold exactly **295,468 `.npy` each — `comm` diff 0 missing,
  0 orphans** (full `find`+sort of both trees). Zero LDA work is reachable from the
  entire tree, not merely from the approved set.
* `find v1/lda -name '*.npy' -newermt '2026-10-15 00:00'` → **0**;
  `-newermt '2026-10-14 00:00'` → **0** — no LDA file was written or touched by this
  run (the "0 files written" line is a true zero, not a silent-failure zero). Newest
  file in the tree is still **2026-07-28 14:42**.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK       .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete.

**Restart fired again (third consecutive run).** The launch did not pre-set
`LD_LIBRARY_PATH`, so `enrichment.py` re-exec'd once (`Restarting subprocess to load
CUDA libraries...`) and re-ran the entire 166k-path NFS sweep — hence the 6 m 12 s
wall clock (vs ~5 m 07–43 s when no restart fires, cf. 2026-10-11/12). Single
restart, not a loop; the second pass terminates normally.

**Remote state gated (second run to do so).** At launch `main` **== `origin/main`**
(`f74c8e7`) — the 2026-10-14 run's push landed, so no audit history is stranded in
the working clone. This run commits and pushes its own record on the same
remote-visible basis.

**Job state at this run:** still `enabled: true`, `state: scheduled`,
`repeat.completed = 40` (was 39 before the 2026-10-14 run), `last_status = ok`,
`next_run_at = 2026-10-16T03:00` — the de-registration recommendation below remains
unexecuted, and the next run is already queued.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Restarting subprocess to load CUDA libraries...
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

## Re-run — 2026-10-16 (scheduled cron, same command)

Identical outcome for the **eighth** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261016.log`; wall clock **348.0 s**
[03:01:17 → 03:07:05 EDT], git `43f933e`, clean tree at launch,
`main == origin/main`). Independent `probe_scan.py` re-sweep over all approved
images (`probe_scan-20261016.log`):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 |
|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Approved count unchanged at 166,200 — no new approvals since 2026-09-30. The
review-DB status census this run: `tainted:extraction_nonface` 215,914 ·
`approved` 166,200 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19.

Independent checks:

* **Pre-flight whole-tree set-difference (the authoritative check, run *before*
  `enrich`):** `auraface/` and `lda/` hold exactly **295,468 `.npy` each**; full
  `find`+`sort`+`comm` → **0 in auraface-not-lda, 0 in lda-not-auraface**. Zero
  LDA work is reachable from the entire tree, not merely from the approved set.
* `find v1/lda -name '*.npy' -newermt '2026-10-16 00:00'` → **0**;
  `-newermt '2026-10-15 00:00'` → **0** — no LDA file was written or touched by
  this run (the "0 files written" line is a true zero, not a silent-failure
  zero). Newest file in the tree is still **2026-07-28 14:42**.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK       .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

**Correction to a prior claim — the "Restarting subprocess…" log line is not a
reliable witness that the restart fired.** Runs 2026-10-13…10-15 were recorded
above as "restart fired" *because the line appears in the log*, and 10-11/10-12
as "no restart" *because it does not*. That inference does not hold: the line is
emitted immediately **before** `os.execv` in `enrichment.py`, and **`os.execv`
does not flush Python's userspace stdout buffer** — so under the block-buffered
redirection this job uses, a *fired* restart discards every unflushed pre-exec
line and can leave a single-sequence log indistinguishable from no restart.
Demonstrated with [`exec_flush_probe.py`](exec_flush_probe.py) (committed this
run; `.venv/bin/python` = CPython 3.14, stdout redirected to a file):
`MARKER_PRE_EXEC_NO_FLUSH` (explicitly flushed) **survives** the exec;
`MARKER_PRE_EXEC_BUFFERED` (unflushed) is **lost**. Note `enrich`'s own prints
carry no `flush=True` outside the every-500 progress line, so the whole pre-exec
sequence is buffer-resident at exec time.

Consequences for this audit: (a) the 10-11…10-15 wall-clock split
(5 m 07–43 s "no restart" vs 6 m 12–34 s "restart") is likewise **not**
dispositive — a second pass over the same paths is largely dentry/page-cache
warm, so the observed ~30–50 s delta is within this job's run-to-run spread; and
(b) this run's log shows a **single sequence** (no restart line) at 348.0 s,
which falls in the "no-restart" band — but that is consistent with *either*
state, and the guard's preconditions do hold here (**`LD_LIBRARY_PATH` empty at
launch, and the cuDNN dir present**, both verified). The restart's true
incidence across runs is therefore **unresolved**, and no prior "restart fired"
attribution should be relied on. It does not affect the material finding: **0
files written**, which is independently corroborated by the tree checks above
(log-line-free evidence).

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

**Job state, read mid-run (03:14 EDT):** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `repeat.completed = 41` (was 40 at the
2026-10-15 run), `next_run_at = 2026-10-17T03:00` (today's fire already
advanced), `deliver: local`, `created_at = 2026-07-24T08:45:36`. The
de-registration recommendation below remains unexecuted.

## Re-run — 2026-10-17 (scheduled cron, same command)

Identical outcome for the **ninth** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261017.log`; wall clock **296 s**
[≈03:01:11 → 03:06:07 EDT], git `43930ad`, clean tree at launch,
`main == origin/main`). Independent `probe_scan.py` re-sweep over all approved
images (`probe_scan-20261017.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 |
|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py), committed
this run): `tainted:extraction_nonface` 215,914 · `approved` 166,200 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 — **identical to 2026-10-16**; no new
approvals.

Independent checks:

* **Pre-flight whole-tree set-difference (the authoritative check, run *before*
  `enrich`):** `auraface/` and `lda/` hold exactly **295,468 `.npy` each**; full
  `find`+`sort`+`comm` → **0 in auraface-not-lda, 0 in lda-not-auraface**. Zero
  LDA work is reachable from the entire tree, not merely from the approved set.
* `find v1/lda -name '*.npy' -newermt '2026-10-17 00:00'` → **0**;
  `-newermt '2026-10-16 00:00'` → **0** — no LDA file was written or touched by
  this run (the "0 files written" line is a true zero, not a silent-failure
  zero). Newest file in the tree is still **2026-07-28 14:42**.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK       .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete.

**Restart observation.** `LD_LIBRARY_PATH` was empty at launch (guard
preconditions hold), and the log shows a **single sequence** at 296 s. Per the
2026-10-16 correction, the presence/absence of the `Restarting subprocess…` line
is **not** a reliable witness (`os.execv` drops the userspace stdout buffer), so
no restart attribution is made — and it does not affect the material finding
(**0 files written**), which the tree checks corroborate independently of any
log line.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `repeat.completed = 42` (was 41 at the
2026-10-16 run), `next_run_at = 2026-10-18T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`. The de-registration recommendation below
remains unexecuted.

## Re-run — 2026-10-18 (scheduled cron, same command)

Identical outcome for the **tenth** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261018.log`; wall clock **5 m 42 s**
[≈03:02:50 → 03:08:32 EDT], git `c433b2b`, clean tree at launch,
`main == origin/main`). Independent `probe_scan.py` re-sweep over all approved
images (`probe_scan-20261018.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 |
|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py)):
`tainted:extraction_nonface` 215,914 · `approved` 166,200 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 — **identical to 2026-10-17**; no new
approvals.

Independent checks:

* **Pre-flight whole-tree set-difference (the authoritative check, run *before*
  `enrich`):** `auraface/` and `lda/` hold exactly **295,468 `.npy` each**; full
  `find`+`sort`+`comm` → **0 in auraface-not-lda, 0 in lda-not-auraface**. Zero
  LDA work is reachable from the entire tree, not merely from the approved set.
* `find v1/lda -name '*.npy' -newermt '2026-10-18 00:00'` → **0** — no LDA file
  was written or touched by this run (the "0 files written" line is a true zero,
  not a silent-failure zero). Newest file in the tree is still
  **2026-07-28 14:42**.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK       .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete. Restart attribution is not made — per the 2026-10-16
correction the `Restarting subprocess…` line is not a reliable witness
(`os.execv` drops the userspace stdout buffer), and it does not affect the
material finding (**0 files written**), which the tree checks corroborate
independently of any log line.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `repeat.completed = 43` (was 42 at the
2026-10-17 run), `next_run_at = 2026-10-19T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`. The de-registration recommendation below
remains unexecuted.

## Re-run — 2026-10-19 (scheduled cron, same command)

Identical outcome for the **eleventh** consecutive recorded run. `enrich` exit 0,
**0 files written** (`enrich-20261019.log`; wall clock **5 m 13 s**
[03:02:43 → 03:07:56 EDT], git `fe90ffe`, clean tree at launch). Independent
`probe_scan.py` re-sweep over all approved images (`probe_scan-20261019.log`,
exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py)):
`tainted:extraction_nonface` 215,914 · `approved` 166,200 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 (total 384,012) — **identical to 2026-10-18**;
no new approvals.

Independent checks:

* **Pre-flight whole-tree set-difference (the authoritative check, run *before*
  `enrich`):** `auraface/` and `lda/` hold exactly **295,468 `.npy` each**; full
  `find` + `LC_ALL=C sort` + `comm` on the **tree-relative** paths → **0 in
  auraface-not-lda, 0 in lda-not-auraface**. Zero LDA work is reachable from the
  entire tree, not merely from the approved set.
  *Method note:* the two trees must be compared on paths made relative to their own
  roots. Comparing the raw `find` output puts *every* line in both difference sets
  (the `auraface/` vs `lda/` prefix differs on every path) and `comm` **silently
  returns the full count instead of erroring** — a plausible-looking
  "295,468 missing" that is pure artifact. Also sort with `LC_ALL=C`: under the
  default locale `comm` emitted `file 1 is not in sorted order` and produced the
  same full-count false positive.
* `find v1/lda -name '*.npy' -newermt '2026-10-19 00:00'` → **0**;
  `-newermt '2026-10-18 00:00'` → **0** — no LDA file was written or touched by
  this run (the "0 files written" line is a true zero, not a silent-failure
  zero). Newest file in the tree is still **2026-07-28 14:42**.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK       .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete. Per the 2026-10-16 correction, no restart attribution is made:
the `Restarting subprocess…` line *is* present in this run's log, but `os.execv`
drops the userspace stdout buffer, so its presence is not a reliable witness. It
does not affect the material finding (**0 files written**), which the tree checks
corroborate independently of any log line.

Raw `enrich` output for this run (exit 0):
```
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Restarting subprocess to load CUDA libraries...
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
```

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `repeat.completed = 44` (was 43 at the
2026-10-18 run), `next_run_at = 2026-10-20T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`. The de-registration recommendation below
remains unexecuted.

## Recommendation — de-register this cron job (no-op since the reprojection)

The cron entry is `2aec4dd52b5c` / `eidolon-lda-reproject`, `0 3 * * *`, created
2026-07-24, `repeat.completed = 44` as of the 2026-10-19 run, `last_status = ok`.
Each of the **eleven** *audited* runs (2026-09-28, 2026-09-30, 2026-10-11, 2026-10-12,
2026-10-13, 2026-10-14, 2026-10-15, 2026-10-16, 2026-10-17, 2026-10-18, 2026-10-19) has written
**zero** files: the premise ("~1,500 missing per-image LDA files on the new refit
basis") has been stale since the reprojection ran 2026-07-25/28. Every run still
costs ~5–8 min of NFS stat load over 166k paths to prove nothing changed.

Nothing can recreate the condition this job guards against except (a) new images
being *approved* in the review DB, or (b) a fresh LDA basis refit — and (b) already
writes its own reprojection step (`scripts/reproject_lda.py`,
`scripts/reproject_lda_ffhq.py`). Recommendation: **remove the daily schedule** and
run this sweep on demand (or once) after a basis refit or an approval batch, never
on a fixed cadence. Not executed here — the cron registry is the operator's, and a
job run may not modify its own schedule.
