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

## Re-run — 2026-10-20 (scheduled cron, same command)

Identical outcome for the **twelfth** consecutive audited run. `enrich` exit 0,
**0 files written** (`enrich-20261020.log`; wall clock **4 m 53 s**
[03:01 → 03:06 EDT], git `64f39f7`, clean tree at launch — an earlier plain
invocation the same morning also wrote 0, in 5 m 28 s). Independent
`probe_scan.py` re-sweep over all approved images (`probe_scan-20261020.log`,
exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census: identical to 2026-10-19 — `approved` 166,200 ·
`tainted:extraction_nonface` 215,914 · `tainted:contamination` 1,626 ·
`tainted:unusable` 253 · `tainted:approved_bad_geometry` 19 (total 384,012).
**No new approvals.**

Independent checks:

* **Pre-flight whole-tree set-difference (the authoritative check):** `auraface/`
  and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13` on
  **tree-relative** paths → **0** and **0** (same-cwd subshells, `LC_ALL=C`
  sort — see the 2026-10-19 method note). Zero LDA work is reachable from the
  entire tree, independent of the review DB.
* `find v1/lda -name '*.npy' -newermt '2026-10-20 00:00'` → **0**; newest file in
  the tree is still **2026-07-28 14:42**. The "0 files written" line is a true
  zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  .../v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`BASIS_FINGERPRINT.json` stamped 2026-09-24, git `b35a3e6`).
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `repeat.completed = 46`,
`next_run_at = 2026-10-21T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`. A `last_failure` entry dated
`2026-10-19T03:08` records a restart-safe worker exit 1 with no durable terminal
state; the 2026-10-19 material finding (0 files written) stands as recorded —
the failure is in the worker's terminal-state bookkeeping, not in the sweep.

The job remains complete. Recommendation below unchanged.

## Re-run — 2026-10-21 (scheduled cron, same command)

Identical outcome for the **thirteenth** consecutive audited run. `enrich` exit 0,
**0 files written** (`enrich-20261021.log`; wall clock **6 m 23.9 s**
[03:02:23 → ≈03:08:47 EDT], git `f20d62e`, clean tree at launch). Independent
`probe_scan.py` re-sweep over all approved images (`probe_scan-20261021.log`,
exit 0 — run as a background job: the scan's ~830k NFS `stat` calls exceed the
420 s foreground window on this host):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py)) — **set
strictly identical to 2026-10-20**, including the grand total, so no row was added,
removed or re-statused:

| status | count |
|---|---|
| `approved` | 166,200 |
| `tainted:extraction_nonface` | 215,914 |
| `tainted:contamination` | 1,626 |
| `tainted:unusable` | 253 |
| `tainted:approved_bad_geometry` | 19 |
| **TOTAL** | **384,012** |

Independent checks:

* **Pre-flight whole-tree set-difference (the authoritative check, run *before*
  `enrich`):** `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm
  -23` / `comm -13` on **tree-relative** paths (same-cwd subshells, `LC_ALL=C`
  sort — see the 2026-10-19 method note) → **0** and **0**. Zero LDA work is
  reachable from the entire tree, independent of the review DB.
* `find v1/lda -name '*.npy' -newermt '2026-10-21 00:00'` → **0**;
  `-newermt '2026-10-20 00:00'` → **0**. Newest file in the tree is still
  **2026-07-28 14:42** (`…/lezhan-in/lezhan-in-sunset/lezhan-in-sunset-16-3000px_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

No new approvals since 2026-09-30 (approved count unchanged at 166,200). The job
remains complete. Per the 2026-10-16 correction, no restart attribution is made:
this run's log shows a **single sequence** and its `/usr/bin/time -v` profile
(186,272 file-system inputs, 529 major page faults) is consistent with either a
cold-cache single sweep or a restart's second pass. It does not affect the
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

**Remote state gated.** At launch `main` (`f20d62e`, the 2026-10-20 record) was
**ahead of `origin/main` (`64f39f7`, the 2026-10-19 record) by one commit** — the
2026-10-20 run committed its record locally but did not push it. This run pushes
both.

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 47` (was 46 at the 2026-10-20 run),
`last_run_at = 2026-10-20T03:26`, `next_run_at = 2026-10-22T03:00`,
`deliver: local`, `created_at = 2026-07-24T08:45:36`, `last_failure` still the
2026-10-19 worker-bookkeeping entry. The de-registration recommendation below
remains unexecuted, and the next run is already queued.

## Re-run — 2026-10-22 (scheduled cron, same command)

Identical outcome for the **fourteenth** consecutive audited run. `enrich` exit 0,
**0 files written** (`enrich-20261022.log`; wall clock **5 m 04.4 s**
[03:00:25 → ≈03:05:29 EDT], git `3ea637a`, clean tree at launch — `main` and
`origin/main` both at `3ea637a`). Independent `probe_scan.py` re-sweep over all
approved images (`probe_scan-20261022.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py)) — **set
strictly identical to 2026-10-21**, including the grand total, so no row was added,
removed or re-statused: `approved` 166,200 · `tainted:extraction_nonface` 215,914 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference (run *before* `enrich`):** `auraface/` and
  `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13` on
  **tree-relative** paths (`LC_ALL=C` sort) → **0** and **0**. Zero LDA work is
  reachable from the entire tree, independent of the review DB.
* Post-run write verification via [`verify_writes.sh`](verify_writes.sh)
  (`verify_writes-20261022.log`): `find v1/lda -name '*.npy' -newermt '2026-10-22 00:00'`
  → **0**; `-newermt '2026-10-21 00:00'` → **0**; `-mmin -1440` → **0**. Newest file in the
  tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The
  "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`, `float32`,
  norms 152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0) — note the self-restart, the same
non-deterministic behaviour recorded on 2026-10-19 (the 10-20/10-21 runs did not
restart); it does not change the material finding:
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

No eidolon process held the GPU during either pass (checked mid-run: the card was
held only by `image_embed` and `ollama` — other projects, not ours).

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 48` (was 47 at the 2026-10-21 run),
`next_run_at = 2026-10-23T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`, `last_failure` still the 2026-10-19
worker-bookkeeping entry. The de-registration recommendation below remains
unexecuted, and the next run is already queued.

## Re-run — 2026-10-23 (scheduled cron, same command)

Identical outcome for the **fifteenth** consecutive audited run. `enrich` exit 0,
**0 files written** (`enrich-20261023.log`; wall clock **6 m 27 s**
[03:03:26 → 03:09:53 EDT], git `d95b9fe`, clean tree at launch — `main` and
`origin/main` both at `d95b9fe`). Independent `probe_scan.py` re-sweep over all
approved images (`probe_scan-20261023.log`, exit 0; 03:22 → 03:31:13 EDT, the
slowest recorded sweep, itself a reminder that these two NFS walks on 166k paths
are wall-clock-unbounded):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py)) — **set
strictly identical to 2026-10-22**, including the grand total, so no row was added,
removed or re-statused: `approved` 166,200 · `tainted:extraction_nonface` 215,914 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference (run *before* `enrich`):** `auraface/` and
  `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13` on
  **tree-relative** paths (`LC_ALL=C` sort) → **0** and **0**. Zero LDA work is
  reachable from the entire tree, independent of the review DB.
  **Instrumentation note (new this run):** the two *incorrect* invocations of this
  check each emit `295468` / `295468` — an output that reads as "nothing has ever
  been projected". (a) `comm` fed a locale-ordered `sort` →
  `comm: file 1 is not in sorted order` + 295,468/295,468; (b) even C-sorted,
  comparing the raw `find auraface …` and `find lda …` listings **without stripping
  the top-level prefix** compares `auraface/faces/…` against `lda/faces/…` →
  295,468/295,468 again. The check only carries the meaning claimed here in the
  tree-relative, C-locale form captured in `preflight_setdiff-20261023.log`. This
  is the `mirror-preserves-the-centroid` failure mode in miniature: a check whose
  *broken* output is a plausible extreme rather than an obvious error.
* Post-run write verification via [`verify_writes.sh`](verify_writes.sh)
  (`verify_writes-20261023.log`): `find v1/lda -name '*.npy' -newermt '2026-10-23 00:00'`
  → **0**; `-newermt '2026-10-22 00:00'` → **0**; `-mmin -1440` → **0**; tree counts
  **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file in the tree is
  still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The
  "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`, `float32`,
  norms 152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0) — a single clean sequence, **no
self-restart** (unlike 2026-10-19 and 2026-10-22, where the CUDA-library re-exec
fired; the run is therefore the cheapest of the recent audited runs):
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

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` at 03:23 showed
only `xdg-desktop-portal-gnome` (12 MiB) — no eidolon process at all.

**Evidence-tracking caveat (applies to every section above, including this one).**
The `.log` files cited by name throughout this README are **not tracked** — the repo
carries a global `*.log` ignore (`.gitignore:59`), and `git ls-files | grep -c '\.log$'`
is **0** on the whole tree. The *scripts* that produce the numbers are committed
(`probe_scan.py`, `verify_writes.sh`, `status_census.py`, `scan_all_lda.py`), and the
raw `enrich` output of each run is inlined above, which is what makes the record
auditable in-repo. But a citation to `enrich-20261023.log` resolves only on this
machine. Suggested (not executed): either force-add the per-run logs
(`git add -f`) or drop the bare filenames in favour of the inlined output.

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 49` (was 48 at the 2026-10-22 run),
`last_run_at = 2026-10-22T03:15:58`, `next_run_at = 2026-10-24T03:00`,
`deliver: local`, `created_at = 2026-07-24T08:45:36`, `last_failure` still the
2026-10-19 worker-bookkeeping entry (failure-strike 0 since). `last_dispatch` for
this run: `scheduled_at 2026-10-23T03:00:00`, `dispatched_at 03:00:33`,
`lateness_seconds 33.9`, `kind: on_time`. The de-registration recommendation below
remains unexecuted, and the next run is already queued.

## Recommendation — de-register this cron job (no-op since the reprojection)

The cron entry is `2aec4dd52b5c` / `eidolon-lda-reproject`, `0 3 * * *`, created
2026-07-24, `repeat.completed = 49` as of the 2026-10-23 run, `last_status = ok`.
Each of the **fifteen** *audited* runs (2026-09-28, 2026-09-30, 2026-10-11, 2026-10-12,
2026-10-13, 2026-10-14, 2026-10-15, 2026-10-16, 2026-10-17, 2026-10-18, 2026-10-19,
2026-10-20, 2026-10-21, 2026-10-22, 2026-10-23) has written
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

## Re-run — 2026-10-26 (scheduled cron, same command)

Identical outcome for the **sixteenth** audited run. `enrich` exit 0, **0 files
written** (`enrich-20261026.log`; wall clock **5 m 04 s** [03:02:13 → 03:07:17 EDT],
git `15db7b1`, clean tree at launch — `main` at `15db7b1`; remote sync state not
re-checked this run). Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261026.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Independent checks:

* Post-run write verification via [`verify_writes.sh`](verify_writes.sh)
  (`verify_writes-20261026.log`, exit 0): `find v1/lda -name '*.npy' -newermt '2026-10-26 00:00'`
  → **0**; `-newermt '2026-10-25 00:00'` → **0**; `-mmin -1440` → **0**; tree counts
  **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file in the tree is
  still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The
  "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261026.log`).
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`, `float32`,
  norms 152.41–155.34 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart fired
(also seen 2026-10-19 and 2026-10-22); it does not change the material finding:
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

**Audit-coverage gap noted this run:** the cron registry shows the job's last run
was 2026-10-25T03:12 (`ok`) and its next is 2026-10-27T03:00, so runs on **10-24 and
10-25 did fire but were not appended to this README** (no `enrich-20261024.log` /
`enrich-20261025.log` exist here). They are presumed no-ops by continuity but are
**not audited**. The job remains `[active]`, `deliver: local`, `repeat: ∞`.

**Job state:** `enabled: true`, `state: scheduled`, `last_status: ok`,
`failure_streak: 0`; the 2026-10-19 worker-bookkeeping failure remains recovered.
The de-registration recommendation above remains unexecuted.

## Re-run — 2026-10-27 (scheduled cron, same command)

Identical outcome for the **seventeenth** audited run. `enrich` exit 0, **0 files
written** (`enrich-20261027.log`; wall clock **6 m 24 s** [03:01:05 → 03:07:29 EDT],
git `49db71f`, clean tree at launch — `main` and `origin/main` both at `49db71f`).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261027.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Independent checks:

* **Pre-flight whole-tree set-difference** (tree-relative, `LC_ALL=C`), re-run this
  sweep (`preflight_setdiff-20261027.log`): `auraface/` and `lda/` each hold exactly
  **295,468 `.npy`**; `comm -23` / `comm -13` → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree, independent of the review DB. (Invocation copied
  from the 2026-10-23 run's corrected form — see that run's instrumentation note.)
* Post-run write verification via [`verify_writes.sh`](verify_writes.sh)
  (`verify_writes-20261027.log`, exit 0): `find v1/lda -name '*.npy' -newermt '2026-10-27 00:00'`
  → **0**; `-newermt '2026-10-26 00:00'` → **0**; `-mmin -1440` → **0**; tree counts
  **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file in the tree is
  still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The
  "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261027.log`).
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`, `float32`,
  norms 152.61–154.79 — refit raw-coords convention, unchanged.
* **Audit-coverage gap from the 2026-10-26 run — CLOSED.** That run noted that the
  10-24 and 10-25 occurrences fired but were never appended here. Two independent
  sources now account for them: (a) `cron/executions.db` (read-only) shows both as
  `status = completed` with an empty `error` column — 10-24 `completed` 03:00:33 →
  03:15:21, 10-25 `completed` 03:00:33 → 03:12:27, i.e. same shape and duration as
  every audited no-op; and (b) `gapcheck-20261027.log`:
  `find lda -name '*.npy' -newermt '2026-10-24 00:00'` → **0** and
  `-newermt '2026-10-20 00:00'` → **0** — **no LDA file was written on any day since
  2026-10-20**, so the two unaudited occurrences provably wrote nothing. The gap was
  a *recording* gap, not a *measurement* gap.

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261027.log`) — **set identical to every prior audited run**, store
total included: `approved` 166,200 · `tainted:extraction_nonface` 215,914 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart fired
(also seen 2026-10-19, 2026-10-22, 2026-10-26); it does not change the material
finding:
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

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` at 03:08 showed
only `xdg-desktop-portal-gnome` (12 MiB) and `ollama` (20,256 MiB — another
project's process, not eidolon). No `insightface` CUDA work therefore ran, so this
job needed no GPU-scheduler reservation.

### Instrumentation notes (new this run)

Two *tooling* failures occurred while producing the audit itself. Both are recorded
because each one's failure state is a plausible-looking value, not an obvious error
— the `mirror-preserves-the-centroid` pattern again.

1. **A typo'd shell assignment silently produced an empty scan with exit 127.** The
   first `probe_scan.py` invocation was written as
   `.venv_py=$(echo /path/to/python)` — the leading `.` makes bash treat the line as
   `source`, so `probe_scan.py` never ran and `probe_scan-20261027.log` contained
   **only** `PROBE_EXIT=127`. Had the exit marker not been appended, an empty log
   would have read as "a scan that found nothing". Fix: invoke the interpreter by
   absolute path directly; never build it via a variable whose name can be
   mis-parsed as a command. **Lesson: an empty result file is not evidence of an
   empty result — check the exit code.**
2. **`set -u` aborts under the runner's login+interactive shell.** The pre-flight
   set-difference command began with `set -u` and died with
   `rvm_bash_nounset: unbound variable` (the runner invokes `bash -lic`, which
   sources rvm). A committed script that sets `set -u` *inside itself* (like
   `verify_writes.sh`) is fine; `set -u` on the command line of a backgrounded
   `bash -lic` command is not. Fix: drop `set -u` from the command line.

**Evidence-tracking caveat (carried, unresolved).** As recorded for the
2026-10-23 run, the `.log` files cited by name in this section are **not tracked**
(global `*.log` ignore, `.gitignore:59`). The raw `enrich` output is inlined above
and every producing script (`probe_scan.py`, `verify_writes.sh`, `status_census.py`)
is committed, so the record is auditable in-repo; but a citation to
`enrich-20261027.log` resolves only on this machine. Force-adding the per-run logs
remains a suggested, unexecuted operator decision — not changed unilaterally here.

**Job state, read after the run:** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 53` (was 49 at the 2026-10-23 run — the 10-24, 10-25, 10-26
occurrences plus this one), `last_run_at = 2026-10-26T03:16:58`,
`next_run_at = 2026-10-28T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`, `last_failure` still the 2026-10-19
worker-bookkeeping entry (failure-strike 0 since). `last_dispatch` for this run:
`scheduled_at 2026-10-27T03:00:00`, `dispatched_at 03:00:39`, `lateness_seconds
39.7`, `kind: on_time`. This run's own execution row is `status = running`
(finished on completion). The de-registration recommendation above remains
unexecuted, and the next run is already queued for 2026-10-28.

## Re-run — 2026-10-28 (occurrence fired, enrich KILLED — no-op proved indirectly)

The **2026-10-28** occurrence is closed here retrospectively: it appended no
section (the agent run was terminated at 03:02:10, see below), but its own
working-tree artifacts survive in this directory and are sufficient to record it.

* `preflight_setdiff-20261028.log` (03:02:00): `auraface/` and `lda/` each
  **295,468**; `comm -23` / `comm -13` → **0** and **0**.
* `status_census-20261028.log`: **identical** to every prior audited run —
  `approved` 166,200 · `tainted:extraction_nonface` 215,914 ·
  `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
  `tainted:approved_bad_geometry` 19 · **TOTAL 384,012**.
* `enrich-20261028.log` — **the sweep was killed**: the log holds the launch
  banner, the two `Skipping Stratum` / `Running AuraFace-only enrichment` lines,
  then
  ```
  bash: [796096: 1 (255)] tcsetattr: Inappropriate ioctl for device
  ENRICH_EXIT=143 WALL=3s END=2026-10-28 03:02:10 EDT
  ```
  **exit 143 = SIGTERM after 3 s**: the process was torn down together with its
  controlling terminal, before it ever reached the projection loop. There is no
  `Enrichment complete.` line.
* `cron/executions.db` (read-only) nevertheless records the 10-28 occurrence as
  `status = completed`, `error = NULL`, `started_at 03:00:17` →
  `finished_at 03:02:10` — **113 s**, one third of the fastest genuine sweep.

**Consequence — 10-28 is a *false* "completed".** The occurrence's status field
says `completed` while its `enrich` never ran: the finding is inherited from the
10-29 run's independent tree checks (`gapcheck-20261029.log`,
`verify_writes-20261029.log`): `find lda -name '*.npy' -newermt '2026-10-28 00:00'`
→ **0**, i.e. the 10-28 occurrence provably wrote nothing. It is a recording and
process failure, **not** a measurement failure — the material claim (0 files
written on 10-28) is independently corroborated, so 10-28 counts as a no-op on
evidence, but it was **not** a successful sweep. (`Completion status ≠ completion
evidence`: an exit-0-shaped registry row can wrap a SIGTERMed child.)

**Mitigation applied this run.** The kill appears to be launcher-side terminal
teardown, so the sweep is now launched through
[`run_enrich.sh`](run_enrich.sh) (detached; no pty attached to the child). The
2026-10-29 sweep ran to completion under it (337 s — see below), where the
pty-tracked 10-28 launch died at 3 s. Re-labelling: only *prior* runs (which
completed their sweep) carry a full audit; 10-28 is a killed-enrich occurrence
recorded for completeness.

## Re-run — 2026-10-29 (scheduled cron, same command)

Identical outcome for the **eighteenth** audited run. `enrich` exit 0, **0 files
written** (`enrich-20261029.log`; wall clock **337 s** [03:02:24 → 03:08:01 EDT],
git `b1542b4`, clean tree at launch — `main` and `origin/main` both at `b1542b4`;
launched detached via [`run_enrich.sh`](run_enrich.sh), see the 10-28 note).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261029.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py)) — **set
strictly identical to every prior audited run**, store total included: `approved`
166,200 · `tainted:extraction_nonface` 215,914 · `tainted:contamination` 1,626 ·
`tainted:unusable` 253 · `tainted:approved_bad_geometry` 19 · **TOTAL 384,012**.
No row added, removed, or re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`; now itself a
  committed script — [`preflight_setdiff.sh`](preflight_setdiff.sh) — so the
  C-locale, tree-relative form can no longer vary between runs):
  `preflight_setdiff-20261029.log` → `auraface/` and `lda/` each **295,468 `.npy`**;
  `comm -23` / `comm -13` → **0** and **0**. Zero LDA work is reachable from the
  *entire* tree, independent of the review DB.
* Post-run write verification via [`verify_writes.sh`](verify_writes.sh)
  (`verify_writes-20261029.log`, exit 0): `find v1/lda -name '*.npy' -newermt
  '2026-10-29 00:00'` → **0**; `-newermt '2026-10-28 00:00'` → **0**;
  `-mmin -1440` → **0**; tree counts **295,468** (`auraface/`) and **295,468**
  (`lda/`). Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261029.log`).
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`, `float32`,
  norms 152.61–154.79 — refit raw-coords convention, unchanged.
* `gapcheck-20261029.log`: `find lda -name '*.npy' -newermt '2026-10-20 00:00'`
  → **0** — no LDA file has been written on **any** day since 2026-10-20, which
  is what independently retires the 10-28 killed occurrence (above).

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart fired
(also seen 2026-10-19, 10-22, 10-26, 10-27); it does not change the material
finding:
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
`ENRICH_EXIT=0 WALL=337s` — the restart **is** visible this time as a doubled
sequence, consistent with the 2026-10-16 note that the line is an unreliable
witness in the other direction (a fired restart can *hide* the line by dropping
the userspace stdout buffer at `os.execv`); here both passes' output survived.

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` at 03:14:53
showed only `xdg-desktop-portal-gnome` (12 MiB). No `insightface` CUDA work ran,
so this job needed no GPU-scheduler reservation.

**Job state, read mid-run (03:1x EDT):** `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 55`, `last_run_at = 2026-10-28T03:02:10`,
`next_run_at = 2026-10-30T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`. This run completes as occurrence **56**.
The de-registration recommendation below remains unexecuted, and the next run is
already queued for 2026-10-30.

### Scripts committed this run

[`preflight_setdiff.sh`](preflight_setdiff.sh) (canonicalises the pre-flight
set-difference in its one correct form), [`run_enrich.sh`](run_enrich.sh)
(detached sweep launcher — removes the pty-teardown SIGTERM seen on 10-28),
[`cron_state.py`](cron_state.py) (read-only dump of this job's
`cron/executions.db` rows + registry block). Every number in this section is
produced by one of these or the pre-existing `probe_scan.py` / `verify_writes.sh`
/ `status_census.py`.

## Re-run — 2026-10-30 (scheduled cron, same command)

Identical outcome for the **nineteenth** audited run. `enrich` exit 0, **0 files
written** (`enrich-20261030.log`; wall clock **408 s** [03:02:00 → 03:08:48 EDT],
git `9b4cf43`, clean tree at launch — `main` and `origin/main` both at `9b4cf43`;
launched via [`run_enrich.sh`](run_enrich.sh)). Independent `probe_scan.py`
re-sweep over all approved images (`probe_scan-20261030.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261030.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the
  committed [`preflight_setdiff.sh`](preflight_setdiff.sh)): see below —
  `preflight_setdiff-20261030.log`. This run the pre-flight was executed *first*
  and separately; the write-verification half of the checks was moved into a new
  background script (below), because the pre-flight + `enrich` + verify + probe
  sequence no longer fits inside a single 420 s foreground window.
* Post-run write verification (`verify_writes-20261030.log`, exit 0):
  `find v1/lda -name '*.npy' -newermt '2026-10-30 00:00'` → **0**;
  `-newermt '2026-10-29 00:00'` → **0**; `-mmin -1440` → **0**; tree counts
  **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file in the tree is
  still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261030.log`).
* Sampled existing LDA files (five, `probe_scan` output): shape `(64,)`, `float32`,
  norms 152.61–154.79 — refit raw-coords convention, unchanged.
* `gapcheck-20261030.log`: `find lda -name '*.npy' -newermt '2026-10-20 00:00'`
  → **0** and `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any
  day since 2026-10-20, which independently retires the 10-28 killed occurrence
  and *this* run.

### Pre-flight observation — the earlier combined command was cut by the 420 s window

The pre-flight set-difference and the write verification both walk the full NFS
tree (~295k paths each). On this run the **first** attempt bundled
`basis-verify + status-census + verify_writes + gapcheck` into one foreground
command; the tool's foreground cap fired at **420 s** with the combined command
still on its *first* `find -printf` over `lda/`. Two consequences, both fixed
this run:

* The scale-verify and census commands printed
  `.venv/bin/python: No such file or directory` because they ran from the audit
  directory (the repo's `.venv` is only resolvable from the repo root). Re-run
  from `/home/tim/source/activity/eidolon` with `.venv/bin/python` → both
  succeeded (`basis_verify-20261030.log`, `status_census-20261030.log`).
  **Lesson: `enrich`'s own launcher `cd`s to the repo root for you; hand-run
  audit commands do not.**
* The NFS-heavy post-checks are now a committed background script,
  [`postchecks.sh`](postchecks.sh) (`bash postchecks.sh YYYYMMDD`), so the whole
  verify + gapcheck + `probe_scan` sequence runs detached as one tracked job
  instead of racing the foreground window.

Pre-flight result (`preflight_setdiff-20261030.log`, run **before** `enrich`):
`auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm
-13` on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work
is reachable from the *entire* tree, independent of the review DB.

Raw `enrich` output for this run (exit 0) — a **single clean sequence** (no
CUDA-library re-exec line; per the 2026-10-16 correction that line is *not* a
reliable witness in either direction, and it does not change the material
finding):
```
=== enrich launch Fri Oct 30 03:02:00 AM EDT 2026 (git 9b4cf43, LD_LIBRARY_PATH='<empty>') ===
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
ENRICH_EXIT=0 WALL=408s END=Fri Oct 30 03:08:48 AM EDT 2026
```

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` at 03:1x showed
only `xdg-desktop-portal-gnome` (12 MiB). No `insightface` CUDA work ran, so this
job needed no GPU-scheduler reservation.

**Job state, read after the run** (via [`cron_state.py`](cron_state.py),
`cron_state-20261030.log`): `enabled: true`, `state: scheduled`,
`last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 56` (the 10-24 / 10-25 / 10-26 / 10-27 / 10-28 / 10-29
occurrences plus their audit commits), `last_run_at = 2026-10-29T03:24:56`,
`next_run_at = 2026-10-31T03:00`, `deliver: local`,
`created_at = 2026-07-24T08:45:36`. This run's own `executions.db` row shows
`status = running` as of 03:0x (finishes on completion); occurrence **57**.
The 10-24 and 10-25 rows are now visible in `executions.db` as `completed`
(closing the recording gap noted on 2026-10-26); the 10-28 row is `completed`
with a 113 s duration — the **false** completed recorded in that section, which
this run's `gapcheck` again retires on evidence.

The de-registration recommendation below remains unexecuted, and the next run is
already queued for 2026-10-31.

### Scripts committed this run

[`postchecks.sh`](postchecks.sh) — the detached, NFS-heavy post-check runner
(`verify_writes` block + gapcheck + `probe_scan.py`), so the full verify sequence
is one background job rather than several foreground commands racing the 420 s
window. No other scripts changed.
