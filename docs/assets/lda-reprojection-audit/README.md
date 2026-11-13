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

## Re-run — 2026-10-31 (bare command FAILED — review-DB host unreachable; recovered via the local DB)

This is the **twentieth** audited run and the **first** in which the bare command
failed. Two findings this run, one mechanical and one structural.

### Finding 1 — the command as specified cannot reach the review DB

The launch `tools.hegre_dataset enrich --dataset …/hegre-faces/v1 --status approved
--skip-stratum` exited **1 in 6 s** (`enrich-20261031.bare-fail.log`) with:

```
psycopg2.OperationalError: connection to server at "192.168.86.49", port 5432
failed: No route to host
```

Independent probes: `ping 192.168.86.49` → `Destination Host Unreachable`;
`nc -zv 192.168.86.49 5432` → `No route to host`; ARP `192.168.86.49 dev wlo1 FAILED`.

### Finding 2 (root cause) — `192.168.86.49` is *this host's own* former Ethernet address

`journalctl -b -1 -u NetworkManager` shows, throughout the prior successful-run
window:

```
game NetworkManager[…]: dhcp4 (enp42s0): state changed new lease, address=192.168.86.49
  2026-10-27 … 2026-10-30
```

The box (`game`) **rebooted 2026-10-31 09:33**. On this boot `enp42s0` is `DOWN`
(no carrier) and the box is on **WiFi** (`wlo1`, profile `935-401`) at
**192.168.86.248**. `config/database.yml` hardcodes `host: 192.168.86.49`, so it now
points at a dead interface. The review DB is **local** to `game` (Postgres 18.6 on
`localhost:5432`) and its census is exactly the audited set below — i.e. the same DB
every prior audit read while `.49` resolved to this host. (The DB is untouched; this
is a name-resolution failure, not a data failure.)

### Recovery — identical sweep against `localhost` (no file, schedule, or DB mutation)

Re-ran the same command with the **documented** env override
`EIDOLON_DB_URL=postgresql+psycopg2://tim:***@localhost/eidolon` → the same local
Postgres over loopback instead of the host's own (ex-)LAN IP. `ENRICH_EXIT=0`, wall
clock **1451 s** (`enrich-20261031.log`). No config file, no schedule, and no DB row
was written. Result: **0 LDA files projected** —
*"All approved images already have AuraFace-LDA data."* — same as all 19 prior runs.

### Numbers (columns as before)

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261031.log`) — **set identical to every prior audited run**, store
total included: `approved` 166,200 · `tainted:extraction_nonface` 215,914 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused. (Read locally over loopback this run, since `.49` is down.)

### Independent checks (all before/after the run)

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh)): `preflight_setdiff-20261031.log` →
  `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13`
  on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree — **and this check is DB-independent**, so it holds
  even on a run where the DB is unreachable.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261031.log`, exit 0): `find v1/lda -name '*.npy' -newermt
  '2026-10-31 00:00'` → **0**; `-newermt '2026-10-30 00:00'` → **0**;
  `-mmin -1440` → **0**; tree counts **295,468** (`auraface/`) and **295,468**
  (`lda/`). Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261031.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261031.log`).
* `probe_scan.py` re-sweep over all approved images (`probe_scan-20261031.log`, exit 0):
  `missing auraface .npy: 23` · `missing LDA .npy: 23` (`of those no auraface: 23,
  not projectable`) · `missing z_g .npy: 9` · `missing pose.npy: 0`. Existing LDA
  samples (five): shape `(64,)`, `float32`, norms 152.61–154.79 — refit raw-coords
  convention, unchanged.

### Timing / occurrence notes

* **The occurrence fired at 09:41:31 EDT, not 03:00.** The box was off/rebooting at
  the scheduled time (boot 09:33), so the scheduler ran the missed occurrence
  post-boot. `executions.db` row for 10-31: `started_at 2026-10-31T09:41:31`.
* **The sweep is ~3.5–4× slower on WiFi than on the wired link** — **1451 s** vs the
  ~300–410 s of every 10-11…10-30 run — because the box's NFS mount to
  `192.168.86.43` now goes over `wlo1`. Measured mid-run: process in state `Dl`
  (`rpc_wait_bit_killable`), ~34 s CPU over 15 min elapsed; it is NFS-latency-bound,
  not compute-bound.
* The CUDA self-restart **fired** (visible as a doubled sequence in the log below),
  so the tree was walked twice — consistent with the 2026-10-16 note that the line is
  unreliable in the *other* direction, but here both passes' output survived.

Raw `enrich` output for this run (exit 0, via the `EIDOLON_DB_URL` override):
```
=== enrich launch Sat Oct 31 09:49:38 AM EDT 2026 (git 0e2a0e1, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=1451s END=Sat Oct 31 10:13:49 AM EDT 2026
```

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` showed **no**
compute apps (idle card). No `insightface` CUDA work ran, so this job needed no
GPU-scheduler reservation.

### Job state, read after the run ([`cron_state.py`](cron_state.py), `cron_state-20261031.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 57`, `last_run_at = 2026-10-30T03:37:23`,
`next_run_at = 2026-11-01T03:00`, `deliver: local`, `created_at = 2026-07-24T08:45:36`.
`executions.db` since 2026-10-29: 10-29 `completed`, 10-30 `completed`, 10-31
`running` (this occurrence; finishes on completion). `main == origin/main` at
`0e2a0e1` at launch; this run commits and pushes its own record.

### Recommendation — TWO operator actions, both unexecuted here

1. **De-register this cron job** (re-stated from the section above; still no-op).
2. **Fix the DB host reference — this is new and now blocking.** `config/database.yml`
   hardcodes `host: 192.168.86.49`, which is `game`'s *own* Ethernet address and is
   dead whenever `enp42s0` is down (WiFi-only boot, unplugged cable, or a DHCP change
   like today's). The DB is local, so the robust value is `localhost` (or `127.0.0.1`).
   Until then, **every** run of this job fails unless `EIDOLON_DB_URL` is exported —
   and the failure is a hard stop (exit 1), not the usual benign no-op. Suggested
   one-line change: `config/database.yml` `host: 192.168.86.49` → `host: localhost`.
   Not executed here — a committed shared-config change is the operator's call, and a
   job run may not edit its own inputs.

### Interpretation caveat

The material finding (**0 files written**) does **not** depend on the DB: it is
corroborated by the DB-independent pre-flight set-difference (295,468 == 295,468,
0/0) and the post-run `find -newermt` scans. The DB was needed only to enumerate the
approved set, which the census shows is unchanged. So this run's verdict — *job
complete, nothing to project* — stands on evidence even though the bare command
could not run.

## Re-run — 2026-11-01 (scheduled cron, same command)

Identical outcome for the **twenty-first** audited run. Unlike 2026-10-31, the
**bare command ran as specified** (no `EIDOLON_DB_URL` override) — the DB-host
blockage had cleared by environment, not by repair (Finding below). `enrich` exit 0,
**0 files written** (`enrich-20261101.log`; wall clock **462 s** [03:03:05 → 03:10:47
EST], git `154079b`, clean tree at launch — `main` and `origin/main` both at
`154079b`; launched detached via [`run_enrich.sh`](run_enrich.sh)). Independent
`probe_scan.py` re-sweep over all approved images (`probe_scan-20261101.log`, exit 0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 | 11-01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261101.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh); 03:01:24 → 03:03:05, 101 s):
  `preflight_setdiff-20261101.log` → `auraface/` and `lda/` each hold exactly
  **295,468 `.npy`**; `comm -23` / `comm -13` on tree-relative, `LC_ALL=C`-sorted
  paths → **0** and **0**. Zero LDA work is reachable from the *entire* tree,
  independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261101.log`, exit 0): `find v1/lda -name '*.npy' -newermt
  '2026-11-01 00:00'` → **0**; `-newermt '2026-10-31 00:00'` → **0**;
  `-mmin -1440` → **0**; tree counts **295,468** (`auraface/`) and **295,468**
  (`lda/`). Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261101.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261101.log`).
* `probe_scan.py` re-sweep (`probe_scan-20261101.log`, PROBE_EXIT=0): `missing
  auraface .npy: 23` · `missing LDA .npy: 23` (`of those no auraface: 23, not
  projectable`) · `missing z_g .npy: 9` · `missing pose.npy: 0`. Existing LDA
  samples (five): shape `(64,)`, `float32`, norms 152.61–154.79 — refit raw-coords
  convention, unchanged.

### Finding — the 2026-10-31 DB-host blockage cleared by the environment, not by a repair

The blocker recorded on 2026-10-31 was that `config/database.yml` hardcodes
`host: 192.168.86.49` — `game`'s **own** Ethernet address — which went dead when
`enp42s0` was down on the post-reboot WiFi-only boot. This run:

* `enp42s0` is **UP** again at `192.168.86.49/24` and `wlo1` is **DOWN**;
  `nc -zv 192.168.86.49 5432` → **succeeded**; the bare command therefore reached
  the local review DB with no `EIDOLON_DB_URL` override. Route to the NAS
  (`192.168.86.43`) is now `dev enp42s0 src 192.168.86.49`.
* **`config/database.yml` still reads `host: 192.168.86.49`** (line 8, unchanged).
  So the 2026-10-31 operator action is **still unapplied** — this run succeeded only
  because the wired link happened to return. The latent bug is intact: any future
  WiFi-only boot, unplugged cable, or DHCP change re-breaks the **bare** command
  (exit 1), with the same local-`localhost` workaround. Today is evidence the fault
  is *environment-dependent*, not *fixed*.
* **Stale NFS mount attribute:** `mount` still reports
  `clientaddr=192.168.86.248` (the 10-31 WiFi address) even though `wlo1` is now
  DOWN and all NAS traffic flows over `enp42s0` (`.49`). The mount is functional
  (every `find`/`probe_scan` walk above completed), so this is cosmetic mount
  metadata left over from the 10-31 automount, not a live misconfiguration —
  recorded because a stale `clientaddr` on a *down* interface is exactly the kind of
  value that reads as a fault when it is not one.
* **Timing:** 462 s wall clock sits in the wired band (~300–410 s of 10-11…10-30)
  plus the CUDA self-restart's second sweep — **not** the 1451 s WiFi band of
  10-31. Consistent with the wired link being back.

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart fired
(also seen 10-19, 10-22, 10-26, 10-27, 10-29, 10-31); per the 2026-10-16 correction
the `Restarting subprocess…` line is *not* a reliable witness in either direction,
and it does not change the material finding:
```
=== enrich launch Sun Nov  1 03:03:05 AM EST 2026 (git 154079b, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=462s END=Sun Nov  1 03:10:47 AM EST 2026
```

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` showed only
`xdg-desktop-portal-gnome` (33 MiB). No `insightface` CUDA work ran, so this job
needed no GPU-scheduler reservation.

### Timing / occurrence notes

* **The occurrence fired on time at 03:00:35 EST** (unlike 10-31, which fired late
  post-boot at 09:41): `executions.db` row for 11-01 → `started_at
  2026-11-01T03:00:35`, `status = running` (finishes on completion). Occurrence
  **59**.

### Job state, read after the run ([`cron_state.py`](cron_state.py), `cron_state-20261101.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 58`, `last_run_at = 2026-10-31T10:25:01`,
`next_run_at = 2026-11-02T03:00`, `deliver: local`, `created_at = 2026-07-24T08:45:36`.
`executions.db` since 2026-10-29: 10-29 `completed`, 10-30 `completed`, 10-31
`completed`, 11-01 `running` (this occurrence). `main == origin/main` at `154079b` at
launch; this run commits and pushes its own record.

The de-registration recommendation below remains unexecuted, and the next run is
already queued for 2026-11-02. The DB-host repair (2026-10-31 action 2) also remains
unexecuted and is again the live risk — today's run only masks it.

## Re-run — 2026-11-02 (scheduled cron, same command)

Identical outcome for the **twenty-second** audited run. The **bare command ran as
specified** (no `EIDOLON_DB_URL` override) — the wired link is up again, so `.49`
resolves. `enrich` exit 0, **0 files written** (`enrich-20261102.log`; wall clock
**547 s** [03:02:30 → 03:11:37 EST], git `53856c4`, clean tree at launch — `main` and
`origin/main` both at `53856c4`; launched detached via [`run_enrich.sh`](run_enrich.sh)).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261102.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 | 11-01 | 11-02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261102.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh); 03:00:54): `preflight_setdiff-20261102.log`
  → `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13`
  on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree, independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261102.log`, exit 0): `find v1/lda -name '*.npy' -newermt
  '2026-11-02 00:00'` → **0**; `-newermt '2026-11-01 00:00'` → **0**; `-mmin -1440`
  → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file
  in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The \"0 files written\" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261102.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261102.log`).
* `probe_scan.py` re-sweep (`probe_scan-20261102.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g
  .npy: 9` · `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart fired
(doubled sequence; also seen 10-19, 10-22, 10-26, 10-27, 10-29, 10-31, 11-01). Per the
2026-10-16 correction the `Restarting subprocess…` line is *not* a reliable witness in
either direction, and it does not change the material finding:
```
=== enrich launch Mon Nov  2 03:02:30 AM EST 2026 (git 53856c4, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=547s END=Mon Nov  2 03:11:37 AM EST 2026
```

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` at 03:29 showed
only `xdg-desktop-portal-gnome` (33 MiB). No `insightface` CUDA work ran, so this job
needed no GPU-scheduler reservation.

### Environment / occurrence notes

* **Wired link up; the bare command works.** `enp42s0` **UP** at `192.168.86.49/24`
  (`wlo1` down) — route to the NAS (`192.168.86.43`) is `dev enp42s0 src
  192.168.86.49`; `nc -zv 192.168.86.49 5432` → **succeeded**, so the review DB was
  reached with no `EIDOLON_DB_URL` override. As on 11-01, **`config/database.yml` still
  reads `host: 192.168.86.49`** — the 2026-10-31 operator action remains unapplied and
  the latent bug is intact (any WiFi-only boot / unplugged cable / DHCP change
  re-breaks the bare command with exit 1).
* **Stale NFS mount attribute persists (cosmetic).** `mount` still reports
  `clientaddr=192.168.86.248` (the 10-31 WiFi address) although all NAS traffic flows
  over `enp42s0` (`.49`). The mount is functional — every `find`/`probe_scan` walk above
  completed — so this is leftover automount metadata, not a live misconfiguration.
* **Timing:** 547 s sits just above the wired band (~300–462 s of 10-11…11-01) and
  well below the 1451 s WiFi band of 10-31; consistent with the wired link plus the
  restart's second sweep (the doubled sequence is visible above).
* **The occurrence fired on time at 03:00:30 EST** (`executions.db` row for 11-02 →
  `started_at 2026-11-02T03:00:30`, `status = running`, finishes on completion).
  Occurrence **60**.

### Job state, read after the run ([`cron_state.py`](cron_state.py), `cron_state-20261102.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 59`, `last_run_at = 2026-11-01T03:29:16`,
`next_run_at = 2026-11-03T03:00`, `deliver: local`, `created_at = 2026-07-24T08:45:36`.
`executions.db` since 2026-10-29: 10-29 `completed`, 10-30 `completed`, 10-31
`completed`, 11-01 `completed`, 11-02 `running` (this occurrence). `main == origin/main`
at `53856c4` at launch; this run commits and pushes its own record.

The de-registration recommendation (2026-10-23, restated 10-31) remains unexecuted, and
the next run is already queued for 2026-11-03. The DB-host repair (2026-10-31 action 2)
also remains unexecuted — today's clean run only masks it.

## Re-run — 2026-11-03 (scheduled cron, same command)

Identical outcome for the **twenty-third** audited run. The **bare command ran as
specified** (no `EIDOLON_DB_URL` override) — the wired link is up, so `.49` resolves.
`enrich` exit 0, **0 files written** (`enrich-20261103.log`; wall clock **389 s**
[03:09:20 → 03:15:49 EST], git `e5f5e51`, clean tree at launch — `main` and
`origin/main` both at `e5f5e51`; launched via [`run_enrich.sh`](run_enrich.sh)).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261103.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 | 11-01 | 11-02 | 11-03 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261103.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh)): `preflight_setdiff-20261103.log`
  → `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13`
  on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree, independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261103.log`): `find v1/lda -name '*.npy' -newermt
  '2026-11-03 00:00'` → **0**; `-newermt '2026-11-02 00:00'` → **0**; `-mmin -1440`
  → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file
  in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261103.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261103.log`).
* `probe_scan.py` re-sweep (`probe_scan-20261103.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g
  .npy: 9` · `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart did **not**
fire this time (single sequence; contrast 10-19, 10-22, 10-26, 10-27, 10-29, 10-31,
11-01, 11-02), which per the 2026-10-16 correction is not a reliable witness in either
direction and does not change the material finding:
```
=== enrich launch Tue Nov  3 03:09:20 AM EST 2026 (git e5f5e51, LD_LIBRARY_PATH='<empty>') ===
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
ENRICH_EXIT=0 WALL=389s END=Tue Nov  3 03:15:49 AM EST 2026
```

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` showed only
`xdg-desktop-portal-gnome` (33 MiB). No `insightface` CUDA work ran, so this job
needed no GPU-scheduler reservation.

### Environment / occurrence notes

* **Wired link up; the bare command works.** `enp42s0` **UP** at `192.168.86.49/24`
  (`wlo1` down); `config/database.yml` reads `host: 192.168.86.49`, so the review DB
  was reached with no `EIDOLON_DB_URL` override. The 2026-10-31 operator action
  (its 2 remaining items) is still unapplied — the latent bug is intact (any
  WiFi-only boot / unplugged cable / DHCP change re-breaks the bare command with
  exit 1). Today's clean run only masks it.
* **Timing:** 389 s sits squarely inside the wired band (~300–462 s of 10-11…11-01),
  well below the 1451 s WiFi band of 10-31.
* **The occurrence fired on time at 03:00:29 EST** (`executions.db` row for 11-03 →
  `started_at 2026-11-03T03:00:29`, `status = running`, finishes on completion).
  Occurrence **61**.

### Job state, read after the run ([`cron_state.py`](cron_state.py), `cron_state-20261103.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 60`, `last_run_at = 2026-11-02T03:29:52`,
`next_run_at = 2026-11-04T03:00`, `deliver: local`. `executions.db` since 2026-11-01:
11-01 `completed`, 11-02 `completed`, 11-03 `running` (this occurrence). Job id
`2aec4dd52b5c`. `main == origin/main` at `e5f5e51` at launch; this run commits and
pushes its own record.

The de-registration recommendation (2026-10-23, restated 10-31) remains unexecuted, and
the next run is already queued for 2026-11-04. The DB-host repair (2026-10-31 action 2)
also remains unexecuted — today's clean run only masks it.


## Re-run — 2026-11-04 (scheduled cron, same command)

Identical outcome for the **twenty-fourth** audited run. The **bare command ran as
specified** (no `EIDOLON_DB_URL` override) — the wired link is up, so `.49` resolves.
`enrich` exit 0, **0 files written** (`enrich-20261104.log`; wall clock **252 s**
[07:15:12 → 07:19:24 EST], git `9b377a7`, clean tree at launch — `main` and
`origin/main` both at `9b377a7`; launched via [`run_enrich.sh`](run_enrich.sh)).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261104.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 | 11-01 | 11-02 | 11-03 | 11-04 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261104.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh)): `preflight_setdiff-20261104.log`
  → `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13`
  on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree, independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261104.log`): `find v1/lda -name '*.npy' -newermt
  '2026-11-04 00:00'` → **0**; `-newermt '2026-11-03 00:00'` → **0**; `-mmin -1440`
  → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file
  in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261104.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261104.log`).
* `probe_scan.py` re-sweep (`probe_scan-20261104.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g
  .npy: 9` · `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0):
```
=== enrich launch Wed Nov  4 07:15:12 AM EST 2026 (git 9b377a7, LD_LIBRARY_PATH='<empty>') ===
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
ENRICH_EXIT=0 WALL=252s END=Wed Nov  4 07:19:24 AM EST 2026
```

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` showed a single
foreign `python` (18,205 MiB — the ComfyUI server on `:8188`), not this job. No
`insightface` CUDA work ran, so this job needed no GPU-scheduler reservation.

### Environment / occurrence notes

* **Wired link up; the bare command works.** `enp42s0` **UP** at `192.168.86.49/24`;
  `config/database.yml` reads `host: 192.168.86.49`, so the review DB was reached
  with no `EIDOLON_DB_URL` override. The 2026-10-31 operator action (its 2 remaining
  items) is still unapplied — the latent bug is intact (any WiFi-only boot / unplugged
  cable / DHCP change re-breaks the bare command with exit 1). Today's clean run only
  masks it.
* **Timing:** 252 s — the fastest of the recent wired runs (prior wired band
  ~300–462 s of 10-11…11-01); well below the 1451 s WiFi band of 10-31.
* **The occurrence fired late at 07:04:29 EST** (`executions.db` row for 11-04 →
  `started_at 2026-11-04T07:04:29`, `status = running`), **not** at its scheduled
  `03:00` — the box was evidently not up/awake at 03:00 and the scheduler caught up
  when the session started. Occurrence **62**.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261104.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 61`, `last_run_at = 2026-11-03T03:33:40`,
`next_run_at = 2026-11-05T03:00`, `deliver: local`. `executions.db` since 2026-11-01:
11-01 `completed`, 11-02 `completed`, 11-03 `completed`, 11-04 `running` (this
occurrence). Job id `2aec4dd52b5c`. `main == origin/main` at `9b377a7` at launch; this
run commits and pushes its own record.

The de-registration recommendation (2026-10-23, restated 10-31) remains unexecuted, and
the next run is already queued for 2026-11-05. The DB-host repair (2026-10-31 action 2)
also remains unexecuted — today's clean run only masks it.


## Re-run — 2026-11-05 (scheduled cron, same command)

Identical outcome for the **twenty-fifth** audited run. The **bare command ran as
specified** (no `EIDOLON_DB_URL` override) — the wired link is up, so `.49` resolves.
`enrich` exit 0, **0 files written** (`enrich-20261105.log`; wall clock **404 s**
[03:01:59 → 03:08:43 EST], git `bd3955c`, clean tree at launch — `main` and
`origin/main` both at `bd3955c`; launched via [`run_enrich.sh`](run_enrich.sh)).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261105.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 | 11-01 | 11-02 | 11-03 | 11-04 | 11-05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261105.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh)): `preflight_setdiff-20261105.log`
  → `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13`
  on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree, independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261105.log`): `find v1/lda -name '*.npy' -newermt
  '2026-11-05 00:00'` → **0**; `-newermt '2026-11-04 00:00'` → **0**; `-mmin -1440`
  → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file
  in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261105.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261105.log`).
* `probe_scan.py` re-sweep (`probe_scan-20261105.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g
  .npy: 9` · `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0):
```
=== enrich launch Thu Nov  5 03:01:59 AM EST 2026 (git bd3955c, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=404s END=Thu Nov  5 03:08:43 AM EST 2026
```

**Behavioral difference vs. every prior audited run:** the run now takes the
CUDA-lib re-exec branch in `tools/hegre_dataset/enrichment.py:216-227` —
`Restarting subprocess to load CUDA libraries...` — which it did **not** take on
11-04. The branch fires only when `…/experiments/geometry_pca/.venv/lib/python3.14/
site-packages/nvidia/cudnn/lib` **exists** *and* is absent from `LD_LIBRARY_PATH`;
if so it `os.execv`s the whole process with cuDNN/cuBLAS on the loader path. That
path evidently became present since the last run. The re-exec doubles the visible
pass and accounts for the wall-clock step from 11-04's **252 s** to **404 s**
(≈ +150 s). It changes no outcome: the restarted subprocess still hits
`import insightface` → `ImportError` → `Skipping AuraFace extraction`, so the 23
AuraFace-less images remain untouched and no GPU work runs. Recorded here because
**if `insightface` were ever importable in this venv, this branch would launch
AuraFace GPU extraction outside the GPU scheduler** — a latent path worth watching,
not a fault that fired today.

No eidolon process held the GPU: `nvidia-smi --query-compute-apps` showed a single
foreign `python` (14,061 MiB — the ComfyUI server on `:8188`, `main.py --listen
0.0.0.0 --enable-manager …`, uid `tim`, ~20 h uptime), not this job. No `insightface`
CUDA work ran (the re-exec only set the loader path; the subprocess then failed on
import before any CUDA call), so this job needed no GPU-scheduler reservation.

### Environment / occurrence notes

* **Wired link up; the bare command works.** `enp42s0` **UP** at `192.168.86.49/24`;
  `config/database.yml` reads `host: 192.168.86.49`, so the review DB was reached
  with no `EIDOLON_DB_URL` override. The 2026-10-31 operator action (its 2 remaining
  items) is still unapplied — the latent bug is intact (any WiFi-only boot / unplugged
  cable / DHCP change re-breaks the bare command with exit 1). Today's clean run only
  masks it.
* **Timing:** 404 s — inside the wired band (~300–462 s of 10-11…11-01), but no
  longer the fastest wired run: the CUDA re-exec (above) added ~150 s over 11-04's
  252 s. Still well below the 1451 s WiFi band of 10-31.
* **The occurrence fired on time at 03:00:15 EST** (`executions.db` row for 11-05 →
  `started_at 2026-11-05T03:00:15`, `status = running`), back at its scheduled
  `03:00` — unlike 11-04's late 07:04:29 catch-up. Occurrence **63**.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261105.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 62`, `last_run_at = 2026-11-04T07:33:11`,
`next_run_at = 2026-11-06T03:00`, `deliver: local`. `executions.db` since 2026-11-01:
11-01 `completed`, 11-02 `completed`, 11-03 `completed`, 11-04 `completed`, 11-05
`running` (this occurrence). Job id `2aec4dd52b5c`. `main == origin/main` at `bd3955c`
at launch; this run commits and pushes its own record.

The de-registration recommendation (2026-10-23, restated 10-31) remains unexecuted, and
the next run is already queued for 2026-11-06. The DB-host repair (2026-10-31 action 2)
also remains unexecuted — today's clean run only masks it.


## Re-run — 2026-11-07 (scheduled cron, same command)

Identical outcome for the **twenty-sixth** audited run. The **bare command ran as
specified** (no `EIDOLON_DB_URL` override) — the wired link is up, so `.49` resolves.
`enrich` exit 0, **0 files written** (`enrich-20261107.log`; wall clock **371 s**
[03:27:37 → 03:33:48 EST], git `1f252d6`, clean tree at launch — `main` and
`origin/main` both at `1f252d6`; launched detached via [`run_enrich.sh`](run_enrich.sh)).
Independent `probe_scan.py` re-sweep over all approved images
(`probe_scan-20261107.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | 10-11 | 10-12 | 10-13 | 10-14 | 10-15 | 10-16 | 10-17 | 10-18 | 10-19 | 10-20 | 10-21 | 10-22 | 10-23 | 10-26 | 10-27 | 10-28 | 10-29 | 10-30 | 10-31 | 11-01 | 11-02 | 11-03 | 11-04 | 11-05 | 11-07 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261107.log`) — **set strictly identical to every prior audited
run**, store total included: `approved` 166,200 · `tainted:extraction_nonface`
215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or
re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh)): `preflight_setdiff-20261107.log`
  → `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13`
  on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is
  reachable from the *entire* tree, independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261107.log`): `find v1/lda -name '*.npy' -newermt
  '2026-11-07 00:00'` → **0**; `-newermt '2026-11-06 00:00'` → **0**; `-mmin -1440`
  → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file
  in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261107.log`): `find lda … -newermt '2026-10-20 00:00'`
  → **0**; `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day
  since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261107.log`).
* `probe_scan.py` re-sweep (`probe_scan-20261107.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g
  .npy: 9` · `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`,
  `float32`, norms 152.61–154.79 — refit raw-coords convention, unchanged.
* **Numeric basis recompute added this run** (not in the standard probe set): for 50
  approved images, `project_to_lda(clean_auraface(auraface.npy))` with the basis now
  in `experiments/geometry_pca/output/` was compared against the on-disk `lda/*.npy`
  — `max_abs_diff` 9.2e-07 … 1.9e-06 (float32 quantisation only; 0 bit-exact, 50/50
  shape `(64,)`). Confirms by direct recomputation what the fingerprint stamp asserts.
  A separate sweep of the **2,220 pre-refit files** (mtime 2026-07-03→07-06) matched
  the 2026-09-28 finding: **0 approved** images hold a pre-refit LDA file (2,219
  `tainted:extraction_nonface` + 1 `tainted:contamination`).

Raw `enrich` output for this run (exit 0):
```
=== enrich launch Sat Nov  7 03:27:37 AM EST 2026 (git 1f252d6, LD_LIBRARY_PATH='<empty>') ===
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Restarting subprocess to load CUDA libraries...
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
ENRICH_EXIT=0 WALL=371s END=Sat Nov  7 03:33:48 AM EST 2026
```

The CUDA-lib re-exec branch (`tools/hegre_dataset/enrichment.py:216-227`) fired again
— `Restarting subprocess to load CUDA libraries...` — as on 10-19, 10-22 and 11-05.
The restarted subprocess still hits `import insightface` → `ImportError`, so no GPU
work ran and the 23 AuraFace-less images remain untouched. The latent path stands: **if
`insightface` were ever importable in this venv, this branch would launch AuraFace GPU
extraction outside the GPU scheduler.**

### Audit-coverage gap — the 2026-11-06 occurrence

The 11-06 occurrence fired on schedule (`executions.db` row → `started_at
2026-11-06T03:00:38`, `finished_at 03:08:44`, `status = completed`) but left **no
record here and no commit** — no `*-20261106.log` exists in this dir and the newest
commit remains the 11-05 record (`1f252d6`). Its outcome is presumed a no-op by
continuity (the tree is unchanged — newest file still 2026-07-28) but it is **not
audited**. This run is therefore counted as the **26th audited**, not the 27th.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261107.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 64`, `last_run_at = 2026-11-06T03:08:44`,
`next_run_at = 2026-11-08T03:00`, `deliver: local`. `executions.db` since 2026-11-01:
11-01 … 11-06 `completed`, 11-07 `running` (this occurrence). Job id `2aec4dd52b5c`.
`main == origin/main` at `1f252d6` at launch; this run commits and pushes its own record.

**Timing:** 371 s — inside the wired band and the fastest wired run recorded (the
CUDA re-exec still doubles the visible pass, but ~7 m under the 1451 s WiFi band of
10-31).

The de-registration recommendation (2026-10-23, restated 10-31) remains unexecuted, and
the next run is already queued for 2026-11-08. The DB-host repair (2026-10-31 action 2)
also remains unexecuted — today's clean run only masks it.

## Re-run — 2026-11-09 (scheduled cron, same command)

Identical outcome for the **twenty-seventh** audited run. The **bare command ran as
specified** (no `EIDOLON_DB_URL` override) — the wired link (`enp42s0`, `192.168.86.49`)
is up, `nc -zv 192.168.86.49 5432` succeeded, so `.49` resolves and `config/database.yml`
reaches the local review DB unchanged. `enrich` exit 0, **0 files written**
(`enrich-20261109.log`; wall clock **402 s** [03:04:11 → 03:10:53 EST], git `f109517`,
clean tree at launch — `main` and `origin/main` both at `f109517`; launched detached via
[`run_enrich.sh`](run_enrich.sh)). Independent `probe_scan.py` re-sweep over all approved
images (`probe_scan-20261109.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | … | 11-05 | 11-07 | 11-09 |
|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | … | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | … | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | … | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | … | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | … | 0 | 0 | **0** |

(All intermediate audited runs 10-11…11-05 are identical; full columns are in their own
sections above.)

Review-DB status census (via [`status_census.py`](status_census.py)) — **set strictly
identical to every prior audited run**, store total included: `approved` 166,200 ·
`tainted:extraction_nonface` 215,914 · `tainted:contamination` 1,626 ·
`tainted:unusable` 253 · `tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row
added, removed, or re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (run *before* `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh), 03:03:08): `auraface/` and `lda/` each
  hold exactly **295,468 `.npy`**; `comm -23` / `comm -13` on tree-relative, `LC_ALL=C`-sorted
  paths → **0** and **0**. Zero LDA work is reachable from the *entire* tree, independent
  of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261109.log`): `find v1/lda -name '*.npy' -newermt '2026-11-09 00:00'`
  → **0**; `-newermt '2026-11-08 00:00'` → **0**; `-mmin -1440` → **0**; tree counts
  **295,468** (`auraface/`) and **295,468** (`lda/`). Newest file in the tree is still
  **2026-07-28 14:42** (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`).
  The "0 files written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261109.log`): `find lda … -newermt '2026-10-20 00:00'` → **0**;
  `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match.
* `probe_scan.py` re-sweep (`probe_scan-20261109.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g .npy: 9`
  · `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`, `float32`, norms
  152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for this run (exit 0) — the CUDA-library self-restart fired (doubled
sequence; also seen 10-19, 10-22, 10-26, 10-27, 10-29, 10-31, 11-01, 11-02, 11-05, 11-07);
per the 2026-10-16 correction the `Restarting subprocess…` line is *not* a reliable witness
in either direction, and it does not change the material finding:
```
=== enrich launch Mon Nov  9 03:04:11 AM EST 2026 (git f109517, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=402s END=Mon Nov  9 03:10:53 AM EST 2026
```

No eidolon process held the GPU: the run's only GPU-relevant branch was the CUDA-lib
re-exec, whose restarted subprocess still failed at `import insightface` (`ImportError`)
before any CUDA call, so no GPU work ran and no GPU-scheduler reservation was needed. The
latent path stands (unchanged): **if `insightface` were ever importable in this venv, this
branch would launch AuraFace GPU extraction outside the GPU scheduler.**

### Audit-coverage gaps (carried)

The **11-06** and **11-08** occurrences fired on schedule (`executions.db` rows →
`completed`, `finished_at` 03:08:44 and 03:20:05 respectively) but left **no README record
and no commit**. Both are presumed no-ops by continuity (the tree is unchanged — newest
file still 2026-07-28; the 11-08 delivery report confirms 0 projected) but are **not
audited**. This run is therefore counted as the **27th audited**, not the 28th.

### Job state, read during the run ([`cron_state.py`](cron_state.py))

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 66`, `last_run_at = 2026-11-08T03:20:05`,
`next_run_at = 2026-11-10T03:00`, `deliver: local`, `created_at = 2026-07-24T08:45:36`.
`executions.db` since 2026-11-01: 11-01 … 11-06 `completed`, 11-07 `completed`, 11-08
`completed`, 11-09 `running` (this occurrence, `started_at 2026-11-09T03:00:39`). Job id
`2aec4dd52b5c`. `main == origin/main` at `f109517` at launch; this run commits and pushes
its own record.

**Timing:** 402 s — inside the wired band (~250–547 s of 10-11…11-07), far below the
1451 s WiFi band of 10-31. The CUDA re-exec doubles the visible pass; the occurrence fired
on time at 03:00:39 EST.

The de-registration recommendation (2026-10-23, restated 10-31 and every run since) remains
unexecuted, and the next run is already queued for 2026-11-11. The DB-host repair (2026-10-31
action 2) also remains unexecuted — today's clean run only masks it.

## Re-run — 2026-11-10 (fired 08:00 on reboot, not 03:00 — host down overnight; first attempt hit a recovering DB)

Identical **material** outcome for the **twenty-eighth** audited run — `enrich` exit 0,
**0 files written** — but this occurrence is *not* routine: **two availability events** bracket
it, and the first `enrich` attempt failed.

### Event 1 — the 03:00 occurrence never fired; the host was powered off ~7 h

`journalctl --list-boots` places the end of the previous boot session at
**2026-11-10 00:45:05 EST** and the start of the current one at **07:57:47 EST**: `game` was
**down ≈ 7 h 12 m**. The scheduled `03:00 EST` occurrence therefore never ran — `executions.db`
carries **no 11-10 row until the catch-up fire at `08:00:32`**, and `repeat.completed` advanced
66 → 67 only on that fire. (Forensic caution: the kernel line `PM: RTC time: 05:45:55, date:
2026-11-10` shows the box came up with a ~7 h-stale RTC, so **`uptime -s` misreports the boot as
`00:45:53`** — the *previous* session's end. `who -b` and the journal boot-id list agree on
**07:57:47**; after a stale-RTC boot, trust those, not `uptime -s`.)

This is the **third** audited occurrence to fire off its `03:00` slot — after **10-31** (09:41)
and **11-04** (07:04) — and each time it fired **on reboot**, i.e. the scheduler runs the missed
occurrence when the box comes up. The pattern is now three-for-three: a reboot before 03:00
converts the nightly fire into a morning catch-up.

### Event 2 — the first `enrich` attempt died on a PostgreSQL in crash recovery

Attempt 1 (launched `08:02:10`) exited **`ENRICH_EXIT=1` in 8 s** with
`FATAL: the database system is starting up` — preserved in
[`enrich-20261110.fail.log`](enrich-20261110.fail.log). The unclean shutdown left postgres 18
crash-recovering (`database system was interrupted; last known up at 2026-11-10 00:08:49 EST`);
the unit began listening at `07:58:02` and reported `Started` at `07:59:01` (with
`pg_ctl: server did not start in time`), but **rejected connections until recovery finished
≈ 08:02:4x**. Critically, **`nc -zv 192.168.86.49 5432` succeeded throughout** — the port was
open the whole time — so the readiness probe used in every earlier audit was *blind* to this
mode; only `pg_isready` (`rejecting connections` → `accepting connections` at `08:02:42`)
distinguishes "listening" from "ready". The sweep was re-launched at **08:02:46** against the
recovered DB.

**Readiness rule (new, recorded here):** after a reboot of this box, gate the bare command on
`pg_isready -h 192.168.86.49 -p 5432`, **not** `nc` — a listening-but-recovering postgres passes
the TCP probe and fails the query (the same lesson as the 10-31 section, one layer down: then
the host was unreachable; now the *host* is up and the *database* is not).

### Outcome — second attempt, `enrich` exit 0, 0 files written

`enrich-20261110.log`; wall clock **298 s** [`08:02:46` → `08:07:44 EST`], git `345d517`, clean
tree at launch — `main` and `origin/main` both at `345d517`. Independent `probe_scan.py`
re-sweep over all approved images (`probe_scan-20261110.log`, PROBE_EXIT=0):

| quantity | 09-28 | 09-30 | … | 11-07 | 11-09 | 11-10 |
|---|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | … | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | … | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | … | 9 | 9 | **9** |
| approved missing pose.npy | — | 0 | … | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | … | 0 | 0 | **0** |

(All intermediate audited runs 10-11…11-09 are identical; full columns are in their own sections
above.)

Review-DB status census (via [`status_census.py`](status_census.py),
`status_census-20261110.log`) — **set strictly identical to every prior audited run**, store
total included: `approved` 166,200 · `tainted:extraction_nonface` 215,914 ·
`tainted:contamination` 1,626 · `tainted:unusable` 253 · `tainted:approved_bad_geometry` 19 ·
**TOTAL 384,012**. No row added, removed, or re-statused — so the 7 h outage and the crash
recovery left the review DB in its prior state, with no lost or duplicated row.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (before `enrich`, via the committed
  [`preflight_setdiff.sh`](preflight_setdiff.sh), `preflight_setdiff-20261110.log`, 08:00:59):
  `auraface/` and `lda/` each hold exactly **295,468 `.npy`**; `comm -23` / `comm -13` on
  tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**. Zero LDA work is reachable from the
  *entire* tree, independent of the review DB.
* Post-run write verification via [`postchecks.sh`](postchecks.sh)
  (`verify_writes-20261110.log`): `find v1/lda -name '*.npy' -newermt '2026-11-10 00:00'` →
  **0**; `-newermt '2026-11-09 00:00'` → **0**; `-mmin -1440` → **0**; tree counts **295,468**
  (`auraface/`) and **295,468** (`lda/`). Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The "0 files
  written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261110.log`): `find lda … -newermt '2026-10-20 00:00'` → **0**;
  `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261110.log`). (`v1/auraface/` carries no `BASIS_FINGERPRINT.json` — expected:
  it holds raw AuraFace embeddings, not an LDA basis; only `v1/lda` is stamped, mtime
  2026-09-24.)
* `probe_scan.py` re-sweep (`probe_scan-20261110.log`): `missing auraface .npy: 23` ·
  `missing LDA .npy: 23` (`of those no auraface: 23, not projectable`) · `missing z_g .npy: 9` ·
  `missing pose.npy: 0`. Existing LDA samples (five): shape `(64,)`, `float32`, norms
  152.61–154.79 — refit raw-coords convention, unchanged.

Raw `enrich` output for the successful attempt (exit 0):
```
=== enrich launch Tue Nov 10 08:02:46 AM EST 2026 (git 345d517, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=298s END=Tue Nov 10 08:07:44 AM EST 2026
```

The CUDA-lib re-exec branch (`tools/hegre_dataset/enrichment.py:216-227`) fired again; its
restarted subprocess still fails at `import insightface` (`ImportError`) before any CUDA call, so
no GPU work ran and no GPU-scheduler reservation was needed. The latent path stands (unchanged):
**if `insightface` were ever importable in this venv, this branch would launch AuraFace GPU
extraction outside the GPU scheduler.**

### Audit-coverage gaps (carried + one new)

The **11-06** and **11-08** occurrences remain unaudited (fired `completed`, no README
record/commit). New this run: the **11-10 03:00 occurrence did not fire at all** — the host was
powered off (Event 1) — so there is nothing to audit for it (no `enrich-20261110.log` exists for
a 03:00 run). This 08:00 catch-up run is therefore counted as the **28th audited**, not the
29th.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261110.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`,
`failure_streak: 0`, `repeat.completed = 67`, `last_run_at = 2026-11-09T03:28:17`,
`next_run_at = 2026-11-11T03:00`, `deliver: local`. `executions.db` since 2026-11-09: 11-09
`completed`, 11-10 `running` (this occurrence, `started_at 2026-11-10T08:00:32`). Job id
`2aec4dd52b5c`. `main == origin/main` at `345d517` at launch; this run commits and pushes its
own record.

**Timing:** 298 s — inside the wired band (~250–547 s of 10-11…11-09), far below the 1451 s WiFi
band of 10-31, and the fastest of the recent wired runs. Note the registry shows
`last_status: ok` / `failure_streak: 0` **despite** the missed 03:00 fire and the failed first
attempt: a skipped occurrence and a transient DB-not-ready failure leave **no trace in the
registry's status fields** — only this audit does.

### Standing recommendation — now with a third availability incident on the same host

The de-registration recommendation (2026-10-23, restated 10-31 and every run since) remains
unexecuted. This run adds a **third** availability incident on the review-DB host (`game`),
after the 10-31 address/stall event and the reboot catch-up of 11-04: a ~7 h power-off that
silently skipped a scheduled fire and left postgres crash-recovering on boot. The point of the
recommendation is unchanged and strengthened — the sweep proves nothing on a fixed cadence, and
**a job whose host can be off cannot be trusted to have run**: a missed fire is indistinguishable
from a no-op in the registry (`last_status: ok`, `failure_streak: 0`, no execution row at all).
The DB-host repair (2026-10-31 action 2, `config/database.yml` `host: 192.168.86.49` →
`localhost`) also remains unexecuted.

## Re-run — 2026-11-11 (scheduled cron, in-slot 03:00 — 29th audited)

Identical **material** outcome for the **twenty-ninth** audited run — `enrich` exit 0,
**0 files written** — and, unlike the last three audited occurrences (11-04 catch-up, 11-05
off-slot, 11-10 reboot catch-up), this one **fired in its own slot** (`03:00:31 EST`) on a host
that had **not** rebooted since 2026-11-10 07:57.

The audit adds **one new forensic finding**: the `os.execv` CUDA-lib restart that prior records
read as an *intermittent* branch (fired on some runs, not others) in fact fires on **every** run;
its log line is a **stdout-buffering artifact**.

### Outcome — `enrich` exit 0, 0 files written

`enrich-20261111.log`; wall clock **403 s** [`03:02:35` → `03:09:18 EST`], git `e128d51`, clean
tree at launch — `main` and `origin/main` both at `e128d51`. Independent `probe_scan.py` re-sweep
(`probe_scan-20261111.log`, PROBE_EXIT=0):

| quantity | 11-05 | 11-07 | 11-09 | 11-10 | 11-11 |
|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (`status_census.py`, `status_census-20261111.log`) — set strictly
identical to every prior audited run, store total included: `approved` 166,200 ·
`tainted:extraction_nonface` 215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (before `enrich`, `preflight_setdiff.sh`,
  `preflight_setdiff-20261111.log`): `auraface/` and `lda/` each hold exactly **295,468 `.npy`**;
  `comm -23` / `comm -13` on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**.
* Post-run write verification (`postchecks.sh`, `verify_writes-20261111.log`):
  `find v1/lda -name '*.npy' -newermt '2026-11-11 00:00'` → **0**; `-newermt '2026-11-10 00:00'`
  → **0**; `-mmin -1440` → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`).
  Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The "0 files
  written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261111.log`): `find lda … -newermt '2026-10-20 00:00'` → **0**;
  `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261111.log`).
* `probe_scan.py` (`probe_scan-20261111.log`): `missing auraface .npy: 23` · `missing LDA .npy:
  23` (`of those no auraface: 23, not projectable`) · `missing z_g .npy: 9` · `missing pose.npy:
  0`. Existing LDA samples (five): shape `(64,)`, `float32`, norms 152.61–154.79 — refit
  raw-coords convention, unchanged.

Raw `enrich` output (exit 0):
```
=== enrich launch Wed Nov 11 03:02:35 AM EST 2026 (git e128d51, LD_LIBRARY_PATH='<empty>') ===
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
ENRICH_EXIT=0 WALL=403s END=Wed Nov 11 03:09:18 AM EST 2026
```

### New finding — the `os.execv` restart fires every run; its log line is a buffering artifact

Prior records (from 10-11 onward) read the presence/absence of `Restarting subprocess to load
CUDA libraries...` as evidence that the branch fired or did not fire, and called it intermittent
("the four prior recorded runs did not restart"). It is **not** intermittent — the branch
condition is deterministic and **true on every launch**:

* `run_enrich.sh` never pre-sets `LD_LIBRARY_PATH` (the header prints `'<empty>'` in all 29
  recorded logs), and `experiments/geometry_pca/.venv/lib/python3.14/site-packages/nvidia/cudnn/lib`
  **exists** — verified by [`cuda_branch_probe.py`](cuda_branch_probe.py)
  (`cuda_branch_probe-20261111.log`), which reproduces the exact predicate and prints
  `would_restart: True`.
* The line is missing whenever Python's **block-buffered stdout is discarded by `os.execv`**. The
  `Restarting…` `print` (`tools/hegre_dataset/enrichment.py:218`) has no `flush=True`; if the
  ~8 KB buffer has not filled/flushed before the `execv`, the **entire first-pass output** — that
  line included — is dropped, and the surviving log is the **second pass alone**, which reads as a
  single clean pass ending in `Error: insightface not installed.`
* **Proof.** Re-running the identical command with `PYTHONUNBUFFERED=1`
  (`enrich-20261111.unbuffered.log`, exit 0, wall **379 s**, **0 files written**) prints **both**
  passes *and* the `Restarting subprocess…` line — the branch fires. The buffered run 25 min
  earlier (same host, same git `e128d51`) printed one pass and no line.
* **Corroboration from the existing data.** The restart flag is uncorrelated with wall clock
  across the 26 completed runs (e.g. 11-03 `restarts=0` 389 s vs 11-05 `restarts=1` 404 s; 11-10
  `restarts=1` 298 s; 11-11 `restarts=0` 403 s). A branch that genuinely skipped a whole 166k-stat
  sweep would show a systematic time gap; there is none.

**Consequence.** The sweep has been running the **full NFS scan twice** on essentially every night
of its life (~2×166k stat loops). The prior audit's "the four prior recorded runs did not restart"
reading is a **measurement artifact of the instrument (a log line), not a fact about the code** —
the same class of error the project's own adversarial-pass rule names: *a check that cannot fail
is not a check*. Fix, if the job is kept: add `flush=True` before `os.execv` (or pre-set
`LD_LIBRARY_PATH` in `run_enrich.sh`), so the branch's firing is auditable from the log. This is a
further argument for the standing de-registration recommendation below.

### Audit-coverage gaps (carried)

The **11-06** and **11-08** occurrences remain unaudited (fired `completed`, no README
record/commit), as recorded on 11-10. No new gap this run.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261111.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 68`, `last_run_at = 2026-11-10T08:22:32`, `next_run_at = 2026-11-12T03:00`,
`deliver: local`. `executions.db` since 2026-11-04: 11-04…11-09 in-slot `completed` (03:00:xx),
11-10 catch-up at `08:00:32`, **11-11 in-slot at `03:00:31`** (this occurrence, `running`). Job id
`2aec4dd52b5c`. `main == origin/main` at `e128d51` at launch; this run commits and pushes its own
record.

**Timing:** 403 s — inside the wired band (~250–547 s of 10-11…11-10), consistent with the double
sweep the buffering finding implies. As established on 11-10, the registry's `last_status: ok` /
`failure_streak: 0` leave **no trace** of a skipped fire or a transient DB-not-ready failure; only
this audit does.

### Standing recommendation — de-register (unchanged; now a 4th restatement unexecuted)

The de-registration recommendation (2026-10-23, restated 10-31 and every run since) remains
unexecuted. This run adds no new availability incident — the host stayed up and the fire was
in-slot — but the buffering finding sharpens the case: the job is a proven no-op that has quietly
consumed **two** full NFS sweeps per night, and its own log line is an unreliable witness to which
code path ran. The DB-host repair (2026-10-31 action 2, `config/database.yml` `host:
192.168.86.49` → `localhost`) also remains unexecuted.

## Re-run — 2026-11-12 (scheduled cron, in-slot 03:00 — 30th audited)

Identical **material** outcome for the **thirtieth** audited run — `enrich` exit 0, **0 files
written** — fired in its own slot (`03:00:30 EST`) on a host that has **not** rebooted since
2026-11-10 07:57. **No new forensic finding**; the 11-11 buffering conclusion is corroborated
(below).

### Outcome — `enrich` exit 0, 0 files written

`enrich-20261112.log`; wall clock **395 s** [`03:02:58` → `03:09:33 EST`], git `d29dc74`, clean
tree at launch — `main` and `origin/main` both at `d29dc74`. Independent `probe_scan.py` re-sweep
(`probe_scan-20261112.log`, PROBE_EXIT=0):

| quantity | 11-07 | 11-09 | 11-10 | 11-11 | 11-12 |
|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (`status_census.py`, `status_census-20261112.log`) — set strictly
identical to every prior audited run, store total included: `approved` 166,200 ·
`tainted:extraction_nonface` 215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (before `enrich`, `preflight_setdiff.sh`,
  `preflight_setdiff-20261112.log`): `auraface/` and `lda/` each hold exactly **295,468 `.npy`**;
  `comm -23` / `comm -13` on tree-relative, `LC_ALL=C`-sorted paths → **0** and **0**.
* Post-run write verification (`postchecks.sh`, `verify_writes-20261112.log`):
  `find v1/lda -name '*.npy' -newermt '2026-11-12 00:00'` → **0**; `-newermt '2026-11-11 00:00'`
  → **0**; `-mmin -1440` → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`).
  Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The "0 files
  written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261112.log`): `find lda … -newermt '2026-10-20 00:00'` → **0**;
  `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261112.log`).
* `cuda_branch_probe.py` (`cuda_branch_probe-20261112.log`): `would_restart: True`;
  `insightface ImportError: No module named 'insightface'` — the re-exec branch predicate is again
  true and the restarted pass again fails before any CUDA call, so no GPU work ran and no
  GPU-scheduler reservation was needed.
* `probe_scan.py` (`probe_scan-20261112.log`): `missing auraface .npy: 23` · `missing LDA .npy:
  23` (`of those no auraface: 23, not projectable`) · `missing z_g .npy: 9` · `missing pose.npy:
  0`. Existing LDA samples (five): shape `(64,)`, `float32`, norms 152.61–154.79 — refit
  raw-coords convention, unchanged.

Raw `enrich` output (exit 0):
```
=== enrich launch Thu Nov 12 03:02:58 AM EST 2026 (git d29dc74, LD_LIBRARY_PATH='<empty>') ===
Running AuraFace-only enrichment (--skip-stratum) for approved images...
Skipping Stratum enrichment (--skip-stratum).
Found 9 approved images with pose but missing z_g. Extracting...
z_g extraction complete in 0s. Extracted 0, skipped 9.
Found 23 approved images missing AuraFace data. Extracting...
Error: insightface not installed. Skipping AuraFace extraction.
All approved images already have AuraFace-LDA data.
Enrichment complete.
ENRICH_EXIT=0 WALL=395s END=Thu Nov 12 03:09:33 AM EST 2026
```

### Corroboration of the 11-11 buffering finding (no new finding)

The 11-11 conclusion — the `os.execv` CUDA-lib restart fires on **every** launch and its
`Restarting subprocess…` log line is a **stdout-buffering artifact** — is corroborated. This run's
buffered log again shows a single clean pass ending in `Error: insightface not installed.` (no
`Restarting…` line), and its **395 s** wall clock sits inside the same band as the other buffered
runs (11-11 403 s / same-run unbuffered 379 s; 11-05 404 s; 11-10 298 s), consistent with the full
NFS scan running twice per night.

### Timing note — the 420 s foreground window still binds (operational confirmation)

The `enrich` sweep itself was in-band (395 s). The NFS-heavy `postchecks.sh`
(`verify_writes` + `gapcheck` + `probe_scan.py`) again exceeded the **420 s foreground window**: a
first foreground launch was cut mid-`gapcheck` (launcher-side, exit at 420 s) and the runner was
relaunched as a **tracked background job** — the documented procedure (see the launcher note
around line 1323) — completing at ≈`03:30:45`. Leg timings are comparable to 11-11, so there is
**no** NFS-throughput anomaly to report: `verify_writes` finished `03:22:57` (≈6 min into the
background run) and `probe_scan.py` walked all 166,200 approved paths in **≈424 s**, vs
**≈453 s** on 11-11. Recorded only as a live confirmation that the window still binds and the
two-stage (foreground-cut → background) launch is the correct handling.

### Audit-coverage gaps (carried)

The **11-06** and **11-08** occurrences remain unaudited (fired `completed`, no README
record/commit), as recorded on 11-10 and 11-11. No new gap this run.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261112.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 69`, `last_run_at = 2026-11-11T03:38:07`, `next_run_at = 2026-11-13T03:00`,
`deliver: local`. `executions.db` since 2026-11-04: 11-04…11-11 in-slot `completed`, **11-12
in-slot at `03:00:30`** (this occurrence, `running`). Job id `2aec4dd52b5c`. `main == origin/main`
at `d29dc74` at launch; this run commits and pushes its own record.

**Timing:** 395 s — inside the wired band (~250–547 s of 10-11…11-11). As established on 11-10,
the registry's `last_status: ok` / `failure_streak: 0` leave **no trace** of a skipped fire or a
transient DB-not-ready failure; only this audit does.

### Standing recommendation — de-register (unchanged; now a 5th restatement unexecuted)

The de-registration recommendation (2026-10-23, restated every run since 10-31) remains
unexecuted. This run adds no new availability incident — the host stayed up and the fire was
in-slot — and no new finding; the job remains a proven no-op on a fixed nightly cadence. The
DB-host repair (2026-10-31 action 2, `config/database.yml` `host: 192.168.86.49` → `localhost`)
also remains unexecuted.

## Re-run — 2026-11-13 (scheduled cron, in-slot 03:00 — 31st audited)

Identical **material** outcome for the **thirty-first** audited run — `enrich` exit 0, **0 files
written** — fired in its own slot (`03:00:30 EST`) on a host that has **not** rebooted since
2026-11-10 07:57. **No new forensic finding.** The 11-11 buffering conclusion is corroborated
again, this time in the *positive* direction (below): the `Restarting subprocess…` line **was**
present in this run's buffered log, both passes visible.

### Outcome — `enrich` exit 0, 0 files written

`enrich-20261113.log`; wall clock **368 s** [`03:01:28` → `03:07:36 EST`], git `93cbc60`, clean
tree at launch — `main` and `origin/main` both at `93cbc60`, pushed before the run. Independent
`probe_scan.py` re-sweep (`probe_scan-20261113.log`, PROBE_EXIT=0):

| quantity | 11-09 | 11-10 | 11-11 | 11-12 | 11-13 |
|---|---|---|---|---|---|
| approved images | 166,200 | 166,200 | 166,200 | 166,200 | **166,200** |
| approved missing LDA | 23 | 23 | 23 | 23 | **23** (all 23 also missing AuraFace ⇒ unprojectable) |
| approved missing z_g | 9 | 9 | 9 | 9 | **9** |
| approved missing pose.npy | 0 | 0 | 0 | 0 | **0** |
| LDA files written by the run | 0 | 0 | 0 | 0 | **0** |

Review-DB status census (`status_census.py`, `status_census-20261113.log`) — set strictly
identical to every prior audited run, store total included: `approved` 166,200 ·
`tainted:extraction_nonface` 215,914 · `tainted:contamination` 1,626 · `tainted:unusable` 253 ·
`tainted:approved_bad_geometry` 19 · **TOTAL 384,012**. No row added, removed, or re-statused.

Independent checks (all before/after the run):

* **Pre-flight whole-tree set-difference** (before `enrich`, `preflight_setdiff.sh`,
  `preflight_setdiff-20261113.log`, 03:00:55 → 03:01:21, 26 s): `auraface/` and `lda/` each hold
  exactly **295,468 `.npy`**; `comm -23` / `comm -13` on tree-relative, `LC_ALL=C`-sorted paths →
  **0** and **0**. Zero LDA work is reachable from the *entire* tree, independent of the review DB.
* Post-run write verification (`postchecks.sh`, `verify_writes-20261113.log`):
  `find v1/lda -name '*.npy' -newermt '2026-11-13 00:00'` → **0**; `-newermt '2026-11-12 00:00'`
  → **0**; `-mmin -1440` → **0**; tree counts **295,468** (`auraface/`) and **295,468** (`lda/`).
  Newest file in the tree is still **2026-07-28 14:42**
  (`…/radka-in/radka-in-summer-dress/radka-in-summer-dress-poster_face1.npy`). The "0 files
  written" line is a true zero, not a silent-failure zero.
* `gapcheck` (`verify_writes-20261113.log`): `find lda … -newermt '2026-10-20 00:00'` → **0**;
  `-newermt '2026-10-29 00:00'` → **0** — no LDA file written on any day since 2026-10-20.
* `hegre-dataset basis-fingerprint verify --dataset v1/lda` →
  `OK  …/v1/lda  120e1c5a1dc4f423  refit basis, raw coords (norm ~153)` — match
  (`basis_verify-20261113.log`).
* `cuda_branch_probe.py` (`cuda_branch_probe-20261113.log`): `would_restart: True`;
  `insightface ImportError: No module named 'insightface'` — the re-exec branch predicate is
  again true and the restarted pass again fails before any CUDA call, so no GPU work ran and no
  GPU-scheduler reservation was needed.
* `probe_scan.py` (`probe_scan-20261113.log`): `missing auraface .npy: 23` · `missing LDA .npy:
  23` (`of those no auraface: 23, not projectable`) · `missing z_g .npy: 9` · `missing pose.npy:
  0`. Existing LDA samples (five): shape `(64,)`, `float32`, norms 152.61–154.79 — refit
  raw-coords convention, unchanged.

Raw `enrich` output (exit 0):
```
=== enrich launch Fri Nov 13 03:01:28 AM EST 2026 (git 93cbc60, LD_LIBRARY_PATH='<empty>') ===
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
ENRICH_EXIT=0 WALL=368s END=Fri Nov 13 03:07:36 AM EST 2026
```

### Corroboration of the 11-11 buffering finding — this run shows the line

The 11-11 conclusion — the `os.execv` CUDA-lib restart fires on **every** launch and its
`Restarting subprocess…` line is a **stdout-buffering artifact** — is corroborated from the
opposite side. The 11-12 run showed a *single* clean pass (line absent: the buffer was dropped at
`execv`); **this run's buffered log shows both passes with the `Restarting subprocess…` line
present**, at an identical launcher, identical empty `LD_LIBRARY_PATH`, and the same
deterministic branch predicate (`would_restart: True` above). The line's presence is therefore
non-deterministic across runs of *identical* conditions — exactly what a buffer-flush race
predicts, and what the 11-11 `PYTHONUNBUFFERED=1` proof established. Wall clock **368 s** sits
inside the same band as the other (double-sweep) runs (11-11 403 s / same-run unbuffered 379 s;
11-12 395 s; 11-10 298 s).

### Environment / occurrence notes

* **Wired link up; the bare command works.** `enp42s0` **UP** at `192.168.86.49/24` (`wlo1`
  down); `config/database.yml` reads `host: 192.168.86.49` (line 8, unchanged), and
  `pg_isready -h 192.168.86.49 -p 5432` → **accepting connections**, so the review DB was reached
  with no `EIDOLON_DB_URL` override. The 2026-10-31 operator action (its 2 remaining items) is
  still unapplied — the latent bug is intact.
* **The NFS mount's stale `clientaddr` has cleared.** Prior records (11-01…11-12) noted `mount`
  reporting `clientaddr=192.168.86.248` (the 10-31 WiFi address) while all traffic flowed over
  `enp42s0`. This run `mount` reports **`clientaddr=192.168.86.49`** — the automount has
  re-established with the live wired address, so the cosmetic mismatch is gone.
* **The occurrence fired on time at 03:00:30 EST** (`executions.db` row for 11-13 →
  `started_at 2026-11-13T03:00:30`, `status = running`, finishes on completion). Occurrence **71**.
* **No eidolon process held the GPU:** `nvidia-smi --query-compute-apps` showed a single foreign
  `llama-server` (4,842 MiB — an `ollama` process, not eidolon). The CUDA-lib re-exec's restarted
  subprocess again failed at `import insightface` (`ImportError`) before any CUDA call, so no
  `insightface` CUDA work ran.
* **Timing / window:** the NFS-heavy `postchecks.sh` (`verify_writes` + `gapcheck` +
  `probe_scan.py`) again exceeds the **420 s foreground window**, so — as on 11-12 — it was
  launched as a **tracked background job** (the documented two-stage handling). `verify_writes` +
  `gapcheck` completed first, then `probe_scan.py` walked all 166,200 approved paths to
  `PROBE_EXIT=0`; the full postcheck run completed by ≈`03:25:20 EST`. No NFS-throughput anomaly.

### Audit-coverage gaps (carried)

The **11-06** and **11-08** occurrences remain unaudited (fired `completed`, no README
record/commit), as recorded on 11-10 through 11-12. No new gap this run.

### Job state, read during the run ([`cron_state.py`](cron_state.py), `cron_state-20261113.log`)

`enabled: true`, `state: scheduled`, `last_status: ok`, `last_error: null`, `failure_streak: 0`,
`repeat.completed = 70`, `last_run_at = 2026-11-12T03:30:57`, `next_run_at = 2026-11-14T03:00`,
`deliver: local`, `created_at = 2026-07-24T08:45:36`. `executions.db` since 2026-11-04: 11-04…11-12
in-slot `completed`, **11-13 in-slot at `03:00:30`** (this occurrence, `running`). Job id
`2aec4dd52b5c`. `main == origin/main` at `93cbc60` at launch; this run commits and pushes its own
record.

**Timing:** 368 s — inside the wired band (~250–547 s of 10-11…11-12). As established on 11-10,
the registry's `last_status: ok` / `failure_streak: 0` leave **no trace** of a skipped fire or a
transient DB-not-ready failure; only this audit does.

### Standing recommendation — de-register (unchanged; now a 6th restatement unexecuted)

The de-registration recommendation (2026-10-23, restated every run since 10-31) remains
unexecuted. This run adds no new availability incident and no new finding; the job remains a
proven no-op on a fixed nightly cadence, consuming two full NFS sweeps per night. The DB-host
repair (2026-10-31 action 2, `config/database.yml` `host: 192.168.86.49` → `localhost`) also
remains unexecuted.
