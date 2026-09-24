"""
Interactive review UI for hegre face datasets.

Shows actual MTCNN face crops for visual verification.
Port-configurable Flask server.

DONE SEMANTICS (INTENDED — do not "fix" into uniformity)
========================================================
DONE (api_done, below) is NOT a neutral "save and continue". Its meaning depends
on the active mode, and that asymmetry is deliberate:

  First Pass (unreviewed) : apply brush taints, then BULK-APPROVE every remaining
                            image in the shown batch. Approve-by-default.
  Review / Audit          : apply brush taints only. Approve nothing; just deal a
                            new random sample of already-approved images.

Rationale: First Pass reviews status='unreviewed' images, where approve is the safe
default and the reviewer signals exceptions by brushing. Review/Audit review
already-approved images, so approving them is a no-op and DONE exists solely to
apply corrections and rotate the sample.

Consequences worth knowing when operating this tool:
  * shown_ids is the FULL 20-image batch, and cards are loading="lazy" in a
    responsive grid — off-screen images are approved without ever being seen.
    The batch, not the individual image, is the unit of review.
  * The `remaining` count in the response is GLOBAL, not persona-scoped
    (it filters on status only, with no persona_id).
  * Enter is bound globally to donePersona() — a stray Enter submits the batch.
  * DONE spawns a background compute-geometry job (_maybe_spawn_geometry_compute)
    which can write further taint labels minutes after DONE returns.
  * The Unreview button is NOT an undo: it picks a random persona and resets 10
    random tainted images, unrelated to your last DONE.

Design system: Lawrenz Admin Dark (DESIGN.md)
Tokens defined as CSS custom properties; no framework dependency.

COORDINATE SYSTEM COUPLING: The @300px convention
==================================================
THUMB_SIZE (300, 300), the pixel average generation, UV mapping, and 3D
FLAME texture all share a hard 300×300 resolution.  Changing THUMB_SIZE
without updating the matching constants in flame_projector.py
(compute_uv_coordinates, generate_textured_mesh) and procrustes.py
(generate_pixel_average) will silently misalign textures.
"""
import io
import os
import sqlite3
import subprocess
import sys
import threading
import numpy as np
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image, ImageDraw

from ..dataset import HegreDataset

# ── Subprocess dedup: track active geometry compute jobs per persona ──
_active_geometry_jobs: dict[int, subprocess.Popen] = {}
_jobs_lock = threading.Lock()


def _maybe_spawn_geometry_compute(
    persona_id: int,
    faces_root: Path,
    encoder_path: Path,
) -> subprocess.Popen | None:
    """Spawn a geometry-compute subprocess unless one is already running.

    Sets EIDOLON_SKIP_REVIEWDB_GUARD=1 so the subprocess skips the
    review.db existence check (it uses PostgreSQL via config).
    Returns the Popen handle if spawned, None if skipped.
    """
    with _jobs_lock:
        for pid, proc in list(_active_geometry_jobs.items()):
            if proc.poll() is not None:
                del _active_geometry_jobs[pid]
        if persona_id in _active_geometry_jobs:
            return None

        env = os.environ.copy()
        env["EIDOLON_SKIP_REVIEWDB_GUARD"] = "1"
        proc = subprocess.Popen([
            sys.executable, "-m", "tools.hegre_dataset", "review", "compute-geometry",
            "--dataset", str(faces_root),
            "--encoder", str(encoder_path),
            "--persona", str(persona_id),
            "--metric", "both"
        ], stdout=sys.stdout, stderr=sys.stderr, env=env)

        _active_geometry_jobs[persona_id] = proc
        return proc


def create_app(db_path: Path, faces_root: Path) -> Flask:
    """Create the Flask application.

    Args:
        db_path: Path to review.db (unused — HegreDataset auto-discovers it from faces_root).
        faces_root: Dataset root directory.
    """
    faces_root = faces_root.resolve()
    ds = HegreDataset(faces_root)
    app = Flask(__name__)
    
    _thumb_cache = {}
    THUMB_SIZE = (300, 300)
    
    def _load_thumb(image_path_rel: str, persona_name: str, draw_skel: bool = False) -> bytes:
        """Load and resize a face crop to thumbnail size."""
        full_path = (faces_root / image_path_rel).resolve()
        if not full_path.is_relative_to(faces_root.resolve()):
            return _get_placeholder()
            
        if not full_path.exists():
            return _get_placeholder()
        
        img = Image.open(full_path).convert("RGB")
        
        if draw_skel:
            p = Path(image_path_rel)
            base_pname = persona_name.split("_cluster_")[0]
            stratum_dir = ds.stratum_dir / base_pname
            pose_path = None
            if stratum_dir.exists():
                for pth in stratum_dir.rglob(f"{p.stem}/pose.npy"):
                    pose_path = pth
                    break
                
            if pose_path and pose_path.exists():
                try:
                    pose = np.load(pose_path)
                    
                    # Stratum DWPose has 133 points. The face keypoints are 23:91.
                    # Column 0: X (normalized)
                    # Column 1: Y (normalized)
                    # Column 2: Confidence [0, 1]
                    face_points = pose[23:91]
                    
                    img_w, img_h = img.size
                    
                    draw = ImageDraw.Draw(img)
                    for point in face_points:
                        if len(point) >= 3:
                            x, y, conf = point[0], point[1], point[2]
                        else:
                            x, y = point[0], point[1]
                            conf = 1.0  # Fallback if confidence isn't present
                            
                        # Stratum seems to output coordinates centered around (0,0) with scales extending past [-1, 1].
                        px = (x / 2.0 + 0.5) * img_w
                        py = (y / 2.0 + 0.5) * img_h
                        
                        if 0 <= px <= img_w and 0 <= py <= img_h and conf > 0.05:
                            # Map confidence to radius (higher confidence = bigger dot, but ensure a visible minimum)
                            # e.g., conf 0.0 -> r=2, conf 1.0 -> r=6
                            r = 2 + (conf * 4)
                            
                            # Map confidence to opacity (alpha). We need an RGBA image or to just draw RGB.
                            # Since we are drawing directly on RGB, we can blend colors, but for simplicity:
                            # We can just draw it solid but vary the radius.
                            draw.ellipse([px-r, py-r, px+r, py+r], fill="lime")
                except Exception as e:
                    print(f"XRAY ERROR: {e}")
                    pass
                        
        resample_filter = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
        img.thumbnail(THUMB_SIZE, resample_filter)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
        
    def _get_placeholder() -> bytes:
        placeholder = Image.new("RGB", THUMB_SIZE, (60, 60, 60))
        buf = io.BytesIO()
        placeholder.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
    
    @app.route("/api/thumb/<int:image_id>")
    def api_thumb(image_id):
        draw_skel = request.args.get("skel", "0") == "1"
        # Force cache bypass if we're debugging, or make sure cache key is robust
        cache_key = f"{image_id}_{draw_skel}"
        
        if cache_key not in _thumb_cache:
            row = ds.db.execute(
                "SELECT i.image_path, p.name FROM images i JOIN personas p ON i.persona_id = p.id WHERE i.id = ?", (image_id,)
            ).fetchone()
            if not row:
                return "", 404
            _thumb_cache[cache_key] = _load_thumb(row["image_path"], row["name"], draw_skel=draw_skel)
            if len(_thumb_cache) > 400:
                _thumb_cache.pop(next(iter(_thumb_cache)))
        return send_file(io.BytesIO(_thumb_cache[cache_key]), mimetype="image/jpeg")
    
    @app.route("/api/pixel/<persona_name>")
    def api_pixel(persona_name):
        try:
            base_pname = persona_name.split("_cluster_")[0]
            pixel_path = ds.stratum_dir / base_pname / f"pixel_{persona_name}.jpg"
            if pixel_path.exists():
                return send_file(str(pixel_path), mimetype="image/jpeg")
            return "File not found at " + str(pixel_path), 404
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/random_persona")
    def api_random_persona():
        mode = request.args.get("mode", "unreviewed")
        force_persona = request.args.get("persona", None)
        status_filter = "approved" if mode in ["review", "audit"] else "unreviewed"
        
        if force_persona:
            row = ds.db.execute(f"SELECT p.id, p.name FROM personas p JOIN images i ON i.persona_id = p.id WHERE i.status = ? AND p.name = ? GROUP BY p.id LIMIT 1", (status_filter, force_persona)).fetchone()
        else:
            row = ds.db.execute(f"SELECT p.id, p.name FROM personas p JOIN images i ON i.persona_id = p.id WHERE i.status = ? GROUP BY p.id ORDER BY RANDOM() LIMIT 1", (status_filter,)).fetchone()
        
        if not row:
            msg = "ALL REVIEWED" if mode in ["review", "audit"] else "ALL DONE"
            return jsonify({"persona_id": None, "persona_name": msg, "image_ids": [], "mode": mode})
            
        pid, pname = row["id"], row["name"]
        
        # Check for reference images
        refs = ds.db.execute("SELECT id, status FROM images WHERE persona_id = ? AND status IN ('unreviewed', 'approved') AND zg_distance IS NOT NULL ORDER BY CAST(zg_distance AS REAL) ASC LIMIT 3", (pid,)).fetchall()
        reference_ids = [r["id"] for r in refs]
        approved_ref_ids = set(r["id"] for r in refs if r["status"] == "approved")
        
        total_for_persona = ds.db.execute("SELECT COUNT(*) FROM images WHERE persona_id = ? AND status = ?", (pid, status_filter)).fetchone()[0]
        
        has_af = ds.db.execute("SELECT COUNT(af_distance) FROM images WHERE persona_id = ? AND af_distance IS NOT NULL", (pid,)).fetchone()[0] > 0
        has_zg = ds.db.execute("SELECT COUNT(zg_distance) FROM images WHERE persona_id = ? AND zg_distance IS NOT NULL", (pid,)).fetchone()[0] > 0

        if mode == "audit":
            # Audit: pick approved images with the highest af distance
            if has_af:
                order_clause = "ORDER BY CAST(af_distance AS REAL) DESC NULLS FIRST LIMIT 20"
                dist_col = "af_distance"
            elif has_zg:
                order_clause = "ORDER BY CAST(zg_distance AS REAL) DESC NULLS FIRST LIMIT 20"
                dist_col = "zg_distance"
            else:
                order_clause = "ORDER BY RANDOM() LIMIT 20"
                dist_col = None
        elif mode == "unreviewed":
            if has_af:
                order_clause = "ORDER BY CAST(af_distance AS REAL) DESC NULLS FIRST LIMIT 20"
                dist_col = "af_distance"
            elif has_zg:
                order_clause = "ORDER BY CAST(zg_distance AS REAL) DESC NULLS FIRST LIMIT 20"
                dist_col = "zg_distance"
            else:
                order_clause = "ORDER BY RANDOM() LIMIT 20"
                dist_col = None
        else:
            # Review: pick a random set of approved images, then sort by af distance
            order_clause = "ORDER BY RANDOM() LIMIT 20"
            if has_af:
                dist_col = "af_distance"
            elif has_zg:
                dist_col = "zg_distance"
            else:
                dist_col = None

        all_imgs = ds.db.execute(f"SELECT id, status, face_index, image_path, zg_distance, af_distance FROM images WHERE persona_id = ? AND status = ? {order_clause}", (pid, status_filter)).fetchall()

        # Mix in best images only if unreviewed (to prevent drift)
        if mode == "unreviewed":
            if dist_col:
                best_imgs = ds.db.execute(f"SELECT id, status, face_index, image_path, zg_distance, af_distance FROM images WHERE persona_id = ? AND status = ? ORDER BY CAST({dist_col} AS REAL) ASC NULLS FIRST LIMIT 5", (pid, status_filter)).fetchall()
            else:
                best_imgs = []
        else:
            best_imgs = []

        combined_ids = []
        for r in best_imgs:
            if r["id"] not in approved_ref_ids:
                combined_ids.append(r["id"])

        for img in all_imgs:
            if img["id"] not in combined_ids and img["id"] not in approved_ref_ids:
                combined_ids.append(img["id"])
            if len(combined_ids) >= 20:
                break

        final_imgs = []
        for img in best_imgs + all_imgs:
            if img["id"] in combined_ids and img["id"] not in [r["id"] for r in final_imgs]:
                final_imgs.append(img)

        if mode == "review":
            if dist_col == "af_distance":
                final_imgs.sort(key=lambda x: x["af_distance"] if x["af_distance"] is not None else -1.0, reverse=True)
            elif dist_col == "zg_distance":
                final_imgs.sort(key=lambda x: x["zg_distance"] if x["zg_distance"] is not None else -1.0, reverse=True)

        # Build distance map with metric prefix
        distances = {}
        for r in final_imgs:
            if dist_col == "af_distance" and r["af_distance"] is not None:
                distances[r["id"]] = float(r["af_distance"])
            elif dist_col == "zg_distance" and r["zg_distance"] is not None:
                distances[r["id"]] = float(r["zg_distance"])
            elif r["af_distance"] is not None:
                distances[r["id"]] = float(r["af_distance"])
            elif r["zg_distance"] is not None:
                distances[r["id"]] = float(r["zg_distance"])

        return jsonify({
            "persona_id": pid,
            "persona_name": pname,
            "total_for_persona": total_for_persona,
            "image_ids": [r["id"] for r in final_imgs],
            "reference_ids": reference_ids,
            "unreviewed_ids": [r["id"] for r in final_imgs if r["status"] == status_filter],
            "statuses": {r["id"]: r["status"] for r in final_imgs},
            "labels": {r["id"]: f"face{r['face_index']}" for r in final_imgs},
            "distances": distances,
            "distance_metric": dist_col or "none",
            "mode": mode,
        })
        
    @app.route("/api/done", methods=["POST"])
    def api_done():
        data = request.get_json()
        pid = data["persona_id"]
        tainted = data.get("tainted", {})
        mode = data.get("mode", "unreviewed")
        shown_ids = data.get("shown_ids", [])
        db = ds.db_writable
        
        for img_id_str, reason in tainted.items():
            db.execute("UPDATE images SET status = ?, reviewed_at = NOW() WHERE id = ?", (reason, int(img_id_str)))
            
        if mode == "unreviewed":
            approved_ids = [int(i) for i in shown_ids if str(i) not in tainted]
            if approved_ids:
                placeholders = ",".join("?" * len(approved_ids))
                db.execute(f"UPDATE images SET status = 'approved', reviewed_at = NOW() WHERE id IN ({placeholders})", approved_ids)
                
        db.commit()
        status_filter = "approved" if mode in ["review", "audit"] else "unreviewed"
        remaining = db.execute("SELECT COUNT(*) FROM images WHERE status = ?", (status_filter,)).fetchone()[0]
        
        # Fire off a background process
        # Dedup: skip if one is already running for this persona.
        encoder_path = Path(__file__).parent.parent.parent.parent / "experiments/geometry_pca/output/encoder_production.npz"
        _maybe_spawn_geometry_compute(int(pid), faces_root, encoder_path)
        
        return jsonify({"remaining": remaining, "mode": mode})

    @app.route("/api/unreview_random", methods=["POST"])
    def api_unreview_random():
        """Pick a random persona, find up to 10 tainted (non-approved) images, reset them to unreviewed."""
        db = ds.db_writable

        # Find a random persona that has tainted images
        row = db.execute(
            "SELECT p.id, p.name FROM personas p "
            "JOIN images i ON i.persona_id = p.id "
            "WHERE i.status LIKE 'tainted:%' "
            "GROUP BY p.id "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()

        if not row:
            return jsonify({"reset": 0, "persona_name": None, "message": "No tainted images found"})

        pid, pname = row["id"], row["name"]

        # Pick up to 10 random tainted images for this persona
        tainted = db.execute(
            "SELECT id FROM images "
            "WHERE persona_id = ? AND status LIKE 'tainted:%' "
            "ORDER BY RANDOM() LIMIT 10",
            (pid,)
        ).fetchall()

        count = len(tainted)
        if count > 0:
            ids = [r["id"] for r in tainted]
            placeholders = ",".join("?" * len(ids))
            db.execute(
                f"UPDATE images SET status = 'unreviewed', reviewed_at = NULL WHERE id IN ({placeholders})",
                ids
            )
            db.commit()

        return jsonify({
            "reset": count,
            "persona_name": pname,
            "persona_id": pid,
            "message": f"Reset {count} images for {pname} back to unreviewed"
        })

    HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eidolon | Hegre Face Review</title>
    <style>
        /* ═══════════════════════════════════════════════════════════════
           Lawrenz Admin Dark — DESIGN.md tokens as CSS custom properties
           ═══════════════════════════════════════════════════════════════ */
        :root {
            /* Colors */
            --color-page: #0D1117;
            --color-surface: #161B22;
            --color-surface-elevated: #1C2128;
            --color-hover: #1C2128;
            --color-input: #0D1117;
            --color-disabled: #21262D;
            --color-overlay: #0D1117E6;
            --color-border-default: #30363D;
            --color-border-muted: #21262D;
            --color-border-focus: #C9A85C;
            --color-text-primary: #E6EDF3;
            --color-text-secondary: #8B949E;
            --color-text-tertiary: #6E7681;
            --color-text-on-accent: #0D1117;
            --color-accent: #C9A85C;
            --color-accent-hover: #D4B56E;
            --color-accent-muted: #2A2416;

            /* Signal colors */
            --color-red: #F85149;
            --color-red-muted-bg: #490202;
            --color-red-muted-text: #FF7B72;
            --color-orange: #D29922;
            --color-orange-muted-bg: #341A00;
            --color-orange-muted-text: #E3B341;
            --color-green: #3FB950;
            --color-green-muted-bg: #04260F;
            --color-green-muted-text: #56D364;
            --color-info: #58A6FF;
            --color-info-muted: #0C2D6B;

            /* Typography */
            --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', 'Helvetica Neue', Arial, sans-serif;
            --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cousine', monospace;

            /* Spacing (4px baseline) */
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 12px;
            --space-lg: 16px;
            --space-xl: 24px;
            --space-2xl: 32px;
            --space-3xl: 48px;
            --touch-target: 44px;

            /* Rounded */
            --radius-sm: 4px;
            --radius-md: 6px;
            --radius-lg: 8px;
            --radius-full: 9999px;
        }

        /* ═══════════════════════════════════════════════════════════════
           Reset & base
           ═══════════════════════════════════════════════════════════════ */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { background: var(--color-page); color: var(--color-text-primary); font-family: var(--font-sans); font-size: 16px; line-height: 1.5; }
        body { min-height: 100dvh; display: flex; flex-direction: column; }

        /* ═══════════════════════════════════════════════════════════════
           Header
           ═══════════════════════════════════════════════════════════════ */
        .header {
            position: sticky; top: 0; z-index: 50;
            background: var(--color-page);
            border-bottom: 1px solid var(--color-border-default);
            padding: var(--space-lg);
        }
        .header-top {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: var(--space-lg);
            flex-wrap: wrap; gap: var(--space-sm);
        }
        .header-info h1 {
            font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em;
            color: var(--color-text-primary);
        }
        .header-info h1 .persona-name {
            color: var(--color-green-muted-text);
        }
        .header-info .status-line {
            font-size: 0.75rem; color: var(--color-text-secondary); margin-top: var(--space-xs);
        }
        .header-controls {
            display: flex; align-items: center; gap: var(--space-md);
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.05em;
            text-transform: uppercase; color: var(--color-text-secondary);
            flex-wrap: wrap;
        }

        /* ═══════════════════════════════════════════════════════════════
           Buttons
           ═══════════════════════════════════════════════════════════════ */
        .btn {
            display: inline-flex; align-items: center; justify-content: center;
            border: none; cursor: pointer; font-family: var(--font-sans);
            font-weight: 500; font-size: 0.875rem; line-height: 1.5;
            transition: background-color 150ms ease-out, color 150ms ease-out;
            white-space: nowrap;
            text-decoration: none;
        }
        .btn:disabled { cursor: not-allowed; }

        /* Primary — the single high-emphasis action */
        .btn-primary {
            background: var(--color-accent); color: var(--color-text-on-accent);
            border-radius: var(--radius-md); padding: var(--space-sm) var(--space-lg);
            font-weight: 600; font-size: 0.8125rem;
            min-height: var(--touch-target);
        }
        .btn-primary:hover { background: var(--color-accent-hover); }
        .btn-primary:disabled { background: var(--color-disabled); color: var(--color-text-secondary); }

        /* Secondary */
        .btn-secondary {
            background: var(--color-surface); color: var(--color-text-primary);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-md); padding: var(--space-sm) var(--space-lg);
            min-height: var(--touch-target);
        }
        .btn-secondary:hover { background: var(--color-hover); }

        /* Ghost — toolbars, low-emphasis */
        .btn-ghost {
            background: transparent; color: var(--color-text-secondary);
            border-radius: var(--radius-md); padding: var(--space-sm) var(--space-md);
            min-height: var(--touch-target);
        }
        .btn-ghost:hover { background: var(--color-hover); color: var(--color-text-primary); }

        /* Danger */
        .btn-danger {
            background: var(--color-red-muted-bg); color: var(--color-red-muted-text);
            border: 1px solid var(--color-red-muted-bg);
            border-radius: var(--radius-md); padding: var(--space-sm) var(--space-lg);
            min-height: var(--touch-target);
        }
        .btn-danger:hover { background: var(--color-red); color: var(--color-page); }

        /* ═══════════════════════════════════════════════════════════════
           Tabs (mode selector)
           ═══════════════════════════════════════════════════════════════ */
        .tabs { display: flex; border-bottom: 1px solid var(--color-border-default); }
        .tab {
            background: transparent; border: none; cursor: pointer;
            color: var(--color-text-tertiary);
            font-family: var(--font-sans); font-size: 0.8125rem; font-weight: 500;
            padding: var(--space-sm) var(--space-lg);
            border-bottom: 2px solid transparent;
            min-height: var(--touch-target);
            display: inline-flex; align-items: center;
            transition: color 150ms ease-out, border-color 150ms ease-out;
        }
        .tab:hover { color: var(--color-text-primary); }
        .tab--active {
            color: var(--color-text-primary);
            border-bottom-color: var(--color-accent);
        }

        /* ═══════════════════════════════════════════════════════════════
           Divider
           ═══════════════════════════════════════════════════════════════ */
        .divider {
            width: 1px; background: var(--color-border-muted);
            align-self: stretch;
        }

        /* ═══════════════════════════════════════════════════════════════
           Brush indicators
           ═══════════════════════════════════════════════════════════════ */
        .brush-indicator {
            display: inline-flex; align-items: center; gap: var(--space-xs);
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .brush-indicator kbd {
            font-family: var(--font-mono); font-size: 0.6875rem;
            background: var(--color-surface); color: var(--color-text-primary);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-sm); padding: 1px 5px;
        }
        .brush-active {
            background: var(--color-accent-muted) !important;
            color: var(--color-accent) !important;
        }

        /* ═══════════════════════════════════════════════════════════════
           Anchor bar (reference images + pixel average)
           ═══════════════════════════════════════════════════════════════ */
        .anchor-bar {
            display: flex; gap: var(--space-md); height: 128px;
        }
        .anchor-bar-label {
            display: flex; flex-direction: column; gap: var(--space-xs);
            width: 128px; flex-shrink: 0; justify-content: center;
            border-right: 1px solid var(--color-border-default);
            padding-right: var(--space-lg);
        }
        .anchor-bar-label .title {
            font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.05em;
            text-transform: uppercase; color: var(--color-green-muted-text);
        }
        .anchor-bar-label .subtitle {
            font-size: 0.6875rem; color: var(--color-text-tertiary);
            line-height: 1.3;
        }
        .anchor-card {
            position: relative; width: 128px; height: 128px; flex-shrink: 0;
            background: var(--color-surface);
            border: 2px solid var(--color-green);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }
        .anchor-card img {
            width: 100%; height: 100%; object-fit: cover;
            border-radius: calc(var(--radius-lg) - 2px); opacity: 0.85;
        }
        .anchor-card .tag {
            position: absolute; bottom: var(--space-xs); right: var(--space-xs);
            background: rgba(13, 17, 23, 0.85);
            font-size: 0.5625rem; font-weight: 600; letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 2px 6px; border-radius: var(--radius-sm);
            color: var(--color-green-muted-text);
        }

        /* ═══════════════════════════════════════════════════════════════
           Image cards
           ═══════════════════════════════════════════════════════════════ */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: var(--space-lg);
            padding: var(--space-lg);
        }
        .card {
            position: relative; aspect-ratio: 1;
            background: var(--color-surface);
            border: 2px solid var(--color-border-muted);
            border-radius: var(--radius-lg);
            overflow: hidden; cursor: pointer;
            transition: border-color 150ms ease-out, opacity 150ms ease-out;
        }
        .card:focus-visible {
            outline: 2px solid var(--color-accent);
            outline-offset: 2px;
        }
        .card img {
            width: 100%; height: 100%; object-fit: cover;
            user-select: none; -webkit-user-drag: none;
            transition: opacity 150ms ease-out;
        }
        .card .dist-chip {
            position: absolute; top: var(--space-sm); left: var(--space-sm);
            display: flex; gap: var(--space-xs);
        }
        .card .dist-chip span {
            font-family: var(--font-mono); font-size: 0.6875rem;
            background: rgba(13, 17, 23, 0.85);
            padding: 2px 6px; border-radius: var(--radius-sm);
        }

        /* Taint states — using signal colors per DESIGN.md */
        .card.unreviewed     { border-color: var(--color-border-default); opacity: 1; }
        .card.approved       { border-color: var(--color-green); opacity: 1; }

        /* extraction_nonface → red (error: quality failure) */
        .card.tainted-nonface {
            border-color: var(--color-red) !important;
            opacity: 0.35; filter: grayscale(80%);
        }
        /* contamination → orange (warning: data quality) */
        .card.tainted-contamination {
            border-color: var(--color-orange) !important;
            opacity: 0.35; filter: grayscale(80%);
        }
        /* unusable → muted (neutral bad) */
        .card.tainted-unusable {
            border-color: var(--color-text-tertiary) !important;
            opacity: 0.35; filter: grayscale(80%);
        }
        /* approved_bad_geometry → info blue (informational) */
        .card.tainted-approved_bad_geometry {
            border-color: var(--color-info) !important;
            opacity: 0.45; filter: grayscale(80%);
        }

        /* ═══════════════════════════════════════════════════════════════
           X-ray skeleton overlay
           ═══════════════════════════════════════════════════════════════ */
        .show-xray .card img { /* skeleton is drawn server-side when ?skel=1 is appended */ }

        /* ═══════════════════════════════════════════════════════════════
           Empty state
           ═══════════════════════════════════════════════════════════════ */
        .empty-state {
            display: flex; align-items: center; justify-content: center;
            padding: var(--space-3xl); min-height: 40vh;
        }
        .empty-state p {
            font-size: 1.375rem; font-weight: 600;
            color: var(--color-green-muted-text);
        }

        /* ═══════════════════════════════════════════════════════════════
           Utility
           ═══════════════════════════════════════════════════════════════ */
        .flex-1 { flex: 1; }
        .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
        .persona-link { color: var(--color-green-muted-text); text-decoration: none; }
        .persona-link:hover { color: var(--color-green); text-decoration: underline; }
        .unlock-link { color: var(--color-text-tertiary); text-decoration: none; font-size: 0.6875rem; margin-left: var(--space-sm); }
        .unlock-link:hover { color: var(--color-red-muted-text); }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-top">
            <div class="header-info">
                <h1>Persona: <span class="persona-name" id="persona_name">loading&hellip;</span></h1>
                <p class="status-line" id="status"></p>
            </div>
            <div class="header-controls">
                <button class="btn btn-primary" onclick="donePersona()" title="Submit (Enter)">DONE</button>
                <div class="divider"></div>
                <div class="tabs" id="mode-tabs">
                    <button id="btn_unreviewed" class="tab tab--active" onclick="switchMode('unreviewed')">First Pass</button>
                    <button id="btn_review" class="tab" onclick="switchMode('review')">Review</button>
                    <button id="btn_audit" class="tab" onclick="switchMode('audit')">Audit</button>
                </div>
                <div class="divider"></div>
                <span class="btn-ghost brush-indicator" id="btn_nonface" onclick="setBrush('tainted:extraction_nonface')">
                    <kbd>1</kbd> Non-Face
                </span>
                <span class="btn-ghost brush-indicator" id="btn_contam" onclick="setBrush('tainted:contamination')">
                    <kbd>2</kbd> Contam
                </span>
                <span class="btn-ghost brush-indicator" id="btn_unusable" onclick="setBrush('tainted:unusable')">
                    <kbd>3</kbd> Unusable
                </span>
                <span class="btn-ghost brush-indicator" id="btn_badgeom" onclick="setBrush('tainted:approved_bad_geometry')">
                    <kbd>4</kbd> Bad Geo
                </span>
                <span class="btn-ghost brush-indicator" id="btn_skel" onclick="toggleSkel()">
                    <kbd>X</kbd> X-Ray
                </span>
                <div class="divider"></div>
                <button class="btn btn-secondary" onclick="unreviewRandom()" title="Pick random persona, reset 10 tainted images to unreviewed" style="font-size:0.75rem;">Unreview</button>
            </div>
        </div>
        <div class="anchor-bar" id="reference-anchors" style="display:none;"></div>
    </header>
    <main class="flex-1">
        <div class="grid" id="grid"></div>
    </main>

    <script>
        let personaId = null, brush = 'tainted:extraction_nonface', tainted = {}, mode = 'unreviewed', shownIds = [], showSkel = false;
        const urlParams = new URLSearchParams(window.location.search);
        const forcePersona = urlParams.get('persona');

        setBrush(brush);

        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'x' && !e.repeat) {
                showSkel = true;
                document.querySelectorAll('.card img').forEach(img => {
                    if (!img.src.includes('skel=1')) img.src = img.src + (img.src.includes('?') ? '&' : '?') + 'skel=1';
                });
            }
            if (e.key === '1') setBrush('tainted:extraction_nonface');
            if (e.key === '2') setBrush('tainted:contamination');
            if (e.key === '3') setBrush('tainted:unusable');
            if (e.key === '4') setBrush('tainted:approved_bad_geometry');
            if (e.key === 'Enter') donePersona();
        });
        document.addEventListener('keyup', (e) => {
            if (e.key.toLowerCase() === 'x') {
                showSkel = false;
                document.querySelectorAll('.card img').forEach(img => {
                    img.src = img.src.replace(/[?&]skel=1/, '');
                });
            }
        });

        function toggleSkel() {
            showSkel = !showSkel;
            document.querySelectorAll('.card img').forEach(img => {
                if (showSkel) {
                    if (!img.src.includes('skel=1')) img.src = img.src + (img.src.includes('?') ? '&' : '?') + 'skel=1';
                } else {
                    img.src = img.src.replace(/[?&]skel=1/, '');
                }
            });
        }

        function switchMode(m) {
            mode = m;
            ['unreviewed', 'review', 'audit'].forEach(mod => {
                const btn = document.getElementById('btn_' + mod);
                if (btn) {
                    btn.className = (mod === m) ? 'tab tab--active' : 'tab';
                }
            });
            loadPersona();
        }

        function setBrush(b) {
            brush = b;
            document.querySelectorAll('.brush-indicator').forEach(e => e.classList.remove('brush-active'));
            if (b === 'tainted:extraction_nonface') document.getElementById('btn_nonface').classList.add('brush-active');
            if (b === 'tainted:contamination') document.getElementById('btn_contam').classList.add('brush-active');
            if (b === 'tainted:unusable') document.getElementById('btn_unusable').classList.add('brush-active');
            if (b === 'tainted:approved_bad_geometry') document.getElementById('btn_badgeom').classList.add('brush-active');
        }

        let g_data = null;
        async function loadPersona() {
            let url = '/api/random_persona?mode=' + mode;
            if (forcePersona) url += '&persona=' + encodeURIComponent(forcePersona);
            const resp = await fetch(url);
            const data = await resp.json();
            g_data = data;
            if (!data.persona_id) {
                const grid = document.getElementById('grid');
                grid.innerHTML = '<div class="empty-state"><p>' + data.persona_name + '!</p></div>';
                return;
            }
            personaId = data.persona_id;
            shownIds = data.image_ids;

            let nameHtml = '<a href="/?persona=' + encodeURIComponent(data.persona_name) + '" class="persona-link" title="Lock to this persona">' + data.persona_name + '</a>';
            if (forcePersona) {
                nameHtml += ' <a href="/" class="unlock-link" title="Unlock persona">[Unlock]</a>';
            }
            document.getElementById('persona_name').innerHTML = nameHtml;
            const n = data.unreviewed_ids.length;
            document.getElementById('status').innerText = 'Mode: ' + mode.toUpperCase() + ' | Total: ' + data.total_for_persona + ' | Unreviewed: ' + n;
            tainted = {};
            renderReferences(data.reference_ids);
            renderGrid(data.image_ids, data.statuses, data.labels, data.distances);
        }

        function renderReferences(ids) {
            const container = document.getElementById('reference-anchors');
            container.innerHTML = '';
            container.style.display = 'flex';

            const label = document.createElement('div');
            label.className = 'anchor-bar-label';
            label.innerHTML = '<span class="title">Centroid Anchors</span><span class="subtitle">Closest to z<sub>g</sub> center.</span>';
            container.appendChild(label);

            // Pixel Average (Procrustes Warping)
            if (g_data && g_data.persona_name) {
                const pix = document.createElement('div');
                pix.className = 'anchor-card';
                pix.title = 'Pixel Average (Procrustes Warping)';
                pix.innerHTML = '<img src="/api/pixel/' + g_data.persona_name + '?t=' + Date.now() + '" onerror="this.parentElement.style.display=\\'none\\'" /><div class="tag">Pixel</div>';
                container.appendChild(pix);
            }

            if (!ids || ids.length === 0) {
                if (!g_data || !g_data.persona_name) {
                    container.style.display = 'none';
                }
                return;
            }

            for (const id of ids) {
                const card = document.createElement('div');
                card.className = 'anchor-card';
                card.innerHTML = '<img src="/api/thumb/' + id + (showSkel ? '?skel=1' : '') + '" /><div class="tag">Ref</div>';
                container.appendChild(card);
            }
        }

        function renderGrid(ids, statuses, labels, distances) {
            const grid = document.getElementById('grid'); grid.innerHTML = '';
            for (const id of ids) {
                const s = statuses[id] || 'unreviewed';
                const dist = distances[id];

                if (mode === 'unreviewed' && s !== 'unreviewed') continue;
                if (mode === 'audit' && s !== 'approved' && !s.startsWith('tainted:approved_')) continue;

                const wrapper = document.createElement('div');
                wrapper.tabIndex = 0;
                wrapper.className = 'card';
                if (s.startsWith('tainted:')) wrapper.classList.add('tainted-' + s.replace('tainted:extraction_nonface', 'nonface').replace('tainted:', ''));
                else if (s === 'approved') wrapper.classList.add('approved');
                else wrapper.classList.add('unreviewed');

                wrapper.onclick = () => toggleTaint(wrapper, id, s);

                let distHtml = '';
                if (dist !== null && dist !== undefined) {
                    const metricLabel = (g_data && g_data.distance_metric === 'af_distance') ? 'af' : 'zg';
                    distHtml = '<span class="dist-chip"><span style="color:var(--color-red-muted-text);border:1px solid var(--color-red-muted-bg);">' + metricLabel + ': ' + parseFloat(dist).toFixed(4) + '</span></span>';
                }

                wrapper.innerHTML = '<img src="/api/thumb/' + id + (showSkel ? '?skel=1' : '') + '" loading="lazy" draggable="false" />' + distHtml;
                grid.appendChild(wrapper);
            }
        }

        function toggleTaint(el, id, defaultStatus) {
            if (tainted[id]) {
                delete tainted[id];
                el.className = 'card ' + ((mode === 'review' || mode === 'audit') ? 'approved' : 'unreviewed');
            } else {
                tainted[id] = brush;
                el.className = 'card tainted-' + brush.replace('tainted:extraction_nonface', 'nonface').replace('tainted:', '');
            }
        }

        async function donePersona() {
            const t = Object.keys(tainted).length;
            const resp = await fetch('/api/done', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ persona_id: personaId, tainted: tainted, mode: mode, shown_ids: shownIds }) });
            const data = await resp.json();
            document.getElementById('status').innerText = 'Saved. ' + data.remaining + ' remaining. Loading next...';
            setTimeout(loadPersona, 400);
        }

        async function unreviewRandom() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '...';
            try {
                const resp = await fetch('/api/unreview_random', { method: 'POST' });
                const data = await resp.json();
                document.getElementById('status').innerText = data.message;
                if (data.reset > 0) {
                    if (personaId === data.persona_id) {
                        loadPersona();
                    }
                }
            } catch (e) {
                document.getElementById('status').innerText = 'Unreview failed: ' + e;
            }
            btn.disabled = false;
            btn.textContent = 'Unreview';
        }
        loadPersona();
    </script>
</body>
</html>"""
    @app.route("/")
    def index():
        return render_template_string(HTML)
    
    return app