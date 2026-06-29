"""
Interactive review UI for hegre face datasets.

Shows actual MTCNN face crops for visual verification.
Brush-to-taint, DONE-to-approve. Port-configurable Flask server.

COORDINATE SYSTEM COUPLING: The @300px convention
==================================================
THUMB_SIZE (300, 300), the pixel average generation, UV mapping, and 3D
FLAME texture all share a hard 300×300 resolution.  Changing THUMB_SIZE
without updating the matching constants in flame_projector.py
(compute_uv_coordinates, generate_textured_mesh) and procrustes.py
(generate_pixel_average) will silently misalign textures.
"""
import io
import sqlite3
import subprocess
import sys
import numpy as np
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image, ImageDraw


def get_db(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    return db


def create_app(db_path: Path, faces_root: Path) -> Flask:
    """Create the Flask application."""
    faces_root = faces_root.resolve()
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
            
            # Use the explicit persona_name from the DB to build the stratum path
            # The persona_name might have a _cluster_ suffix from DBSCAN, which isn't in the image_path!
            base_pname = persona_name.split("_cluster_")[0]
            
            stratum_dir = faces_root / "stratum" / base_pname
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
    
    @app.route("/api/pixel/<persona_name>")
    def api_pixel(persona_name):
        try:
            base_pname = persona_name.split("_cluster_")[0]
            pixel_path = faces_root / "stratum" / base_pname / f"pixel_{persona_name}.jpg"
            if pixel_path.exists():
                return send_file(str(pixel_path), mimetype="image/jpeg")
            return "File not found at " + str(pixel_path), 404
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/ghost/<persona_name>")
    def api_ghost(persona_name):
        try:
            base_pname = persona_name.split("_cluster_")[0]
            ghost_path = faces_root / "stratum" / base_pname / f"ghost_{persona_name}.png"
            if ghost_path.exists():
                return send_file(str(ghost_path), mimetype="image/png")
            return "File not found at " + str(ghost_path), 404
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/3d/<persona_name>")
    def api_3d(persona_name):
        try:
            base_pname = persona_name.split("_cluster_")[0]
            gif_path = faces_root / "stratum" / base_pname / f"3d_{persona_name}.gif"
            if gif_path.exists():
                return send_file(str(gif_path), mimetype="image/gif")
            return "File not found at " + str(gif_path), 404
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/thumb/<int:image_id>")
    def api_thumb(image_id):
        draw_skel = request.args.get("skel", "0") == "1"
        # Force cache bypass if we're debugging, or make sure cache key is robust
        cache_key = f"{image_id}_{draw_skel}"
        
        if cache_key not in _thumb_cache:
            db = get_db(db_path)
            row = db.execute(
                "SELECT i.image_path, p.name FROM images i JOIN personas p ON i.persona_id = p.id WHERE i.id = ?", (image_id,)
            ).fetchone()
            db.close()
            
            if not row:
                return "", 404
            _thumb_cache[cache_key] = _load_thumb(row["image_path"], row["name"], draw_skel=draw_skel)
            if len(_thumb_cache) > 400:
                _thumb_cache.pop(next(iter(_thumb_cache)))
        return send_file(io.BytesIO(_thumb_cache[cache_key]), mimetype="image/jpeg")
    
    @app.route("/api/random_persona")
    def api_random_persona():
        mode = request.args.get("mode", "unreviewed")
        force_persona = request.args.get("persona", None)
        status_filter = "approved" if mode in ["review", "audit"] else "unreviewed"
        db = get_db(db_path)
        
        if force_persona:
            row = db.execute(f"SELECT p.id, p.name FROM personas p JOIN images i ON i.persona_id = p.id WHERE i.status = ? AND p.name = ? GROUP BY p.id LIMIT 1", (status_filter, force_persona)).fetchone()
        else:
            row = db.execute(f"SELECT p.id, p.name FROM personas p JOIN images i ON i.persona_id = p.id WHERE i.status = ? GROUP BY p.id ORDER BY RANDOM() LIMIT 1", (status_filter,)).fetchone()
        
        if not row:
            msg = "ALL REVIEWED" if mode in ["review", "audit"] else "ALL DONE"
            db.close()
            return jsonify({"persona_id": None, "persona_name": msg, "image_ids": [], "mode": mode})
            
        pid, pname = row["id"], row["name"]
        
        # Check for reference images
        # We only want references that exist, have a valid distance, and are NOT rejected
        refs = db.execute("SELECT id, status FROM images WHERE persona_id = ? AND status IN ('unreviewed', 'approved') AND zg_distance IS NOT NULL ORDER BY zg_distance ASC LIMIT 3", (pid,)).fetchall()
        reference_ids = [r["id"] for r in refs]
        approved_ref_ids = set(r["id"] for r in refs if r["status"] == "approved")
        
        total_for_persona = db.execute("SELECT COUNT(*) FROM images WHERE persona_id = ? AND status = ?", (pid, status_filter)).fetchone()[0]
        
        # Determine sorting strategy - prefer af_distance, fall back to zg_distance
        has_af = db.execute("SELECT COUNT(af_distance) FROM images WHERE persona_id = ? AND af_distance IS NOT NULL", (pid,)).fetchone()[0] > 0
        has_zg = db.execute("SELECT COUNT(zg_distance) FROM images WHERE persona_id = ? AND zg_distance IS NOT NULL", (pid,)).fetchone()[0] > 0

        if mode == "audit":
            # Audit: pick approved images with the highest af distance
            if has_af:
                order_clause = "ORDER BY af_distance DESC NULLS LAST LIMIT 20"
                dist_col = "af_distance"
            elif has_zg:
                order_clause = "ORDER BY zg_distance DESC NULLS LAST LIMIT 20"
                dist_col = "zg_distance"
            else:
                order_clause = "ORDER BY RANDOM() LIMIT 20"
                dist_col = None
        elif mode == "unreviewed":
            if has_af:
                order_clause = "ORDER BY af_distance DESC NULLS LAST LIMIT 20"
                dist_col = "af_distance"
            elif has_zg:
                order_clause = "ORDER BY zg_distance DESC NULLS LAST LIMIT 20"
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

        all_imgs = db.execute(f"SELECT id, status, face_index, image_path, zg_distance, af_distance FROM images WHERE persona_id = ? AND status = ? {order_clause}", (pid, status_filter)).fetchall()

        # Mix in best images only if unreviewed (to prevent drift)
        if mode == "unreviewed":
            if dist_col:
                best_imgs = db.execute(f"SELECT id, status, face_index, image_path, zg_distance, af_distance FROM images WHERE persona_id = ? AND status = ? ORDER BY {dist_col} ASC NULLS LAST LIMIT 5", (pid, status_filter)).fetchall()
            else:
                best_imgs = []
        else:
            best_imgs = []

        db.close()

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
        db = get_db(db_path)
        
        for img_id_str, reason in tainted.items():
            db.execute("UPDATE images SET status = ?, reviewed_at = datetime('now') WHERE id = ?", (reason, int(img_id_str)))
            
        if mode == "unreviewed":
            approved_ids = [int(i) for i in shown_ids if str(i) not in tainted]
            if approved_ids:
                placeholders = ",".join("?" * len(approved_ids))
                db.execute(f"UPDATE images SET status = 'approved', reviewed_at = datetime('now') WHERE id IN ({placeholders})", approved_ids)
                
        db.commit()
        status_filter = "approved" if mode in ["review", "audit"] else "unreviewed"
        remaining = db.execute("SELECT COUNT(*) FROM images WHERE status = ?", (status_filter,)).fetchone()[0]
        db.close()
        
        # Fire off a background process to recalculate the centroid for JUST this persona.
        # This isolates the heavy PCA logic and SQLite lock from the Flask thread.
        encoder_path = Path(__file__).parent.parent.parent.parent / "experiments/geometry_pca/output/encoder_production.npz"
        subprocess.Popen([
            sys.executable, "-m", "tools.hegre_dataset", "review", "compute-geometry",
            "--dataset", str(faces_root),
            "--encoder", str(encoder_path),
            "--persona", str(pid),
            "--metric", "both"
        ], stdout=sys.stdout, stderr=sys.stderr)
        
        return jsonify({"remaining": remaining, "mode": mode})

    HTML = """<!DOCTYPE html>
<html lang="en" class="dark bg-zinc-950 text-zinc-300">
<head>
    <meta charset="UTF-8">
    <title>Eidolon | Hegre Face Review</title>
    <script src="https://unpkg.com/@tailwindcss/browser@4"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;600&display=swap');
        body { font-family: 'Geist Mono', monospace; }
        .skel-layer { opacity: 0; transition: opacity 0.1s; pointer-events: none; }
        .show-xray .skel-layer { opacity: 1; }
        .image-card:focus-within { outline: 2px solid #a1a1aa; outline-offset: 2px; }
        .tainted-nonface { border-color: #f43f5e !important; opacity: 0.3; filter: grayscale(80%); }
        .tainted-contamination { border-color: #f59e0b !important; opacity: 0.3; filter: grayscale(80%); }
        .tainted-unusable { border-color: #52525b !important; opacity: 0.3; filter: grayscale(80%); }
        .tainted-approved_bad_geometry { border-color: #60a5fa !important; opacity: 0.4; filter: grayscale(80%); }
        .approved { border-color: #10b981 !important; opacity: 1.0; }
        .unreviewed { border-color: #3f3f46; opacity: 1.0; }
        .brush-active { outline: 2px solid white; outline-offset: 2px; }
    </style>
</head>
<body class="min-h-[100dvh] flex flex-col antialiased selection:bg-zinc-800">
    <header class="sticky top-0 z-50 bg-zinc-950/90 backdrop-blur border-b border-zinc-800 p-4 shrink-0">
        <div class="flex items-center justify-between mb-4">
            <div>
                <h1 class="text-sm tracking-widest uppercase text-zinc-100">Persona: <span class="text-emerald-400" id="persona_name">loading...</span></h1>
                <p class="text-xs text-zinc-500 mt-1" id="status"></p>
            </div>
            <div class="flex gap-4 text-[10px] text-zinc-500 uppercase tracking-wider items-center">
                <button class="px-3 py-1.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded hover:bg-emerald-500/40" onclick="donePersona()">[ENTER] DONE</button>
                <div class="h-4 w-px bg-zinc-800"></div>
                <span class="cursor-pointer" onclick="toggleSkel()"><kbd class="border border-zinc-700 px-1 rounded text-zinc-300">Hold X</kbd> X-Ray</span>
                <span id="btn_nonface" class="brush-indicator cursor-pointer" onclick="setBrush('tainted:extraction_nonface')"><kbd class="border border-zinc-700 px-1 rounded text-zinc-300">1</kbd> Non-Face</span>
                <span id="btn_contam" class="brush-indicator cursor-pointer" onclick="setBrush('tainted:contamination')"><kbd class="border border-zinc-700 px-1 rounded text-zinc-300">2</kbd> Contam</span>
                <span id="btn_unusable" class="brush-indicator cursor-pointer" onclick="setBrush('tainted:unusable')"><kbd class="border border-zinc-700 px-1 rounded text-zinc-300">3</kbd> Unusable</span>
                <span id="btn_badgeom" class="brush-indicator cursor-pointer" onclick="setBrush('tainted:approved_bad_geometry')"><kbd class="border border-zinc-700 px-1 rounded text-zinc-300">4</kbd> Bad Geo</span>
                <div class="h-4 w-px bg-zinc-800"></div>
                <button id="btn_unreviewed" class="px-2 py-1 bg-zinc-700 rounded text-zinc-100 font-bold" onclick="switchMode('unreviewed')">First Pass</button>
                <button id="btn_review" class="px-2 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300" onclick="switchMode('review')">Review</button>
                <button id="btn_audit" class="px-2 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300" onclick="switchMode('audit')">Audit</button>
            </div>
        </div>
        <div class="flex gap-4 h-32" id="reference-anchors" style="display:none;"></div>
    </header>
    <main class="flex-1 p-4">
        <div class="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4" id="grid"></div>
    </main>

    <script>
        let personaId=null, brush='tainted:extraction_nonface', tainted={}, mode='unreviewed', shownIds=[], showSkel=false;
        const urlParams = new URLSearchParams(window.location.search);
        const forcePersona = urlParams.get('persona');

        setBrush(brush);

        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'x' && !e.repeat) {
                document.body.classList.add('show-xray');
                showSkel = true;
                document.querySelectorAll('img').forEach(img => {
                    if(!img.src.includes('skel=1')) img.src = img.src + (img.src.includes('?') ? '&' : '?') + 'skel=1';
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
                document.body.classList.remove('show-xray');
                showSkel = false;
                document.querySelectorAll('img').forEach(img => {
                    img.src = img.src.replace(/[\\?&]skel=1/, '');
                });
            }
        });

        function toggleSkel() {
            showSkel = !showSkel;
            if(showSkel) {
                document.body.classList.add('show-xray');
                document.querySelectorAll('img').forEach(img => {
                    if(!img.src.includes('skel=1')) img.src = img.src + (img.src.includes('?') ? '&' : '?') + 'skel=1';
                });
            } else {
                document.body.classList.remove('show-xray');
                document.querySelectorAll('img').forEach(img => {
                    img.src = img.src.replace(/[\\?&]skel=1/, '');
                });
            }
        }

        function switchMode(m) {
            mode = m;
            ['unreviewed', 'review', 'audit'].forEach(mod => {
                const btn = document.getElementById('btn_' + mod);
                if (btn) {
                    if (mod === m) {
                        btn.className = "px-2 py-1 bg-zinc-700 rounded text-zinc-100 font-bold transition-colors";
                    } else {
                        btn.className = "px-2 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300 transition-colors";
                    }
                }
            });
            loadPersona();
        }

        function setBrush(b){
            brush=b;
            document.querySelectorAll('.brush-indicator').forEach(e=>e.classList.remove('brush-active'));
            if(b==='tainted:extraction_nonface')document.getElementById('btn_nonface').classList.add('brush-active');
            if(b==='tainted:contamination')document.getElementById('btn_contam').classList.add('brush-active');
            if(b==='tainted:unusable')document.getElementById('btn_unusable').classList.add('brush-active');
            if(b==='tainted:approved_bad_geometry')document.getElementById('btn_badgeom').classList.add('brush-active');
        }

        let g_data = null;
        async function loadPersona(){
            let url = '/api/random_persona?mode=' + mode;
            if (forcePersona) url += '&persona=' + encodeURIComponent(forcePersona);
            const resp=await fetch(url);
            const data=await resp.json();
            g_data = data;
            if(!data.persona_id){document.getElementById('grid').innerHTML='<p class="text-2xl text-emerald-400 p-4">'+data.persona_name+'!</p>';return;}
            personaId=data.persona_id;
            shownIds=data.image_ids;
            let nameHtml = `<a href="/?persona=${encodeURIComponent(data.persona_name)}" class="hover:text-emerald-300 hover:underline transition-colors" title="Lock to this persona">${data.persona_name}</a>`;
            if (forcePersona) {
                nameHtml += ` <a href="/" class="text-zinc-500 hover:text-rose-400 text-[10px] ml-2 no-underline" title="Unlock persona">[Unlock]</a>`;
            }
            document.getElementById('persona_name').innerHTML = nameHtml;
            const n=data.unreviewed_ids.length;
            document.getElementById('status').innerText=`Mode: ${mode.toUpperCase()} | Total: ${data.total_for_persona} | Unreviewed: ${n}`;
            tainted={};
            renderReferences(data.reference_ids);
            renderGrid(data.image_ids,data.statuses,data.labels,data.distances);
        }

        function renderReferences(ids) {
            const container = document.getElementById('reference-anchors');
            container.innerHTML = `
            <div class="flex flex-col gap-1 w-32 border-r border-zinc-800 pr-4 justify-center shrink-0">
                <span class="text-[10px] text-emerald-400 tracking-widest uppercase">Centroid Anchors</span>
                <span class="text-[10px] text-zinc-500 leading-tight">These crops are closest to the zg center.</span>
            </div>`;
            
            // Render Ghost Average & Pixel Average
            if (g_data && g_data.persona_name) {
                container.innerHTML += `
                <div class="relative w-32 h-32 bg-zinc-900 border-2 border-emerald-500 rounded shrink-0" title="Ghost Average (Inverse PCA)">
                    <img src="/api/ghost/${g_data.persona_name}?t=${Date.now()}" class="w-full h-full object-cover rounded opacity-90" onerror="this.parentElement.style.display='none'" />
                    <div class="absolute bottom-1 right-1 bg-zinc-950/80 text-[9px] px-1 rounded backdrop-blur text-emerald-400">Ghost</div>
                </div>
                <div class="relative w-32 h-32 bg-zinc-900 border-2 border-emerald-500 rounded shrink-0" title="Pixel Average (Procrustes Warping)">
                    <img src="/api/pixel/${g_data.persona_name}?t=${Date.now()}" class="w-full h-full object-cover rounded opacity-90" onerror="this.parentElement.style.display='none'" />
                    <div class="absolute bottom-1 right-1 bg-zinc-950/80 text-[9px] px-1 rounded backdrop-blur text-emerald-400">Pixel</div>
                </div>
                <div class="relative w-32 h-32 bg-zinc-900 border-2 border-emerald-500 rounded shrink-0" title="3D Volume (FLAME + Pixel Average)">
                    <img src="/api/3d/${g_data.persona_name}?t=${Date.now()}" class="w-full h-full object-cover rounded opacity-90" onerror="this.parentElement.style.display='none'" />
                    <div class="absolute bottom-1 right-1 bg-zinc-950/80 text-[9px] px-1 rounded backdrop-blur text-emerald-400">Volume</div>
                </div>`;
            }

            if (!ids || ids.length === 0) {
                // We still show the ghost if it exists
                if (!g_data || !g_data.persona_name) {
                    container.style.display = 'none';
                } else {
                    container.style.display = 'flex';
                }
                return;
            }
            container.style.display = 'flex';
            for(const id of ids) {
                container.innerHTML += `
                <div class="relative w-32 h-32 bg-zinc-900 border-2 border-emerald-500 rounded shrink-0">
                    <img src="/api/thumb/${id}${showSkel ? "?skel=1" : ""}" class="w-full h-full object-cover rounded opacity-80" />
                    <div class="absolute bottom-1 right-1 bg-zinc-950/80 text-[9px] px-1 rounded backdrop-blur text-emerald-400">Ref</div>
                </div>`;
            }
        }

        function renderGrid(ids,statuses,labels,distances){
            const grid=document.getElementById('grid');grid.innerHTML='';
            for(const id of ids){
                const s=statuses[id]||'unreviewed';
                let lbl=labels[id]||'';
                const dist=distances[id];
                
                if(mode==='unreviewed' && s!=='unreviewed') continue;
                if(mode==='audit' && s!=='approved' && !s.startsWith('tainted:approved_')) continue;
                
                const wrapper=document.createElement('div');
                wrapper.tabIndex = 0;
                wrapper.className = 'image-card group relative aspect-square bg-zinc-900 border-2 border-zinc-800 rounded cursor-pointer overflow-hidden transition-colors focus:outline-none';
                if(s.startsWith('tainted:')) wrapper.classList.add('tainted-'+s.replace('tainted:extraction_nonface','nonface').replace('tainted:',''));
                else if(s==='approved') wrapper.classList.add('approved');
                else wrapper.classList.add('unreviewed');
                
                wrapper.onclick = () => toggleTaint(wrapper, id, s);

                let distHtml = '';
                if (dist !== null && dist !== undefined) {
                    const metricLabel = (g_data && g_data.distance_metric === 'af_distance') ? 'af' : 'zg';
                    distHtml = `<span class="bg-zinc-950/80 backdrop-blur text-[10px] px-1.5 py-0.5 rounded border border-zinc-800 text-rose-300">${metricLabel}: ${parseFloat(dist).toFixed(4)}</span>`;
                }

                wrapper.innerHTML = `
                    <img src="/api/thumb/${id}${showSkel ? "?skel=1" : ""}" class="w-full h-full object-cover transition-opacity" loading="lazy" />
                    <div class="absolute top-2 left-2 flex flex-col gap-1">
                        ${distHtml}
                    </div>
                `;
                grid.appendChild(wrapper);
            }
        }

        function toggleTaint(el,id, defaultStatus){
            if(tainted[id]){
                delete tainted[id];
                el.className = 'image-card group relative aspect-square bg-zinc-900 border-2 rounded cursor-pointer overflow-hidden transition-colors focus:outline-none ' + 
                               ((mode==='review'||mode==='audit') ? 'approved' : 'unreviewed');
            }
            else{
                tainted[id]=brush;
                el.className = 'image-card group relative aspect-square bg-zinc-900 border-2 rounded cursor-pointer overflow-hidden transition-colors focus:outline-none tainted-' + brush.replace('tainted:extraction_nonface','nonface').replace('tainted:','');
            }
        }

        async function donePersona(){
            const t=Object.keys(tainted).length;
            const resp=await fetch('/api/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({persona_id:personaId,tainted:tainted,mode:mode,shown_ids:shownIds})});
            const data=await resp.json();
            document.getElementById('status').innerText='Saved. '+data.remaining+' remaining. Loading next...';
            setTimeout(loadPersona,400);
        }
        loadPersona();
    </script>
</body>
</html>"""
    @app.route("/")
    def index():
        return render_template_string(HTML)
    
    return app
