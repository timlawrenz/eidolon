"""Interactive review UI for hegre face datasets.

Shows actual MTCNN face crops for visual verification.
Brush-to-taint, DONE-to-approve. Port-configurable Flask server.
"""
import io
import sqlite3
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
    app = Flask(__name__)
    
    _thumb_cache = {}
    THUMB_SIZE = (120, 120)
    
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
                        
                        if 0 <= px <= img_w and 0 <= py <= img_h:
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
        refs = db.execute("SELECT id FROM images WHERE persona_id = ? AND status IN ('unreviewed', 'approved') AND zg_distance IS NOT NULL ORDER BY zg_distance ASC LIMIT 3", (pid,)).fetchall()
        reference_ids = [r["id"] for r in refs]
        
        total_for_persona = db.execute("SELECT COUNT(*) FROM images WHERE persona_id = ? AND status = ?", (pid, status_filter)).fetchone()[0]
        
        # Determine sorting strategy
        has_distances = db.execute("SELECT COUNT(zg_distance) FROM images WHERE persona_id = ? AND zg_distance IS NOT NULL", (pid,)).fetchone()[0] > 0
        
        if mode == "audit":
            order_clause = "ORDER BY zg_distance DESC NULLS LAST LIMIT 20"
        elif has_distances and mode == "unreviewed":
            order_clause = "ORDER BY zg_distance DESC NULLS LAST LIMIT 20"
        else:
            order_clause = "ORDER BY RANDOM() LIMIT 20"
        
        all_imgs = db.execute(f"SELECT id, status, face_index, image_path, zg_distance FROM images WHERE persona_id = ? AND status = ? {order_clause}", (pid, status_filter)).fetchall()
        
        # Mix in best images only if unreviewed (to prevent drift)
        if mode == "unreviewed":
            best_imgs = db.execute(f"SELECT id, status, face_index, image_path, zg_distance FROM images WHERE persona_id = ? AND status = ? ORDER BY zg_distance ASC NULLS LAST LIMIT 5", (pid, status_filter)).fetchall()
        else:
            best_imgs = []
            
        db.close()
        
        combined_ids = [r["id"] for r in best_imgs]
        for img in all_imgs:
            if img["id"] not in combined_ids:
                combined_ids.append(img["id"])
            if len(combined_ids) >= 20:
                break
                
        final_imgs = []
        for img in best_imgs + all_imgs:
            if img["id"] in combined_ids and img["id"] not in [r["id"] for r in final_imgs]:
                final_imgs.append(img)
        
        return jsonify({
            "persona_id": pid,
            "persona_name": pname,
            "total_for_persona": total_for_persona,
            "image_ids": [r["id"] for r in final_imgs],
            "reference_ids": reference_ids,
            "unreviewed_ids": [r["id"] for r in final_imgs if r["status"] == status_filter],
            "statuses": {r["id"]: r["status"] for r in final_imgs},
            "labels": {r["id"]: f"face{r['face_index']}" for r in final_imgs},
            "distances": {r["id"]: r["zg_distance"] for r in final_imgs},
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
        
        return jsonify({"remaining": remaining, "mode": mode})

    HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body { background:#1a1a1a; color:#ddd; font-family:sans-serif; margin:20px; }
h2 { color:#4CAF50; }
.grid { display:flex; flex-wrap:wrap; gap:6px; }
.thumb { width:120px; height:120px; object-fit:cover; border:2px solid #333; cursor:pointer; transition:border-color 0.2s,opacity 0.2s; }
.thumb.tainted-black { border-color:#000; opacity:0.5; }
.thumb.tainted-nonface { border-color:#f44336; opacity:0.5; }
.thumb.tainted-contamination { border-color:#e91e63; opacity:0.5; }
.thumb.tainted-unusable { border-color:#9C27B0; opacity:0.5; }
.thumb.tainted-approved_bad_geometry { border-color:#FF9800; opacity:0.8; }
.thumb.approved { border-color:#4CAF50; opacity:0.6; }
.thumb.unreviewed { border-color:#555; }
.tools { margin:12px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.brush { padding:8px 16px; border:none; cursor:pointer; font-weight:bold; border-radius:4px; color:white; }
.brush.nonface { background:#f44336; }
.brush.contamination { background:#e91e63; }
.brush.unusable { background:#9C27B0; }
.brush.badgeom { background:#FF9800; color:black; }
.brush.done { background:#4CAF50; font-size:16px; padding:10px 24px; }
.brush.mode { background:#555; font-size:12px; }
.brush.xray { background:#2196F3; }
.brush:hover { opacity:0.85; } .brush.active { outline:3px solid white; }
.key { font-size:12px; color:#666; margin:8px 0; }
.label { font-size:10px; color:#888; text-align:center; margin-top:2px; }
.thumb-wrapper { display:flex; flex-direction:column; align-items:center; }
</style></head><body>
<h2><span id="persona_name">loading...</span></h2>
<div class="key"><span style="color:#555">■ unreviewed</span> <span style="color:#4CAF50">■ approved</span> <span style="color:#f44336">■ non-face</span> <span style="color:#e91e63">■ contamination</span> <span style="color:#9C27B0">■ unusable</span> <span style="color:#FF9800">■ bad geometry</span></div>
<div class="tools">
  <span style="color:#aaa">Brush:</span>
  <button class="brush nonface active" id="btn_nonface" onclick="setBrush('tainted:extraction_nonface')">Non-face</button>
  <button class="brush contamination" id="btn_contam" onclick="setBrush('tainted:contamination')">Contamination</button>
  <button class="brush unusable" id="btn_unusable" onclick="setBrush('tainted:unusable')">Unusable</button>
  <button class="brush badgeom" id="btn_badgeom" onclick="setBrush('tainted:approved_bad_geometry')">Bad Geometry</button>
  <button class="brush done" onclick="donePersona()">&#x2713; DONE</button>
  
  <span style="color:#aaa; margin-left:10px;">Modes:</span>
  <button class="brush mode" id="btn_unreviewed_mode" style="display:none" onclick="switchMode('unreviewed')">&#x21E0; First Pass</button>
  <button class="brush mode" id="btn_review_mode" onclick="switchMode('review')">&#x21BB; Review Pass</button>
  <button class="brush mode" id="btn_audit_mode" onclick="switchMode('audit')">🔍 Audit Pass</button>
  <button class="brush xray" onclick="toggleSkel()" id="btn_xray">🦴 X-Ray</button>
  
  <span id="status" style="margin-left:10px;"></span>
</div>
<div id="reference-anchors" style="display:flex; gap:10px; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #444;"></div>
<div class="grid" id="grid"></div>
<script>
let personaId=null,brush='tainted:extraction_nonface',tainted={},mode='unreviewed',shownIds=[],showSkel=false;
let g_data = null;

const urlParams = new URLSearchParams(window.location.search);
const forcePersona = urlParams.get('persona');

function toggleSkel() {
    showSkel = !showSkel;
    document.getElementById('btn_xray').style.outline = showSkel ? "3px solid white" : "none";
    
    // Dynamically update the src attributes to toggle the skel=1 query param
    document.querySelectorAll('img.thumb').forEach(img => {
        let currentSrc = img.src;
        if (showSkel) {
            if (!currentSrc.includes('skel=1')) {
                img.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 'skel=1';
            }
        } else {
            img.src = currentSrc.replace(/[\?&]skel=1/, '');
        }
    });
}

function switchMode(m){
    mode=m;
    document.getElementById('btn_review_mode').style.display=m==='review'?'none':'inline-block';
    document.getElementById('btn_audit_mode').style.display=m==='audit'?'none':'inline-block';
    document.getElementById('btn_unreviewed_mode').style.display=m==='unreviewed'?'none':'inline-block';
    loadPersona();
}

function setBrush(b){
    brush=b;
    document.querySelectorAll('.brush').forEach(e=>e.classList.remove('active'));
    if(b==='tainted:extraction_nonface')document.getElementById('btn_nonface').classList.add('active');
    if(b==='tainted:contamination')document.getElementById('btn_contam').classList.add('active');
    if(b==='tainted:unusable')document.getElementById('btn_unusable').classList.add('active');
    if(b==='tainted:approved_bad_geometry')document.getElementById('btn_badgeom').classList.add('active');
}

async function loadPersona(){
  let url = '/api/random_persona?mode=' + mode;
  if (forcePersona) {
      url += '&persona=' + encodeURIComponent(forcePersona);
  }
  const resp=await fetch(url);
  const data=await resp.json();
  g_data = data;
  if(!data.persona_id){document.getElementById('grid').innerHTML='<p style="font-size:24px;color:#4CAF50">'+data.persona_name+'!</p>';return;}
  personaId=data.persona_id;
  shownIds=data.image_ids;
  document.getElementById('persona_name').innerText=data.persona_name + ' (showing ' + shownIds.length + ' of ' + data.total_for_persona + ' remaining)';
  const n=data.unreviewed_ids.length;document.getElementById('status').innerText=n+' '+(mode==='unreviewed'?'unreviewed':'approved');
  tainted={};
  renderReferences(data.reference_ids);
  renderGrid(data.image_ids,data.statuses,data.labels,data.distances);
}

function renderReferences(ids) {
    const container = document.getElementById('reference-anchors');
    container.innerHTML = '<div style="margin-top:40px; margin-right: 15px; color: #aaa;"><strong>True Identity<br>Anchors:</strong></div>';
    if (!ids || ids.length === 0) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'flex';
    for(const id of ids) {
        const wrapper = document.createElement('div');
        wrapper.className = 'thumb-wrapper';
        const img = document.createElement('img');
        img.src = '/api/thumb/' + id + (showSkel ? "?skel=1" : "");
        img.className = 'thumb approved';
        img.style.borderColor = '#FFD700'; 
        img.style.borderWidth = '3px';
        img.style.opacity = '1';
        wrapper.appendChild(img);
        const lbl = document.createElement('div');
        lbl.className = 'label';
        lbl.innerText = 'Reference';
        lbl.style.color = '#FFD700';
        wrapper.appendChild(lbl);
        container.appendChild(wrapper);
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
    
    if (dist !== null && dist !== undefined) {
        lbl += ` (zg: ${parseFloat(dist).toFixed(1)})`;
    }
    
    const wrapper=document.createElement('div');wrapper.className='thumb-wrapper';
    const img=document.createElement('img');
    img.src='/api/thumb/'+id + (showSkel ? "?skel=1" : "");
    img.dataset.id=id;img.className='thumb';
    if(s.startsWith('tainted:')){img.classList.add('tainted-'+s.replace('tainted:extraction_','').replace('tainted:',''));}
    else if(s==='approved'){img.classList.add('approved');if(mode==='review'||mode==='audit')img.onclick=()=>toggleTaint(img,id);}
    else{img.classList.add('unreviewed');img.onclick=()=>toggleTaint(img,id);}
    wrapper.appendChild(img);
    if(lbl){const label=document.createElement('div');label.className='label';label.innerText=lbl;wrapper.appendChild(label);}
    grid.appendChild(wrapper);
  }
}

function toggleTaint(el,id){
  if(tainted[id]){
      delete tainted[id];
      el.className='thumb';
      if(mode==='review'||mode==='audit') el.classList.add('approved');
      else el.classList.add('unreviewed');
  }
  else{
      tainted[id]=brush;
      el.className='thumb';
      el.classList.add('tainted-'+brush.replace('tainted:extraction_','').replace('tainted:',''));
  }
}

async function donePersona(){
  const t=Object.keys(tainted).length;
  const resp=await fetch('/api/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({persona_id:personaId,tainted:tainted,mode:mode,shown_ids:shownIds})});
  const data=await resp.json();
  document.getElementById('status').innerText='Saved. '+data.remaining+' total items remaining. Loading next...';
  setTimeout(loadPersona,400);
}
loadPersona();
</script></body></html>"""

    @app.route("/")
    def index():
        return render_template_string(HTML)
    
    return app
