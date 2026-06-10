#!/usr/bin/env python3
"""Review UI: random persona, random image order, brush-to-taint, DONE button.
Modes: first pass (unreviewed) and review pass (re-check approved).
Run: .venv/bin/python scripts/16_review_ui.py --port 5100"""
import os, sys, random, sqlite3, io, time
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_pca.constants import FACE_SLICE

ROOT = "/mnt/nas-ai-models/training-data/loras/hegre-14000px"
ENRICHED = "data/hegre_enriched"
DB = "data/review.db"
MAX_DIM = 1024

app = Flask(__name__)

def get_db():
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    return db

def _crop_face(image_path):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    s = min(1.0, MAX_DIM / max(w, h))
    if s < 1.0:
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    img_arr = np.asarray(im)
    h_a, w_a = img_arr.shape[:2]
    enriched_dir = image_path.replace(ROOT, ENRICHED).replace('.jpg', '')
    pose_path = os.path.join(enriched_dir, "pose.npy")
    if not os.path.exists(pose_path):
        return im.resize((120, 120))
    pose = np.load(pose_path)
    face = pose[FACE_SLICE]
    px_x = (face[:, 0] + 1.0) / 2.0 * w_a
    px_y = (face[:, 1] + 1.0) / 2.0 * h_a
    span_x = px_x.max() - px_x.min()
    span_y = px_y.max() - px_y.min()
    if span_x < 0.01 or span_y < 0.01:
        return Image.new("RGB", (120, 120), (40, 40, 40))
    mn_x, mx_x = np.percentile(px_x, [5, 95])
    mn_y, mx_y = np.percentile(px_y, [5, 95])
    cx, cy = (mn_x + mx_x) / 2, (mn_y + mx_y) / 2
    span = max(mx_x - mn_x, mx_y - mn_y) * 1.1
    x0 = int(max(0, cx - span/2)); x1 = int(min(w_a, cx + span/2))
    y0 = int(max(0, cy - span/2)); y1 = int(min(h_a, cy + span/2))
    if x1 <= x0 or y1 <= y0:
        return im.resize((120, 120))
    crop = Image.fromarray(img_arr[y0:y1, x0:x1])
    return crop.resize((120, 120))

_thumb_cache = {}

@app.route("/api/thumb/<int:image_id>")
def api_thumb(image_id):
    if image_id in _thumb_cache:
        img = _thumb_cache[image_id]
    else:
        db = get_db()
        row = db.execute("SELECT image_path FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            return "", 404
        img = _crop_face(row["image_path"])
        _thumb_cache[image_id] = img
        if len(_thumb_cache) > 200:
            _thumb_cache.pop(next(iter(_thumb_cache)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route("/api/random_persona")
def api_random_persona():
    mode = request.args.get("mode", "unreviewed")
    status_filter = "approved" if mode == "review" else "unreviewed"
    db = get_db()
    row = db.execute(f"""
        SELECT p.id, p.name FROM personas p
        JOIN images i ON i.persona_id = p.id
        WHERE i.status = '{status_filter}'
        GROUP BY p.id
        ORDER BY RANDOM() LIMIT 1
    """).fetchone()
    if not row:
        msg = "ALL REVIEWED" if mode == "review" else "ALL DONE"
        return jsonify({"persona_id": None, "persona_name": msg, "image_ids": [], "mode": mode})
    pid, pname = row["id"], row["name"]
    imgs = db.execute(f"""
        SELECT id FROM images WHERE persona_id = ? AND status = '{status_filter}' ORDER BY RANDOM()
    """, (pid,)).fetchall()
    unreviewed_ids = [r["id"] for r in imgs]
    all_imgs = db.execute("""
        SELECT id, status FROM images WHERE persona_id = ? ORDER BY RANDOM()
    """, (pid,)).fetchall()
    return jsonify({
        "persona_id": pid, "persona_name": pname,
        "image_ids": [r["id"] for r in all_imgs],
        "unreviewed_ids": unreviewed_ids,
        "statuses": {r["id"]: r["status"] for r in all_imgs},
        "mode": mode
    })

@app.route("/api/done", methods=["POST"])
def api_done():
    data = request.get_json()
    pid = data["persona_id"]
    tainted = data.get("tainted", {})
    mode = data.get("mode", "unreviewed")
    db = get_db()
    tainted_count = 0
    for img_id_str, reason in tainted.items():
        img_id = int(img_id_str)
        db.execute(
            "UPDATE images SET status = ?, reviewed_at = datetime('now') WHERE id = ?",
            (reason, img_id))
        changed = db.execute("SELECT changes()").fetchone()[0]
        tainted_count += changed
    if mode == "review":
        approved = 0
    else:
        db.execute(
            "UPDATE images SET status = 'approved', reviewed_at = datetime('now') WHERE persona_id = ? AND status = 'unreviewed'",
            (pid,))
        approved = db.execute("SELECT changes()").fetchone()[0]
    db.commit()
    remaining = db.execute(
        "SELECT COUNT(*) FROM images WHERE status = 'approved'" if mode == "review"
        else "SELECT COUNT(*) FROM images WHERE status = 'unreviewed'"
    ).fetchone()[0]
    return jsonify({"approved": approved, "tainted": tainted_count, "remaining": remaining, "mode": mode})

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
.thumb.approved { border-color:#4CAF50; opacity:0.6; }
.thumb.unreviewed { border-color:#555; }
.tools { margin:12px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.brush { padding:8px 16px; border:none; cursor:pointer; font-weight:bold; border-radius:4px; color:white; }
.brush.black { background:#000; } .brush.nonface { background:#f44336; }
.brush.contamination { background:#e91e63; } .brush.unusable { background:#9C27B0; }
.brush.done { background:#4CAF50; font-size:16px; padding:10px 24px; }
.brush.mode { background:#555; font-size:12px; }
.brush:hover { opacity:0.85; } .brush.active { outline:3px solid white; }
.key { font-size:12px; color:#666; margin:8px 0; }
#mode_indicator { color:#FF9800; font-weight:bold; margin-left:8px; }
</style></head><body>
<h2><span id="persona_name">loading...</span> <span id="mode_indicator">FIRST PASS</span></h2>
<div class="key">
  <span style="color:#555">■ unreviewed</span> <span style="color:#4CAF50">■ approved</span>
  <span style="color:#000">■ black</span> <span style="color:#f44336">■ non-face</span>
  <span style="color:#e91e63">■ contamination</span> <span style="color:#9C27B0">■ unusable</span>
</div>
<div class="tools">
  <span style="color:#aaa">Brush:</span>
  <button class="brush black active" id="btn_black" onclick="setBrush('tainted:extraction_black')">Black</button>
  <button class="brush nonface" id="btn_nonface" onclick="setBrush('tainted:extraction_nonface')">Non-face</button>
  <button class="brush contamination" id="btn_contam" onclick="setBrush('tainted:contamination')">Contamination</button>
  <button class="brush unusable" id="btn_unusable" onclick="setBrush('tainted:unusable')">Unusable</button>
  <button class="brush done" onclick="donePersona()">&#x2713; DONE</button>
  <button class="brush mode" id="btn_review_mode" onclick="switchMode('review')">&#x21BB; Review Pass</button>
  <button class="brush mode" id="btn_unreviewed_mode" style="display:none" onclick="switchMode('unreviewed')">&#x21E0; First Pass</button>
  <span id="status"></span>
</div>
<div class="grid" id="grid"></div>
<script>
let personaId=null,brush='tainted:extraction_black',tainted={},mode='unreviewed';
function switchMode(m){mode=m;document.getElementById('mode_indicator').innerText=m==='review'?'REVIEW PASS':'FIRST PASS';document.getElementById('btn_review_mode').style.display=m==='review'?'none':'inline-block';document.getElementById('btn_unreviewed_mode').style.display=m==='review'?'inline-block':'none';loadPersona();}
function setBrush(b){brush=b;document.querySelectorAll('.brush').forEach(e=>e.classList.remove('active'));if(b==='tainted:extraction_black')document.getElementById('btn_black').classList.add('active');if(b==='tainted:extraction_nonface')document.getElementById('btn_nonface').classList.add('active');if(b==='tainted:contamination')document.getElementById('btn_contam').classList.add('active');if(b==='tainted:unusable')document.getElementById('btn_unusable').classList.add('active');}
async function loadPersona(){
  const resp=await fetch('/api/random_persona?mode='+mode);
  const data=await resp.json();
  if(!data.persona_id){document.getElementById('grid').innerHTML='<p style="font-size:24px;color:#4CAF50">'+data.persona_name+'!</p>';document.getElementById('persona_name').innerText='';document.getElementById('status').innerText='';return;}
  personaId=data.persona_id;document.getElementById('persona_name').innerText=data.persona_name;tainted={};
  let n=data.unreviewed_ids.length;document.getElementById('status').innerText=n+' '+(mode==='review'?'approved':'unreviewed');
  renderGrid(data.image_ids,data.statuses);
}
function renderGrid(ids,statuses){
  const grid=document.getElementById('grid');grid.innerHTML='';
  for(const id of ids){
    const s=statuses[id]||'unreviewed';const img=document.createElement('img');img.src='/api/thumb/'+id;img.dataset.id=id;img.className='thumb';
    if(s.startsWith('tainted:')){img.classList.add('tainted-'+s.replace('tainted:extraction_','').replace('tainted:',''));}
    else if(s==='approved'){img.classList.add('approved');if(mode==='review')img.onclick=()=>toggleTaint(img,id);}
    else{img.classList.add('unreviewed');img.onclick=()=>toggleTaint(img,id);}
    grid.appendChild(img);
  }
}
function toggleTaint(el,id){
  if(tainted[id]){delete tainted[id];el.className='thumb';if(mode==='review')el.classList.add('approved');else el.classList.add('unreviewed');}
  else{tainted[id]=brush;el.className='thumb';el.classList.add('tainted-'+brush.replace('tainted:extraction_','').replace('tainted:',''));}
}
async function donePersona(){
  const t=Object.keys(tainted).length;
  const msg=mode==='review'?('Re-taint '+t+' images? (rest stay approved)'):('Approve all unreviewed images? ('+t+' tainted, rest approved)');
  if(!confirm(msg))return;
  const resp=await fetch('/api/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({persona_id:personaId,tainted:tainted,mode:mode})});
  const data=await resp.json();
  document.getElementById('status').innerText='Saved. '+data.remaining+' images remaining. Loading next...';
  setTimeout(loadPersona,400);
}
loadPersona();
</script></body></html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5100)
    args = ap.parse_args()
    print(f"Starting review UI on http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
