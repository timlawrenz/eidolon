import io
import sqlite3
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image

def get_db(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    return db

def create_app(db_path: Path, faces_root: Path) -> Flask:
    app = Flask(__name__)
    _thumb_cache = {}
    THUMB_SIZE = (120, 120)
    resample_filter = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
    
    def _load_thumb(image_path_rel: str) -> bytes:
        full_path = faces_root / image_path_rel
        if not full_path.exists():
            placeholder = Image.new("RGB", THUMB_SIZE, (60, 60, 60))
            buf = io.BytesIO()
            placeholder.save(buf, format="JPEG", quality=75)
            return buf.getvalue()
        
        img = Image.open(full_path).convert("RGB")
        img.thumbnail(THUMB_SIZE, resample_filter)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()

    @app.route("/api/thumb/<int:image_id>")
    def api_thumb(image_id):
        if image_id not in _thumb_cache:
            db = get_db(db_path)
            row = db.execute("SELECT image_path FROM images WHERE id = ?", (image_id,)).fetchone()
            db.close()
            if not row:
                return "", 404
            _thumb_cache[image_id] = _load_thumb(row["image_path"])
            if len(_thumb_cache) > 200:
                _thumb_cache.pop(next(iter(_thumb_cache)))
        return send_file(io.BytesIO(_thumb_cache[image_id]), mimetype="image/jpeg")

    @app.route("/api/random_persona")
    def api_random_persona():
        mode = request.args.get("mode", "unreviewed")
        status_filter = "approved" if mode == "review" else "unreviewed"
        db = get_db(db_path)
        row = db.execute(f"SELECT p.id, p.name FROM personas p JOIN images i ON i.persona_id = p.id WHERE i.status = ? GROUP BY p.id ORDER BY RANDOM() LIMIT 1", (status_filter,)).fetchone()
        
        if not row:
            msg = "ALL REVIEWED" if mode == "review" else "ALL DONE"
            db.close()
            return jsonify({"persona_id": None, "persona_name": msg, "image_ids": [], "mode": mode})
            
        pid, pname = row["id"], row["name"]
        
        total_for_persona = db.execute("SELECT COUNT(*) FROM images WHERE persona_id = ? AND status = ?", (pid, status_filter)).fetchone()[0]
        
        all_imgs = db.execute("SELECT id, status, face_index, image_path FROM images WHERE persona_id = ? AND status = ? ORDER BY RANDOM() LIMIT 20", (pid, status_filter)).fetchall()
        db.close()
        
        return jsonify({
            "persona_id": pid,
            "persona_name": pname,
            "total_for_persona": total_for_persona,
            "image_ids": [r["id"] for r in all_imgs],
            "unreviewed_ids": [r["id"] for r in all_imgs if r["status"] == status_filter],
            "statuses": {r["id"]: r["status"] for r in all_imgs},
            "labels": {r["id"]: f"face{r['face_index']}" for r in all_imgs},
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
            
        if mode != "review":
            approved_ids = [int(i) for i in shown_ids if str(i) not in tainted]
            if approved_ids:
                placeholders = ",".join("?" * len(approved_ids))
                db.execute(f"UPDATE images SET status = 'approved', reviewed_at = datetime('now') WHERE id IN ({placeholders})", approved_ids)
                
        db.commit()
        status_filter = "approved" if mode == "review" else "unreviewed"
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
.thumb.approved { border-color:#4CAF50; opacity:0.6; }
.thumb.unreviewed { border-color:#555; }
.tools { margin:12px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.brush { padding:8px 16px; border:none; cursor:pointer; font-weight:bold; border-radius:4px; color:white; }
.brush.nonface { background:#f44336; }
.brush.contamination { background:#e91e63; }
.brush.unusable { background:#9C27B0; }
.brush.done { background:#4CAF50; font-size:16px; padding:10px 24px; }
.brush.mode { background:#555; font-size:12px; }
.brush:hover { opacity:0.85; } .brush.active { outline:3px solid white; }
.key { font-size:12px; color:#666; margin:8px 0; }
.label { font-size:10px; color:#888; text-align:center; margin-top:2px; }
.thumb-wrapper { display:flex; flex-direction:column; align-items:center; }
</style></head><body>
<h2><span id="persona_name">loading...</span></h2>
<div class="key"><span style="color:#555">■ unreviewed</span> <span style="color:#4CAF50">■ approved</span> <span style="color:#f44336">■ non-face</span> <span style="color:#e91e63">■ contamination</span> <span style="color:#9C27B0">■ unusable</span></div>
<div class="tools">
  <span style="color:#aaa">Brush:</span>
  <button class="brush nonface active" id="btn_nonface" onclick="setBrush('tainted:extraction_nonface')">Non-face</button>
  <button class="brush contamination" id="btn_contam" onclick="setBrush('tainted:contamination')">Contamination</button>
  <button class="brush unusable" id="btn_unusable" onclick="setBrush('tainted:unusable')">Unusable</button>
  <button class="brush done" onclick="donePersona()">&#x2713; DONE</button>
  <button class="brush mode" id="btn_review_mode" onclick="switchMode('review')">&#x21BB; Review Pass</button>
  <button class="brush mode" id="btn_unreviewed_mode" style="display:none" onclick="switchMode('unreviewed')">&#x21E0; First Pass</button>
  <span id="status"></span>
</div>
<div class="grid" id="grid"></div>
<script>
let personaId=null,brush='tainted:extraction_nonface',tainted={},mode='unreviewed',shownIds=[];
function switchMode(m){mode=m;document.getElementById('btn_review_mode').style.display=m==='review'?'none':'inline-block';document.getElementById('btn_unreviewed_mode').style.display=m==='review'?'inline-block':'none';loadPersona();}
function setBrush(b){brush=b;document.querySelectorAll('.brush').forEach(e=>e.classList.remove('active'));if(b==='tainted:extraction_nonface')document.getElementById('btn_nonface').classList.add('active');if(b==='tainted:contamination')document.getElementById('btn_contam').classList.add('active');if(b==='tainted:unusable')document.getElementById('btn_unusable').classList.add('active');}
async function loadPersona(){
  const resp=await fetch('/api/random_persona?mode='+mode);
  const data=await resp.json();
  if(!data.persona_id){document.getElementById('grid').innerHTML='<p style="font-size:24px;color:#4CAF50">'+data.persona_name+'!</p>';return;}
  personaId=data.persona_id;
  shownIds=data.image_ids;
  document.getElementById('persona_name').innerText=data.persona_name + ' (showing ' + shownIds.length + ' of ' + data.total_for_persona + ' remaining)';
  const n=data.unreviewed_ids.length;document.getElementById('status').innerText=n+' '+(mode==='review'?'approved':'unreviewed');
  tainted={};
  renderGrid(data.image_ids,data.statuses,data.labels);
}
function renderGrid(ids,statuses,labels){
  const grid=document.getElementById('grid');grid.innerHTML='';
  for(const id of ids){
    const s=statuses[id]||'unreviewed';const lbl=labels[id]||'';
    if(mode==='unreviewed' && s!=='unreviewed') continue;
    const wrapper=document.createElement('div');wrapper.className='thumb-wrapper';
    const img=document.createElement('img');img.src='/api/thumb/'+id;img.dataset.id=id;img.className='thumb';
    if(s.startsWith('tainted:')){img.classList.add('tainted-'+s.replace('tainted:extraction_','').replace('tainted:',''));}
    else if(s==='approved'){img.classList.add('approved');if(mode==='review')img.onclick=()=>toggleTaint(img,id);}
    else{img.classList.add('unreviewed');img.onclick=()=>toggleTaint(img,id);}
    wrapper.appendChild(img);
    if(lbl){const label=document.createElement('div');label.className='label';label.innerText=lbl;wrapper.appendChild(label);}
    grid.appendChild(wrapper);
  }
}
function toggleTaint(el,id){
  if(tainted[id]){delete tainted[id];el.className='thumb';if(mode==='review')el.classList.add('approved');else el.classList.add('unreviewed');}
  else{tainted[id]=brush;el.className='thumb';el.classList.add('tainted-'+brush.replace('tainted:extraction_','').replace('tainted:',''));}
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
