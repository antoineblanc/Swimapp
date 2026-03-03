import os, json, wave, struct, math, base64, shutil, tempfile, subprocess, threading
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

MAX_FILE_MB = 500
FRAME_W, FRAME_H = 640, 360
JOBS_DIR = "/tmp/swimjobs"
os.makedirs(JOBS_DIR, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

def job_path(job_id): return os.path.join(JOBS_DIR, f"{job_id}.json")

def load_job(job_id):
    try:
        with open(job_path(job_id)) as f: return json.load(f)
    except: return None

def save_job(job_id, data):
    with open(job_path(job_id), "w") as f: json.dump(data, f)

def upd(job_id, msg):
    job = load_job(job_id)
    if job:
        job["progress"].append(msg)
        save_job(job_id, job)

def get_duration(path):
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_streams",path],
        capture_output=True, text=True, timeout=30)
    for s in json.loads(r.stdout)["streams"]:
        if s["codec_type"] == "video": return float(s["duration"])
    raise ValueError("No video stream found")

def extract_frames(src, out_dir, start, dur, fps, label):
    pattern = os.path.join(out_dir, f"{label}_%04d.jpg")
    subprocess.run(["ffmpeg","-i",src,"-vf",f"fps={fps},scale={FRAME_W}:{FRAME_H}",
        "-ss",str(start),"-t",str(dur),pattern,"-y","-q:v","2","-loglevel","error"],
        check=True, timeout=120)
    files = sorted(Path(out_dir).glob(f"{label}_*.jpg"))
    return [(round(start + i/fps, 3), str(f)) for i,f in enumerate(files)]

def extract_audio_rms(src, out_dir):
    wav = os.path.join(out_dir, "audio.wav")
    subprocess.run(["ffmpeg","-i",src,"-vn","-acodec","pcm_s16le","-ar","44100",
        wav,"-y","-loglevel","error"], check=True, timeout=60)
    with wave.open(wav,"rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        mono = [(samples[i]+samples[i+1])//2 for i in range(0,len(samples),2)] if ch==2 else list(samples)
    chunk = sr//100
    return [(round(i*0.01,3), math.sqrt(sum(x*x for x in mono[i*chunk:(i+1)*chunk])/chunk))
            for i in range(len(mono)//chunk)]

def find_go(rms_data, window=4.0):
    early = [(t,r) for t,r in rms_data if t<=window]
    if not early: return 0.0
    baseline_vals = sorted([r for t,r in early if t<0.3])
    baseline = baseline_vals[len(baseline_vals)//2] if baseline_vals else 100
    threshold = max(baseline*4, 2000)
    for t,r in early:
        if r >= threshold: return round(t,3)
    return round(max(early, key=lambda x:x[1])[0], 3)

def frame_b64(path):
    with open(path,"rb") as f: return base64.standard_b64encode(f.read()).decode()

def ask_claude(frames, question, hint=""):
    content = []
    if hint: content.append({"type":"text","text":hint})
    for t,path in frames:
        content.append({"type":"text","text":f"Frame at {t:.3f}s:"})
        content.append({"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":frame_b64(path)}})
    content.append({"type":"text","text":question})
    resp = client.messages.create(model="claude-opus-4-6", max_tokens=300,
        messages=[{"role":"user","content":content}])
    text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
    try: return json.loads(text)
    except:
        import re
        m = re.search(r'"timestamp"\s*:\s*([\d.]+)', text)
        return {"timestamp": float(m.group(1)) if m else None, "confidence":"low"}

def analyse_one(path, name, job_id):
    tmpdir = tempfile.mkdtemp()
    try:
        upd(job_id, f"[{name}] Reading video...")
        duration = get_duration(path)

        upd(job_id, f"[{name}] Detecting start signal...")
        rms = extract_audio_rms(path, tmpdir)
        go_time = find_go(rms)
        upd(job_id, f"[{name}] Start signal at {go_time:.2f}s")

        upd(job_id, f"[{name}] Detecting water entry...")
        ef = extract_frames(path, tmpdir, max(0,go_time-0.2), 3.5, 10, "entry")
        time.sleep(2)
        entry = ask_claude(ef, 'When do hands FIRST touch water? JSON only: {"timestamp":2.07,"confidence":"high","note":""}', "Swimming dive analysis.")
        entry_t = entry.get("timestamp")
        upd(job_id, f"[{name}] Entry: {entry_t:.2f}s" if entry_t else f"[{name}] Entry: not found")

        upd(job_id, f"[{name}] Detecting end of coulée...")
        surf_start = (entry_t or go_time+1.5) + 1.0
        sf = extract_frames(path, tmpdir, surf_start, 9.0, 5, "surf")
        time.sleep(2)
        surface = ask_claude(sf, f'Swimmer entered at {(entry_t or 0):.2f}s. When does HEAD first break surface? JSON only: {{"timestamp":7.4,"confidence":"high","note":""}}', "Breaststroke coulée analysis.")
        surface_t = surface.get("timestamp")
        upd(job_id, f"[{name}] Coulée end: {surface_t:.2f}s" if surface_t else f"[{name}] Coulée: not found")

        upd(job_id, f"[{name}] Detecting turn...")
        approx = duration * 0.45
        tf = extract_frames(path, tmpdir, max(0,approx-10), 20.0, 3, "turn")
        time.sleep(2)
        wall = ask_claude(tf, f'Near turn wall around {approx:.0f}s. When do hands/feet FIRST TOUCH wall? JSON only: {{"timestamp":25.5,"confidence":"high","note":""}}', "Swim turn analysis.")
        wall_t = wall.get("timestamp")
        upd(job_id, f"[{name}] Wall touch: {wall_t:.2f}s" if wall_t else f"[{name}] Wall: not found")

        pushoff_t = None
        if wall_t:
            upd(job_id, f"[{name}] Detecting push-off...")
            pf = extract_frames(path, tmpdir, wall_t, 3.0, 8, "po")
            time.sleep(2)
            pushoff = ask_claude(pf, f'Touched wall at {wall_t:.2f}s. When do FEET LEAVE wall? JSON only: {{"timestamp":26.1,"confidence":"high","note":""}}', "Push-off analysis.")
            pushoff_t = pushoff.get("timestamp")
            upd(job_id, f"[{name}] Push-off: {pushoff_t:.2f}s" if pushoff_t else f"[{name}] Push-off: not found")

        upd(job_id, f"[{name}] Detecting finish...")
        ff = extract_frames(path, tmpdir, max(0,duration-8), 9.0, 5, "fin")
        time.sleep(2)
        finish = ask_claude(ff, 'When does swimmer TOUCH finish wall? JSON only: {"timestamp":54.2,"confidence":"high","note":""}', "Finish analysis.")
        finish_t = finish.get("timestamp")
        upd(job_id, f"[{name}] Finish: {finish_t:.2f}s" if finish_t else f"[{name}] Finish: not found")

        upd(job_id, f"[{name}] ✓ Done!")
        return {
            "name": name,
            "reaction": round(entry_t-go_time,3) if entry_t else None,
            "coulee":   round(surface_t-entry_t,3) if surface_t and entry_t else None,
            "first_25": round(wall_t-go_time,3) if wall_t else None,
            "turn":     round(pushoff_t-wall_t,3) if pushoff_t and wall_t else None,
            "last_25":  round(finish_t-pushoff_t,3) if finish_t and pushoff_t else None,
            "total":    round(finish_t-go_time,3) if finish_t else None,
            "error": None,
        }
    except Exception as e:
        upd(job_id, f"[{name}] ✗ Error: {e}")
        return {"name":name,"error":str(e),"reaction":None,"coulee":None,"first_25":None,"turn":None,"last_25":None,"total":None}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def run_job(job_id, files):
    results = []
    for name, path in files:
        r = analyse_one(path, name, job_id)
        results.append(r)
        try: os.unlink(path)
        except: pass
    job = load_job(job_id)
    job["results"] = results
    job["status"] = "done"
    job["generated"] = datetime.now().isoformat()
    save_job(job_id, job)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f: return f.read()

@app.post("/analyse")
async def analyse(files: List[UploadFile] = File(...)):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    if len(files) > 5:
        raise HTTPException(400, "Max 5 videos")
    import uuid
    job_id = str(uuid.uuid4())[:8]
    saved = []
    for f in files:
        ext = Path(f.filename).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(await f.read())
        tmp.close()
        saved.append((Path(f.filename).stem, tmp.name))
    save_job(job_id, {"status":"running","progress":[],"results":[],"generated":None})
    threading.Thread(target=run_job, args=(job_id, saved), daemon=True).start()
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = load_job(job_id)
    if not job: raise HTTPException(404, "Job not found")
    return job

@app.get("/health")
async def health(): return {"ok": True}
