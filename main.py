import os
import json
import wave
import struct
import math
import base64
import shutil
import tempfile
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic

app = FastAPI(title="SwimTimer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

MAX_FILE_MB = 500
FRAME_W, FRAME_H = 640, 360
AUDIO_SPIKE_MULT = 4
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True, timeout=30
    )
    for s in json.loads(r.stdout)["streams"]:
        if s["codec_type"] == "video":
            return float(s["duration"])
    raise ValueError("No video stream found")

def extract_frames(src, out_dir, start, dur, fps, label):
    pattern = os.path.join(out_dir, f"{label}_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-i", src,
        "-vf", f"fps={fps},scale={FRAME_W}:{FRAME_H}",
        "-ss", str(start), "-t", str(dur),
        pattern, "-y", "-q:v", "2", "-loglevel", "error"
    ], check=True, timeout=120)
    files = sorted(Path(out_dir).glob(f"{label}_*.jpg"))
    return [(round(start + i / fps, 3), str(f)) for i, f in enumerate(files)]

def extract_audio_rms(src, out_dir):
    wav = os.path.join(out_dir, "audio.wav")
    subprocess.run([
        "ffmpeg", "-i", src, "-vn", "-acodec", "pcm_s16le", "-ar", "44100",
        wav, "-y", "-loglevel", "error"
    ], check=True, timeout=60)
    with wave.open(wav, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        mono = [(samples[i]+samples[i+1])//2 for i in range(0, len(samples), 2)] if ch == 2 else list(samples)
    chunk = sr // 100
    return [(round(i * 0.01, 3), math.sqrt(sum(x*x for x in mono[i*chunk:(i+1)*chunk]) / chunk))
            for i in range(len(mono) // chunk)]

def find_go(rms_data, window=4.0):
    early = [(t, r) for t, r in rms_data if t <= window]
    if not early: return 0.0
    baseline_vals = sorted([r for t, r in early if t < 0.3])
    baseline = baseline_vals[len(baseline_vals)//2] if baseline_vals else 100
    threshold = max(baseline * AUDIO_SPIKE_MULT, 2000)
    for t, r in early:
        if r >= threshold:
            return round(t, 3)
    return round(max(early, key=lambda x: x[1])[0], 3)

def frame_b64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()

def ask_claude(frames_with_times, question, hint=""):
    content = []
    if hint:
        content.append({"type": "text", "text": hint})
    for t, path in frames_with_times:
        content.append({"type": "text", "text": f"Frame at {t:.3f}s:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64(path)}})
    content.append({"type": "text", "text": question})
    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": content}]
    )
    text = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        import re
        m = re.search(r'"timestamp"\s*:\s*([\d.]+)', text)
        return {"timestamp": float(m.group(1)) if m else None, "confidence": "low", "note": text[:120]}

def analyse_video_sync(path, name, job_id):
    tmpdir = tempfile.mkdtemp(prefix=f"swim_{name}_")
    def upd(msg):
        jobs[job_id]["progress"].append(f"[{name}] {msg}")
    try:
        upd("Reading video info...")
        duration = get_duration(path)
        upd("Detecting start signal from audio...")
        rms = extract_audio_rms(path, tmpdir)
        go_time = find_go(rms)
        upd(f"Start signal at {go_time:.2f}s")
        upd("Detecting water entry...")
        entry_frames = extract_frames(path, tmpdir, max(0, go_time - 0.2), 3.5, 30, "entry")
        entry = ask_claude(entry_frames,
            'Find when hands FIRST touch water. Respond ONLY as JSON: {"timestamp": 2.07, "confidence": "high", "note": "..."}',
            "Analysing a swimming dive entry.")
        entry_t = entry.get("timestamp")
        upd(f"Water entry: {entry_t:.2f}s" if entry_t else "Water entry: not found")
        upd("Detecting end of coulee...")
        surf_start = (entry_t or go_time + 1.5) + 1.0
        surf_frames = extract_frames(path, tmpdir, surf_start, 9.0, 15, "surf")
        surface = ask_claude(surf_frames,
            f'Swimmer entered at {(entry_t or 0):.3f}s. Find when HEAD first breaks surface and first stroke begins. Respond ONLY as JSON: {{"timestamp": 7.4, "confidence": "high", "note": "..."}}',
            "Analysing breaststroke coulee phase.")
        surface_t = surface.get("timestamp")
        upd(f"End of coulee: {surface_t:.2f}s" if surface_t else "Coulee end: not found")
        upd("Detecting turn wall touch...")
        approx_turn = duration * 0.45
        turn_frames = extract_frames(path, tmpdir, max(0, approx_turn - 10), 20.0, 10, "turn")
        wall = ask_claude(turn_frames,
            f'Swimmer near turn wall around {approx_turn:.0f}s. Find when hands/feet FIRST TOUCH wall. Respond ONLY as JSON: {{"timestamp": 25.5, "confidence": "high", "note": "..."}}',
            "Analysing swim turn.")
        wall_t = wall.get("timestamp")
        upd(f"Wall touch: {wall_t:.2f}s" if wall_t else "Wall touch: not found")
        pushoff_t = None
        if wall_t:
            upd("Detecting push-off...")
            po_frames = extract_frames(path, tmpdir, wall_t, 3.0, 30, "po")
            pushoff = ask_claude(po_frames,
                f'Swimmer touched wall at {wall_t:.3f}s. Find when FEET LEAVE wall. Respond ONLY as JSON: {{"timestamp": 26.1, "confidence": "high", "note": "..."}}',
                "Analysing turn push-off.")
            pushoff_t = pushoff.get("timestamp")
            upd(f"Push-off: {pushoff_t:.2f}s" if pushoff_t else "Push-off: not found")
        upd("Detecting finish...")
        fin_frames = extract_frames(path, tmpdir, max(0, duration - 8), 9.0, 15, "fin")
        finish = ask_claude(fin_frames,
            'Find when swimmer TOUCHES the finish wall. Respond ONLY as JSON: {"timestamp": 54.2, "confidence": "high", "note": "..."}',
            "Analysing swim finish.")
        finish_t = finish.get("timestamp")
        upd(f"Finish: {finish_t:.2f}s" if finish_t else "Finish: not found")
        upd("Analysis complete!")
        return {
            "name": name, "duration": round(duration, 3), "go_time": go_time,
            "reaction": round(entry_t - go_time, 3) if entry_t else None,
            "coulee":   round(surface_t - entry_t, 3) if surface_t and entry_t else None,
            "first_25": round(wall_t - go_time, 3) if wall_t else None,
            "turn":     round(pushoff_t - wall_t, 3) if pushoff_t and wall_t else None,
            "last_25":  round(finish_t - pushoff_t, 3) if finish_t and pushoff_t else None,
            "total":    round(finish_t - go_time, 3) if finish_t else None,
            "error": None,
        }
    except Exception as e:
        upd(f"Error: {e}")
        return {"name": name, "error": str(e), "reaction": None, "coulee": None,
                "first_25": None, "turn": None, "last_25": None, "total": None}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

jobs = {}

def run_job_thread(job_id, files):
    jobs[job_id]["status"] = "running"
    results = []
    for name, path in files:
        try:
            r = analyse_video_sync(path, name, job_id)
        except Exception as e:
            r = {"name": name, "error": str(e)}
        results.append(r)
        try: os.unlink(path)
        except: pass
    jobs[job_id]["results"] = results
    jobs[job_id]["status"] = "done"
    jobs[job_id]["generated"] = datetime.now().isoformat()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/analyse")
async def analyse(files: List[UploadFile] = File(...)):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY not configured on server")
    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 videos at once")
    import uuid
    job_id = str(uuid.uuid4())[:8]
    saved = []
    for f in files:
        if f.size and f.size > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(400, f"{f.filename} exceeds {MAX_FILE_MB}MB limit")
        ext = Path(f.filename).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(await f.read())
        tmp.close()
        saved.append((Path(f.filename).stem, tmp.name))
    jobs[job_id] = {"status": "queued", "progress": [], "results": [], "generated": None}
    threading.Thread(target=run_job_thread, args=(job_id, saved), daemon=True).start()
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "results": job["results"] if job["status"] == "done" else [],
        "generated": job["generated"],
    }

@app.get("/health")
async def health():
    return {"ok": True}
