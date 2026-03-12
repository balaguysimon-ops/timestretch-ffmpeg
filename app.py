from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import subprocess, tempfile, os, json, uuid, urllib.request, shutil, time, threading

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE = "/tmp/audio_out"
os.makedirs(STORE, exist_ok=True)

# Cleanup: supprime les fichiers > 10 min toutes les 5 min
def _cleanup_store():
    while True:
        time.sleep(300)
        try:
            now = time.time()
            for f in os.listdir(STORE):
                fp = os.path.join(STORE, f)
                if os.path.isfile(fp) and now - os.path.getmtime(fp) > 600:
                    os.remove(fp)
        except Exception:
            pass

threading.Thread(target=_cleanup_store, daemon=True).start()

class Req(BaseModel):
    audio_url: str
    target_duration_ms: int
    preserve_pitch: bool = True
    format_out: Literal["mp3","wav"] = "mp3"
    bitrate_kbps: int = 192

def sh(cmd: list[str]) -> tuple[str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout, p.stderr

def ffprobe_duration_ms(path: str) -> int:
    out, _ = sh([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1", path
    ])
    return int(round(float(out.strip()) * 1000))

def _run_pipeline(src_path: str, target_ms: int, format_out: str, bitrate_kbps: int) -> dict:
    """Pipeline de stretch partagé entre /process (URL) et /process-upload (fichier)."""
    MIN_F, MAX_F = 0.8, 1.25

    with tempfile.TemporaryDirectory() as td:
        step1 = os.path.join(td, "step1.wav")
        norm  = os.path.join(td, "norm.wav")
        final = os.path.join(td, f"final.{format_out}")

        # 1) Pivot 24k mono + stretch principal
        in_ms = ffprobe_duration_ms(src_path)
        F = target_ms / in_ms  # ratio cible/source (ex: 0.8 = raccourcir)
        if not (MIN_F <= F <= MAX_F):
            raise HTTPException(400, f"Stretch factor {F:.3f} outside [{MIN_F},{MAX_F}]")

        # atempo = vitesse de lecture : >1 accélère (raccourcit), <1 ralentit (allonge)
        atempo_main = in_ms / target_ms
        sh([
            "ffmpeg","-y","-i", src_path,
            "-af", f"aformat=sample_fmts=fltp:sample_rates=24000:channel_layouts=mono,atempo={atempo_main:.8f}",
            step1
        ])

        # 2) Correction
        step1_ms = ffprobe_duration_ms(step1)
        atempo_corr = step1_ms / target_ms

        # 3) Loudnorm analyse (2-pass EBU R128) — passe 1
        _o, err = sh([
            "ffmpeg","-y","-i", step1,
            "-af", f"atempo={atempo_corr:.8f},loudnorm=I=-23:LRA=7:TP=-1:print_format=json",
            "-f","null","-"
        ])
        j0, j1 = err.find("{"), err.rfind("}")
        if j0 == -1 or j1 == -1:
            raise HTTPException(500, "loudnorm analysis failed")
        stats = json.loads(err[j0:j1+1])

        # 4) Correction + loudnorm apply + fades
        ln = (
            "loudnorm=I=-23:LRA=7:TP=-1:"
            f"measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}:"
            f"measured_TP={stats['input_tp']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:linear=true:print_format=summary"
        )
        fade_out_start = max(0, target_ms / 1000.0 - 0.01)
        sh([
            "ffmpeg","-y","-i", step1,
            "-af", f"atempo={atempo_corr:.8f},{ln},afade=t=in:st=0:d=0.01,afade=t=out:st={fade_out_start:.4f}:d=0.01",
            "-ar","24000","-ac","1",
            norm
        ])

        # 5) Ajustement exact de la durée (trim/pad silence)
        norm_ms = ffprobe_duration_ms(norm)
        delta_ms = target_ms - norm_ms
        if abs(delta_ms) > 2:
            trimmed = os.path.join(td, "trimmed.wav")
            if delta_ms < 0:
                sh(["ffmpeg","-y","-i", norm, "-af",
                    f"atrim=0:{target_ms/1000.0:.6f},asetpts=N/SR/TB", trimmed])
            else:
                pad_sec = delta_ms/1000.0
                sh(["ffmpeg","-y","-i", norm, "-af",
                    f"apad=pad_dur={pad_sec:.6f},atrim=0:{target_ms/1000.0:.6f},asetpts=N/SR/TB", trimmed])
            shutil.move(trimmed, norm)

        # 6) Encodage final
        if format_out == "mp3":
            sh(["ffmpeg","-y","-i", norm, "-c:a","libmp3lame","-b:a", f"{bitrate_kbps}k", final])
        else:
            shutil.copyfile(norm, final)

        out_ms = ffprobe_duration_ms(final)
        # MP3 encoder ajoute un padding incompressible (~26-60ms)
        tolerance_ms = 80 if format_out == "mp3" else 2
        if abs(out_ms - target_ms) > tolerance_ms:
            raise HTTPException(500, f"Final duration mismatch {out_ms} vs {target_ms}")

        # 7) Expose via /dl/{name}
        fid = uuid.uuid4().hex
        suggested = f"chronique_{fid}_{target_ms}.{format_out}"
        store_path = os.path.join(STORE, suggested)
        shutil.copyfile(final, store_path)

        return {
            "download_url": f"/dl/{suggested}",
            "final_duration_ms": out_ms,
            "factor": round(F, 6),
            "factor_correction": round(atempo_corr, 6),
            "pipeline": "ffmpeg_atempo_double + loudnorm2 + fades + exact_trim",
            "meta": {"input_duration_ms": in_ms, "post_norm_ms": norm_ms}
        }

@app.post("/process")
def process(req: Req):
    """Stretch via URL (existant)."""
    if req.target_duration_ms <= 0 or req.target_duration_ms > 3*60*1000:
        raise HTTPException(413, "target_duration_ms out of bounds (0, 180000]")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in")
        try:
            urllib.request.urlretrieve(req.audio_url, src)
        except Exception as e:
            raise HTTPException(400, f"Cannot download source: {e}")
        return _run_pipeline(src, req.target_duration_ms, req.format_out, req.bitrate_kbps)

@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    target_duration_ms: int = Form(...),
    format_out: str = Form("mp3"),
    bitrate_kbps: int = Form(192),
):
    """Stretch via upload direct (pas besoin d'URL publique)."""
    if target_duration_ms <= 0 or target_duration_ms > 3*60*1000:
        raise HTTPException(413, "target_duration_ms out of bounds (0, 180000]")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.wav")
        with open(src, "wb") as f:
            content = await file.read()
            f.write(content)
        return _run_pipeline(src, target_duration_ms, format_out, bitrate_kbps)

@app.get("/dl/{name}")
def dl(name: str):
    path = os.path.join(STORE, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    mime = "audio/mpeg" if name.endswith(".mp3") else "audio/wav"
    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": f'inline; filename="{name}"'})

@app.get("/")
def root():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
