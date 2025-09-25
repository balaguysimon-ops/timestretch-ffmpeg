from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Literal
import subprocess, tempfile, os, json, uuid, urllib.request, shutil

app = FastAPI()

STORE = "/tmp/audio_out"          # stockage temporaire pour /dl/{name}
os.makedirs(STORE, exist_ok=True)

class Req(BaseModel):
    audio_url: str
    target_duration_ms: int
    preserve_pitch: bool = True            # atempo préserve le pitch
    format_out: Literal["mp3","wav"] = "mp3"
    bitrate_kbps: int = 192                # 192 ou 256

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

@app.post("/process")
def process(req: Req):
    if req.target_duration_ms <= 0 or req.target_duration_ms > 3*60*1000:
        raise HTTPException(413, "target_duration_ms out of bounds (0, 180000]")
    MIN_F, MAX_F = 0.8, 1.25

    with tempfile.TemporaryDirectory() as td:
        src   = os.path.join(td, "in")
        pivot = os.path.join(td, "pivot.wav")
        step1 = os.path.join(td, "step1.wav")
        step2 = os.path.join(td, "step2.wav")
        norm  = os.path.join(td, "norm.wav")
        final = os.path.join(td, f"final.{req.format_out}")

        # 0) Download
        try:
            urllib.request.urlretrieve(req.audio_url, src)
        except Exception as e:
            raise HTTPException(400, f"Cannot download source: {e}")

        # 1) Pivot 48k mono float
        sh(["ffmpeg","-y","-i", src, "-ac","1","-ar","48000","-sample_fmt","fltp", pivot])
        in_ms = ffprobe_duration_ms(pivot)

        # 2) Stretch principal
        target_ms = req.target_duration_ms
        F = target_ms / in_ms
        if not (MIN_F <= F <= MAX_F):
            raise HTTPException(400, f"Stretch factor {F:.3f} outside [{MIN_F},{MAX_F}]")
        sh([
            "ffmpeg","-y","-i", pivot,
            "-af", f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=mono,atempo={F:.8f}",
            step1
        ])

        # 3) Correction
        step1_ms = ffprobe_duration_ms(step1)
        F_corr = target_ms / step1_ms
        sh(["ffmpeg","-y","-i", step1, "-af", f"atempo={F_corr:.8f}", step2])

        # 4) Loudnorm (EBU R128) 2-pass + fades 10 ms
        _o, err = sh([
            "ffmpeg","-y","-i", step2,
            "-af","loudnorm=I=-23:LRA=7:TP=-1:print_format=json",
            "-f","null","-"
        ])
        j0, j1 = err.find("{"), err.rfind("}")
        if j0 == -1 or j1 == -1:
            raise HTTPException(500, "loudnorm analysis failed")
        stats = json.loads(err[j0:j1+1])

        ln = (
            "loudnorm=I=-23:LRA=7:TP=-1:"
            f"measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}:"
            f"measured_TP={stats['input_tp']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:linear=true:print_format=summary"
        )
        sh([
            "ffmpeg","-y","-i", step2,
            "-af", f"{ln},afade=t=in:st=0:d=0.01,afade=t=out:st=0:d=0.01",
            "-sample_fmt","fltp","-ar","48000","-ac","1",
            norm
        ])

        # 5) Ajustement exact de la durée (trim/pad silence)
        norm_ms = ffprobe_duration_ms(norm)
        delta_ms = target_ms - norm_ms
        if abs(delta_ms) > 2:
            if delta_ms < 0:
                sh(["ffmpeg","-y","-i", norm, "-af",
                    f"atrim=0:{target_ms/1000.0:.6f},asetpts=N/SR/TB", norm])
            else:
                pad_sec = delta_ms/1000.0
                sh(["ffmpeg","-y","-i", norm, "-af",
                    f"apad=pad_dur={pad_sec:.6f},atrim=0:{target_ms/1000.0:.6f},asetpts=N/SR/TB", norm])

        # 6) Encodage final
        if req.format_out == "mp3":
            sh(["ffmpeg","-y","-i", norm, "-c:a","libmp3lame","-b:a", f"{req.bitrate_kbps}k", final])
        else:
            shutil.copyfile(norm, final)

        out_ms = ffprobe_duration_ms(final)
        if abs(out_ms - target_ms) > 1:
            raise HTTPException(500, f"Final duration mismatch {out_ms} vs {target_ms}")

        # 7) Expose via /dl/{name}
        fid = uuid.uuid4().hex
        suggested = f"chronique_{fid}_{target_ms}.{req.format_out}"
        store_path = os.path.join(STORE, suggested)
        shutil.copyfile(final, store_path)

        return {
            "download_url": f"/dl/{suggested}",
            "final_duration_ms": out_ms,
            "factor": round(F, 6),
            "factor_correction": round(F_corr, 6),
            "pipeline": "ffmpeg_atempo_double + loudnorm2 + fades + exact_trim",
            "meta": {"input_duration_ms": in_ms, "post_norm_ms": norm_ms}
        }

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
