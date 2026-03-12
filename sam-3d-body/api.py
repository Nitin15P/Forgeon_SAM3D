# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SAM3D Body Estimation API",
    description="Upload a video to generate 3D body mesh, IK, and Inverse Dynamics using SAM3D.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
PIPELINE_OUTPUT_DIR = BASE_DIR / "pipeline_output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PIPELINE_OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = str(BASE_DIR / "model.ckpt")
YOLOV8_WEIGHTS = str(BASE_DIR / "yolov8n.pt")
MHR_PATH = str(BASE_DIR / "assets" / "mhr_model.pt")
SAM_PYTHON = "/home/tech/sam_env/bin/python"

_smpl_npz = BASE_DIR / "MHRtoSMPL" / "SMPL_NEUTRAL.npz"
_smpl_pkl = BASE_DIR / "MHRtoSMPL" / "SMPL_NEUTRAL.pkl"
SMPL_MODEL_PATH = str(_smpl_npz if _smpl_npz.exists() else _smpl_pkl)

# In-memory job tracking
jobs: dict = {}
bio_jobs: dict = {}


def run_demo(job_id: str, video_path: str, output_video: str, output_folder: str, skip_fov: bool = True):
    """Run demo.py with XVFB in a subprocess."""
    jobs[job_id]["status"] = "processing"
    log_path = os.path.join(output_folder, "job.log")
    jobs[job_id]["log_path"] = log_path

    cmd = [
        "xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x720x24",
        "/home/tech/sam_env/bin/python", "-u", str(BASE_DIR / "demo.py"),
        "--video_path", video_path,
        "--output_video", output_video,
        "--output_folder", output_folder,
        "--checkpoint_path", CHECKPOINT_PATH,
        "--detector_name", "yolov8",
        "--yolov8_weights", YOLOV8_WEIGHTS,
        "--mhr_path", MHR_PATH,
        "--bbox_thresh", "0.5",
        "--det_class_id", "0",
    ]

    # Skip FOV estimator for faster processing (uses fixed FOV instead of MoGe per-frame)
    if skip_fov:
        cmd += ["--fov_name", ""]

    env = os.environ.copy()
    env["DISPLAY"] = ":99"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                env=env,
            )
            process.wait(timeout=3600)

        if process.returncode == 0:
            # Crop to front 3D mesh panel only (panel 3 of 4, i.e. x offset = 2*original_width)
            cropped_video = output_video.replace("_output.mp4", "_rendered.mp4")
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height", "-of", "csv=p=0", output_video],
                    capture_output=True, text=True
                )
                w, h = map(int, probe.stdout.strip().split(","))
                panel_w = w // 4  # each of 4 panels
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_video,
                    "-vf", f"crop={panel_w}:{h}:{panel_w*2}:0",
                    "-c:v", "libx264", "-crf", "18", cropped_video
                ], capture_output=True, check=True)
                jobs[job_id]["output_video"] = cropped_video
            except Exception:
                jobs[job_id]["output_video"] = output_video  # fallback to full video
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_folder"] = output_folder
        else:
            with open(log_path) as f:
                content = f.read()
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = content[-3000:]
    except subprocess.TimeoutExpired:
        process.kill()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Processing timed out (1 hour limit)"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/process-video", summary="Upload a video for 3D body mesh generation")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input video file (mp4, avi, mov, etc.)"),
    mesh_only: Optional[bool] = False,
    skip_fov: Optional[bool] = True,
):
    """
    Upload a video file to run SAM3D body estimation.

    - Returns a **job_id** immediately.
    - Poll `/status/{job_id}` to check progress.
    - Download results via `/download/{job_id}/video` or `/download/{job_id}/mesh`.
    """
    allowed_types = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed_types}")

    job_id = str(uuid.uuid4())
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_upload_dir.mkdir(parents=True)
    job_output_dir.mkdir(parents=True)

    input_path = str(job_upload_dir / file.filename)
    output_video = str(job_output_dir / f"{Path(file.filename).stem}_output.mp4")

    # Save uploaded file
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "output_video": None,
        "output_folder": str(job_output_dir),
        "error": None,
        "log": None,
    }

    background_tasks.add_task(run_demo, job_id, input_path, output_video, str(job_output_dir), skip_fov)

    return JSONResponse({"job_id": job_id, "status": "queued", "message": "Processing started. Poll /status/{job_id} for updates."})


@app.get("/status/{job_id}", summary="Check processing status of a job")
async def get_status(job_id: str):
    """
    Returns the current status of a job:
    - `queued` — waiting to start
    - `processing` — running SAM3D inference
    - `completed` — done, ready to download
    - `failed` — error occurred (see `error` field)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(jobs[job_id])


@app.get("/download/{job_id}/video", summary="Download the output video")
async def download_video(job_id: str):
    """Download the rendered output video with 3D body overlays."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    video_path = job["output_video"]
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Output video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=os.path.basename(video_path))


@app.get("/download/{job_id}/mesh", summary="Download mesh data as a zip archive")
async def download_mesh(job_id: str):
    """Download all mesh data (NPZ + JSON files) as a zip archive."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")

    mesh_folder = Path(job["output_folder"]) / "mesh_data"
    if not mesh_folder.exists():
        raise HTTPException(status_code=404, detail="Mesh data folder not found")

    zip_path = str(Path(job["output_folder"]) / f"{job_id}_mesh_data")
    shutil.make_archive(zip_path, "zip", str(mesh_folder))

    return FileResponse(
        zip_path + ".zip",
        media_type="application/zip",
        filename=f"{job_id}_mesh_data.zip",
    )


@app.get("/logs/{job_id}", summary="Stream live logs for a job")
async def get_logs(job_id: str, last_n_lines: int = 50):
    """Returns the last N lines of the live processing log for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = jobs[job_id].get("log_path")
    if not log_path or not os.path.exists(log_path):
        return JSONResponse({"log": "Log not available yet."})
    with open(log_path) as f:
        lines = f.readlines()
    return JSONResponse({"status": jobs[job_id]["status"], "log": "".join(lines[-last_n_lines:])})


@app.get("/jobs", summary="List all jobs")
async def list_jobs():
    """Returns all job IDs and their statuses."""
    return JSONResponse({jid: {"status": j["status"], "filename": j["filename"]} for jid, j in jobs.items()})


@app.get("/file/{job_id}", summary="Download rendered video by job folder (works after restart)")
async def download_file_direct(job_id: str):
    """Download the rendered video directly by job_id even after API restart."""
    base = OUTPUT_DIR / job_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Job folder not found")
    # Prefer cropped rendered video, fallback to full output
    for pattern in ["*_rendered.mp4", "*_output.mp4"]:
        matches = list(base.glob(pattern))
        if matches:
            return FileResponse(str(matches[0]), media_type="video/mp4", filename=matches[0].name)
    raise HTTPException(status_code=404, detail="No video found in job folder")


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "model": CHECKPOINT_PATH, "xvfb": shutil.which("xvfb-run") is not None}


# ─── Biomechanics Pipeline ─────────────────────────────────────────────────────

def _detect_fps(video_path: str) -> float:
    """Use ffprobe to detect video FPS; fallback to 30."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            num, den = r.stdout.strip().split("/")
            return round(float(num) / float(den), 3)
    except Exception:
        pass
    return 30.0


def _read_mass_from_subject_json(subject_folder: str) -> Optional[float]:
    """Parse massKg from smpl2addbio's _subject.json."""
    p = Path(subject_folder) / "_subject.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        val = data.get("massKg") or data.get("mass_kg")
        return float(val) if val is not None else None
    except Exception:
        return None


def _find_ik_mot_files(engine_input: str) -> list:
    ik_dir = Path(engine_input) / "osim_results" / "IK"
    if not ik_dir.exists():
        return []
    return sorted(str(p) for p in ik_dir.glob("*.mot"))


def _find_osim(engine_input: str) -> Optional[str]:
    base = Path(engine_input)
    # Priority order: scaled model > unscaled-but-optimized > raw unscaled
    search_dirs = [
        base / "osim_results" / "Models",
        base,
    ]
    preferred = [
        "match_markers_but_ignore_physics.osim",
        "optimized_scale_and_markers.osim",
        "unscaled_generic.osim",
        "unscaled_generic_raw.osim",
    ]
    for name in preferred:
        for d in search_dirs:
            p = d / name
            if p.exists():
                return str(p)
    for d in search_dirs:
        candidates = list(d.glob("*.osim"))
        if candidates:
            return str(candidates[0])
    return None


def _run_biomechanics_pipeline(
    job_id: str,
    video_path: str,
    fps: float,
    gender: str,
    mass: Optional[float],
):
    """Full pipeline: SAM3D → MHR-SMPL → SMPL sequence → TRC → IK → ID."""
    job = bio_jobs[job_id]
    output_dir = Path(job["output_dir"])
    log_path = output_dir / "pipeline.log"
    job["log_path"] = str(log_path)

    def log(msg: str):
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    def run_stage(name: str, cmd: list, env: dict = None, timeout: int = 7200):
        job["stage"] = name
        log(f"\n{'='*60}\n[{name}] Starting…\n{'='*60}\n")
        with open(log_path, "a") as lf:
            p = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                env=env or os.environ.copy(),
            )
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                raise RuntimeError(f"Stage '{name}' timed out after {timeout}s")
        if p.returncode != 0:
            raise RuntimeError(f"Stage '{name}' failed (exit code {p.returncode})")
        log(f"[{name}] Done.\n")

    try:
        job["status"] = "processing"
        engine_input = str(output_dir / "engine_input")
        id_output = str(output_dir / "id_results")

        log(f"Job {job_id} | video={video_path} fps={fps} gender={gender} mass={mass}")

        # ── Stages 1–6: video → mesh → SMPL → TRC → IK ──────────────────────
        pipeline_env = os.environ.copy()
        pipeline_env.update({
            "AB_SKIP_SCALETOOL": "1",
            "PYTHONUNBUFFERED": "1",
            "DISPLAY": ":99",
            "SAM3D_MHR_PATH": MHR_PATH,
        })
        pipeline_cmd = [
            "xvfb-run", "-a", "-s", "-screen 0 1920x1080x24",
            SAM_PYTHON,
            str(BASE_DIR / "scripts" / "run_full_pipeline.py"),
            "--video_path", video_path,
            "--output_dir", str(output_dir),
            "--checkpoint_path", CHECKPOINT_PATH,
            "--yolov8_weights", YOLOV8_WEIGHTS,
            "--smpl-model", SMPL_MODEL_PATH,
            "--fps", str(fps),
            "--gender", gender,
            "--keep-intermediates",
        ]
        run_stage("sam3d_to_ik", pipeline_cmd, env=pipeline_env, timeout=7200)

        # ── Locate IK outputs ─────────────────────────────────────────────────
        osim_path = _find_osim(engine_input)
        mot_files = _find_ik_mot_files(engine_input)
        if not osim_path:
            raise RuntimeError("No .osim model found after IK pass")
        if not mot_files:
            raise RuntimeError("No IK .mot file found after IK pass")
        job["osim_path"] = osim_path
        job["mot_files"] = mot_files
        log(f"IK: {len(mot_files)} trial(s) | osim: {osim_path}")

        # ── Resolve subject mass ──────────────────────────────────────────────
        if mass is None:
            subject_folder = str(output_dir / "OUT_addbio" / "smpl_seq_trial")
            mass = _read_mass_from_subject_json(subject_folder) or 70.0
            log(f"Mass from subject.json: {mass:.2f} kg")
        job["mass_kg"] = mass

        # ── Stage 7: Inverse Dynamics (one run per .mot segment) ─────────────
        os.makedirs(id_output, exist_ok=True)
        id_env = os.environ.copy()
        id_env["PYTHONUNBUFFERED"] = "1"
        for mot_file in mot_files:
            seg_name = Path(mot_file).stem  # e.g. smpl_sequence_segment_0_ik
            seg_out = os.path.join(id_output, seg_name)
            id_cmd = [
                SAM_PYTHON,
                str(BASE_DIR / "run_inverse_dynamics.py"),
                "--osim", osim_path,
                "--mot", mot_file,
                "--out", seg_out,
                "--mass", str(mass),
            ]
            run_stage(f"id_{seg_name}", id_cmd, env=id_env, timeout=600)

        job["status"] = "completed"
        job["id_output"] = id_output
        log(f"\nPipeline complete. IK: {engine_input}  ID: {id_output}")

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        log(f"\n[ERROR] {exc}")


# ─── Biomechanics endpoints ────────────────────────────────────────────────────

@app.post("/biomechanics/process", summary="Run full biomechanics pipeline (SAM3D → IK → ID)")
async def process_biomechanics(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Input video (mp4, avi, mov, mkv, webm)"),
    fps: Optional[float] = Query(None, description="Video framerate. Auto-detected if omitted."),
    gender: Optional[str] = Query("neutral", description="SMPL gender: neutral | male | female"),
    mass: Optional[float] = Query(None, description="Subject mass in kg. Estimated from SMPL shape if omitted."),
):
    """
    Upload a video to run the full biomechanics pipeline:
    1. SAM3D body mesh estimation
    2. MHR → SMPL conversion
    3. SMPL → TRC marker generation (smpl2addbio)
    4. Inverse Kinematics (AddBiomechanics engine)
    5. Inverse Dynamics (static GRF estimation)

    Returns a **job_id**. Poll `/biomechanics/{job_id}/status` for progress.
    Download results via `/biomechanics/{job_id}/download/{ik|id|all}`.
    """
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    if gender not in ("neutral", "male", "female"):
        raise HTTPException(status_code=400, detail="gender must be neutral | male | female")

    job_id = str(uuid.uuid4())
    job_dir = PIPELINE_OUTPUT_DIR / job_id
    upload_path = job_dir / file.filename
    job_dir.mkdir(parents=True)

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    actual_fps = fps or _detect_fps(str(upload_path))

    bio_jobs[job_id] = {
        "status": "queued",
        "stage": None,
        "filename": file.filename,
        "fps": actual_fps,
        "gender": gender,
        "mass_kg": mass,
        "output_dir": str(job_dir),
        "log_path": None,
        "osim_path": None,
        "mot_files": None,
        "id_output": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_biomechanics_pipeline, job_id, str(upload_path), actual_fps, gender, mass
    )
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "fps_detected": actual_fps,
        "message": f"Pipeline started. Poll /biomechanics/{job_id}/status for updates.",
    })


@app.get("/biomechanics/{job_id}/status", summary="Pipeline job status")
async def biomechanics_status(job_id: str):
    """
    Returns job status and current stage:
    - `queued` → `processing` (stage: sam3d_to_ik | id_*) → `completed` | `failed`
    """
    if job_id not in bio_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = bio_jobs[job_id]
    return JSONResponse({
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "filename": job["filename"],
        "fps": job["fps"],
        "gender": job["gender"],
        "mass_kg": job.get("mass_kg"),
        "mot_files": job.get("mot_files"),
        "error": job.get("error"),
    })


@app.get("/biomechanics/{job_id}/logs", summary="Tail pipeline logs")
async def biomechanics_logs(job_id: str, last_n: int = Query(80, description="Number of log lines to return")):
    """Returns the last N lines of the pipeline log."""
    if job_id not in bio_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = bio_jobs[job_id].get("log_path")
    if not log_path or not os.path.exists(log_path):
        return JSONResponse({"log": "Log not available yet."})
    with open(log_path) as f:
        lines = f.readlines()
    return JSONResponse({
        "status": bio_jobs[job_id]["status"],
        "stage": bio_jobs[job_id]["stage"],
        "log": "".join(lines[-last_n:]),
    })


@app.get("/biomechanics/{job_id}/download/{artifact}", summary="Download pipeline results")
async def biomechanics_download(job_id: str, artifact: str):
    """
    Download results for a completed job.

    `artifact` options:
    - `ik`    — IK .mot files (zip)
    - `id`    — Inverse Dynamics .sto files (zip)
    - `video` — rendered mesh video (mp4)
    - `all`   — everything zipped together
    """
    if job_id not in bio_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = bio_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job['status']})")

    output_dir = Path(job["output_dir"])

    if artifact == "video":
        for pattern in ["*_rendered.mp4", "*_output.mp4"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                return FileResponse(str(matches[0]), media_type="video/mp4", filename=matches[0].name)
        raise HTTPException(status_code=404, detail="No video found")

    if artifact == "ik":
        ik_dir = output_dir / "engine_input" / "osim_results" / "IK"
        if not ik_dir.exists():
            raise HTTPException(status_code=404, detail="IK output not found")
        zip_base = str(output_dir / f"{job_id}_ik")
        shutil.make_archive(zip_base, "zip", str(ik_dir))
        return FileResponse(zip_base + ".zip", media_type="application/zip", filename=f"{job_id}_ik.zip")

    if artifact == "id":
        id_dir = Path(job.get("id_output", ""))
        if not id_dir.exists():
            raise HTTPException(status_code=404, detail="ID output not found")
        zip_base = str(output_dir / f"{job_id}_id")
        shutil.make_archive(zip_base, "zip", str(id_dir))
        return FileResponse(zip_base + ".zip", media_type="application/zip", filename=f"{job_id}_id.zip")

    if artifact == "all":
        # Collect: IK .mot, ID .sto, osim model, video
        staging = output_dir / "_download_staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()

        ik_dir = output_dir / "engine_input" / "osim_results" / "IK"
        if ik_dir.exists():
            shutil.copytree(str(ik_dir), str(staging / "IK"))

        id_dir = Path(job.get("id_output", ""))
        if id_dir.exists():
            shutil.copytree(str(id_dir), str(staging / "ID"))

        osim = job.get("osim_path")
        if osim and os.path.exists(osim):
            shutil.copy2(osim, str(staging / Path(osim).name))

        for pattern in ["*_rendered.mp4", "*_output.mp4"]:
            matches = list(output_dir.glob(pattern))
            if matches:
                shutil.copy2(str(matches[0]), str(staging / matches[0].name))
                break

        zip_base = str(output_dir / f"{job_id}_all")
        shutil.make_archive(zip_base, "zip", str(staging))
        shutil.rmtree(staging, ignore_errors=True)
        return FileResponse(zip_base + ".zip", media_type="application/zip", filename=f"{job_id}_all.zip")

    raise HTTPException(status_code=400, detail="artifact must be: ik | id | video | all")


@app.get("/biomechanics", summary="List all biomechanics jobs")
async def list_bio_jobs():
    return JSONResponse({
        jid: {"status": j["status"], "stage": j["stage"], "filename": j["filename"]}
        for jid, j in bio_jobs.items()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
