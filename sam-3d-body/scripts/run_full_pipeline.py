#!/usr/bin/env python3
"""
Single entry-point: one video → SAM 3D mesh → MHR-SMPL → SMPL sequence → AddBiomechanics → IK.

Runs the full pipeline so it can be triggered with a single video input. Steps:
  1. demo.py (subprocess): video → mesh_data (per-frame .npz)
  2. sam3d_dir_to_smpl.py (subprocess, with LD_LIBRARY_PATH for libtorch): mesh_data → smpl_fit_results.npz
  3. smpl_fit_to_sequence.py (subprocess): smpl_fit_results.npz → subject_smpl_sequence.npz (smpl2ab format)
  4. smpl2addbio.py (subprocess): SMPL sequence → synthetic markers (TRC) + _subject.json
  5. prepare_engine_input.py (subprocess): AddBiomechanics input layout
  6. run_until_ik.py (subprocess): kinematics pass → IK .mot files

Usage:
  # Full pipeline (video → IK)
  python scripts/run_full_pipeline.py --video_path /path/to/video.mp4 --smpl-model /path/to/SMPL_male.pkl

  # Stop after a given stage (for debugging or running MHR-SMPL only on VM)
  python scripts/run_full_pipeline.py --video_path /path/to/video.mp4 --smpl-model /path/to/SMPL_male.pkl --stop-after mhr_smpl

  # Mesh-only (no rendered video), then run rest from existing mesh_data
  python scripts/run_full_pipeline.py --video_path /path/to/video.mp4 --smpl-model /path/to/SMPL_male.pkl --mesh-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    cwd = cwd or _repo_root()
    env = env or os.environ
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: video → SAM 3D mesh → MHR-SMPL → SMPL sequence → AddBio → IK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Base output directory (default: ./output/<video_basename>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="SAM 3D Body checkpoint path (or use --checkpoint_hf)",
    )
    parser.add_argument(
        "--checkpoint_hf",
        type=str,
        default="",
        help="HuggingFace model id, e.g. facebook/sam-3d-body-dinov3",
    )
    parser.add_argument(
        "--smpl-model",
        type=str,
        required=True,
        help="Path to SMPL .pkl (or .npz) for MHR→SMPL conversion",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="male",
        choices=("male", "female", "neutral"),
        help="SMPL gender (default: male)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Mocap framerate for sequence (default: 30)",
    )
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Step 1: only save mesh data, no output video",
    )
    parser.add_argument(
        "--person-idx",
        type=int,
        default=0,
        help="Person index in multi-person mesh_data (default: 0)",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        choices=("mesh", "mhr_smpl", "sequence", "addbio", "ik"),
        default="ik",
        help="Stop after this stage (default: ik)",
    )
    parser.add_argument(
        "--yolov8_weights",
        type=str,
        default="",
        help="Path to YOLOv8 weights if using yolov8 detector",
    )
    parser.add_argument(
        "--enable-fov",
        action="store_true",
        help="Enable FOV estimator (MoGe2). By default FOV is disabled so 'moge' is not required.",
    )
    parser.add_argument(
        "--keep-only-final",
        action="store_true",
        default=True,
        help="Keep only mesh video and IK files; remove intermediate dirs (default: True)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep all intermediate outputs (overrides --keep-only-final)",
    )
    args = parser.parse_args()

    repo = _repo_root()
    video_path = os.path.abspath(args.video_path)
    if not os.path.isfile(video_path):
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = args.output_dir or os.path.join(repo, "output", base_name)
    output_dir = os.path.abspath(output_dir)
    mesh_data_dir = os.path.join(output_dir, "mesh_data")
    mhr_smpl_dir = os.path.join(output_dir, "mhr_smpl")
    smpl_seq_folder = os.path.join(output_dir, "smpl_seq_trial")
    seq_npz = os.path.join(smpl_seq_folder, "subject_smpl_sequence.npz")
    addbio_out = os.path.join(output_dir, "OUT_addbio")
    engine_input = os.path.join(output_dir, "engine_input")
    smpl_model_path = os.path.abspath(args.smpl_model)
    if not os.path.isfile(smpl_model_path):
        print(f"Error: SMPL model not found: {smpl_model_path}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Video → mesh_data ----
    if not args.checkpoint_path and not args.checkpoint_hf:
        print("Error: provide --checkpoint_path or --checkpoint_hf for SAM 3D Body", file=sys.stderr)
        return 1
    # Mesh video path: inside output_dir so we can keep only it + IK when cleaning
    mesh_video_path = os.path.join(output_dir, f"{base_name}_output.mp4")
    demo_cmd = [
        sys.executable,
        "demo.py",
        "--video_path", video_path,
        "--output_folder", output_dir,
        "--output_video", mesh_video_path,
    ]
    if args.checkpoint_hf:
        demo_cmd.extend(["--checkpoint_hf", args.checkpoint_hf])
    else:
        demo_cmd.extend(["--checkpoint_path", args.checkpoint_path])
    if args.mesh_only:
        demo_cmd.append("--mesh_only")
    if not args.enable_fov:
        # Disable FOV estimator (MoGe2) by default, matching API behavior and avoiding 'moge' dependency
        demo_cmd.extend(["--fov_name", ""])
    if args.yolov8_weights:
        demo_cmd.extend(["--detector_name", "yolov8", "--yolov8_weights", args.yolov8_weights])
    # Use xvfb for headless (same behavior as API)
    use_xvfb = os.environ.get("SAM3D_USE_XVFB", "1") == "1"
    run_cmd = (
        ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"] + demo_cmd
        if use_xvfb
        else demo_cmd
    )
    print("[1/6] Running SAM 3D Body (video → mesh_data)...")
    _run(run_cmd, cwd=repo)
    if args.stop_after == "mesh":
        print(f"Stopping after mesh. Mesh data: {mesh_data_dir}")
        return 0
    if not os.path.isdir(mesh_data_dir):
        print(f"Error: mesh_data not found: {mesh_data_dir}", file=sys.stderr)
        return 1

    # ---- 2. mesh_data → MHR-SMPL (subprocess with LD_LIBRARY_PATH so pymomentum finds libtorch.so) ----
    mhr_conv_dir = os.path.join(repo, "MHRtoSMPL", "tools", "mhr_smpl_conversion")
    mhr_root = os.path.join(repo, "MHRtoSMPL")
    env_mhr = os.environ.copy()
    env_mhr["PYTHONPATH"] = mhr_root + (os.pathsep + env_mhr.get("PYTHONPATH", ""))
    # pymomentum-cpu loads libtorch.so; ensure the loader can find it
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
            capture_output=True,
            text=True,
            cwd=repo,
            env=env_mhr,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            torch_lib = r.stdout.strip()
            if os.path.isdir(torch_lib):
                env_mhr["LD_LIBRARY_PATH"] = torch_lib + (os.pathsep + env_mhr.get("LD_LIBRARY_PATH", ""))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    sam3d_cmd = [
        sys.executable,
        "sam3d_dir_to_smpl.py",
        "--sam3d-dir", mesh_data_dir,
        "--smpl-model", smpl_model_path,
        "--gender", args.gender,
        "--out-dir", mhr_smpl_dir,
        "--person-idx", str(args.person_idx),
        "--device", "cpu",
    ]
    print("[2/6] Running MHR→SMPL (mesh_data → smpl_fit_results.npz)...")
    _run(sam3d_cmd, cwd=mhr_conv_dir, env=env_mhr)
    smpl_fit_npz = os.path.join(mhr_smpl_dir, "smpl_fit_results.npz")
    if not os.path.isfile(smpl_fit_npz):
        print(f"Error: {smpl_fit_npz} not found", file=sys.stderr)
        return 1
    if args.stop_after == "mhr_smpl":
        print(f"Stopping after MHR-SMPL. Output: {smpl_fit_npz}")
        return 0

    # ---- 3. smpl_fit_results.npz → sequence .npz ----
    os.makedirs(smpl_seq_folder, exist_ok=True)
    seq_cmd = [
        sys.executable,
        "smpl_fit_to_sequence.py",
        "--input", smpl_fit_npz,
        "--output", seq_npz,
        "--fps", str(args.fps),
        "--gender", args.gender,
    ]
    print("[3/6] Converting to SMPL sequence (smpl2ab format)...")
    _run(seq_cmd, cwd=mhr_conv_dir, env=env_mhr)
    if args.stop_after == "sequence":
        print(f"Stopping after sequence. Sequence: {seq_npz}")
        return 0

    # ---- 4. SMPL sequence → AddBiomechanics input (TRC + _subject.json) ----
    smpl2ab_cmd = [
        sys.executable,
        os.path.join(repo, "smpl2ab", "smpl2addbio.py"),
        "-i", smpl_seq_folder,
        "-o", addbio_out,
        "--body_model", "smpl",
    ]
    subject_folder = os.path.join(addbio_out, os.path.basename(smpl_seq_folder))
    print("[4/6] Running smpl2addbio (SMPL sequence → TRC markers)...")
    _run(smpl2ab_cmd, cwd=repo)
    if not os.path.isdir(subject_folder):
        print(f"Error: smpl2addbio output not found: {subject_folder}", file=sys.stderr)
        return 1
    if args.stop_after == "addbio":
        print(f"Stopping after addbio. Output: {subject_folder}")
        return 0

    # ---- 5. Prepare engine input ----
    prep_cmd = [
        sys.executable,
        os.path.join(repo, "scripts", "prepare_engine_input.py"),
        "-i", subject_folder,
        "-o", engine_input,
    ]
    print("[5/6] Preparing AddBiomechanics engine input...")
    _run(prep_cmd, cwd=repo)

    # ---- 6. Run kinematics pass → IK .mot ----
    ik_cmd = [
        sys.executable,
        os.path.join(repo, "scripts", "run_until_ik.py"),
        "-i", engine_input,
        "--prepared",
    ]
    print("[6/6] Running IK (kinematics pass)...")
    _run(ik_cmd, cwd=repo)
    ik_src_dir = os.path.join(engine_input, "osim_results", "IK")

    # Keep only mesh video + IK: copy IK to output_dir/IK/, then remove intermediates
    if args.keep_only_final and not args.keep_intermediates:
        final_ik_dir = os.path.join(output_dir, "IK")
        os.makedirs(final_ik_dir, exist_ok=True)
        for f in os.listdir(ik_src_dir):
            if f.endswith(".mot"):
                shutil.copy2(os.path.join(ik_src_dir, f), os.path.join(final_ik_dir, f))
        for d in ["mesh_data", "mhr_smpl", "smpl_seq_trial", "OUT_addbio", "engine_input"]:
            path = os.path.join(output_dir, d)
            if os.path.isdir(path):
                shutil.rmtree(path)
        kept = [final_ik_dir + "/ (IK .mot)"]
        if os.path.isfile(mesh_video_path):
            kept.insert(0, mesh_video_path + " (mesh video)")
        print(f"Done. Kept only: {', '.join(kept)}")
    else:
        print(f"Done. Mesh video: {mesh_video_path}; IK .mot files: {ik_src_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
