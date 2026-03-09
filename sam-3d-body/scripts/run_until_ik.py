#!/usr/bin/env python3
"""
End-to-end helper: take SMPL2AddBiomechanics output and run through IK.

This script intentionally stops after the AddBiomechanics *kinematics* stage and
writes out IK .mot files, without attempting to generate the scaled OpenSim XML
via OpenSim ScaleTool (which requires `opensim-cmd` and can fail on some inputs).

Typical usage (one command):
  python scripts/run_until_ik.py -i OUT_addbio/OUT

If you already have a prepared engine input folder:
  python scripts/run_until_ik.py -i engine_input --prepared
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional, Tuple


def _paths_from_repo_root() -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    engine_dir = os.path.join(repo_root, "AddBiomechanics", "server", "engine")
    engine_src = os.path.join(engine_dir, "src")
    return repo_root, engine_dir, engine_src


def _maybe_reexec_into_engine_venv() -> None:
    """
    Use current Python if nimblephysics is available (single-venv mode).
    Otherwise re-exec into AddBiomechanics engine venv when it exists.
    Set SMPL2AB_NO_ENGINE_VENV=1 to always use current Python.
    """
    if os.environ.get("SMPL2AB_NO_ENGINE_VENV", "") != "":
        return
    try:
        import nimblephysics  # noqa: F401
        return  # Single venv: nimblephysics already available
    except ImportError:
        pass
    _, engine_dir, _ = _paths_from_repo_root()
    engine_venv_python = os.path.join(engine_dir, "venv", "bin", "python")
    if (
        os.path.exists(engine_venv_python)
        and os.path.abspath(sys.executable) != os.path.abspath(engine_venv_python)
    ):
        os.execv(engine_venv_python, [engine_venv_python, os.path.abspath(__file__), *sys.argv[1:]])


def _run_prepare_engine_input(
    smpl2ab_out: str, engine_input: str, osim_path: Optional[str]
) -> None:
    repo_root, _, _ = _paths_from_repo_root()
    prepare_script = os.path.join(repo_root, "scripts", "prepare_engine_input.py")
    cmd = [sys.executable, prepare_script, "-i", smpl2ab_out, "-o", engine_input]
    if osim_path:
        cmd.extend(["-m", osim_path])
    subprocess.run(cmd, check=True)


def _ensure_poses_shape_matches_timestamps(poses, n_timestamps: int):
    import numpy as np

    poses_np = np.asarray(poses)
    if poses_np.ndim == 1:
        poses_np = poses_np.reshape((-1, 1))

    # Expected by OpenSimParser.saveMot: (ndofs, T)
    if poses_np.ndim != 2:
        raise RuntimeError(f"Unexpected pose array shape: {poses_np.shape}")
    if poses_np.shape[1] == n_timestamps:
        return poses_np
    if poses_np.shape[0] == n_timestamps:
        return poses_np.T
    raise RuntimeError(
        f"Pose shape {poses_np.shape} does not match timestamps length {n_timestamps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare input and run AddBiomechanics kinematics pass through IK (.mot) only"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to SMPL2AddBiomechanics output folder (OUT_addbio/OUT) OR prepared engine_input (with --prepared)",
    )
    parser.add_argument(
        "--prepared",
        action="store_true",
        help="Treat --input as an already-prepared engine input folder (skip prepare step)",
    )
    parser.add_argument(
        "-e",
        "--engine-input",
        type=str,
        default="engine_input",
        help="Where to create the engine input folder (only used when not --prepared)",
    )
    parser.add_argument(
        "-m",
        "--osim",
        type=str,
        default=None,
        help="Optional OpenSim model (.osim) to use when preparing input",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        type=str,
        default="osim_results",
        help="Output folder name under the engine input (default: osim_results)",
    )
    args = parser.parse_args()

    repo_root, engine_dir, engine_src = _paths_from_repo_root()
    if not os.path.isdir(engine_src):
        raise FileNotFoundError(
            f"AddBiomechanics engine not found at {engine_src}. Clone it first:\n"
            f"  git clone https://github.com/keenon/AddBiomechanics"
        )

    if args.prepared:
        engine_input_path = os.path.abspath(args.input)
    else:
        smpl2ab_out = os.path.abspath(args.input)
        engine_input_path = os.path.abspath(args.engine_input)
        _run_prepare_engine_input(smpl2ab_out, engine_input_path, args.osim)

    # Run from engine dir so any relative paths inside AddBiomechanics resolve normally
    os.chdir(engine_dir)
    sys.path.insert(0, engine_src)

    import numpy as np
    import nimblephysics as nimble
    from kinematics_pass.subject import Subject
    from kinematics_pass.trial import ProcessingStatus

    data_folder_path = os.path.abspath(os.path.join(engine_dir, "..", "data"))

    subject = Subject()
    subject.load_folder(engine_input_path, data_folder_path)
    subject.segment_trials()
    subject.run_kinematics_pass(data_folder_path)

    ik_dir = os.path.join(engine_input_path, args.output_name, "IK")
    os.makedirs(ik_dir, exist_ok=True)

    written = 0
    for trial in subject.trials:
        for seg_idx, segment in enumerate(trial.segments):
            if getattr(segment, "kinematics_status", None) != ProcessingStatus.FINISHED:
                continue
            trial_segment_name = f"{trial.trial_name}_segment_{seg_idx}"
            mot_path = os.path.join(ik_dir, f"{trial_segment_name}_ik.mot")
            timestamps = list(getattr(segment, "timestamps", []))
            if len(timestamps) == 0:
                start_time = trial.timestamps[segment.start] if len(trial.timestamps) > segment.start else 0.0
                timestamps = [start_time + i * trial.timestep for i in range(segment.end - segment.start)]

            poses = _ensure_poses_shape_matches_timestamps(
                getattr(segment, "kinematics_poses", np.zeros((subject.skeleton.getNumDofs(), 0))),
                len(timestamps),
            )

            nimble.biomechanics.OpenSimParser.saveMot(
                subject.kinematics_skeleton, mot_path, timestamps, poses
            )
            written += 1

    if written == 0:
        raise RuntimeError(
            "No IK .mot files were written. This usually means kinematics fitting failed for all segments."
        )

    print(f"Wrote {written} IK .mot file(s) to: {ik_dir}")


if __name__ == "__main__":
    _maybe_reexec_into_engine_venv()
    main()

