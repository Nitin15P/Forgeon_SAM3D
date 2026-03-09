"""
Convert smpl_fit_results.npz (from sam3d_dir_to_smpl.py) to a single SMPL sequence .npz
in the format expected by smpl2addbio (poses, trans, betas, gender, mocap_framerate).
"""

import argparse
import numpy as np
from pathlib import Path


def run_smpl_fit_to_sequence(
    input_npz: str | Path,
    output_npz: str | Path,
    fps: float = 30.0,
    gender: str = "male",
) -> Path:
    """
    Convert smpl_fit_results.npz to subject_smpl_sequence.npz for smpl2addbio.
    Callable from an integrated pipeline; returns path to the written sequence .npz.
    """
    inp = Path(input_npz).expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input file not found: {inp}")

    data = np.load(inp, allow_pickle=True)
    global_orient = np.asarray(data["global_orient"], dtype=np.float32)
    body_pose = np.asarray(data["body_pose"], dtype=np.float32)
    transl = np.asarray(data["transl"], dtype=np.float32)
    betas = np.asarray(data["betas"], dtype=np.float32)

    if global_orient.ndim == 1:
        global_orient = global_orient.reshape(1, -1)
    if body_pose.ndim == 1:
        body_pose = body_pose.reshape(1, -1)
    if transl.ndim == 1:
        transl = transl.reshape(1, -1)

    n_frames = global_orient.shape[0]
    if body_pose.shape[0] != n_frames or transl.shape[0] != n_frames:
        raise ValueError(
            f"Frame count mismatch: global_orient {global_orient.shape[0]}, "
            f"body_pose {body_pose.shape[0]}, transl {transl.shape[0]}"
        )

    poses = np.concatenate(
        [global_orient.reshape(n_frames, -1), body_pose.reshape(n_frames, -1)],
        axis=1,
    )
    if poses.shape[1] != 72:
        raise ValueError(f"Expected poses (N, 72), got {poses.shape}")

    trans = transl.reshape(n_frames, 3)
    if betas.ndim == 1:
        betas = np.resize(betas.flatten(), 10).astype(np.float32)
    elif betas.shape[0] > 1:
        betas = np.asarray(betas[0], dtype=np.float32).flatten()[:10]
    else:
        betas = np.asarray(betas.flat[:10], dtype=np.float32)

    out = {
        "poses": poses,
        "trans": trans,
        "betas": betas,
        "gender": np.array(gender, dtype=f"<U{max(len(gender), 1)}"),
        "mocap_framerate": np.array(fps, dtype=np.float64),
    }
    out_path = Path(output_npz).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved SMPL sequence: {out_path} (frames={n_frames})")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert smpl_fit_results.npz to SMPL sequence for smpl2addbio."
    )
    p.add_argument("--input", type=str, required=True, help="Path to smpl_fit_results.npz.")
    p.add_argument("--output", type=str, default="subject_smpl_sequence.npz", help="Output sequence .npz path.")
    p.add_argument("--fps", type=float, default=30.0, help="Mocap framerate (default: 30).")
    p.add_argument("--gender", type=str, default="male", choices=("neutral", "male", "female"), help="Gender.")
    args = p.parse_args()
    run_smpl_fit_to_sequence(args.input, args.output, fps=args.fps, gender=args.gender)


if __name__ == "__main__":
    main()
