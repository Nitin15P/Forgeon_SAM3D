"""
Convert per-frame SMPL result NPZ files (smpl_results/00000000.npz, ...) into a single
sequence-style NPZ file matching subject_custom_90fps_male.npz format.

Output keys: betas, poses (N, 72), trans (N, 3), gender, mocap_framerate.
Optional: vertices (N, 6890, 3), fitting_error (N,) when --include-extra.
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert smpl_results (per-frame NPZ) to a single sequence NPZ file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="smpl_results",
        help="Directory containing 00000000.npz, 00000001.npz, ...",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="subject_smpl_sequence.npz",
        help="Output path for the sequence NPZ file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Mocap framerate to store (default: 30).",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=("neutral", "male", "female"),
        help="Gender for the sequence (default: neutral).",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also include vertices and fitting_error in the output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    # Collect and sort frame files (8-digit zero-padded)
    npz_files = sorted(input_dir.glob("*.npz"), key=lambda p: p.stem)
    if not npz_files:
        raise SystemExit(f"No .npz files in {input_dir}")

    # Load first frame to get shapes and betas
    first = np.load(npz_files[0], allow_pickle=True)
    betas = np.asarray(first["betas"], dtype=np.float32).flatten()
    if betas.shape[0] != 10:
        betas = np.resize(betas, 10)

    list_poses = []
    list_trans = []
    list_vertices = [] if args.include_extra else None
    list_errors = [] if args.include_extra else None

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        go = np.asarray(data["global_orient"], dtype=np.float32).flatten()  # (3,)
        bp = np.asarray(data["body_pose"], dtype=np.float32).flatten()     # (69,)
        poses_row = np.concatenate([go, bp], axis=0)  # (72,)
        list_poses.append(poses_row)
        list_trans.append(np.asarray(data["transl"], dtype=np.float32).flatten())
        if args.include_extra:
            list_vertices.append(np.asarray(data["vertices"], dtype=np.float32))
            list_errors.append(float(np.asarray(data["fitting_error"]).ravel()[0]))

    poses = np.stack(list_poses, axis=0)   # (N, 72)
    trans = np.stack(list_trans, axis=0)   # (N, 3)

    out = {
        "betas": betas,
        "poses": poses,
        "trans": trans,
        "gender": np.array(args.gender, dtype=f"<U{len(args.gender)}"),
        "mocap_framerate": np.array(args.fps, dtype=np.float64),
    }
    if args.include_extra:
        out["vertices"] = np.stack(list_vertices, axis=0)
        out["fitting_error"] = np.array(list_errors, dtype=np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)
    print(f"Saved sequence: {output_path}")
    print(f"  Frames: {poses.shape[0]}, poses {poses.shape}, trans {trans.shape}, betas {betas.shape}")


if __name__ == "__main__":
    main()
