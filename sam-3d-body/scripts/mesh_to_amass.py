#!/usr/bin/env python3
"""
Convert SAM 3D Body mesh_data (MHR outputs) to AMASS-style SMPL sequences
for use with SMPL2AddBiomechanics / OpenSim.

Input:  mesh_data/ folder with per-frame .npz files (frame_XXXX_personN.npz or {name}_personN.npz)
Output: subject folder with AMASS-format .npz sequences (poses, trans, betas, gender, mocap_framerate)

Usage:
    python scripts/mesh_to_amass.py -i /path/to/mesh_data -o /path/to/pravas_walk
    python scripts/mesh_to_amass.py -i /path/to/sam_out/mesh_data -o ./output_pravas --fps 30
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np


# MHR70 keypoint indices -> SMPL 24 joint indices
# SMPL 24 order (smplx): pelvis, L_hip, R_hip, spine1, L_knee, R_knee, spine2, L_ankle, R_ankle,
#   spine3, L_foot, R_foot, neck, L_collar, R_collar, head, L_shoulder, R_shoulder,
#   L_elbow, R_elbow, L_wrist, R_wrist, L_hand, R_hand
# MHR70: 0=nose, 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow, 9=L_hip, 10=R_hip,
#   11=L_knee, 12=R_knee, 13=L_ankle, 14=R_ankle, 15-20=feet, 41=R_wrist, 62=L_wrist, 69=neck
MHR70_TO_SMPL24 = {
    0: 15,   # nose -> head
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 1,    # left_hip
    10: 2,   # right_hip
    11: 4,   # left_knee
    12: 5,   # right_knee
    13: 7,   # left_ankle
    14: 8,   # right_ankle
    17: 10,  # left_heel -> left_foot
    20: 11,  # right_heel -> right_foot
    41: 21,  # right_wrist
    62: 20,  # left_wrist
    69: 12,  # neck
}
# Pelvis = midpoint of hips; spine/collar approximated
PELVIS_MHR_IDXS = [9, 10]  # left_hip, right_hip
SPINE1_MHR_IDXS = [9, 10, 69]  # hips + neck
L_COLLAR_MHR_IDXS = [5, 69]   # L_shoulder, neck
R_COLLAR_MHR_IDXS = [6, 69]
SPINE2_MHR_IDXS = [9, 10, 69]
SPINE3_MHR_IDXS = [9, 10, 69]
# Hands: use wrist as proxy for hand
L_HAND_MHR_IDX = 62
R_HAND_MHR_IDX = 41


def build_smpl24_from_mhr70(kpts: np.ndarray) -> np.ndarray:
    """Build (24, 3) SMPL joint positions from (70, 3) MHR keypoints."""
    out = np.full((24, 3), np.nan, dtype=np.float32)
    for mhr_idx, smpl_idx in MHR70_TO_SMPL24.items():
        out[smpl_idx] = kpts[mhr_idx]
    # Pelvis
    out[0] = np.nanmean(kpts[PELVIS_MHR_IDXS], axis=0)
    # Spine1, 2, 3 - interpolate
    out[3] = np.nanmean(kpts[[9, 10, 69]], axis=0) * 0.5 + out[0] * 0.5
    out[6] = np.nanmean(kpts[[9, 10, 69]], axis=0) * 0.7 + out[0] * 0.3
    out[9] = np.nanmean(kpts[[9, 10, 69]], axis=0) * 0.85 + out[0] * 0.15
    # Collars
    out[13] = (kpts[5] + kpts[69]) / 2 if not np.any(np.isnan(kpts[5])) else out[16]
    out[14] = (kpts[6] + kpts[69]) / 2 if not np.any(np.isnan(kpts[6])) else out[17]
    # Hands (use wrist + small offset along forearm)
    out[22] = kpts[62] if not np.any(np.isnan(kpts[62])) else out[20]
    out[23] = kpts[41] if not np.any(np.isnan(kpts[41])) else out[21]
    # Fill any remaining NaN with pelvis
    out[np.any(np.isnan(out), axis=1)] = out[0]
    return out


def fit_smpl_to_joints_simple(
    target_joints: np.ndarray,
    smpl_model_path: str,
    gender: str = "neutral",
    num_betas: int = 10,
    num_iters: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit SMPL to target (24, 3) joints. Returns (poses 72, betas 10, trans 3).
    Uses smplx + scipy optimization. poses = global_orient (3) + body_pose (69).
    """
    try:
        import torch
        import smplx
    except ImportError as e:
        raise ImportError(
            "mesh_to_amass requires: pip install smplx torch\n"
            "Also download SMPL models from https://smpl-body.is.tue.mpg.de/ and set --smpl_model_path"
        ) from e

    from scipy.optimize import minimize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.tensor(target_joints, dtype=torch.float32, device=device).unsqueeze(0)

    body_model = smplx.create(
        smpl_model_path,
        model_type="smpl",
        gender=gender,
        num_betas=num_betas,
        batch_size=1,
    ).to(device)

    def loss_fn(params):
        global_orient = params[:3].reshape(1, 3)
        body_pose = params[3:72].reshape(1, 69)
        betas = params[72 : 72 + num_betas].reshape(1, num_betas)
        trans = params[72 + num_betas :].reshape(1, 3)
        go_t = torch.tensor(global_orient, dtype=torch.float32, device=device)
        bp_t = torch.tensor(body_pose, dtype=torch.float32, device=device)
        be_t = torch.tensor(betas, dtype=torch.float32, device=device)
        tr_t = torch.tensor(trans, dtype=torch.float32, device=device)
        output = body_model(
            global_orient=go_t,
            body_pose=bp_t,
            betas=be_t,
            transl=tr_t,
        )
        pred_joints = output.joints[0, :24]
        diff = (pred_joints - target[0, :24]).norm(dim=-1).mean()
        return float(diff)

    # Initialize: zero pose (axis-angle), zero betas, trans = pelvis of target
    x0 = np.zeros(72 + num_betas + 3, dtype=np.float32)
    x0[72 + num_betas :] = target_joints[0]  # pelvis as initial trans

    res = minimize(loss_fn, x0, method="L-BFGS-B", options={"maxiter": num_iters})
    pose = np.concatenate([res.x[:3], res.x[3:72]], axis=0).astype(np.float32)
    betas = res.x[72 : 72 + num_betas].astype(np.float32)
    trans = res.x[72 + num_betas :].astype(np.float32)
    return pose, betas, trans


def fit_smpl_batch_smplfitter(
    joints_batch: np.ndarray,
    smpl_model_path: str,
    gender: str = "neutral",
    num_betas: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Fit SMPL to batch of joints using smplfitter (faster). Returns None if not available.
    joints_batch: (T, 24, 3)
    Returns: poses (T,72), betas (10,), trans (T,3) or None
    """
    try:
        import torch
        from smplfitter.pt import BodyModel, BodyFitter
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # smplfitter expects DATA_ROOT/body_models/smpl/...
    smpl_dir = Path(smpl_model_path)
    data_root = smpl_dir.parent.parent if smpl_dir.name == "smpl" else smpl_dir.parent
    os.environ.setdefault("DATA_ROOT", str(data_root))

    body_model = BodyModel("smpl", gender, num_betas=num_betas).to(device)
    fitter = BodyFitter(body_model).to(device)
    vertices_dummy = torch.zeros(joints_batch.shape[0], 6890, 3, device=device)
    joints_t = torch.tensor(joints_batch, dtype=torch.float32, device=device)
    fit_res = fitter.fit(vertices_dummy, joints_t, num_iter=3, beta_regularizer=1)
    # smplfitter returns body_pose (69) + global_orient (3) = 72
    body_pose = fit_res["body_pose"].cpu().numpy()
    global_orient = fit_res.get("global_orient", np.zeros((joints_batch.shape[0], 3)))
    if global_orient.ndim == 1:
        global_orient = global_orient.reshape(1, 3).repeat(joints_batch.shape[0], axis=0)
    poses = np.concatenate([global_orient, body_pose], axis=-1)
    betas = fit_res["betas"].cpu().numpy()
    if betas.ndim == 2:
        betas = betas.mean(axis=0)
    trans = fit_res["transl"].cpu().numpy()
    return poses, betas, trans


def load_mesh_sequence(mesh_data_dir: Path, person_id: int = 0) -> tuple[list[np.ndarray], list[dict], float]:
    """
    Load per-frame npz files for one person, sorted by frame order.
    Returns (list of pred_keypoints_3d, list of full frame data, fps_estimate).
    """
    pattern = re.compile(r"frame_(\d+)_person(\d+)|(.+)_person(\d+)")
    files = []
    for f in mesh_data_dir.glob("*.npz"):
        m = pattern.match(f.stem)
        if m is None:
            continue
        if m.group(1) is not None:  # frame_XXXX_personN
            frame_num = int(m.group(1))
            pid = int(m.group(2))
            key = (frame_num, pid)
        else:  # name_personN
            name = m.group(3)
            pid = int(m.group(4))
            key = (name, pid)
        if pid != person_id:
            continue
        files.append((key, f))

    files.sort(key=lambda x: x[0])
    if not files:
        return [], [], 30.0

    keypoints_list = []
    frame_data_list = []
    for _, fp in files:
        data = np.load(fp, allow_pickle=True)
        kpts = data["pred_keypoints_3d"]
        if isinstance(kpts, np.ndarray):
            keypoints_list.append(np.asarray(kpts, dtype=np.float32))
        else:
            keypoints_list.append(np.asarray(kpts.tolist(), dtype=np.float32))
        frame_data_list.append(dict(data))

    # FPS: assume 30 if from video; could parse from metadata if available
    fps = 30.0
    return keypoints_list, frame_data_list, fps


def convert_sequence_to_amass(
    keypoints_list: list[np.ndarray],
    frame_data_list: list[dict],
    fps: float,
    smpl_model_path: str,
    gender: str = "neutral",
    use_smplfitter: bool = True,
) -> dict:
    """Convert one sequence to AMASS-style dict."""
    T = len(keypoints_list)
    if T == 0:
        raise ValueError("Empty sequence")

    # Build SMPL 24 joints per frame
    joints_24 = np.stack([build_smpl24_from_mhr70(k) for k in keypoints_list], axis=0)

    # Fit SMPL
    if use_smplfitter:
        result = fit_smpl_batch_smplfitter(joints_24, smpl_model_path, gender=gender)
    else:
        result = None
    if result is None:
        poses_list = []
        betas_list = []
        trans_list = []
        for t in range(T):
            p, b, tr = fit_smpl_to_joints_simple(
                joints_24[t], smpl_model_path, gender=gender
            )
            poses_list.append(p)
            betas_list.append(b)
            trans_list.append(tr)
        poses = np.stack(poses_list, axis=0)
        betas = np.stack(betas_list, axis=0).mean(axis=0)
        trans = np.stack(trans_list, axis=0)
    else:
        poses, betas, trans = result

    # Ensure poses (T, 72), trans (T, 3), betas (10,)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    if trans.ndim == 1:
        trans = trans.reshape(1, -1)
    if betas.ndim == 2:
        betas = betas.mean(axis=0)
    if betas.size > 10:
        betas = betas[:10]

    return {
        "poses": poses.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": gender,
        "mocap_framerate": float(fps),
    }


def discover_sequences(mesh_data_dir: Path) -> dict[tuple[str, int], list[Path]]:
    """Group npz files by (subject_name, person_id)."""
    pattern = re.compile(r"frame_(\d+)_person(\d+)|(.+)_person(\d+)")
    groups = {}
    for f in mesh_data_dir.glob("*.npz"):
        m = pattern.match(f.stem)
        if m is None:
            continue
        if m.group(1) is not None:
            subj = "sequence"
            pid = int(m.group(2))
        else:
            subj = m.group(3)
            pid = int(m.group(4))
        key = (subj, pid)
        groups.setdefault(key, []).append(f)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda p: p.stem)
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM 3D Body mesh_data to AMASS-style SMPL for SMPL2AddBiomechanics"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to mesh_data folder")
    parser.add_argument("-o", "--output", required=True, help="Output folder (e.g. pravas_walk/)")
    parser.add_argument(
        "--smpl_model_path",
        required=True,
        help="Path to SMPL model folder or .pkl file. Folder should contain SMPL_NEUTRAL.pkl (or SMPL_MALE.pkl, SMPL_FEMALE.pkl). Download from https://smpl-body.is.tue.mpg.de/",
    )
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (if not inferrable)")
    parser.add_argument("--person_id", type=int, default=0, help="Person index to convert")
    parser.add_argument("--no_smplfitter", action="store_true", help="Use scipy fitting instead of smplfitter")
    parser.add_argument("--trial_prefix", default="trial", help="Prefix for output files (trial_01.npz, ...)")
    args = parser.parse_args()

    mesh_dir = Path(args.input)
    out_dir = Path(args.output)
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {mesh_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    keypoints_list, frame_data_list, fps = load_mesh_sequence(mesh_dir, person_id=args.person_id)
    if not keypoints_list:
        raise SystemExit(f"No mesh data found for person {args.person_id} in {mesh_dir}")

    fps = args.fps if args.fps else fps
    amass_data = convert_sequence_to_amass(
        keypoints_list,
        frame_data_list,
        fps=fps,
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        use_smplfitter=not args.no_smplfitter,
    )

    out_file = out_dir / f"{args.trial_prefix}_01_poses.npz"
    np.savez(
        out_file,
        poses=amass_data["poses"],
        trans=amass_data["trans"],
        betas=amass_data["betas"],
        gender=amass_data["gender"],
        mocap_framerate=amass_data["mocap_framerate"],
    )
    print(f"Saved AMASS-style sequence: {out_file}")
    print(f"  poses: {amass_data['poses'].shape}, trans: {amass_data['trans'].shape}, betas: {amass_data['betas'].shape}")


if __name__ == "__main__":
    main()
