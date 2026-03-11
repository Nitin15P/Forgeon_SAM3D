"""
MHR to SMPL Conversion Script

Converts MHR parameters from input directory to SMPL parameters in output directory.

Uses PyMomentum method with single_identity=True and is_tracking=True
"""

import os
import sys
import argparse
import shutil
import tempfile
from pathlib import Path
# MHR code and assets live next to this script (mhr/, tools/, assets/)
MHR_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MHR_ROOT))
sys.path.insert(0, str(MHR_ROOT / "tools" / "mhr_smpl_conversion"))

# Stub out pymomentum_fitting BEFORE any other imports to avoid
# pymomentum.solver + sklearn BLAS conflict (MHR->SMPL uses PyTorch only)
import types as _types, sys as _sys
_pmf = _types.ModuleType('pymomentum_fitting')
_pmf._NUM_RIG_PARAMETERS = 204
_pmf._NUM_BODY_BLENDSHAPES = 20
_pmf._NUM_HEAD_BLENDSHAPES = 20
_pmf._NUM_HAND_BLENDSHAPES = 5
class _DummyPMF:
    def __init__(self, *a, **kw):
        raise RuntimeError("PyMomentumModelFitting not available - use method='pytorch'")
_pmf.PyMomentumModelFitting = _DummyPMF
_sys.modules['pymomentum_fitting'] = _pmf
del _types, _pmf, _DummyPMF, _sys

# IMPORTANT: Import pymomentum BEFORE torch to ensure correct CUDA/library init
import pymomentum.geometry
import pymomentum.torch

# Now import torch and other modules
import numpy as np
import torch
from tqdm import tqdm
import smplx
from mhr.mhr import MHR
from conversion import Conversion


def load_mhr_vertices(mhr_params_dir: str):
    """Load all MHR vertices (and camera translations, if present) from the data files.

    Supports both:
    - Original MHR *_data.npz format with a 'data' object containing 'pred_vertices' / 'pred_cam_t'
    - SAM3D body outputs saved per-frame as frame_XXXX_personY.npz with top-level
      'pred_vertices' / 'pred_cam_t' arrays.
    """
    mhr_params_dir = Path(mhr_params_dir)

    # Find all data files (support both *_data.npz and generic frame_*.npz)
    data_files = sorted(mhr_params_dir.glob("*_data.npz"))
    if not data_files:
        data_files = sorted(mhr_params_dir.glob("*.npz"))
    print(f"Found {len(data_files)} frames in {mhr_params_dir}")

    if len(data_files) == 0:
        print(f"Error: No .npz MHR files found in {mhr_params_dir}")
        sys.exit(1)

    all_vertices = []
    all_cam_t = []
    for data_file in tqdm(data_files, desc="Loading MHR vertices"):
        data = np.load(data_file, allow_pickle=True)

        cam_t = None
        # Case 1: original format with 'data' object
        if "data" in data:
            root = data["data"]
            # root might be an array of objects or a pickled dict
            if isinstance(root, np.ndarray):
                obj = root[0]
            else:
                obj = root.item()
            vertices = obj["pred_vertices"]
            cam_t = obj.get("pred_cam_t", None)
        # Case 2: SAM3D-style outputs with top-level arrays
        elif "pred_vertices" in data:
            verts = data["pred_vertices"]
            # Expect shape (num_people, V, 3); take first person
            if verts.ndim == 3:
                vertices = verts[0]
            else:
                vertices = verts
            cam_t = data.get("pred_cam_t", None)
            if isinstance(cam_t, np.ndarray) and cam_t.ndim > 1:
                # If per-person cam_t is provided, take first person
                cam_t = cam_t[0]
        else:
            raise KeyError(
                f"File {data_file} does not contain 'data' or 'pred_vertices'; "
                "cannot extract MHR vertices."
            )

        all_vertices.append(vertices.astype(np.float32, copy=False))
        all_cam_t.append(cam_t)

    # Stack to (N, 18439, 3)
    all_vertices = np.stack(all_vertices, axis=0)
    print(f"Loaded vertices shape: {all_vertices.shape}")

    return all_vertices, all_cam_t


def main():
    # --- 新增：解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Convert MHR to SMPL parameters")
    parser.add_argument('--input', type=str, default="/root/TVB/test/mhr_params", 
                        help='Input directory containing MHR params')
    parser.add_argument('--output', type=str, default="/root/TVB/MHRtoSMPL/output", 
                        help='Output directory for SMPL results')
    parser.add_argument('--sequence', type=str, default="subject_smpl_sequence.npz",
                        help='Filename for single sequence NPZ in output dir (set empty to skip)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Mocap framerate for the sequence file')
    parser.add_argument('--gender', type=str, default='neutral', choices=('neutral', 'male', 'female'),
                        help='Gender for the sequence file')
    args = parser.parse_args()

    # 使用参数中的路径
    mhr_params_dir = args.input
    output_dir = args.output

    # Prefer local SMPL_NEUTRAL.npz (SMPL-style npz) if available; otherwise fall back to SMPL_NEUTRAL.pkl
    smpl_model_dir = Path(__file__).resolve().parent
    smpl_model_npz_path = smpl_model_dir / "SMPL_NEUTRAL.npz"
    smpl_model_pkl_path = smpl_model_dir / "SMPL_NEUTRAL.pkl"
    mhr_assets_dir = str(MHR_ROOT / "assets")

    print(f"Input Directory:  {mhr_params_dir}")
    print(f"Output Directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Device - PyMomentum only supports CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load MHR model
    print("Loading MHR model...")
    mhr_model = MHR.from_files(folder=Path(mhr_assets_dir), device=device, lod=1)

    # Load SMPL model
    print("Loading SMPL model...")
    if smpl_model_npz_path.exists():
        print(f"  Using SMPL model from npz: {smpl_model_npz_path}")
        smpl_data = np.load(smpl_model_npz_path, allow_pickle=True)

        class _SMPLData:
            pass

        data_struct = _SMPLData()
        for k in smpl_data.files:
            setattr(data_struct, k, smpl_data[k])

        smpl_model = smplx.SMPL(
            model_path=None,
            data_struct=data_struct,
            batch_size=1,
        ).to(device)
    else:
        print(f"  Using SMPL model from pkl: {smpl_model_pkl_path}")
        smpl_model = smplx.SMPL(
            model_path=str(smpl_model_pkl_path),
            batch_size=1,
        ).to(device)

    # Create converter with PyTorch method (PyMomentum only supports SMPL->MHR, not MHR->SMPL)
    print("Creating converter...")
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smpl_model,
        method="pytorch"  # MHR->SMPL only supports pytorch
    )

    # Load MHR vertices (and camera translations, if present)
    print("Loading MHR vertices...")
    mhr_vertices, cam_t_list = load_mhr_vertices(mhr_params_dir)

    # SAM3D outputs pred_vertices in METERS (local/centered), pred_cam_t also in METERS.
    # The MHR<->SMPL conversion library expects MHR vertices in CENTIMETERS (world space).
    # So we must: vertices_cm = (pred_vertices + pred_cam_t) * 100
    print("Converting vertices from meters to centimeters (SAM3D → MHR space)...")
    for i in range(len(mhr_vertices)):
        cam_t = cam_t_list[i]
        if cam_t is not None:
            mhr_vertices[i] = (mhr_vertices[i] + cam_t.astype(np.float32)) * 100.0
        else:
            mhr_vertices[i] = mhr_vertices[i] * 100.0

    # Convert to tensor
    mhr_vertices_tensor = torch.from_numpy(mhr_vertices).float()

    # Run conversion
    print("Starting conversion...")
    print(f"  - Method: PyTorch (GPU if available)")
    print(f"  - Frames: {len(mhr_vertices)}")
    print(f"  - single_identity: True")

    results = converter.convert_mhr2smpl(
        mhr_vertices=mhr_vertices_tensor,
        mhr_parameters=None,
        single_identity=True,
        return_smpl_meshes=False,
        return_smpl_parameters=True,
        return_smpl_vertices=True,
        return_fitting_errors=True,
    )

    # Get results
    smpl_params = results.result_parameters
    smpl_vertices = results.result_vertices
    errors = results.result_errors

    print(f"\nConversion completed!")
    print(f"Fitting errors - mean: {errors.mean():.4f}, max: {errors.max():.4f}")

    # Save per-frame data to a temp dir, then build single sequence and remove temp
    num_frames = len(mhr_vertices)
    temp_dir = tempfile.mkdtemp(prefix="smpl_perframe_", dir=output_dir)
    try:
        print(f"\nSaving per-frame data to temp dir...")
        for i in tqdm(range(num_frames), desc="Saving frames"):
            # Use MHR camera translation for global root motion when available.
            cam_t = cam_t_list[i] if i < len(cam_t_list) else None

            frame_data = {
                'betas': smpl_params['betas'][i].detach().cpu().numpy() if 'betas' in smpl_params else smpl_params['betas'].detach().cpu().numpy(),
                'body_pose': smpl_params['body_pose'][i].detach().cpu().numpy(),
                'global_orient': smpl_params['global_orient'][i].detach().cpu().numpy(),
            }

            if cam_t is not None:
                # MHR uses camera coords (x right, y down, z forward); flip Y to make it y-up.
                cam_t_smpl = cam_t.astype(np.float32, copy=False)
                cam_t_smpl[1] *= 1.0
                frame_data['transl'] = cam_t_smpl
            elif 'transl' in smpl_params:
                frame_data['transl'] = smpl_params['transl'][i].detach().cpu().numpy()

            # Add vertices if available
            if smpl_vertices is not None:
                frame_data['vertices'] = smpl_vertices[i]

            # Add fitting error
            frame_data['fitting_error'] = errors[i]

            frame_path = os.path.join(temp_dir, f"{i:08d}.npz")
            np.savez(frame_path, **frame_data)

        # Build and save a single sequence file (same format as subject_custom_90fps_male.npz)
        if args.sequence:
            betas_np = smpl_params['betas'][0].detach().cpu().numpy() if 'betas' in smpl_params else smpl_params['betas'].detach().cpu().numpy()
            betas_np = np.asarray(betas_np, dtype=np.float32).flatten()
            if betas_np.shape[0] != 10:
                betas_np = np.resize(betas_np, 10)

            go = smpl_params['global_orient'].detach().cpu().numpy()   # (N, 3) or (N, 1, 3)
            bp = smpl_params['body_pose'].detach().cpu().numpy()       # (N, 69) or (N, 23, 3)
            if go.ndim == 3:
                go = go.reshape(num_frames, -1)
            if bp.ndim == 3:
                bp = bp.reshape(num_frames, -1)
            poses = np.concatenate([go, bp], axis=1).astype(np.float32)   # (N, 72)

            trans_list = []
            for i in range(num_frames):
                cam_t = cam_t_list[i] if i < len(cam_t_list) else None
                if cam_t is not None:
                    t = np.asarray(cam_t, dtype=np.float32)
                elif 'transl' in smpl_params:
                    t = smpl_params['transl'][i].detach().cpu().numpy().astype(np.float32)
                else:
                    t = np.zeros(3, dtype=np.float32)
                trans_list.append(t)
            trans = np.stack(trans_list, axis=0)

            sequence_data = {
                'betas': betas_np,
                'poses': poses,
                'trans': trans,
                'gender': np.array(args.gender, dtype=f'<U{len(args.gender)}'),
                'mocap_framerate': np.array(args.fps, dtype=np.float64),
            }
            seq_path = os.path.join(output_dir, args.sequence)
            np.savez(seq_path, **sequence_data)
            print(f"Saved sequence file: {seq_path}  (poses {poses.shape}, trans {trans.shape})")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleared temp per-frame data.")

    print(f"\nDone! Output: {output_dir} (single sequence NPZ only)")


if __name__ == "__main__":
    main()
