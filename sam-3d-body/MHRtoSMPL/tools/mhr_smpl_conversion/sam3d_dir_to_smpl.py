import argparse
import json
from pathlib import Path

# IMPORTANT: pymomentum must be imported before torch (and before smplx, which imports torch)
# to avoid segfaults (see MHRtoSMPL/README.md)
try:
    import pymomentum.geometry  # noqa: F401
    import pymomentum.torch  # noqa: F401
except ImportError as e:
    raise RuntimeError(
        "pymomentum is required for MHR model conversion but is not installed.\n"
        "Install it in your environment, for example:\n"
        "  pip install pymomentum-cpu smplx\n"
    ) from e

import numpy as np
import smplx
import torch
import trimesh

from conversion import Conversion
from mhr.mhr import MHR


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a SAM3D mesh_data directory (.npz) to SMPL via MHR-to-SMPL conversion."
    )
    p.add_argument(
        "--sam3d-dir",
        type=str,
        required=True,
        help="Path to SAM3D output directory containing .npz files (e.g. sam_out_pravas/mesh_data).",
    )
    p.add_argument(
        "--smpl-model",
        type=str,
        required=True,
        help="Path to SMPL .pkl file (e.g. basicmodel_m_lbs_10_207_0_v1.1.0.pkl).",
    )
    p.add_argument(
        "--gender",
        type=str,
        default="male",
        choices=["male", "female", "neutral"],
        help="SMPL gender to load (default: male).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for results (meshes, parameters, errors).",
    )
    p.add_argument(
        "--person-idx",
        type=int,
        default=0,
        help="Only process files whose stored person_idx equals this value (default: 0).",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, only process first N npz files (sorted by name).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu or cuda:0 (default: cpu).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for conversion (default: 256).",
    )
    return p.parse_args()


def _safe_numpy_item(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _load_smpl_model(model_path: Path, gender: str, device: torch.device) -> smplx.SMPL:
    """
    Load SMPL from either:
    - official .npz (preferred; chumpy-free) -> converted to a pickle dict for smplx
    - .pkl (may require chumpy depending on source)
    """
    model_path = model_path.expanduser().resolve()

    if model_path.suffix.lower() == ".npz":
        # Convert official chumpy-free npz to a pickle dict that smplx can read.
        converted_pkl = model_path.with_name(model_path.stem + "_generated_from_npz.pkl")
        smpl_model_npz = np.load(model_path)
        smpl_model_data = dict(smpl_model_npz)

        # If posedirs is stored as (6890*3, 207), transpose to (207, 6890*3),
        # which is what smplx.lbs.lbs expects for the matmul.
        posedirs = smpl_model_data.get("posedirs")
        if posedirs is not None and posedirs.ndim == 2 and posedirs.shape == (6890 * 3, 207):
            smpl_model_data["posedirs"] = posedirs.T

        import pickle

        with open(converted_pkl, "wb") as f:
            pickle.dump(smpl_model_data, f)

        return smplx.SMPL(model_path=str(converted_pkl), gender=gender).to(device)

    try:
        return smplx.SMPL(model_path=str(model_path), gender=gender).to(device)
    except ModuleNotFoundError as e:
        if str(e) == "No module named 'chumpy'":
            raise RuntimeError(
                "Failed to load SMPL .pkl because it depends on 'chumpy'. "
                "Download the official SMPL .npz model file and pass that path to --smpl-model "
                "(the script will auto-convert it to a chumpy-free .pkl)."
            ) from e
        raise


def run_sam3d_dir_to_smpl(
    sam3d_dir: str | Path,
    out_dir: str | Path,
    smpl_model_path: str | Path,
    gender: str = "male",
    person_idx: int = 0,
    device: str = "cpu",
    batch_size: int = 256,
    max_files: int = 0,
) -> Path:
    """
    Convert a SAM3D mesh_data directory to smpl_fit_results.npz (single file).
    Callable from an integrated pipeline; returns path to smpl_fit_results.npz.
    """
    sam3d_dir = Path(sam3d_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    smpl_model = _load_smpl_model(Path(smpl_model_path), gender=gender, device=dev)
    mhr_model = MHR.from_files(lod=1, device=dev)
    converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch")

    npz_files = sorted(sam3d_dir.glob("*.npz"))
    if max_files > 0:
        npz_files = npz_files[:max_files]

    selected = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if "person_idx" in d.files:
            if int(_safe_numpy_item(d["person_idx"])) != person_idx:
                continue
        selected.append(f)

    if not selected:
        raise ValueError(f"No .npz files matched person_idx={person_idx} in {sam3d_dir}")

    sam3d_outputs = []
    meta = []
    for f in selected:
        d = np.load(f, allow_pickle=True)
        record = {k: _safe_numpy_item(d[k]) for k in d.files}
        sam3d_outputs.append(record)
        meta.append(
            {
                "file": f.name,
                "image_name": str(record.get("image_name", "")),
                "person_idx": int(record.get("person_idx", person_idx)) if "person_idx" in record else person_idx,
            }
        )
        if "pred_vertices" in record and "pred_cam_t" in record and "faces" in record:
            v = record["pred_vertices"] + record["pred_cam_t"][None, ...]
            in_mesh = trimesh.Trimesh(v, record["faces"], process=False)
            in_mesh.export(out_dir / f"{f.stem}_sam3d_mhr.ply")

    results = converter.convert_sam3d_output_to_smpl(
        sam3d_outputs=sam3d_outputs,
        return_smpl_meshes=True,
        return_smpl_parameters=True,
        return_smpl_vertices=False,
        return_fitting_errors=True,
        batch_size=batch_size,
    )

    meshes_dir = out_dir / "smpl_meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    for i, mesh in enumerate(results.result_meshes or []):
        mesh.export(meshes_dir / f"{Path(meta[i]['file']).stem}_smpl.ply")

    params_out = {}
    if results.result_parameters is not None:
        for k, v in results.result_parameters.items():
            params_out[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    if results.result_errors is not None:
        params_out["fitting_errors"] = results.result_errors

    out_npz = out_dir / "smpl_fit_results.npz"
    np.savez(out_npz, **params_out)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Processed {len(selected)} frames. Wrote {out_npz}")
    return out_npz


def main() -> None:
    args = _parse_args()
    run_sam3d_dir_to_smpl(
        sam3d_dir=args.sam3d_dir,
        out_dir=args.out_dir,
        smpl_model_path=args.smpl_model,
        gender=args.gender,
        person_idx=args.person_idx,
        device=args.device,
        batch_size=args.batch_size,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()

