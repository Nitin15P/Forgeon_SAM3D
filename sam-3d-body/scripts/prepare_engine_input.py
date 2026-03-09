#!/usr/bin/env python3
"""
Prepare SMPL2AddBiomechanics output for the AddBiomechanics engine.

Converts the flat structure (trials/*.trc) to the engine format
(trials/{name}/markers.trc) and copies the osim model as unscaled_generic.osim.

BSM Geometry uses .vtp.ply (PLY). Nimblephysics loads these; ScaleTool/OpenSim needs .vtp (VTK).
We provide both: .vtp.ply for loading, .vtp (converted via pyvista) for ScaleTool.

Usage:
  python scripts/prepare_engine_input.py -i OUT_addbio/OUT -o engine_input
  python scripts/prepare_engine_input.py -i OUT_addbio/OUT -o engine_input -m models/bsm/bsm.osim
"""

import argparse
import os
import shutil


def _convert_ply_to_vtp(ply_path: str, vtp_path: str) -> bool:
    """Convert PLY to VTK .vtp using pyvista. OpenSim requires ASCII format."""
    try:
        import pyvista as pv
        mesh = pv.read(ply_path)
        mesh.save(vtp_path, binary=False)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SMPL2AddBiomechanics output for AddBiomechanics engine"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to SMPL2AddBiomechanics output folder (contains _subject.json and trials/)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output folder for engine input (will be created)",
    )
    parser.add_argument(
        "-m", "--osim",
        type=str,
        default=None,
        help="Path to OpenSim model (.osim). Default: models/bsm/bsm.osim",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    # Resolve osim path
    if args.osim:
        osim_path = os.path.abspath(args.osim)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        osim_path = os.path.join(project_root, "models", "bsm", "bsm.osim")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input folder not found: {input_path}")
    if not os.path.exists(osim_path):
        raise FileNotFoundError(f"OpenSim model not found: {osim_path}")

    subject_json = os.path.join(input_path, "_subject.json")
    trials_dir = os.path.join(input_path, "trials")

    if not os.path.exists(subject_json):
        raise FileNotFoundError(f"_subject.json not found in {input_path}")
    if not os.path.exists(trials_dir):
        raise FileNotFoundError(f"trials/ not found in {input_path}")

    os.makedirs(output_path, exist_ok=True)
    trials_out = os.path.join(output_path, "trials")
    os.makedirs(trials_out, exist_ok=True)

    # Copy _subject.json
    shutil.copy(subject_json, os.path.join(output_path, "_subject.json"))
    print(f"Copied _subject.json")

    # Copy osim as unscaled_generic.osim
    shutil.copy(osim_path, os.path.join(output_path, "unscaled_generic.osim"))
    print(f"Copied {osim_path} -> unscaled_generic.osim")

    # Set up Geometry from the same folder as the osim (e.g. models/bsm/Geometry for BSM)
    osim_dir = os.path.dirname(osim_path)
    geometry_src = os.path.join(osim_dir, "Geometry")
    geometry_dst = os.path.join(output_path, "Geometry")

    if os.path.exists(geometry_dst):
        if os.path.islink(geometry_dst):
            os.unlink(geometry_dst)
        else:
            shutil.rmtree(geometry_dst)

    if os.path.isdir(geometry_src):
        os.makedirs(geometry_dst, exist_ok=True)
        for f in os.listdir(geometry_src):
            src = os.path.join(geometry_src, f)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(geometry_dst, f)
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
            # ScaleTool expects .vtp; convert .vtp.ply to .vtp for OpenSim
            if f.endswith(".vtp.ply"):
                vtp_name = f[:-4]
                vtp_dst = os.path.join(geometry_dst, vtp_name)
                if not os.path.exists(vtp_dst) and _convert_ply_to_vtp(src, vtp_dst):
                    pass  # converted
        print(f"Set up Geometry from {geometry_src}")
    else:
        print(f"Warning: Geometry folder not found at {geometry_src} - OpenSim may fail to load meshes")

    # Convert trials: each .trc -> trials/{name}/markers.trc
    trc_files = [f for f in os.listdir(trials_dir) if f.endswith(".trc")]
    if not trc_files:
        raise FileNotFoundError(f"No .trc files in {trials_dir}")

    for trc_file in trc_files:
        name = os.path.splitext(trc_file)[0]
        trial_dir = os.path.join(trials_out, name)
        os.makedirs(trial_dir, exist_ok=True)
        shutil.copy(
            os.path.join(trials_dir, trc_file),
            os.path.join(trial_dir, "markers.trc"),
        )
        print(f"  trials/{name}/markers.trc <- {trc_file}")

    print(f"\nEngine input ready at: {output_path}")
    print(f"Run: python AddBiomechanics/server/engine/src/engine.py {output_path}")


if __name__ == "__main__":
    main()
