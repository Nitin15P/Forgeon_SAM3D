#!/usr/bin/env python3
"""
Run the AddBiomechanics engine on prepared input.

Requires AddBiomechanics to be cloned in this project (AddBiomechanics/).
Use scripts/prepare_engine_input.py first to create the input folder.

Usage:
  python scripts/run_addbio_engine.py -i engine_input
  python scripts/run_addbio_engine.py -i engine_input -o my_results
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run AddBiomechanics engine on prepared input"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to prepared engine input folder",
    )
    parser.add_argument(
        "-o", "--output-name",
        type=str,
        default="osim_results",
        help="Output name for results (default: osim_results)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    engine_dir = os.path.join(project_root, "AddBiomechanics", "server", "engine")
    engine_script = os.path.join(engine_dir, "src", "engine.py")

    if not os.path.exists(engine_script):
        print(
            "Error: AddBiomechanics engine not found. Clone it first:\n"
            "  git clone https://github.com/keenon/AddBiomechanics\n"
            f"Expected at: {engine_script}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prefer engine's venv python (has nimblephysics 0.10.52.1)
    engine_venv_python = os.path.join(engine_dir, "venv", "bin", "python")
    if os.path.exists(engine_venv_python):
        python_exe = engine_venv_python
        print("Using engine venv (AddBiomechanics/server/engine/venv)")
    else:
        python_exe = sys.executable
        print(
            "Warning: Engine venv not found. Run from AddBiomechanics/server/engine:\n"
            "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt",
            file=sys.stderr,
        )

    input_path = os.path.abspath(args.input)
    if not input_path.endswith("/"):
        input_path += "/"

    if not os.path.exists(input_path):
        print(f"Error: Input folder not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Run from engine directory so relative paths (../../data) resolve correctly
    cmd = [
        python_exe,
        "src/engine.py",
        input_path,
        args.output_name,
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"From: {engine_dir}\n")

    # Add OPENSIM_BIN_PATH to PATH if set (for venv users who install OpenSim via conda)
    env = os.environ.copy()
    opensim_bin = env.get("OPENSIM_BIN_PATH")
    if opensim_bin and os.path.isdir(opensim_bin):
        env["PATH"] = opensim_bin + os.pathsep + env.get("PATH", "")

    result = subprocess.run(cmd, cwd=engine_dir, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
