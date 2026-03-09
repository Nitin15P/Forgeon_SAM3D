#!/bin/bash
# Setup the full IK pipeline: engine venv + OpenSim (conda)
# Run from project root: bash scripts/setup_ik_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "=== 1. Clone AddBiomechanics (if needed) ==="
if [ ! -d AddBiomechanics ]; then
    git clone --depth 1 https://github.com/keenon/AddBiomechanics
fi

echo ""
echo "=== 2. Engine venv (Python 3.11) ==="
ENGINE_DIR=AddBiomechanics/server/engine
PY311="${PY311:-}"
if [ -z "$PY311" ]; then
    PY311=$(command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || true)
fi
if [ -z "$PY311" ]; then
    PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    for v in 3.11.14 3.11.13 3.11 3.10.14 3.10; do
        p="$PYENV_ROOT/versions/$v/bin/python"
        if [ -x "$p" ]; then
            PY311="$p"
            break
        fi
    done
fi
if [ -z "$PY311" ]; then
    echo "Python 3.10 or 3.11 not found. Install with: pyenv install 3.11.14"
    exit 1
fi
cd "$ENGINE_DIR"
rm -rf venv
"$PY311" -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ../..

echo ""
echo "=== 3. OpenSim (conda) ==="
if ! command -v conda &>/dev/null; then
    echo "Conda not found. Install Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
if ! conda env list | grep -q opensim_cmd; then
    conda create -n opensim_cmd -c opensim-org opensim -y
else
    echo "opensim_cmd env already exists"
fi

echo ""
echo "=== Done ==="
echo "To run the pipeline:"
echo "  1. python scripts/prepare_engine_input.py -i OUT_addbio/OUT -o engine_input"
echo "  2. export OPENSIM_BIN_PATH=\$(conda run -n opensim_cmd which opensim-cmd | xargs dirname)"
echo "  3. python scripts/run_addbio_engine.py -i engine_input"
echo ""
echo "Or with conda run (no PATH needed):"
echo "  The engine will auto-detect opensim-cmd via 'conda run -n opensim_cmd which opensim-cmd'"
