# Scripts

## Checkpoints and assets (not in repo)

The pipeline needs two kinds of assets; neither are “checkpoints” in the usual sense, but both are required:

### 1. SAM 3D Body model (step 1: video → mesh)

- **Not in repo:** `backend/app/sam-3d-body/checkpoints/` is gitignored. You must provide the model one of two ways:
  - **Recommended:** Use HuggingFace so nothing is stored locally:
    ```bash
    --checkpoint_hf facebook/sam-3d-body-dinov3
    ```
    This downloads the checkpoint and the MHR head asset (`mhr_model.pt`) automatically.
  - **Local:** Download the [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) checkpoint (and optional MHR head asset), then:
    ```bash
    --checkpoint_path /path/to/checkpoints/model.ckpt
    # If the MHR head is separate (e.g. from the same release):
    --mhr_path /path/to/assets
    # or set env: SAM3D_MHR_PATH=/path/to/assets
    ```

### 2. MHR body model for MHR→SMPL (step 2)

- **Location:** `MHRtoSMPL/assets/` (repo root for MHRtoSMPL = `backend/app/sam-3d-body/MHRtoSMPL/assets/`).
- **Required files:** `lod1.fbx`, `compact_v6_1.model`, `corrective_blendshapes_lod1.npz`, `corrective_activation.npz`.
- These come from the **MHR release** (e.g. Meta’s MHR “assets.zip”). If they are missing in your clone (e.g. not committed or on another branch), you must obtain that release and place those files in `MHRtoSMPL/assets/`.

You also need a **SMPL model** (e.g. `SMPL_male.pkl` or official `.npz`) for MHR→SMPL; that path is passed as `--smpl-model`.

---

## Single video → full pipeline (mesh → MHR-SMPL → SMPL sequence → IK)

One entry-point runs the entire pipeline from a single video:

```bash
# Full pipeline (video → SAM 3D mesh → MHR-SMPL → SMPL sequence → AddBio markers → IK .mot)
python scripts/run_full_pipeline.py --video_path /path/to/video.mp4 --smpl-model /path/to/SMPL_male.pkl --checkpoint_hf facebook/sam-3d-body-dinov3
```

**Stages:**  
1. **SAM 3D Body** (`demo.py`): video → `output/<name>/mesh_data/` (per-frame .npz).  
2. **MHR→SMPL** (`sam3d_dir_to_smpl.py`): mesh_data → `smpl_fit_results.npz` (single file). Uses PyTorch backend by default (no `pymomentum-cpu` required); runs on VM.  
3. **SMPL sequence** (`smpl_fit_to_sequence.py`): `smpl_fit_results.npz` → sequence .npz (poses, trans, betas, gender, mocap_framerate) for smpl2ab.  
4. **smpl2addbio**: SMPL sequence → synthetic markers (TRC) + `_subject.json`.  
5. **prepare_engine_input**: Layout for AddBiomechanics engine.  
6. **run_until_ik**: Kinematics pass only → IK `.mot` files (no scaling/OpenSim CLI needed).

**Options:**

- `--stop-after mesh|mhr_smpl|sequence|addbio|ik` — stop after that stage (e.g. `--stop-after mhr_smpl` to run only up to MHR-SMPL on the VM).  
- `--mesh-only` — step 1 saves only mesh data (no output video).  
- `--output_dir <dir>` — base output directory (default: `./output/<video_basename>`).  
- `--gender male|female|neutral`, `--fps 30`, `--person-idx 0`.

**SMPL sequence → IK:** The pipeline uses `run_until_ik.py`, which runs only the AddBiomechanics **kinematics** pass (marker fitting → IK .mot). It does not run scaling or the full engine; it uses the engine’s Python API and the BSM/osim model from `models/bsm/`. The engine venv (Python 3.10/3.11) is used automatically when available.

---

## AddBiomechanics Engine (Local IK)

Run the AddBiomechanics processing engine locally for scaling + inverse kinematics.

### Setup (one-time)

**Quick setup (all-in-one):**
```bash
bash scripts/setup_ik_pipeline.sh
```

**Manual setup:**

1. **Clone AddBiomechanics** (if not already present):
   ```bash
   git clone https://github.com/keenon/AddBiomechanics
   ```

2. **Create engine virtual environment** (requires Python 3.10 or 3.11; 3.12 is incompatible):
   ```bash
   cd AddBiomechanics/server/engine
   python3.11 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cd ../..
   ```

3. **Install OpenSim** (for `opensim-cmd` used in scaling). PyOpenSim does not include the CLI; you need the full OpenSim distribution:

   **Option A – Minimal conda (venv + conda hybrid):** Use conda only for `opensim-cmd`, keep venv for the rest:
   ```bash
   conda create -n opensim_cmd -c opensim-org opensim
   # Point to the conda env's bin (adjust path if your conda is elsewhere):
   export OPENSIM_BIN_PATH="$HOME/miniconda3/envs/opensim_cmd/bin"
   python scripts/run_addbio_engine.py -i engine_input
   ```
   The run script reads `OPENSIM_BIN_PATH` and adds it to PATH for the engine subprocess.

   **Option B – OpenSim GUI:** Install the [OpenSim GUI](https://github.com/opensim-org/opensim-gui/releases) for your platform; it includes `opensim-cmd`. Add its install directory to PATH.

### Usage

1. **Generate SMPL2AddBiomechanics output** (if you haven't already):
   ```bash
   python smpl2ab/smpl2addbio.py -i OUT -o OUT_addbio
   ```

2. **Prepare engine input**:
   ```bash
   python scripts/prepare_engine_input.py -i OUT_addbio/OUT -o engine_input
   ```
   Optionally specify a custom OpenSim model: `-m models/bsm/bsm.osim`

3. **Run the engine** (activate the engine venv first):
   ```bash
   source AddBiomechanics/server/engine/venv/bin/activate
   python scripts/run_addbio_engine.py -i engine_input
   ```

Results will be written to the input folder (e.g. `engine_input/osim_results_*.mot`, scaled model, etc.).
