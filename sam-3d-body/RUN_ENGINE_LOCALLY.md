# Running the AddBiomechanics Engine Locally

This guide explains how to run the AddBiomechanics processing engine on your machine, so you can do scaling + IK without the web app or the broken `fitMarkersToWorldPositions` in nimblephysics.

**Quick summary (using project scripts):**
```bash
# 1. AddBiomechanics is already cloned in this project
cd AddBiomechanics/server/engine
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare input from SMPL2AddBiomechanics output
python scripts/prepare_engine_input.py -i OUT_addbio/OUT -o engine_input

# 3. Run the engine
python scripts/run_addbio_engine.py -i engine_input
```

**Manual summary:**
1. Clone AddBiomechanics, `cd server/engine`, create venv, `pip install -r requirements.txt`
2. Restructure your data: `subject/unscaled_generic.osim`, `subject/_subject.json`, `subject/trials/{trial_name}/markers.trc`
3. Run: `python src/engine.py /path/to/subject_folder`

## Prerequisites

- Python 3.8+
- Git

## Step 1: Clone AddBiomechanics

```bash
git clone https://github.com/keenon/AddBiomechanics
cd AddBiomechanics/server/engine
```

## Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies

The engine pins specific versions:

```bash
pip install -r requirements.txt
```

From `requirements.txt`:
```
numpy == 1.24.4
nimblephysics == 0.10.52.1
pandas
matplotlib
scipy
```

**Important:** The engine uses `nimblephysics == 0.10.52.1`. This version may differ from what you have installed elsewhere. The segfault we saw was with a different nimblephysics version—0.10.52.1 might work, or it might hit the same bug.

## Step 4: Prepare Input Data

The engine expects a folder with this structure (from AddBiomechanics `InterchangeFormats.md`):

```
{SUBJECT_FOLDER}/
├── unscaled_generic.osim    # OpenSim model (generic, unscaled)
├── _subject.json            # Subject metadata (mass, height, sex, etc.)
└── trials/
    ├── trial1/
    │   └── markers.trc      # Marker data (or markers.c3d)
    └── trial2/
        └── markers.trc
```

### Converting SMPL2AddBiomechanics Output

SMPL2AddBiomechanics produces:
```
OUT_addbio/OUT/
├── _subject.json
└── trials/
    └── smpl_fit_results.trc   # Single file per trial
```

You need to restructure so each trial is a **folder** with `markers.trc` inside:

```bash
# Example: convert SMPL2AddBiomechanics output for engine
ENGINE_INPUT=/path/to/engine_input
mkdir -p $ENGINE_INPUT/trials/smpl_fit_results
cp OUT_addbio/OUT/_subject.json $ENGINE_INPUT/
cp models/bsm/bsm.osim $ENGINE_INPUT/unscaled_generic.osim

# Each .trc file becomes a trial folder with markers.trc inside
for trc in OUT_addbio/OUT/trials/*.trc; do
  name=$(basename "$trc" .trc)
  mkdir -p "$ENGINE_INPUT/trials/$name"
  cp "$trc" "$ENGINE_INPUT/trials/$name/markers.trc"
done
```

**Note:** The BSM model (`bsm.osim`) is typically already scaled. AddBiomechanics usually expects a *generic* model like Rajagopal2015. If the engine fails or gives poor results, you may need to use a proper generic model. The engine's `data/` folder contains `PresetSkeletons/` (Rajagopal2015, LaiUhlrich2022, etc.)—use `skeletonPreset: "vicon"` or `"cmu"` in `_subject.json` to use those instead of a custom model.

## Step 5: Run the Engine

From `AddBiomechanics/server/engine/`:

```bash
python src/engine.py /path/to/SUBJECT_FOLDER
```

Optional arguments:
- `sys.argv[2]`: output name (default: `osim_results`)
- `sys.argv[3]`: href for web integration (default: empty)

Example with custom output name:
```bash
python src/engine.py /path/to/subject_folder my_results
```

## Step 6: Get Test Data (Optional)

The repo's test data lives on AWS S3. To fetch it (requires AWS credentials and access):

```bash
./get_test_data.sh
```

Or use the `test_data` folder from the [Google Drive link](https://drive.google.com/drive/folders/1jGfgM1m13ksqLZByKUEoUwsy22OVtEza) mentioned in the main README (ask Keenon for access).

## Troubleshooting

### Nimblephysics Segfault

If you still get a segfault with `nimblephysics==0.10.52.1`, the engine uses `MarkerFitter` (a different code path than `fitMarkersToWorldPositions`). It may or may not hit the same bug. If it does:

- Try a different nimblephysics version
- Report the issue to the [nimblephysics repo](https://github.com/keenon/nimblephysics)
- Fall back to the AddBiomechanics web app for processing

### Missing Geometry/Data

The engine references `../../data` (resolves to `AddBiomechanics/server/data/`) which must contain:
- `Geometry/` — mesh files for visualization
- `PresetSkeletons/` — optional preset models (Rajagopal2015, etc.)

You must run from within the full AddBiomechanics clone; the engine cannot be run in isolation. The `load_model_files` step will symlink `Geometry` into your subject folder if missing.

### Engine Entry Point

The engine is designed to be called by the AddBiomechanics server, not as a standalone CLI. You may need to:

1. Find the server code that invokes `engine.py`
2. Replicate that invocation (including `output_name`, `href`, etc.)
3. Or adapt `engine.py` to accept a simple path argument

Check `server/app/` for how the server triggers processing.
