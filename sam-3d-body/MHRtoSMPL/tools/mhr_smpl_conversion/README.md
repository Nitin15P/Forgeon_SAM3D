# SMPL-MHR Conversion Tool

A Python toolkit for converting between SMPL/SMPLX and MHR body model representations. This tool provides bidirectional conversion capabilities with multiple optimization backends and identity handling options.

## Overview

The SMPL-MHR Conversion Tool enables seamless conversion between different 3D human body model formats:

- **SMPL/SMPLX → MHR**: Convert SMPL/SMPLX model parameters/meshes to MHR format
- **MHR → SMPL/SMPLX**: Convert MHR parameters back to SMPL/SMPLX format
- **SAM3D Outputs (MHR) → SMPL/SMPLX**: Convert SAM3D output to SMPL/SMPLX model parameters

The tool uses barycentric interpolation for topology mapping and offers multiple optimization backends for parameter fitting.

## Features

### 🔄 **Bidirectional Conversion**
- SMPL/SMPLX parameters or vertices → MHR parameters
- MHR parameters or vertices → SMPL/SMPLX parameters

### ⚙️ **Multiple Optimization Methods**
- **PyMomentum**: CPU-based Gauss-Newton hierarchical optimization with robust fitting
- **PyTorch**: GPU-accelerated optimization with edge and vertex loss

### 🎭 **Identity Handling**
- **Single Identity**: Use consistent shape parameters across all frames
- **Multiple Identities**: Unique shape parameters for each frame

### 📊 **Output Options**
- Return meshes in trimesh.Trimesh format
- Return parameter dictionaries
- Return vertex as numpy arrays
- Return fitting errors

## Installation

### Prerequisites

```bash
# On top of MHR
pixi add --pypi trimesh scikit-learn tqdm smplx
```

### SMPL/SMPLX Model Files

You'll need the official SMPL/SMPLX model files:

1. **SMPL**: Download from [SMPL website](https://smpl.is.tue.mpg.de/)
2. **SMPLX**: Download from [SMPLX website](https://smpl-x.is.tue.mpg.de/)

**Note**: If you run into issues with the `.pkl` SMPL(X) model file, try the official `.npz` files instead.

### Model files used by this pipeline

- **MHR**: `assets/` at repo root (from MHR release `assets.zip`: e.g. `lod1.fbx`).
- **Conversion assets**: `tools/mhr_smpl_conversion/assets/` (e.g. `mhr2smpl_mapping.npz`, `mhr2smplx_mapping.npz`, etc.).
- **SMPL** (for `sam3d_dir_to_smpl.py`): one official SMPL `.npz` you provide (e.g. `SMPL_NEUTRAL.npz`).
- **SMPL-X** (for `example.py` SAM3D→SMPL-X): path to a folder containing `SMPLX_*.npz` (e.g. `models_smplx_v1_1/models/smplx`).

### Model files not used (safe to remove)

These are **not** used by the conversion scripts and can be removed to save space:

- `SMPL_python_v.1.1.0.zip` and the extracted `SMPL_python_v.1.1.0/` folder (chumpy-based `.pkl`; the pipeline uses official `.npz` instead).
- Any USMPL Unreal `.npz` (e.g. `basicmodel_m_lbs_10_207_0_v1.0.0.npz`) if you have switched to the official SMPL `.npz`.
- Generated cache files next to an `.npz` (e.g. `*_generated_from_npz.pkl`) — can be deleted; they are recreated when needed.

## Quick Start

### Run Examples

```bash
pixi run python example.py --smpl path/to/smpl/model.pkl --smplx path/to/smplx/model.pkl -o output_dir
```
Three conversions will be conducted and the results will be exported in three folders under the output_dir. The results folders are named as {source_model}\_{input_format}2{target_model}\_{method}(_{if_single_identity_sequence}):

```
mhr_smpl_conversion/
├── output_dir
│   ├── smpl_para2mhr_pymomentum
│   ├── mhr_para2smplx_pytorch_single_identity
│   ├── smplx_mesh2mhr_pytorch_single_identity
│   ├── sam3d_output_to_smplx
└──...
```

Meshes of the source and the target model can be found under each conversion results folder.

### Programmatic Usage

```python
import torch
from mhr.mhr import MHR
from smpl_mhr import Conversion
import smplx

# Initialize models
mhr_model = MHR.from_files(lod=1, device="cuda")
smplx_model = smplx.SMPLX(model_path="path/to/smplx", gender="neutral")

# Create converter
converter = Conversion(
    mhr_model=mhr_model,
    smpl_model=smplx_model,
    method="pytorch"  # or "pymomentum"
)

# Convert SAM3D output to SMPL(X)
results = converter.convert_sam3d_output_to_smplx(
    sam3d_outputs=sam3d_outputs,
    return_smpl_meshes=True,
    return_smpl_parameters=True,
    return_fitting_errors=True
)

# Convert SMPLX to MHR
results = converter.convert_smpl2mhr(
    smpl_parameters=smplx_params,
    single_identity=True,
    return_mhr_meshes=True,
    return_mhr_parameters=True
)

# Convert MHR back to SMPLX
smplx_results = converter.convert_mhr2smpl(
    mhr_parameters=results.result_parameters,
    return_smpl_meshes=True
)
```

## API Reference

### `Conversion` Class

The main conversion class that handles all transformations between SMPL/SMPLX and MHR formats.

#### Constructor

```python
Conversion(mhr_model, smpl_model, method="pytorch")
```

**Parameters:**
- `mhr_model` (MHR): MHR body model instance
- `smpl_model` (smplx.SMPLX | smplx.SMPL): SMPL or SMPLX model instance
- `method` (str): Optimization method ("pytorch" or "pymomentum")

#### `convert_smpl2mhr()`

Convert SMPL/SMPLX data to MHR format.

**Parameters:**
- `smpl_vertices` (torch.Tensor, optional): SMPL vertex positions [B, V, 3]
- `smpl_parameters` (dict, optional): SMPL parameter dictionary
- `single_identity` (bool): Use single identity across frames (default: True)
- `is_tracking` (bool): Use temporal tracking for optimization (default: False)
- `return_mhr_meshes` (bool): Return mesh objects (default: False)
- `return_mhr_parameters` (bool): Return parameter dictionary (default: False)
- `return_mhr_vertices` (bool): Return vertex arrays (default: False)
- `return_fitting_errors` (bool): Return fitting error metrics (default: True)

**Returns:**
- `ConversionResult`: Object containing requested outputs

#### `convert_mhr2smpl()`

Convert MHR data to SMPL/SMPLX format.

**Parameters:**
- `mhr_vertices` (torch.Tensor, optional): MHR vertex positions [B, V, 3]
- `mhr_parameters` (dict, optional): MHR parameter dictionary
- `single_identity` (bool): Use single identity across frames (default: True)
- `is_tracking` (bool): Use temporal tracking for optimization, only for SMPL(X)-> MHR with PyMomentum (default: False)
- `return_smpl_meshes` (bool): Return mesh objects (default: False)
- `return_smpl_parameters` (bool): Return parameter dictionary (default: False)
- `return_smpl_vertices` (bool): Return vertex arrays (default: False)

**Returns:**
- `ConversionResult`: Object containing requested outputs

### `ConversionResult` Class

Container for conversion results.

**Attributes:**
- `result_meshes` (list[trimesh.Trimesh]): Generated mesh objects
- `result_vertices` (np.ndarray): Vertex positions [B, V, 3]
- `result_parameters` (dict): Model parameter dictionary
- `result_errors` (np.ndarray): Per-frame fitting errors


## How MHR → SMPL conversion works (high level)

1. **Input: MHR-style data**  
   For SAM3D, each frame provides an MHR mesh:
   - `pred_vertices`: \((V_\text{MHR}, 3)\) in meters  
   - `pred_cam_t`: \((3,)\) camera translation  
   These are converted to MHR vertices in centimeters (the internal MHR unit):  
   \[
   \text{mhr\_vertices} = 100 \cdot \text{pred\_vertices} + 100 \cdot \text{pred\_cam\_t}
   \]
   producing an array of shape \((N, V_\text{MHR}, 3)\) in cm.

2. **Topology mapping: MHR vertices → SMPL vertex space**  
   MHR and SMPL have different meshes (vertex count and connectivity). The converter uses a precomputed mapping in `assets/mhr2smpl_mapping.npz`:
   - For each SMPL vertex it stores which MHR triangle to sample and the corresponding barycentric coordinates.  
   Using barycentric interpolation, each frame’s MHR surface is projected onto the SMPL topology, yielding a set of **target points in SMPL vertex space** (no SMPL parameters yet).

3. **Fitting SMPL parameters to the targets**  
   Given these target 3D positions, the PyTorch backend in `pytorch_fitting.py` optimizes SMPL parameters:
   - Variables: `global_orient`, `body_pose`, `betas` (and, for SMPL-X, additional DOFs).  
   - Loss: primarily vertex distance (plus optional edge terms) between the SMPL mesh and the target vertices from step 2.  
   Optimization updates pose and shape until the SMPL mesh matches the targets as well as possible, producing one set of SMPL parameters per frame.

4. **Outputs: SMPL data**  
   - **SMPL parameters**: stored in `ConversionResult.result_parameters` and, for scripts like `sam3d_dir_to_smpl.py`, in files such as `smpl_fit_results.npz` (`global_orient`, `body_pose`, `betas`, etc.).  
   - **SMPL meshes (optional)**: by forwarding the fitted parameters through the SMPL model, the code can export `.ply` meshes for visualization or downstream processing.

In short, MHR data is converted to SMPL by first mapping MHR vertices into SMPL vertex space using a fixed MHR→SMPL barycentric mapping, then fitting SMPL parameters so that the SMPL mesh best matches those mapped target vertices.


## Optimization Methods

### PyMomentum Backend

- **Pros**: Robust hierarchical optimization, make use of temporal consistency (is_tracking=True)
- **Cons**: CPU-only, may be slower for large batch of temporally inconsistent data. The identity is the average identity across the first sequential processing, not optimized across the whole sequence.

**Features:**
- Hierarchical optimization stages
- Automatic failure case reprocessing
- Temporal tracking support

### PyTorch Backend

- **Pros**: GPU acceleration, faster processing
- **Cons**: Currently process each frame independently, no temporal consistency is leveraged.
- **Best for**: Large-scale conversion of independent poses.

**Features:**
- Edge + vertex loss combination
- Batch processing
