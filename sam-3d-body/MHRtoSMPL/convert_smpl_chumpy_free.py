"""
Convert SMPL pkl file from chumpy format to chumpy-free format.
Run with an env that has chumpy (e.g. Python 3.8: pip install chumpy numpy scipy).
"""

import pickle
from pathlib import Path

import numpy as np

# NumPy 1.24+ / 2.x no longer expose np.bool, np.object, etc. Chumpy imports them.
# Patch so chumpy can load when pickle deserializes chumpy objects.
for _attr, _fallback in (
    ("bool", getattr(np, "bool_")),
    ("int", getattr(np, "int_")),
    ("float", getattr(np, "float_")),
    ("complex", getattr(np, "complex_")),
    ("object", getattr(np, "object_")),
    ("str", getattr(np, "str_")),
    ("unicode", getattr(np, "unicode_")),
):
    if not hasattr(np, _attr):
        setattr(np, _attr, _fallback)

from scipy.sparse import csc_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_NAME = "SMPL_NEUTRAL_chumpy_free.pkl"
INPUT_CANDIDATES = [
    SCRIPT_DIR / "SMPL_NEUTRAL.pkl",
    SCRIPT_DIR / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
]


def convert_to_numpy(obj):
    """Convert chumpy object to numpy array."""
    if hasattr(obj, 'r'):  # chumpy object has .r attribute for the actual value
        return np.array(obj.r)
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, csc_matrix):
        return obj  # keep sparse matrix as is
    else:
        return obj


def main():
    output_path = SCRIPT_DIR / OUTPUT_NAME
    input_path = None
    data = None

    for p in INPUT_CANDIDATES:
        if not p.exists():
            continue
        print(f"Trying {p.name}...")
        try:
            with open(p, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            input_path = p
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if data is None:
        raise FileNotFoundError(
            f"No valid input found. Place a chumpy SMPL .pkl in MHRtoSMPL as one of: "
            f"{[p.name for p in INPUT_CANDIDATES]}"
        )

    print(f"Loaded from {input_path.name}")

    print('Converting chumpy objects to numpy...')
    new_data = {}
    for k, v in data.items():
        type_name = type(v).__module__ + '.' + type(v).__name__
        if 'chumpy' in type_name.lower():
            print(f'  Converting {k} from chumpy to numpy')
            new_data[k] = convert_to_numpy(v)
        else:
            new_data[k] = v

    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

    print('Done!')

    # Verify
    print('\nVerifying...')
    with open(output_path, 'rb') as f:
        verify_data = pickle.load(f)

    for k, v in verify_data.items():
        type_name = type(v).__module__ + '.' + type(v).__name__
        print(f'  {k}: {type_name}')

if __name__ == '__main__':
    main()
