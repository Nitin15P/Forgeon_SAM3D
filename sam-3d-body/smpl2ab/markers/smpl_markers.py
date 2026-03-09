# Copyright (C) 2024  Max Planck Institute for Intelligent Systems Tuebingen, Marilyn Keller 
 
import numpy as np
from smpl2ab.utils.smpl_utils import smpl_model_fwd


def _write_trc_python(path, timestamps, marker_timesteps, marker_names, fps):
    """Write TRC file in pure Python. Bypasses nimblephysics to avoid segfaults."""
    n_frames = len(timestamps)
    n_markers = len(marker_names)

    with open(path, 'w') as f:
        f.write('PathFileType\t4\t(X/Y/Z)\t' + path.split('/')[-1] + '\n')
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write(f'{fps:.6f}\t{fps:.6f}\t{n_frames}\t{n_markers}\tm\t{fps:.6f}\t1\t{n_frames}\n')
        # Marker names row: Frame# Time M1\t\tM2\t\tM3...
        marker_cols = '\t\t'.join(marker_names)
        f.write(f'Frame#\tTime\t{marker_cols}\n')
        # Coordinate labels: X1 Y1 Z1 X2 Y2 Z2...
        coord_labels = '\t'.join(f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(n_markers))
        f.write(f'\t\t{coord_labels}\n')
        # Data rows
        for frame_idx, (ts, frame_data) in enumerate(zip(timestamps, marker_timesteps)):
            coords = []
            for name in marker_names:
                pos = frame_data[name]
                arr = np.asarray(pos, dtype=np.float64)
                coords.extend([f'{arr[0]:.6f}', f'{arr[1]:.6f}', f'{arr[2]:.6f}'])
            f.write(f'{frame_idx + 1}\t{ts:.6f}\t' + '\t'.join(coords) + '\n')


class SmplMarker():
    """ Class that, given a SMPL sequence, generates a marker sequence from specific SMPL vertices."""
        
    def __init__(self, verts, markers_dict, fps, name):
        self.verts = verts
        self.set = markers_dict  
        self.fps = fps
        self.name = name

        markers_smpl_indices = list(markers_dict.values())
        # Ensure numpy for indexing (verts may be torch tensors)
        verts_np = np.asarray(verts) if not isinstance(verts, np.ndarray) else verts

        self.marker_trajectory = verts_np[:, markers_smpl_indices]
        self.marker_names = list(markers_dict.keys())

    def save_trc(self, path):
        """Save markers to TRC file. Uses pure Python writer to avoid nimblephysics segfaults."""
        marker_timesteps = []
        for frame_id in range(len(self.marker_trajectory)):
            frame_dict = {}
            for marker_id, marker_name in enumerate(self.marker_names):
                pos = self.marker_trajectory[frame_id][marker_id]
                frame_dict[marker_name] = np.asarray(pos, dtype=np.float64)
            marker_timesteps.append(frame_dict)

        timestamps = (1.0 / self.fps) * np.arange(len(marker_timesteps))
        # Handle numpy scalar fps (e.g. from npz)
        fps = float(self.fps)

        _write_trc_python(path, timestamps, marker_timesteps, self.marker_names, fps)
        
    @classmethod
    def from_smpl_data(cls, smpl_data, marker_set_name, markers_dict, smpl_model, fps=None):
        """ Create a marker sequence from a SMPL sequence"""
        
        # FPS for the sequence
        if fps is None and "fps" in smpl_data.keys():
            fps = smpl_data['fps']
        elif fps is not None:
            # Use the value provided as argument
            pass
        else:
            raise ValueError("The motion sequence FPS was not found in smpl_data (expected a key 'fps'), please provide it as argument")
        
        # Per frame SMPL vertices to generate the markers from
        verts = smpl_model_fwd(smpl_model, smpl_data)

        return cls(verts, markers_dict, fps, name=marker_set_name)
    