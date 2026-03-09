#!/usr/bin/env python3
"""
Run inverse kinematics using nimblephysics on TRC marker data.
Produces .mot (OpenSim motion) and optionally .osim (scaled model) outputs.
"""

import argparse
import os
import numpy as np
import nimblephysics as nimble

import smpl2ab.config as cg


def load_trc_marker_observations(trc_path):
    """Load TRC file and return (timestamps, marker_observations, fps)."""
    trc_path = os.path.abspath(trc_path)
    trc = nimble.biomechanics.OpenSimParser.loadTRC(trc_path)
    fps = trc.framesPerSecond
    timestamps = list(trc.timestamps)
    # markerTimesteps: List[Dict[str, np.ndarray]]
    marker_obs = []
    for frame in trc.markerTimesteps:
        frame_dict = {}
        for k, v in frame.items():
            arr = np.asarray(v, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(3, 1)
            frame_dict[k] = arr
        marker_obs.append(frame_dict)
    return timestamps, marker_obs, fps


def run_ik(trc_path, osim_path, output_mot_path, output_osim_path=None):
    """
    Run IK on TRC marker data using the given OpenSim model.
    Uses skeleton.fitMarkersToWorldPositions for per-frame IK.
    """
    osim_path = os.path.abspath(osim_path)
    trc_path = os.path.abspath(trc_path)

    osim = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
    skel = osim.skeleton
    markers_map = osim.markersMap

    timestamps, marker_obs, fps = load_trc_marker_observations(trc_path)
    n_frames = len(timestamps)

    # Build marker list in consistent order (match TRC marker names)
    # Use only markers present in both osim and TRC
    trc_marker_names = list(marker_obs[0].keys())
    marker_list = []  # List[Tuple[BodyNode, offset]]
    marker_names_used = []
    for name in trc_marker_names:
        if name in markers_map:
            body_node, offset = markers_map[name]
            off = np.asarray(offset, dtype=np.float64)
            if off.ndim == 1:
                off = off.reshape(3, 1)
            marker_list.append((body_node, off))
            marker_names_used.append(name)

    n_markers = len(marker_list)
    if n_markers == 0:
        raise ValueError("No matching markers between TRC and OpenSim model. "
                         "Ensure the TRC was generated with BSM markers.")

    print(f"Using {n_markers} markers for IK (of {len(trc_marker_names)} in TRC)")

    # Per-frame IK
    poses_list = []
    for i in range(n_frames):
        frame_dict = marker_obs[i]
        target_positions = np.zeros((n_markers * 3, 1), dtype=np.float64)
        for j, name in enumerate(marker_names_used):
            pos = frame_dict[name]
            target_positions[j * 3:(j + 1) * 3, 0] = pos.flatten()
        weights = np.ones((n_markers * 3, 1), dtype=np.float64)

        # Clone skeleton for each frame to avoid state pollution
        frame_skel = skel.clone()
        frame_markers = []
        for name in marker_names_used:
            body_node, offset = markers_map[name]
            body_name = body_node.getName()
            new_body = frame_skel.getBodyNode(body_name)
            off = np.asarray(offset, dtype=np.float64)
            if off.ndim == 1:
                off = off.reshape(3, 1)
            frame_markers.append((new_body, off))

        loss = frame_skel.fitMarkersToWorldPositions(
            frame_markers, target_positions, weights,
            scaleBodies=False,
            maxStepCount=100,
            numIndependentStarts=3,
        )
        poses_list.append(frame_skel.getPositions().reshape(-1))

    poses = np.array(poses_list)

    # Save .mot file
    output_mot_path = os.path.abspath(output_mot_path)
    os.makedirs(os.path.dirname(output_mot_path) or '.', exist_ok=True)
    nimble.biomechanics.OpenSimParser.saveMot(skel, output_mot_path, timestamps, poses)
    print(f"Saved IK motion to {output_mot_path}")

    if output_osim_path:
        # Save scaled model (skeleton state from last frame - or we could use first)
        # For now, just copy the input osim - full scaling would need MarkerFitter
        output_osim_path = os.path.abspath(output_osim_path)
        os.makedirs(os.path.dirname(output_osim_path) or '.', exist_ok=True)
        import shutil
        shutil.copy(osim_path, output_osim_path)
        print(f"Saved OpenSim model to {output_osim_path} (copy of input; no scaling)")

    return output_mot_path


def main():
    parser = argparse.ArgumentParser(
        description="Run IK on TRC marker data using nimblephysics"
    )
    parser.add_argument(
        "-i", "--trc",
        type=str,
        required=True,
        help="Path to TRC marker file",
    )
    parser.add_argument(
        "-m", "--osim",
        type=str,
        default=cg.osim_model_path,
        help="Path to OpenSim model (.osim)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path for output .mot file",
    )
    parser.add_argument(
        "--osim-out",
        type=str,
        default=None,
        help="Optional path to save OpenSim model",
    )
    args = parser.parse_args()

    run_ik(
        trc_path=args.trc,
        osim_path=args.osim,
        output_mot_path=args.output,
        output_osim_path=args.osim_out,
    )


if __name__ == "__main__":
    main()
