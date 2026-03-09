#!/usr/bin/env python3
"""
Render mesh data (npz files) to a 3D animation video using matplotlib.
Light blue wireframe mesh + red keypoints, no pyrender needed - works headless.
"""
import os
import re
import argparse
import numpy as np

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D


def load_mesh_frame(npz_path: str) -> dict:
    """Load mesh data from .npz file (demo output format)."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "vertices": data["pred_vertices"],
        "faces": data["faces"],
        "keypoints_3d": data["pred_keypoints_3d"],
        "frame_idx": int(data.get("frame_idx", -1)),
        "person_idx": int(data.get("person_idx", 0)),
    }


def render_mesh_to_video(
    mesh_data_folder: str,
    output_video_path: str,
    fps: int = 30,
    person_id: int = 0,
    elev: float = -90,
    azim: float = 270,
    figsize: tuple = (16, 9),
) -> str:
    """
    Create 3D animation video from mesh data folder.
    Light blue wireframe + red keypoints, matplotlib style.
    """
    mesh_data_folder = os.path.abspath(mesh_data_folder)
    output_video_path = os.path.abspath(output_video_path)

    # Get npz files: frame_XXXX_personN.npz or {name}_personN.npz
    all_files = [f for f in os.listdir(mesh_data_folder) if f.endswith(".npz")]

    # Filter by person_id and sort by frame
    def frame_sort_key(f):
        m = re.match(r"frame_(\d+)_person(\d+)", f)
        if m:
            return (int(m.group(2)), int(m.group(1)))
        m = re.match(r"(.+)_person(\d+)", f)
        if m:
            return (int(m.group(2)), m.group(1))
        return (0, f)

    npz_files = sorted(
        [f for f in all_files if frame_sort_key(f)[0] == person_id],
        key=frame_sort_key,
    )

    if not npz_files:
        raise FileNotFoundError(f"No .npz files for person {person_id} in {mesh_data_folder}")

    # Compute bounds from all frames
    all_vertices = []
    for f in npz_files:
        d = load_mesh_frame(os.path.join(mesh_data_folder, f))
        all_vertices.append(d["vertices"])
    all_vertices = np.concatenate(all_vertices, axis=0)
    x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
    z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()
    margin = 0.15
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)
    z_lim = (z_min - margin, z_max + margin)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def animate(frame_idx):
        ax.clear()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Mesh - Frame {frame_idx + 1}/{len(npz_files)}")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        d = load_mesh_frame(os.path.join(mesh_data_folder, npz_files[frame_idx]))
        verts = d["vertices"]
        faces = d["faces"]
        kpts = d["keypoints_3d"]

        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces,
            alpha=0.7,
            color="lightblue",
            edgecolor="none",
        )
        ax.scatter(
            kpts[:, 0], kpts[:, 1], kpts[:, 2],
            c="red",
            s=30,
            marker="o",
            alpha=0.8,
        )
        ax.view_init(elev=elev, azim=azim)

    anim = FuncAnimation(
        fig,
        animate,
        frames=len(npz_files),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="SAM 3D Body"), bitrate=1800)
    anim.save(output_video_path, writer=writer)
    plt.close()
    return output_video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render mesh data to 3D animation video")
    parser.add_argument("-i", "--input", required=True, help="Path to mesh_data folder")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--person", type=int, default=0)
    parser.add_argument("--elev", type=float, default=-90, help="Elevation angle (-90 = side view with XY plane up)")
    parser.add_argument("--azim", type=float, default=270, help="Azimuth angle")
    args = parser.parse_args()
    render_mesh_to_video(
        args.input,
        args.output,
        fps=args.fps,
        person_id=args.person,
        elev=args.elev,
        azim=args.azim,
    )
    print(f"Video saved: {args.output}")
