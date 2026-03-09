# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import numpy as np
import cv2
# Lazy import of Renderer to avoid OpenGL initialization errors in headless environments
# Renderer will be imported only when needed
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def _create_safe_offscreen_renderer(width, height):
    """
    Create a pyrender OffscreenRenderer that is compatible with Xvfb.
    
    Important: pyrender's OffscreenRenderer only supports EGL and OSMesa, NOT GLX.
    Even though Xvfb sets DISPLAY and GLX is available, pyrender cannot use GLX.
    
    Rules:
    1. Do NOT force EGL
    2. Do NOT force OSMesa
    3. Do NOT reject the 'glx' backend (but pyrender doesn't support it anyway)
    4. Always remove PYOPENGL_PLATFORM at runtime inside the renderer creation
    5. Use pyrender.OffscreenRenderer(width, height) with no extra arguments
    6. If the renderer fails, fall back to None
    
    Returns:
        pyrender.OffscreenRenderer or None if creation fails
    """
    try:
        import pyrender
    except (ImportError, OSError):
        return None
    
    # Remove PYOPENGL_PLATFORM at runtime to let pyrender choose between EGL and OSMesa
    # Note: pyrender's OffscreenRenderer does NOT support GLX, even if DISPLAY is set
    # So we remove PYOPENGL_PLATFORM and let pyrender try EGL first, then OSMesa if available
    saved_platform = os.environ.pop("PYOPENGL_PLATFORM", None)
    display_set = os.environ.get("DISPLAY") is not None
    
    try:
        # Use pyrender.OffscreenRenderer(width, height) with no extra arguments
        # pyrender will try EGL first (if available), then OSMesa
        # It will NOT use GLX even if DISPLAY is set, because pyrender doesn't support GLX
        renderer = pyrender.OffscreenRenderer(width, height)
        return renderer
    except Exception as e:
        # If renderer creation fails, try EGL explicitly (pyrender's preferred backend)
        # This is a fallback, not forcing - we're just being explicit after auto-detection failed
        if saved_platform != "egl":
            try:
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                renderer = pyrender.OffscreenRenderer(width, height)
                # Clear it for next time
                os.environ.pop("PYOPENGL_PLATFORM", None)
                return renderer
            except Exception:
                os.environ.pop("PYOPENGL_PLATFORM", None)
                pass
        
        # If EGL fails, try OSMesa as last resort (software rendering)
        if saved_platform != "osmesa":
            try:
                os.environ["PYOPENGL_PLATFORM"] = "osmesa"
                renderer = pyrender.OffscreenRenderer(width, height)
                os.environ.pop("PYOPENGL_PLATFORM", None)
                return renderer
            except Exception:
                os.environ.pop("PYOPENGL_PLATFORM", None)
                pass
        
        # If all attempts fail, log the error and return None
        # This allows graceful fallback to 2D keypoints
        import warnings
        display_info = os.environ.get("DISPLAY", "not set")
        warnings.warn(f"Failed to create OffscreenRenderer (DISPLAY={display_info}): {type(e).__name__}: {e}. "
                     f"Tried EGL and OSMesa. Falling back to 2D keypoint visualization.")
        return None
    finally:
        # Restore PYOPENGL_PLATFORM if it was set originally
        if saved_platform is not None:
            os.environ["PYOPENGL_PLATFORM"] = saved_platform
        elif "PYOPENGL_PLATFORM" in os.environ:
            # Clear if we set it during fallback attempts
            os.environ.pop("PYOPENGL_PLATFORM", None)


class SafeRenderer:
    """
    Safe wrapper around pyrender rendering that is compatible with Xvfb and GLX.
    Falls back to None if rendering fails, allowing 2D keypoint visualization.
    """
    
    def __init__(self, focal_length, faces=None):
        self.focal_length = focal_length
        self.faces = faces
        self._renderer = None  # Will be created on first use
    
    def _get_renderer(self, width, height):
        """Get or create the offscreen renderer for the given dimensions."""
        if self._renderer is None:
            self._renderer = _create_safe_offscreen_renderer(width, height)
        return self._renderer
    
    def __call__(
        self,
        vertices,
        cam_t,
        image,
        full_frame=False,
        imgname=None,
        side_view=False,
        top_view=False,
        rot_angle=90,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        tri_color_lights=False,
        return_rgba=False,
        camera_center=None,
    ):
        """
        Render meshes on input image. Returns None if rendering fails.
        """
        try:
            import pyrender
            import trimesh
        except (ImportError, OSError):
            return None
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)
        image = image / 255.0
        h, w = image.shape[:2]
        
        # Get renderer (may return None if creation fails)
        renderer = self._get_renderer(w, h)
        if renderer is None:
            return None
        
        try:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.0
            
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode="OPAQUE",
                baseColorFactor=(
                    mesh_base_color[2],
                    mesh_base_color[1],
                    mesh_base_color[0],
                    1.0,
                ),  # Swap RGB to BGR for pyrender
            )
            mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
            
            if side_view:
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(rot_angle), [0, 1, 0]
                )
                mesh.apply_transform(rot)
            elif top_view:
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(rot_angle), [1, 0, 0]
                )
                mesh.apply_transform(rot)
            
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            
            scene = pyrender.Scene(
                bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3)
            )
            scene.add(mesh, "mesh")
            
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation
            if camera_center is None:
                camera_center = [image.shape[1] / 2.0, image.shape[0] / 2.0]
            camera = pyrender.IntrinsicsCamera(
                fx=self.focal_length,
                fy=self.focal_length,
                cx=camera_center[0],
                cy=camera_center[1],
                zfar=1e12,
            )
            scene.add(camera, pose=camera_pose)
            
            if tri_color_lights:
                # Import create_raymond_lights from renderer module
                from sam_3d_body.visualization.renderer import create_raymond_lights
                lights = create_raymond_lights()
                for light in lights:
                    scene.add_node(light)
            else:
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0))
            
            color, _rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            if return_rgba:
                return color
            else:
                return color[:, :, :3]
        except Exception:
            # If rendering fails, return None
            return None


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = SafeRenderer(focal_length=person_output["focal_length"], faces=faces)
        img2_result = renderer(
            person_output["pred_vertices"],
            person_output["pred_cam_t"],
            img_mesh.copy(),
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        if img2_result is not None:
            img2 = img2_result * 255
        else:
            # Fallback to 2D keypoints if 3D rendering fails
            img2 = img1.copy()

        white_img = np.ones_like(img_cv2) * 255
        img3_result = renderer(
            person_output["pred_vertices"],
            person_output["pred_cam_t"],
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        if img3_result is not None:
            img3 = img3_result * 255
        else:
            # Fallback to 2D keypoints if 3D rendering fails
            img3 = img1.copy()

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces):
    # Handle empty outputs (no detections)
    if not outputs or len(outputs) == 0:
        # Return original image repeated 4 times (no detections to visualize)
        return np.concatenate([img_cv2, img_cv2, img_cv2, img_cv2], axis=1)
    
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Try to render 3D mesh, fallback to 2D keypoints if rendering fails (headless environments)
    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Use SafeRenderer which handles GLX, EGL, and OSMesa automatically
    renderer = SafeRenderer(focal_length=person_output["focal_length"], faces=all_faces)
    
    # Render front view
    img_mesh_result = renderer(
        all_pred_vertices,
        fake_pred_cam_t,
        img_mesh,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
    )
    if img_mesh_result is not None:
        img_mesh = img_mesh_result * 255
    else:
        # Fallback to 2D keypoints if 3D rendering fails
        img_mesh = img_keypoints.copy()

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side_result = renderer(
        all_pred_vertices,
        fake_pred_cam_t,
        white_img,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        side_view=True,
    )
    if img_mesh_side_result is not None:
        img_mesh_side = img_mesh_side_result * 255
    else:
        # Fallback to 2D keypoints if 3D rendering fails
        img_mesh_side = img_keypoints.copy()

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img
