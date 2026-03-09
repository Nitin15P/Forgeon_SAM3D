# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["sam-3d-body"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import json
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, load_sam_3d_body_hf, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm


def convert_to_json_serializable(data):
    """Convert numpy arrays and other types to JSON-serializable format."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating)):
        return float(data) if isinstance(data, np.floating) else int(data)
    elif isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif data is None:
        return None
    else:
        return data


def main(args):
    if not args.checkpoint_hf and not args.checkpoint_path:
        raise ValueError(
            "Either --checkpoint_path or --checkpoint_hf must be provided. "
            "Example: --checkpoint_hf facebook/sam-3d-body-dinov3"
        )

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # For YOLOv8, use the weights path directly if provided
    if args.detector_name == "yolov8":
        if not args.yolov8_weights:
            raise ValueError(
                "--yolov8_weights is required for YOLOv8 detector. "
                "Example: --yolov8_weights yolov8n.pt"
            )
        detector_path = args.yolov8_weights
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"YOLOv8 weights file not found: {detector_path}")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.checkpoint_hf:
        print(f"Loading model from HuggingFace: {args.checkpoint_hf}")
        model, model_cfg = load_sam_3d_body_hf(args.checkpoint_hf, device=device)
    else:
        model, model_cfg = load_sam_3d_body(
            args.checkpoint_path, device=device, mhr_path=mhr_path
        )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # Set det_cat_id - use provided class ID or default to 0 (person)
    det_cat_id = args.det_class_id

    # Process video if provided
    if args.video_path:
        process_video(args, estimator, det_cat_id)
    # Otherwise process images
    elif args.image_folder:
        process_images(args, estimator, det_cat_id)
    else:
        raise ValueError("Either --image_folder or --video_path must be provided")


def process_video(args, estimator, det_cat_id):
    """Process a video file and output a video with 3D body estimation."""
    video_path = args.video_path
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Determine mesh output folder
    if args.output_folder:
        mesh_data_folder = os.path.join(args.output_folder, "mesh_data")
    elif args.output_video:
        mesh_data_folder = os.path.join(os.path.dirname(args.output_video), "mesh_data")
    else:
        mesh_data_folder = os.path.join("./output", base_name, "mesh_data")

    os.makedirs(mesh_data_folder, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer only when not mesh_only
    out = None
    if not args.mesh_only:
        if args.output_video == "":
            output_video = os.path.join("./output", f"{base_name}_output.mp4")
        else:
            output_video = args.output_video
        os.makedirs(os.path.dirname(output_video) if os.path.dirname(output_video) else "./output", exist_ok=True)
        output_width = width * 4
        output_height = height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))
        print(f"Output video: {output_video}")

    print(f"Processing video: {video_path}")
    print(f"Mesh data folder: {mesh_data_folder}")
    print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
    if args.mesh_only:
        print("Mode: mesh only (no video, no pyrender)")

    frame_count = 0
    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (cv2 reads BGR, but process_one_image expects RGB for numpy arrays)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        outputs = estimator.process_one_image(
            frame_rgb,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            det_cat_id=det_cat_id,
        )

        # Visualize and write video only when not mesh_only
        if not args.mesh_only and out is not None:
            rend_img = visualize_sample_together(frame, outputs, estimator.faces)
            if rend_img.dtype != np.uint8:
                rend_img = rend_img.astype(np.uint8)
            out.write(rend_img)

        # Save 3D mesh data for each detected person (npz only when mesh_only)
        if len(outputs) > 0:
            for person_idx, person_output in enumerate(outputs):
                mesh_data = {
                    "frame_idx": frame_idx,
                    "person_idx": person_idx,
                    "bbox": person_output["bbox"],
                    "focal_length": person_output["focal_length"],
                    "pred_keypoints_3d": person_output["pred_keypoints_3d"],
                    "pred_keypoints_2d": person_output["pred_keypoints_2d"],
                    "pred_vertices": person_output["pred_vertices"],
                    "pred_cam_t": person_output["pred_cam_t"],
                    "faces": estimator.faces,
                    "body_pose_params": person_output.get("body_pose_params", None),
                    "hand_pose_params": person_output.get("hand_pose_params", None),
                    "shape_params": person_output.get("shape_params", None),
                    "scale_params": person_output.get("scale_params", None),
                }

                mesh_filename = f"{mesh_data_folder}/frame_{frame_idx:04d}_person{person_idx}.npz"
                np.savez_compressed(mesh_filename, **mesh_data)

                if not args.mesh_only:
                    json_filename = f"{mesh_data_folder}/frame_{frame_idx:04d}_person{person_idx}.json"
                    json_data = convert_to_json_serializable(mesh_data)
                    with open(json_filename, 'w') as f:
                        json.dump(json_data, f, indent=2)

        frame_count += 1

    cap.release()
    if out is not None:
        out.release()
        print(f"Video processing complete! Output saved to: {output_video}")
    else:
        print(f"Mesh data saved to: {mesh_data_folder}")


def process_images(args, estimator, det_cat_id):
    """Process images from a folder."""
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    # Create subfolder for 3D mesh data
    mesh_data_folder = os.path.join(output_folder, "mesh_data")
    os.makedirs(mesh_data_folder, exist_ok=True)

    for image_path in tqdm(images_list, desc="Processing images"):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            det_cat_id=det_cat_id,
        )

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        
        # Save visualization image
        base_name = os.path.basename(image_path)[:-4]
        cv2.imwrite(
            f"{output_folder}/{base_name}.jpg",
            rend_img.astype(np.uint8),
        )
        
        # Save 3D mesh data for each detected person
        if len(outputs) > 0:
            for person_idx, person_output in enumerate(outputs):
                # Prepare data for saving (convert to numpy arrays)
                mesh_data = {
                    "image_name": base_name,
                    "person_idx": person_idx,
                    "bbox": person_output["bbox"],
                    "focal_length": person_output["focal_length"],
                    "pred_keypoints_3d": person_output["pred_keypoints_3d"],
                    "pred_keypoints_2d": person_output["pred_keypoints_2d"],
                    "pred_vertices": person_output["pred_vertices"],
                    "pred_cam_t": person_output["pred_cam_t"],
                    "faces": estimator.faces,  # Mesh faces for visualization
                    "body_pose_params": person_output.get("body_pose_params", None),
                    "hand_pose_params": person_output.get("hand_pose_params", None),
                    "shape_params": person_output.get("shape_params", None),
                    "scale_params": person_output.get("scale_params", None),
                }
                
                # Save as .npz file (numpy compressed format - easy to load in Colab)
                mesh_filename = f"{mesh_data_folder}/{base_name}_person{person_idx}.npz"
                np.savez_compressed(mesh_filename, **mesh_data)
                
                # Save as .json file
                json_filename = f"{mesh_data_folder}/{base_name}_person{person_idx}.json"
                json_data = convert_to_json_serializable(mesh_data)
                with open(json_filename, 'w') as f:
                    json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Process images:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt
                
                # Process video with YOLOv8:
                python demo.py --video_path ./input.mp4 --checkpoint_path ./checkpoints/model.ckpt --detector_name yolov8 --yolov8_weights /path/to/best.pt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        default="",
        type=str,
        help="Path to folder containing input images (required if --video_path not provided)",
    )
    parser.add_argument(
        "--video_path",
        default="",
        type=str,
        help="Path to input video file (required if --image_folder not provided)",
    )
    parser.add_argument(
        "--output_video",
        default="",
        type=str,
        help="Path to output video file (default: ./output/<input_video_name>_output.mp4)",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        type=str,
        help="Path to SAM 3D Body model checkpoint (required if --checkpoint_hf not set)",
    )
    parser.add_argument(
        "--checkpoint_hf",
        default="",
        type=str,
        help="HuggingFace repo ID to load model from (e.g. facebook/sam-3d-body-dinov3). Overrides --checkpoint_path.",
    )
    parser.add_argument(
        "--detector_name",
        default="yolov8",
        type=str,
        help="Human detection model for demo (default: yolov8).",
    )
    parser.add_argument(
        "--yolov8_weights",
        default="",
        type=str,
        help="Path to YOLOv8 weights file (.pt) - required when using yolov8 detector",
    )
    parser.add_argument(
        "--det_class_id",
        default=0,
        type=int,
        help="Class ID to detect (0=person in COCO, 3=bowler in custom model). Default: 0",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--mesh_only",
        action="store_true",
        default=False,
        help="Only save mesh data (npz files), skip video generation and visualization. Faster, no pyrender needed.",
    )
    args = parser.parse_args()

    main(args)
