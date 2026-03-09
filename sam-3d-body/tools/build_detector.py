# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import cv2
import numpy as np


class HumanDetector:
    def __init__(self, name="yolov8", device="cuda", **kwargs):
        self.device = device

        if name == "yolov8":
            print("########### Using human detector: YOLOv8...")
            self.detector = load_yolov8(**kwargs)
            self.detector_func = run_yolov8
        else:
            raise NotImplementedError(
                f"Detector '{name}' not implemented. Supported: 'yolov8'"
            )

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)


def load_yolov8(path="", **kwargs):
    """
    Load YOLOv8 detector from Ultralytics.
    
    Args:
        path: Path to YOLOv8 weights file (.pt)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package is required for YOLOv8. Install with: pip install ultralytics"
        )
    
    if not path:
        raise ValueError("YOLOv8 requires a weights path. Provide path via detector_path argument.")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLOv8 weights file not found at {path}")
    
    model = YOLO(path)
    return model


def run_yolov8(
    detector,
    img,
    det_cat_id: int = 0,  # Default to class 0 (person in COCO)
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    """
    Run YOLOv8 detection on an image.
    
    Args:
        detector: YOLOv8 model
        img: Input image (numpy array, BGR format)
        det_cat_id: Class ID to filter (default: 0 for person in COCO)
        bbox_thr: Confidence threshold
        nms_thr: NMS threshold (YOLOv8 handles NMS internally, but we can filter)
        default_to_full_image: If True, return full image bbox when no detections
    
    Returns:
        boxes: numpy array of shape [N, 4] with [x1, y1, x2, y2] format
    """
    height, width = img.shape[:2]
    
    # YOLOv8 expects RGB, but we receive BGR from cv2
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    
    # Run inference
    results = detector(img_rgb, conf=bbox_thr, iou=nms_thr, verbose=False)
    
    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        # Extract boxes, scores, and class IDs
        boxes_tensor = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] in x1, y1, x2, y2 format
        scores = results[0].boxes.conf.cpu().numpy()  # [N]
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # [N]
        
        # Filter by class ID
        valid_idx = class_ids == det_cat_id
        
        if valid_idx.sum() > 0:
            boxes = boxes_tensor[valid_idx]
            # Sort boxes to keep a consistent output order
            sorted_indices = np.lexsort(
                (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
            )
            boxes = boxes[sorted_indices]
    
    # If no detections and default_to_full_image, return full image bbox
    if len(boxes) == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
    
    return boxes
