"""Object detection package for EdgeVision AI."""

from cv_engine.detection.detector import DetectionResult, ObjectDetector
from cv_engine.detection.postprocessing import (
    Detection,
    apply_nms,
    calculate_iou,
    clip_box,
    compute_iou_matrix,
    filter_and_format_detections,
)

__all__ = [
    "Detection",
    "DetectionResult",
    "ObjectDetector",
    "apply_nms",
    "calculate_iou",
    "clip_box",
    "compute_iou_matrix",
    "filter_and_format_detections",
]
