"""Bounding box postprocessing, NMS, and standardization for EdgeVision AI."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Detection:
    """
    Standardized detection output representation.

    Attributes:
        bbox: Bounding box coordinates as (x1, y1, x2, y2) in pixels.
        class_id: Numerical class identifier.
        class_name: Human-readable category label.
        confidence: Prediction confidence score in [0.0, 1.0].
    """

    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def clip_box(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    Clip bounding box coordinates to image dimensions [0, width] and [0, height].

    Args:
        bbox: (x1, y1, x2, y2)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Clipped integer bounding box (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    cx1 = max(0, min(int(round(x1)), width - 1))
    cy1 = max(0, min(int(round(y1)), height - 1))
    cx2 = max(0, min(int(round(x2)), width))
    cy2 = max(0, min(int(round(y2)), height))

    # Ensure valid order
    if cx2 < cx1:
        cx1, cx2 = cx2, cx1
    if cy2 < cy1:
        cy1, cy2 = cy2, cy1

    return (cx1, cy1, cx2, cy2)


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """
    Calculate Intersection-over-Union (IoU) between two bounding boxes.

    Boxes format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise IoU matrix between two sets of boxes.

    Args:
        boxes1: Shape (N, 4) in (x1, y1, x2, y2)
        boxes2: Shape (M, 4) in (x1, y1, x2, y2)

    Returns:
        iou_matrix: Shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area1 = np.maximum(0.0, b1_x2 - b1_x1) * np.maximum(0.0, b1_y2 - b1_y1)
    area2 = np.maximum(0.0, b2_x2 - b2_x1) * np.maximum(0.0, b2_y2 - b2_y1)
    union = area1[:, None] + area2[None, :] - intersection

    return np.where(union > 0, intersection / np.maximum(union, 1e-7), 0.0)


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Standard Greedy Non-Maximum Suppression (NMS).

    Args:
        boxes: Array of shape (N, 4) formatted as (x1, y1, x2, y2)
        scores: Array of shape (N,) with detection confidence scores
        iou_threshold: Overlap threshold above which suppression occurs

    Returns:
        List of retained indices sorted by descending confidence.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        ovr = np.where(union > 0, inter / np.maximum(union, 1e-7), 0.0)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def filter_and_format_detections(
    raw_boxes: Union[np.ndarray, List],
    raw_scores: Union[np.ndarray, List],
    raw_labels: Union[np.ndarray, List],
    class_names: Dict[int, str],
    image_width: int,
    image_height: int,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    apply_class_specific_nms: bool = True,
) -> List[Detection]:
    """
    Filter raw detector predictions by confidence, clip coordinates, apply NMS,
    and format into standardized Detection objects.

    Args:
        raw_boxes: Array of shape (N, 4)
        raw_scores: Array of shape (N,)
        raw_labels: Array of shape (N,)
        class_names: Mapping from class_id to human-readable label
        image_width: Frame width for coordinate clipping
        image_height: Frame height for coordinate clipping
        confidence_threshold: Minimum confidence score to retain
        iou_threshold: IoU overlap threshold for NMS
        apply_class_specific_nms: Whether to apply NMS per-class or across all classes

    Returns:
        List of formatted Detection instances.
    """
    if len(raw_boxes) == 0:
        return []

    boxes = np.asarray(raw_boxes, dtype=np.float32)
    scores = np.asarray(raw_scores, dtype=np.float32)
    labels = np.asarray(raw_labels, dtype=np.int64)

    # 1. Filter by confidence
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if len(boxes) == 0:
        return []

    # 2. NMS (class-specific or global)
    final_indices: List[int] = []
    if apply_class_specific_nms:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            class_mask = labels == label
            class_indices = np.where(class_mask)[0]
            cls_boxes = boxes[class_indices]
            cls_scores = scores[class_indices]

            keep = apply_nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)
            final_indices.extend(class_indices[keep])
    else:
        final_indices = apply_nms(boxes, scores, iou_threshold=iou_threshold)

    # 3. Construct Detection objects with coordinate clipping
    detections: List[Detection] = []
    for idx in final_indices:
        clipped_bbox = clip_box(
            (float(boxes[idx, 0]), float(boxes[idx, 1]), float(boxes[idx, 2]), float(boxes[idx, 3])),
            width=image_width,
            height=image_height,
        )
        cid = int(labels[idx])
        cname = class_names.get(cid, f"class_{cid}")
        conf = float(scores[idx])

        # Filter zero-area degenerate boxes
        if (clipped_bbox[2] - clipped_bbox[0]) > 0 and (clipped_bbox[3] - clipped_bbox[1]) > 0:
            detections.append(
                Detection(
                    bbox=clipped_bbox,
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                )
            )

    # Sort descending by confidence
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections
