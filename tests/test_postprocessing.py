"""Unit tests for detection postprocessing, NMS, and bounding box operations."""

import numpy as np
import pytest
from cv_engine.detection.postprocessing import (
    Detection,
    apply_nms,
    calculate_iou,
    clip_box,
    compute_iou_matrix,
    filter_and_format_detections,
)


def test_detection_dataclass() -> None:
    """Test Detection dataclass properties."""
    det = Detection(bbox=(10, 20, 50, 80), class_id=1, class_name="person", confidence=0.92)
    assert det.x1 == 10
    assert det.y1 == 20
    assert det.x2 == 50
    assert det.y2 == 80
    assert det.width == 40
    assert det.height == 60
    assert det.area == 2400
    assert det.center == (30, 50)
    assert det.confidence == 0.92


def test_clip_box() -> None:
    """Test bounding box coordinate clipping."""
    # Box fully inside
    assert clip_box((10, 10, 50, 50), 100, 100) == (10, 10, 50, 50)
    # Box exceeding image boundaries
    assert clip_box((-10, -5, 120, 150), 100, 100) == (0, 0, 100, 100)
    # Inverted coordinates
    assert clip_box((50, 50, 10, 10), 100, 100) == (10, 10, 50, 50)


def test_calculate_iou() -> None:
    """Test IoU calculation for identical, disjoint, and overlapping boxes."""
    # Identical boxes
    box_a = (0.0, 0.0, 10.0, 10.0)
    assert calculate_iou(box_a, box_a) == 1.0

    # Disjoint boxes
    box_b = (20.0, 20.0, 30.0, 30.0)
    assert calculate_iou(box_a, box_b) == 0.0

    # 50% overlap (intersection 50, area1 100, area2 100, union 150 => 50/150 = 1/3)
    box_c = (0.0, 0.0, 10.0, 10.0)
    box_d = (5.0, 0.0, 15.0, 10.0)
    iou = calculate_iou(box_c, box_d)
    assert pytest.approx(iou, rel=1e-3) == (50.0 / 150.0)


def test_compute_iou_matrix() -> None:
    """Test vectorized pairwise IoU matrix computation."""
    boxes1 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)
    boxes2 = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)

    matrix = compute_iou_matrix(boxes1, boxes2)
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0  # Identical
    assert matrix[0, 1] == 0.0  # Disjoint
    assert matrix[1, 1] == 0.0  # Disjoint


def test_apply_nms() -> None:
    """Test Non-Maximum Suppression suppresses overlapping redundant boxes."""
    # Three boxes: Box 0 (high conf), Box 1 (high overlap with 0, lower conf), Box 2 (disjoint)
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],  # Heavy overlap with box 0
            [100.0, 100.0, 150.0, 150.0],  # Disjoint
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.7, 0.85], dtype=np.float32)

    kept = apply_nms(boxes, scores, iou_threshold=0.45)
    # Box 0 and Box 2 should be kept, Box 1 suppressed
    assert 0 in kept
    assert 2 in kept
    assert 1 not in kept
    assert len(kept) == 2


def test_filter_and_format_detections() -> None:
    """Test end-to-end filtering, coordinate clipping, and formatting."""
    raw_boxes = np.array(
        [
            [10.0, 10.0, 60.0, 60.0],  # Conf 0.95 -> Keep
            [12.0, 12.0, 58.0, 58.0],  # Conf 0.80 -> Suppressed by NMS for same class
            [100.0, 100.0, 200.0, 200.0],  # Conf 0.30 -> Filtered by confidence threshold 0.5
            [-20.0, -10.0, 50.0, 50.0],  # Conf 0.85 -> Clipped to [0, 0, 50, 50]
        ],
        dtype=np.float32,
    )
    raw_scores = np.array([0.95, 0.80, 0.30, 0.85], dtype=np.float32)
    raw_labels = np.array([1, 1, 2, 3], dtype=np.int64)
    class_names = {1: "person", 2: "bicycle", 3: "car"}

    dets = filter_and_format_detections(
        raw_boxes=raw_boxes,
        raw_scores=raw_scores,
        raw_labels=raw_labels,
        class_names=class_names,
        image_width=320,
        image_height=240,
        confidence_threshold=0.5,
        iou_threshold=0.45,
    )

    # Expected: Box 0 (person), Box 3 (car). Box 1 suppressed by NMS, Box 2 filtered by score.
    assert len(dets) == 2
    assert dets[0].class_name == "person"
    assert pytest.approx(dets[0].confidence, rel=1e-3) == 0.95
    assert dets[1].class_name == "car"
    assert pytest.approx(dets[1].confidence, rel=1e-3) == 0.85
    # Check clipping on Box 3 (originally -20, -10, 50, 50)
    assert dets[1].x1 == 0
    assert dets[1].y1 == 0
    assert dets[1].x2 == 50
    assert dets[1].y2 == 50
