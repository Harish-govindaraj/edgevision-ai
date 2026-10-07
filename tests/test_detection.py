"""Unit tests for ObjectDetector module and real-time detection pipeline."""

import numpy as np
import pytest
import torch
from cv_engine.detection.detector import DetectionResult, ObjectDetector
from cv_engine.detection.postprocessing import Detection


def test_detector_device_resolution() -> None:
    """Test device selection logic and fallback mechanism."""
    detector = ObjectDetector(device="cpu", pretrained=False)
    assert detector.device == torch.device("cpu")

    # If CUDA is requested on a CPU-only build, it should fall back to CPU gracefully
    if not torch.cuda.is_available():
        detector_cuda_fallback = ObjectDetector(device="cuda", pretrained=False)
        assert detector_cuda_fallback.device == torch.device("cpu")


def test_detector_preprocessing() -> None:
    """Test frame preprocessing from OpenCV BGR to PyTorch float32 tensor."""
    detector = ObjectDetector(device="cpu", pretrained=False)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[10:50, 10:50] = [0, 128, 255]  # BGR color

    tensor, orig_w, orig_h = detector.preprocess(frame)

    assert tensor.shape == (3, 240, 320)
    assert tensor.dtype == torch.float32
    assert orig_w == 320
    assert orig_h == 240
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_detector_empty_frame_raises() -> None:
    """Empty frames should raise ValueError."""
    detector = ObjectDetector(device="cpu", pretrained=False)
    with pytest.raises(ValueError, match="must not be empty"):
        detector.preprocess(np.array([], dtype=np.uint8))


def test_detector_draw_detections() -> None:
    """Test bounding box drawing on OpenCV frame."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        Detection(bbox=(20, 20, 80, 80), class_id=1, class_name="person", confidence=0.88),
        Detection(bbox=(100, 100, 150, 150), class_id=3, class_name="car", confidence=0.75),
    ]

    annotated = ObjectDetector.draw_detections(frame, detections)
    assert annotated.shape == frame.shape
    # Frame should have non-zero pixels where boxes were drawn
    assert np.any(annotated > 0)


def test_detector_inference_on_synthetic_frame() -> None:
    """Test full predict pipeline on a synthetic frame."""
    detector = ObjectDetector(confidence_threshold=0.3, device="cpu", pretrained=True)
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    result = detector.predict(frame)

    assert isinstance(result, DetectionResult)
    assert result.preprocessing_time_ms >= 0.0
    assert result.inference_time_ms > 0.0
    assert result.postprocessing_time_ms >= 0.0
    assert result.total_time_ms > 0.0
    assert result.fps > 0.0
    assert isinstance(result.detections, list)
