"""Lightweight real-time object detector module for EdgeVision AI."""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from cv_engine.detection.postprocessing import Detection, filter_and_format_detections

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Complete output and performance metrics for one frame detection pass."""

    detections: List[Detection] = field(default_factory=list)
    preprocessing_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_time_ms: float = 0.0
    fps: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


class ObjectDetector:
    """
    Lightweight Object Detector tailored for real-time edge computer vision.

    Uses MobileNetV3-Large SSDLite320 pretrained on COCO, with standardized
    preprocessing, device selection, and decoupled postprocessing.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        pretrained: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._resolve_device(device)

        # Load weights and model
        if pretrained:
            self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
            # COCO category labels mapping
            categories = self.weights.meta.get("categories", [])
            self.class_names: Dict[int, str] = {i: cat for i, cat in enumerate(categories)}
        else:
            self.weights = None
            self.model = ssdlite320_mobilenet_v3_large(weights=None)
            self.class_names = {i: f"class_{i}" for i in range(91)}

        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested_device: str) -> torch.device:
        """Resolve target device with fallback to CPU if CUDA is unavailable."""
        req = requested_device.lower().strip()
        if req in ("auto", "cuda"):
            if torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda:0")
            else:
                if req == "cuda":
                    logger.warning("CUDA was requested but torch.cuda.is_available() is False. Falling back to CPU.")
                return torch.device("cpu")
        elif req == "cpu":
            return torch.device("cpu")
        else:
            return torch.device(req)

    def preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Convert BGR OpenCV frame to PyTorch tensor.

        Args:
            frame: (H, W, 3) BGR image array (uint8).

        Returns:
            tensor: (3, H, W) float32 tensor normalized to [0, 1].
            orig_width: Original image width.
            orig_height: Original image height.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame must not be empty.")

        height, width = frame.shape[:2]

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, [0, 255] uint8 -> [0.0, 1.0] float32 tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
        return tensor.to(self.device), width, height

    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Execute full detection pipeline on a single frame.

        Measures preprocessing, raw inference, and postprocessing latencies.

        Args:
            frame: (H, W, 3) BGR frame.

        Returns:
            DetectionResult containing formatted detections and profiling timings.
        """
        t_start = time.perf_counter()

        # 1. Preprocessing
        t0_pre = time.perf_counter()
        tensor, orig_w, orig_h = self.preprocess(frame)
        t_pre = (time.perf_counter() - t0_pre) * 1000.0

        # 2. Raw Inference
        t0_inf = time.perf_counter()
        with torch.no_grad():
            outputs = self.model([tensor])
        t_inf = (time.perf_counter() - t0_inf) * 1000.0

        # 3. Postprocessing
        t0_post = time.perf_counter()
        raw_output = outputs[0]
        raw_boxes = raw_output["boxes"].detach().cpu().numpy()
        raw_scores = raw_output["scores"].detach().cpu().numpy()
        raw_labels = raw_output["labels"].detach().cpu().numpy()

        detections = filter_and_format_detections(
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_labels=raw_labels,
            class_names=self.class_names,
            image_width=orig_w,
            image_height=orig_h,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )
        t_post = (time.perf_counter() - t0_post) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(t_total, 1e-6)

        return DetectionResult(
            detections=detections,
            preprocessing_time_ms=t_pre,
            inference_time_ms=t_inf,
            postprocessing_time_ms=t_post,
            total_time_ms=t_total,
            fps=fps,
        )

    def warmup(self, runs: int = 3) -> None:
        """Warm up model with synthetic frames to stabilize timings."""
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(runs):
            _ = self.predict(dummy_frame)

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Render bounding boxes, class labels, and confidence tags on frame.

        Args:
            frame: BGR frame to draw on (modified in-place or returned).
            detections: List of Detection objects.
            color: BGR color for bounding boxes.
            thickness: Line thickness.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{det.class_name}: {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            # Background pill for readability
            bg_y1 = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(
                annotated,
                (x1, bg_y1),
                (x1 + text_w + 4, bg_y1 + text_h + baseline + 4),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 2, bg_y1 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated
