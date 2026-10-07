"""ONNX Runtime inference engine with provider auto-detection and graceful fallback."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import onnxruntime as ort

from cv_engine.detection.detector import DetectionResult
from cv_engine.detection.postprocessing import Detection, filter_and_format_detections

logger = logging.getLogger(__name__)


class ONNXDetector:
    """
    High-performance ONNX Runtime inference wrapper for edge object detection.

    Supports CPUExecutionProvider and CUDAExecutionProvider with dynamic provider
    negotiation, graceful fallback, and standardized output formatting.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/ssdlite320_mobilenet_v3_large.onnx",
        provider: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Resolve execution provider
        self.actual_provider = self._resolve_provider(provider)

        # Initialize ONNX Runtime Session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=[self.actual_provider],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Standard COCO category mapping
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

        categories = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.meta.get("categories", [])
        self.class_names: Dict[int, str] = {i: cat for i, cat in enumerate(categories)}

    def _resolve_provider(self, requested_provider: str) -> str:
        """Negotiate requested provider against available ONNX Runtime providers."""
        available = ort.get_available_providers()
        req = requested_provider.lower().strip()

        if req in ("auto", "cuda", "cudaexecutionprovider"):
            if "CUDAExecutionProvider" in available:
                logger.info("Selected ONNX CUDAExecutionProvider.")
                return "CUDAExecutionProvider"
            else:
                if req in ("cuda", "cudaexecutionprovider"):
                    logger.warning(
                        "CUDAExecutionProvider requested but not available in current onnxruntime build. "
                        "Falling back to CPUExecutionProvider."
                    )
                return "CPUExecutionProvider"
        elif req in ("cpu", "cpuexecutionprovider"):
            return "CPUExecutionProvider"
        elif req in ("tensorrt", "tensorrtexecutionprovider"):
            if "TensorrtExecutionProvider" in available:
                return "TensorrtExecutionProvider"
            logger.warning("TensorrtExecutionProvider unavailable. Falling back to CPUExecutionProvider.")
            return "CPUExecutionProvider"
        else:
            return "CPUExecutionProvider"

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Convert BGR frame to 320x320 normalized float32 tensor for ONNX inference.

        Returns:
            array: Shape (3, 320, 320) float32 array.
            orig_w: Original frame width.
            orig_h: Original frame height.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame must not be empty.")

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return tensor, width, height

    def predict(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference using ONNX Runtime and measure latency breakdown.

        Args:
            frame: (H, W, 3) BGR image array.

        Returns:
            DetectionResult with standardized detections and profiling metrics.
        """
        t_start = time.perf_counter()

        # 1. Preprocessing
        t0_pre = time.perf_counter()
        inp, orig_w, orig_h = self.preprocess(frame)
        t_pre = (time.perf_counter() - t0_pre) * 1000.0

        # 2. ONNX Runtime Inference
        t0_inf = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: inp})
        t_inf = (time.perf_counter() - t0_inf) * 1000.0

        # 3. Postprocessing
        t0_post = time.perf_counter()
        raw_boxes = outputs[0].copy()
        raw_scores = outputs[1]
        raw_labels = outputs[2]

        # Rescale boxes from 320x320 model space to original image coordinates
        scale_x = orig_w / 320.0
        scale_y = orig_h / 320.0
        raw_boxes[:, 0] *= scale_x
        raw_boxes[:, 1] *= scale_y
        raw_boxes[:, 2] *= scale_x
        raw_boxes[:, 3] *= scale_y

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
        """Warmup ONNX session."""
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(runs):
            _ = self.predict(dummy)
