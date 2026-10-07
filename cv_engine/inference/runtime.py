"""Unified inference runtime abstraction supporting PyTorch and ONNX execution backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from cv_engine.detection.detector import DetectionResult, ObjectDetector
from cv_engine.detection.postprocessing import Detection
from cv_engine.inference.onnx_inference import ONNXDetector
from cv_engine.utils.system import get_system_metrics


class BaseInferenceRuntime(ABC):
    """Abstract interface decoupling detection backends from consumers."""

    @abstractmethod
    def load(self) -> None:
        """Initialize or load underlying model weights/graph."""
        pass

    @abstractmethod
    def warmup(self, runs: int = 3) -> None:
        """Warm up execution caches and runtime buffers."""
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a single BGR frame."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the inference runtime engine (e.g., 'PyTorch', 'ONNX Runtime')."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Underlying hardware execution provider (e.g., 'CPU', 'CUDAExecutionProvider')."""
        pass

    @property
    @abstractmethod
    def precision(self) -> str:
        """Model numerical precision ('FP32', 'FP16', 'INT8')."""
        pass

    def benchmark(
        self,
        iterations: int = 30,
        warmup: int = 5,
        frame_shape: Tuple[int, int] = (480, 640),
    ) -> Dict[str, Any]:
        """
        Execute profiling benchmark on synthetic frames.

        Measures preprocessing, raw inference, postprocessing, total latencies,
        percentiles (P50, P95), FPS, and system metrics.
        """
        self.warmup(runs=warmup)

        h, w = frame_shape
        dummy_frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        pre_ms: List[float] = []
        inf_ms: List[float] = []
        post_ms: List[float] = []
        tot_ms: List[float] = []

        for _ in range(iterations):
            # Synchronize CUDA if active
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            res = self.predict(dummy_frame)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            pre_ms.append(res.preprocessing_time_ms)
            inf_ms.append(res.inference_time_ms)
            post_ms.append(res.postprocessing_time_ms)
            tot_ms.append(res.total_time_ms)

        def calc_stats(vals: List[float]) -> Dict[str, float]:
            arr = np.array(vals)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        sys_metrics = get_system_metrics()

        tot_stats = calc_stats(tot_ms)
        inf_stats = calc_stats(inf_ms)
        fps = 1000.0 / max(tot_stats["p50"], 1e-6)

        return {
            "backend": self.backend_name,
            "provider": self.provider_name,
            "precision": self.precision,
            "iterations": iterations,
            "input_resolution": f"{w}x{h}",
            "preprocessing_ms": calc_stats(pre_ms),
            "inference_ms": inf_stats,
            "postprocessing_ms": calc_stats(post_ms),
            "total_ms": tot_stats,
            "fps_p50": float(fps),
            "system_metrics": {
                "cpu_util_pct": sys_metrics.cpu_utilization_percent,
                "ram_used_mb": sys_metrics.ram_used_mb,
                "gpu_name": sys_metrics.gpu_name,
                "gpu_util_pct": sys_metrics.gpu_utilization_percent,
                "gpu_memory_used_mb": sys_metrics.gpu_memory_used_mb,
            },
        }


class PyTorchRuntime(BaseInferenceRuntime):
    """Inference runtime backed by native PyTorch."""

    def __init__(
        self,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> None:
        self.requested_device = device
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.detector: Optional[ObjectDetector] = None

    def load(self) -> None:
        self.detector = ObjectDetector(
            confidence_threshold=self.conf,
            iou_threshold=self.iou,
            device=self.requested_device,
            pretrained=True,
        )

    def warmup(self, runs: int = 3) -> None:
        if self.detector is None:
            self.load()
        self.detector.warmup(runs=runs)

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.detector is None:
            self.load()
        return self.detector.predict(frame)

    @property
    def backend_name(self) -> str:
        return "PyTorch"

    @property
    def provider_name(self) -> str:
        if self.detector is None:
            return "CPU"
        return "CUDA" if self.detector.device.type == "cuda" else "CPU"

    @property
    def precision(self) -> str:
        return "FP32"


class ONNXRuntime(BaseInferenceRuntime):
    """Inference runtime backed by ONNX Runtime."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/ssdlite320_mobilenet_v3_large.onnx",
        provider: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        precision: str = "FP32",
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_provider = provider
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self._precision = precision
        self.detector: Optional[ONNXDetector] = None

    def load(self) -> None:
        self.detector = ONNXDetector(
            model_path=self.model_path,
            provider=self.requested_provider,
            confidence_threshold=self.conf,
            iou_threshold=self.iou,
        )

    def warmup(self, runs: int = 3) -> None:
        if self.detector is None:
            self.load()
        self.detector.warmup(runs=runs)

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.detector is None:
            self.load()
        return self.detector.predict(frame)

    @property
    def backend_name(self) -> str:
        return "ONNX Runtime"

    @property
    def provider_name(self) -> str:
        if self.detector is None:
            return "CPUExecutionProvider"
        return self.detector.actual_provider

    @property
    def precision(self) -> str:
        return self._precision


def create_inference_runtime(
    backend: str = "pytorch",
    device_or_provider: str = "auto",
    model_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    precision: str = "FP32",
) -> BaseInferenceRuntime:
    """Factory creating configured InferenceRuntime instance."""
    b = backend.lower().strip()
    if b in ("pytorch", "torch"):
        rt = PyTorchRuntime(
            device=device_or_provider,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
    elif b in ("onnx", "onnxruntime", "ort"):
        m_path = model_path or "models/ssdlite320_mobilenet_v3_large.onnx"
        rt = ONNXRuntime(
            model_path=m_path,
            provider=device_or_provider,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            precision=precision,
        )
    else:
        raise ValueError(f"Unsupported inference backend: '{backend}'. Expected 'pytorch' or 'onnx'.")

    rt.load()
    return rt
