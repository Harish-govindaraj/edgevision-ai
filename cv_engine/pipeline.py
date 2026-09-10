"""Core real-time computer vision pipeline."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from cv_engine.preprocessing.filters import preprocess_frame
from cv_engine.utils.fps import FPSMeter


@dataclass
class PipelineResult:
    """Output generated for each processed frame."""

    frame: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray
    fps: float
    processing_time_ms: float


class VisionPipeline:
    """Execute the EdgeVision MVP preprocessing pipeline."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.width = width
        self.height = height
        self.fps_meter = FPSMeter()

    def process(self, frame: np.ndarray) -> PipelineResult:
        """Process one camera frame."""
        start_time = time.perf_counter()

        resized, grayscale, edges = preprocess_frame(
            frame,
            width=self.width,
            height=self.height,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        fps = self.fps_meter.update()

        return PipelineResult(
            frame=resized,
            grayscale=grayscale,
            edges=edges,
            fps=fps,
            processing_time_ms=processing_time_ms,
        )

    def create_display(self, result: PipelineResult) -> np.ndarray:
        """Create a side-by-side visualization of the pipeline output."""

        edges_bgr = cv2.cvtColor(
            result.edges,
            cv2.COLOR_GRAY2BGR,
        )

        display = np.hstack(
            [
                result.frame,
                edges_bgr,
            ]
        )

        cv2.putText(
            display,
            f"FPS: {result.fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"Latency: {result.processing_time_ms:.2f} ms",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return display