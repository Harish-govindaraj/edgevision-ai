import time
from collections import deque
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PipelineMetrics:
    """Snapshot of real-time pipeline telemetry."""
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    dropped_frames: int = 0
    input_queue_depth: int = 0
    
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    analytics_ms: float = 0.0
    rendering_ms: float = 0.0
    total_ms: float = 0.0
    
    # Statistical percentiles for total latency
    total_p50_ms: float = 0.0
    total_p95_ms: float = 0.0

class MetricsTracker:
    """Maintains rolling statistics for pipeline performance."""
    
    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        
        self.total_latencies = deque(maxlen=window_size)
        self.inference_latencies = deque(maxlen=window_size)
        self.tracking_latencies = deque(maxlen=window_size)
        self.analytics_latencies = deque(maxlen=window_size)
        self.rendering_latencies = deque(maxlen=window_size)
        
        self.dropped_frames: int = 0
        
        # FPS Tracking
        self._last_capture_time = time.perf_counter()
        self._capture_frame_count = 0
        self._capture_fps = 0.0
        
        self._last_process_time = time.perf_counter()
        self._process_frame_count = 0
        self._process_fps = 0.0

    def record_capture(self) -> None:
        """Called when a frame is read from the camera."""
        self._capture_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_capture_time
        if elapsed > 1.0:
            self._capture_fps = self._capture_frame_count / elapsed
            self._capture_frame_count = 0
            self._last_capture_time = now

    def record_process(self) -> None:
        """Called when a frame finishes full processing."""
        self._process_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_process_time
        if elapsed > 1.0:
            self._process_fps = self._process_frame_count / elapsed
            self._process_frame_count = 0
            self._last_process_time = now

    def record_drop(self) -> None:
        """Called when a frame is dropped due to queue full."""
        self.dropped_frames += 1

    def record_latencies(
        self,
        inference_ms: float,
        tracking_ms: float,
        analytics_ms: float,
        rendering_ms: float,
        total_ms: float,
    ) -> None:
        self.inference_latencies.append(inference_ms)
        self.tracking_latencies.append(tracking_ms)
        self.analytics_latencies.append(analytics_ms)
        self.rendering_latencies.append(rendering_ms)
        self.total_latencies.append(total_ms)

    def get_snapshot(self, queue_depth: int = 0) -> PipelineMetrics:
        """Calculate and return the current snapshot of metrics."""
        def safe_mean(dq: deque) -> float:
            return float(np.mean(dq)) if len(dq) > 0 else 0.0
            
        def safe_percentile(dq: deque, p: float) -> float:
            return float(np.percentile(dq, p)) if len(dq) > 0 else 0.0

        return PipelineMetrics(
            capture_fps=self._capture_fps,
            processing_fps=self._process_fps,
            dropped_frames=self.dropped_frames,
            input_queue_depth=queue_depth,
            inference_ms=safe_mean(self.inference_latencies),
            tracking_ms=safe_mean(self.tracking_latencies),
            analytics_ms=safe_mean(self.analytics_latencies),
            rendering_ms=safe_mean(self.rendering_latencies),
            total_ms=safe_mean(self.total_latencies),
            total_p50_ms=safe_percentile(self.total_latencies, 50),
            total_p95_ms=safe_percentile(self.total_latencies, 95),
        )
