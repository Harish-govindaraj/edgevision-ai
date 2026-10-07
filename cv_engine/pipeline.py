"""Core real-time asynchronous computer vision pipeline."""

import time
import queue
import threading
import sys
from typing import Optional, Union, Dict, Any, Tuple
import cv2
import numpy as np

from cv_engine.inference.runtime import BaseInferenceRuntime
from cv_engine.tracking.tracker import CentroidTracker
from cv_engine.tracking.analytics import TrackingAnalytics, Tripwire
from cv_engine.utils.metrics_tracker import MetricsTracker, PipelineMetrics


class AsyncRealtimePipeline:
    """
    Bounded producer-consumer pipeline for true zero-lag real-time inference.
    """

    def __init__(
        self,
        source: Union[int, str],
        runtime: BaseInferenceRuntime,
        width: int = 640,
        height: int = 480,
        enable_tracking: bool = True,
        tripwire_y: int = 0,
        headless: bool = False,
    ) -> None:
        self.source = source
        self.runtime = runtime
        self.width = width
        self.height = height
        self.enable_tracking = enable_tracking
        self.headless = headless

        # Threading and synchronization
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        # Bounded Queues (size=1 to drop old frames and prevent lag)
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=2)

        # State & Metrics
        self.metrics_tracker = MetricsTracker()

        # CV Components
        self.tracker = CentroidTracker(max_distance=60.0, max_missed_frames=12, max_history_length=25)

        tw_y = tripwire_y if tripwire_y > 0 else (height // 2)
        self.tripwire = Tripwire(orientation="horizontal", position=tw_y)
        self.analytics = TrackingAnalytics(tripwire=self.tripwire)
        self.summary_stats = {
            "active_tracks_count": 0,
            "total_unique_observed": 0,
            "tripwire_in": 0,
            "tripwire_out": 0
        }

    def start(self) -> None:
        """Start the pipeline threads."""
        self._stop_event.clear()
        self._capture_finished = False

        self.runtime.warmup(runs=1)

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="WorkerThread")

        self._capture_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        """Gracefully stop the pipeline threads."""
        self._stop_event.set()

        # Unblock queues
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass

        try:
            self.output_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        """Producer: Read from source and push to input queue."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[!] Error: Could not open source {self.source}", file=sys.stderr)
            self._stop_event.set()
            return

        # Optional file throttle to prevent blowing through test videos instantly
        is_file = isinstance(self.source, str) and not str(self.source).isdigit()

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            success, raw_frame = cap.read()
            if not success or raw_frame is None:
                # End of file or camera disconnect
                self._capture_finished = True
                break

            self.metrics_tracker.record_capture()

            # Explicit Drop Strategy: If queue is full, pull out the old frame and put the new one
            try:
                self.input_queue.put_nowait(raw_frame)
            except queue.Full:
                self.metrics_tracker.record_drop()
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.input_queue.put_nowait(raw_frame)
                except queue.Full:
                    pass

            if is_file:
                # Throttle to roughly 30 FPS to simulate real-time camera
                elapsed = time.perf_counter() - t0
                if elapsed < 0.033:
                    time.sleep(0.033 - elapsed)

        cap.release()

    def _worker_loop(self) -> None:
        """Consumer: Run inference, tracking, analytics."""
        while not self._stop_event.is_set():
            if self._capture_finished and self.input_queue.empty():
                break

            try:
                try:
                    raw_frame = self.input_queue.get(timeout=0.5)
                    if raw_frame is None:
                        continue
                except queue.Empty:
                    continue

                t_start = time.perf_counter()

                # 0. Resize
                if raw_frame.shape[1] != self.width or raw_frame.shape[0] != self.height:
                    frame = cv2.resize(raw_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                else:
                    frame = raw_frame

                # 1. Inference
                t0_inf = time.perf_counter()
                try:
                    det_result = self.runtime.predict(frame)
                except Exception as e:
                    print(f"[!] Inference failed: {e}", file=sys.stderr)
                    det_result = None
                inf_ms = (time.perf_counter() - t0_inf) * 1000.0

                # 2. Tracking
                t0_track = time.perf_counter()
                tracks = []
                if det_result and self.enable_tracking:
                    try:
                        tracks = self.tracker.update(det_result.detections)
                    except Exception as e:
                        print(f"[!] Tracking failed: {e}", file=sys.stderr)
                track_ms = (time.perf_counter() - t0_track) * 1000.0

                # 3. Analytics
                t0_ana = time.perf_counter()
                if self.enable_tracking:
                    try:
                        self.summary_stats = self.analytics.update(tracks)
                    except Exception as e:
                        print(f"[!] Analytics failed: {e}", file=sys.stderr)
                ana_ms = (time.perf_counter() - t0_ana) * 1000.0

                # 4. Rendering
                t0_ren = time.perf_counter()
                annotated = frame.copy()
                if self.enable_tracking:
                    annotated = TrackingAnalytics.draw_tracks(annotated, tracks, draw_trajectory=True)
                    annotated = self.tripwire.draw(annotated)
                elif det_result:
                    # Just draw detections
                    for d in det_result.detections:
                        x1, y1, x2, y2 = map(int, d.bbox)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"{d.class_name} {d.confidence:.2f}"
                        cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                ren_ms = (time.perf_counter() - t0_ren) * 1000.0

                tot_ms = (time.perf_counter() - t_start) * 1000.0

                # Commit metrics
                self.metrics_tracker.record_latencies(
                    inference_ms=inf_ms,
                    tracking_ms=track_ms,
                    analytics_ms=ana_ms,
                    rendering_ms=ren_ms,
                    total_ms=tot_ms
                )
                self.metrics_tracker.record_process()

                snapshot = self.metrics_tracker.get_snapshot(queue_depth=self.input_queue.qsize())

                try:
                    self.output_queue.put((annotated, snapshot, self.summary_stats), timeout=1.0)
                except queue.Full:
                    pass
            except Exception as e:
                print(f"[WORKER LOOP FATAL EXCEPTION] {e}", flush=True)
                self._stop_event.set()
                break

    def get_output(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, PipelineMetrics, Dict[str, Any]]]:
        """Fetch the next processed frame from the output queue."""
        try:
            res = self.output_queue.get(timeout=timeout)
            if res is None:
                return None
            return res
        except queue.Empty:
            return None
