"""Unit tests for the real-time video detection runner."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from scripts.run_video import draw_hud, parse_source, run_video_pipeline


def test_parse_source() -> None:
    """Test camera index and video filepath parsing."""
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("sample.mp4") == "sample.mp4"
    assert parse_source("/path/to/video.avi") == "/path/to/video.avi"


def test_draw_hud() -> None:
    """Test HUD telemetry rendering produces valid image array."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    from cv_engine.utils.metrics_tracker import PipelineMetrics
    metrics = PipelineMetrics(
        processing_fps=24.5,
        inference_ms=40.2,
        tracking_ms=0.15,
        total_ms=45.1,
        dropped_frames=2
    )
    hud_frame = draw_hud(
        frame=frame,
        metrics=metrics,
        active_count=3,
        total_unique=5,
        tripwire_in=2,
        tripwire_out=1,
        device_name="CPU",
    )
    assert hud_frame.shape == frame.shape
    # Ensure HUD modified top pixels
    assert np.any(hud_frame[:75, :] > 0)


def test_run_video_graceful_failure_on_invalid_source() -> None:
    """Test that non-existent video source exits gracefully without crashing."""
    frames = run_video_pipeline(
        source="non_existent_stream_path_12345.mp4",
        headless=True,
    )
    assert frames == 0


def test_run_video_synthetic_file() -> None:
    """Test real video processing on a generated synthetic video file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "synthetic.mp4"

        # Create a synthetic 10-frame video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Draw moving white rectangle
            cv2.rectangle(frame, (i * 10, 50), (i * 10 + 60, 150), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        # Run pipeline in headless mode for 4 frames
        processed = run_video_pipeline(
            source=str(video_path),
            width=320,
            height=240,
            max_frames=4,
            headless=True,
        )
        assert processed == 4
