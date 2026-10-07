"""
Real-time object detection, multi-object tracking, and analytics runner for EdgeVision AI.

Pipeline:
  Frame Acquisition (Async) -> Preprocessing -> SSDLite320 Detector -> Centroid Tracker
  -> Trajectory History -> Motion Analytics / Tripwire -> Telemetry HUD Rendering
"""

import argparse
from pathlib import Path
import sys
from typing import Union
import cv2
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_engine.inference.runtime import create_inference_runtime
from cv_engine.pipeline import AsyncRealtimePipeline
from cv_engine.utils.metrics_tracker import PipelineMetrics

def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for real-time video execution."""
    parser = argparse.ArgumentParser(
        description="EdgeVision AI — Real-Time Detection & Multi-Object Tracking Runner"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g. 0) or path to video file (e.g. video.mp4)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Processing frame width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Processing frame height (default: 480)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Confidence threshold for detection (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: 'auto', 'cpu', or 'cuda' (default: auto)",
    )
    parser.add_argument(
        "--tripwire-y",
        type=int,
        default=0,
        help="Horizontal tripwire y-position (0 = centered at height // 2)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before exiting (0 = infinite)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying cv2.imshow GUI window (useful for headless servers/testing)",
    )
    return parser.parse_args()


def parse_source(source_str: str) -> Union[int, str]:
    """Convert integer camera index strings to int, preserving file paths."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def draw_hud(
    frame: np.ndarray,
    metrics: PipelineMetrics,
    active_count: int,
    total_unique: int,
    tripwire_in: int,
    tripwire_out: int,
    device_name: str,
) -> np.ndarray:
    """Render a clean multi-line telemetry Heads-Up-Display (HUD) on frame top."""
    hud = frame.copy()
    hud_h = 75
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)

    # Line 1: Title & Real-time FPS
    cv2.putText(
        hud,
        "EdgeVision AI | AsyncRealtimePipeline",
        (15, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    fps_color = (0, 255, 0) if metrics.processing_fps >= 15 else (0, 165, 255)
    cv2.putText(
        hud,
        f"FPS: {metrics.processing_fps:.1f}",
        (frame.shape[1] - 125, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    # Line 2: Latency breakdown
    stats_text = (
        f"Det: {metrics.inference_ms:.1f}ms | Track: {metrics.tracking_ms:.2f}ms | Total: {metrics.total_ms:.1f}ms | Device: {device_name}"
    )
    cv2.putText(
        hud,
        stats_text,
        (15, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Line 3: Tracking & Tripwire counters
    counts_text = (
        f"Active: {active_count} | Seen: {total_unique} | "
        f"Drop: {metrics.dropped_frames} | In={tripwire_in}, Out={tripwire_out}"
    )
    cv2.putText(
        hud,
        counts_text,
        (15, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 255, 100),
        1,
        cv2.LINE_AA,
    )
    return hud


def run_video_pipeline(
    source: Union[int, str],
    width: int = 640,
    height: int = 480,
    confidence_threshold: float = 0.45,
    device: str = "auto",
    tripwire_y: int = 0,
    max_frames: int = 0,
    headless: bool = False,
) -> int:
    source_val = parse_source(str(source))

    print(f"[*] Initializing Runtime on device '{device}'...")
    runtime = create_inference_runtime(
        backend="pytorch",
        device_or_provider=device,
        confidence_threshold=confidence_threshold
    )

    pipeline = AsyncRealtimePipeline(
        source=source_val,
        runtime=runtime,
        width=width,
        height=height,
        tripwire_y=tripwire_y,
        headless=headless
    )

    print(f"[*] Starting Async Real-Time Pipeline from source: {source}")
    print("    Press 'q' in the display window to stop.")
    
    pipeline.start()
    
    frame_count = 0
    
    try:
        while True:
            res = pipeline.get_output(timeout=0.5)
            if res is None:
                # Check if pipeline threads have died indicating stream end
                if not pipeline._capture_thread.is_alive() and pipeline.input_queue.empty() and pipeline.output_queue.empty():
                    print("[*] Stream ended or capture thread died.")
                    break
                continue
                
            annotated_frame, metrics, summary = res
            
            display_frame = draw_hud(
                annotated_frame,
                metrics=metrics,
                active_count=summary.get("active_tracks_count", 0),
                total_unique=summary.get("total_unique_observed", 0),
                tripwire_in=summary.get("tripwire_in", 0),
                tripwire_out=summary.get("tripwire_out", 0),
                device_name=runtime.provider_name
            )

            frame_count += 1

            if not headless:
                cv2.imshow("EdgeVision AI — Async Pipeline", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[*] User quit requested.")
                    break

            if 0 < max_frames <= frame_count:
                print(f"[*] Processed requested frame limit ({max_frames}).")
                break
    except KeyboardInterrupt:
        print("[*] Interrupted by user.")
    finally:
        pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[+] Complete. Rendered {frame_count} frames.")
    return frame_count


def main() -> None:
    args = parse_arguments()
    run_video_pipeline(
        source=args.source,
        width=args.width,
        height=args.height,
        confidence_threshold=args.confidence,
        device=args.device,
        tripwire_y=args.tripwire_y,
        max_frames=args.max_frames,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
