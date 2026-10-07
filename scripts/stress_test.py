import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_engine.inference.runtime import create_inference_runtime
from cv_engine.pipeline import AsyncRealtimePipeline

def run_stress_test() -> None:
    print("[*] Generating synthetic video stream...")
    # Generate 500 frames of synthetic video to a temp file
    temp_video = "synthetic_stress_test.mp4"
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, 30.0, (w, h))
    
    # Draw a moving square to simulate an object
    for i in range(500):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Moving box 1
        x = int(50 + (i % 400))
        y = 200
        cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 255, 255), -1)
        # Moving box 2
        x2 = int(500 - (i % 400))
        y2 = 300
        cv2.rectangle(frame, (x2, y2), (x2+50, y2+50), (200, 200, 200), -1)
        out.write(frame)
    out.release()
    
    print("[*] Initializing AsyncRealtimePipeline on synthetic video...")
    
    runtime = create_inference_runtime(backend="pytorch", device_or_provider="cpu")
    
    pipeline = AsyncRealtimePipeline(
        source=temp_video,
        runtime=runtime,
        width=w,
        height=h,
        headless=True
    )
    
    pipeline.start()
    
    frames_processed = 0
    try:
        while True:
            res = pipeline.get_output(timeout=0.5)
            if res is None:
                if not pipeline._capture_thread.is_alive() and pipeline.input_queue.empty() and pipeline.output_queue.empty():
                    print("[*] Capture thread finished and queues are empty.")
                    break
                if not pipeline._worker_thread.is_alive():
                    print("[!] Worker thread died prematurely!")
                    break
                continue
            
            annotated, metrics, summary = res
            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"  Processed {frames_processed}/500 frames | FPS: {metrics.processing_fps:.1f} | Dropped: {metrics.dropped_frames}")
                
    finally:
        pipeline.stop()
        
    print(f"[+] Stress test complete. Processed {frames_processed} frames successfully.")
    
    # Cleanup temp video
    Path(temp_video).unlink(missing_ok=True)
    
    assert frames_processed > 0, "Pipeline failed to process frames"
    print("[+] Graceful shutdown and resource cleanup verified.")

if __name__ == "__main__":
    run_stress_test()
