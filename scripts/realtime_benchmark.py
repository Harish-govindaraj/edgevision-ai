import time
import json
import csv
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_engine.inference.runtime import create_inference_runtime
from cv_engine.pipeline import AsyncRealtimePipeline
from cv_engine.utils.system import get_system_metrics

def run_realtime_benchmark() -> None:
    print("[*] Generating synthetic video stream for benchmark...")
    temp_video = "synthetic_realtime_bench.mp4"
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, 30.0, (w, h))
    
    frames_to_generate = 300
    for i in range(frames_to_generate):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(50 + (i % 400))
        y = 200
        cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    
    results = []
    
    configs = [
        {"backend": "pytorch", "precision": "FP32"},
        {"backend": "onnx", "precision": "FP32"},
        {"backend": "onnx", "precision": "INT8"},
    ]
    
    print("\n[*] Starting Real-time Benchmark...")
    
    for cfg in configs:
        b = cfg["backend"]
        p = cfg["precision"]
        print(f"\nEvaluating: {b.upper()} ({p})")
        
        kwargs = {
            "backend": b,
            "device_or_provider": "cpu",
            "precision": p
        }
        if b == "onnx":
            if p == "INT8":
                kwargs["model_path"] = "models/ssdlite320_mobilenet_v3_large_int8.onnx"
            else:
                kwargs["model_path"] = "models/ssdlite320_mobilenet_v3_large.onnx"
                
        runtime = create_inference_runtime(**kwargs)
        
        pipeline = AsyncRealtimePipeline(
            source=temp_video,
            runtime=runtime,
            width=w,
            height=h,
            headless=True
        )
        
        pipeline.start()
        
        frames_processed = 0
        final_metrics = None
        
        t0 = time.perf_counter()
        
        try:
            while True:
                res = pipeline.get_output(timeout=0.5)
                if res is None:
                    if not pipeline._capture_thread.is_alive() and pipeline.input_queue.empty() and pipeline.output_queue.empty():
                        break
                    continue
                
                _, metrics, _ = res
                frames_processed += 1
                final_metrics = metrics
        finally:
            pipeline.stop()
            
        elapsed = time.perf_counter() - t0
        
        if final_metrics:
            sys_met = get_system_metrics()
            
            row = {
                "runtime": f"{b.upper()} CPU ({p})",
                "resolution": f"{w}x{h}",
                "processed_frames": frames_processed,
                "capture_fps": round(final_metrics.capture_fps, 1),
                "processing_fps": round(final_metrics.processing_fps, 1),
                "dropped_frames": final_metrics.dropped_frames,
                "total_p50_ms": round(final_metrics.total_p50_ms, 2),
                "total_p95_ms": round(final_metrics.total_p95_ms, 2),
                "inference_mean_ms": round(final_metrics.inference_ms, 2),
                "tracking_mean_ms": round(final_metrics.tracking_ms, 2),
                "cpu_util_pct": sys_met.cpu_utilization_percent
            }
            results.append(row)
            
    # Cleanup temp video
    Path(temp_video).unlink(missing_ok=True)
    
    # Save results
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "realtime_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    json_path = out_dir / "realtime_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    md_path = out_dir / "realtime_benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write("# Real-Time Pipeline Benchmark Summary\n\n")
        f.write("*Measured on Host CPU*\n\n")
        f.write("| Runtime | Res | Processed | Cap FPS | Proc FPS | Dropped | P50 (ms) | P95 (ms) | Inf (ms) | Track (ms) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['runtime']} | {r['resolution']} | {r['processed_frames']} | {r['capture_fps']} | {r['processing_fps']} | {r['dropped_frames']} | {r['total_p50_ms']} | {r['total_p95_ms']} | {r['inference_mean_ms']} | {r['tracking_mean_ms']} |\n")
            
    print(f"\n[+] Benchmark complete. Saved to {out_dir}")

if __name__ == "__main__":
    run_realtime_benchmark()
