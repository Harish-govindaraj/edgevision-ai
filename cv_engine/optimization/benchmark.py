"""Edge AI model benchmarking suite comparing PyTorch, ONNX FP32, and ONNX INT8 runtimes."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from cv_engine.inference.runtime import BaseInferenceRuntime, create_inference_runtime
from cv_engine.optimization.export_onnx import export_detector_to_onnx
from cv_engine.optimization.quantization import quantize_onnx_model
from cv_engine.utils.system import get_system_metrics


def run_comprehensive_edge_benchmark(
    iterations: int = 25,
    warmup: int = 5,
    width: int = 640,
    height: int = 480,
    onnx_fp32_path: str = "models/ssdlite320_mobilenet_v3_large.onnx",
    onnx_int8_path: str = "models/ssdlite320_mobilenet_v3_large_int8.onnx",
    output_dir: str = "experiments/results",
) -> Dict[str, Any]:
    """
    Execute systematic edge benchmark across all available local runtimes.

    Runtimes evaluated:
      - PyTorch CPU (FP32)
      - PyTorch CUDA (FP32) — if torch.cuda.is_available() is True
      - ONNX Runtime CPU (FP32)
      - ONNX Runtime CPU (INT8)
      - ONNX Runtime CUDA — if CUDAExecutionProvider is present in ONNX Runtime

    Saves structured results to:
      experiments/results/edge_benchmark.json
      experiments/results/edge_benchmark.csv
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EdgeVision AI — Comprehensive Edge AI Inference Benchmark")
    print("=" * 65)

    # Step 1: Ensure ONNX FP32 model exists
    onnx_file = Path(onnx_fp32_path)
    if not onnx_file.exists():
        print(f"[*] ONNX FP32 model not found at {onnx_file}. Exporting now...")
        export_detector_to_onnx(output_path=str(onnx_file))
    fp32_size_mb = onnx_file.stat().st_size / (1024.0 * 1024.0)

    # Step 2: Ensure ONNX INT8 model exists
    int8_file = Path(onnx_int8_path)
    if not int8_file.exists():
        print(f"[*] ONNX INT8 model not found at {int8_file}. Quantizing now...")
        quantize_onnx_model(input_model_path=str(onnx_file), output_model_path=str(int8_file))
    int8_size_mb = int8_file.stat().st_size / (1024.0 * 1024.0)

    # PyTorch model size
    pt_size_mb = 13.6  # Standard torchvision SSDLite320 checkpoint

    # Prepare runtime configs to benchmark
    benchmark_targets: List[Dict[str, Any]] = [
        {
            "name": "PyTorch CPU",
            "backend": "pytorch",
            "provider": "cpu",
            "precision": "FP32",
            "model_path": None,
            "model_size_mb": pt_size_mb,
        },
    ]

    if torch.cuda.is_available():
        benchmark_targets.append(
            {
                "name": "PyTorch CUDA",
                "backend": "pytorch",
                "provider": "cuda",
                "precision": "FP32",
                "model_path": None,
                "model_size_mb": pt_size_mb,
            }
        )

    benchmark_targets.append(
        {
            "name": "ONNX Runtime CPU (FP32)",
            "backend": "onnx",
            "provider": "cpu",
            "precision": "FP32",
            "model_path": str(onnx_file),
            "model_size_mb": fp32_size_mb,
        }
    )

    benchmark_targets.append(
        {
            "name": "ONNX Runtime CPU (INT8)",
            "backend": "onnx",
            "provider": "cpu",
            "precision": "INT8",
            "model_path": str(int8_file),
            "model_size_mb": int8_size_mb,
        }
    )

    # Check for ONNX CUDA
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        benchmark_targets.append(
            {
                "name": "ONNX Runtime CUDA",
                "backend": "onnx",
                "provider": "cuda",
                "precision": "FP32",
                "model_path": str(onnx_file),
                "model_size_mb": fp32_size_mb,
            }
        )

    # Execute benchmarks
    benchmark_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for target in benchmark_targets:
        print(f"\n[*] Benchmarking runtime: {target['name']}...")
        rt = create_inference_runtime(
            backend=target["backend"],
            device_or_provider=target["provider"],
            model_path=target["model_path"],
            precision=target["precision"],
        )

        metrics = rt.benchmark(
            iterations=iterations,
            warmup=warmup,
            frame_shape=(height, width),
        )

        entry = {
            "runtime": target["name"],
            "backend": target["backend"],
            "provider": metrics["provider"],
            "precision": target["precision"],
            "model_size_mb": round(target["model_size_mb"], 2),
            "input_resolution": f"{width}x{height}",
            "iterations": iterations,
            "preprocessing": metrics["preprocessing_ms"],
            "inference": metrics["inference_ms"],
            "postprocessing": metrics["postprocessing_ms"],
            "total": metrics["total_ms"],
            "fps_p50": round(metrics["fps_p50"], 1),
            "system_metrics": metrics["system_metrics"],
        }
        benchmark_results.append(entry)

        csv_rows.append(
            {
                "runtime": target["name"],
                "precision": target["precision"],
                "provider": metrics["provider"],
                "preprocess_mean_ms": round(metrics["preprocessing_ms"]["mean"], 3),
                "inference_mean_ms": round(metrics["inference_ms"]["mean"], 3),
                "inference_p50_ms": round(metrics["inference_ms"]["p50"], 3),
                "inference_p95_ms": round(metrics["inference_ms"]["p95"], 3),
                "postprocess_mean_ms": round(metrics["postprocessing_ms"]["mean"], 3),
                "total_p50_ms": round(metrics["total_ms"]["p50"], 3),
                "total_p95_ms": round(metrics["total_ms"]["p95"], 3),
                "fps": round(metrics["fps_p50"], 1),
                "throughput": f"{round(metrics['fps_p50'], 1)} FPS",
                "model_size_mb": round(target["model_size_mb"], 2),
            }
        )

        print(
            f"    -> Inference P50: {metrics['inference_ms']['p50']:.2f} ms | "
            f"Total P50: {metrics['total_ms']['p50']:.2f} ms | "
            f"FPS: {metrics['fps_p50']:.1f} | Model Size: {target['model_size_mb']:.2f} MB"
        )

    # Save JSON
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": get_system_metrics().__dict__,
        "input_resolution": f"{width}x{height}",
        "iterations": iterations,
        "results": benchmark_results,
    }
    json_path = out_path / "edge_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[+] Saved edge benchmark JSON: {json_path}")

    # Save CSV
    df = pd.DataFrame(csv_rows)
    csv_path = out_path / "edge_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved edge benchmark CSV:  {csv_path}")

    print("\n" + "=" * 65)
    print("EDGE AI BENCHMARK SUMMARY TABLE:")
    print("=" * 65)
    print(df.to_string(index=False))

    return final_output
