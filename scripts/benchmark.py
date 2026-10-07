"""
Unified CLI entry point for EdgeVision AI inference benchmarking.

Benchmarks PyTorch and ONNX Runtime backends (FP32 and INT8) on identical workloads,
capturing latency percentiles, FPS, model footprint, and hardware metrics.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_engine.optimization.benchmark import run_comprehensive_edge_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EdgeVision AI Edge Optimization & Runtime Benchmark."
    )
    parser.add_argument("--iterations", type=int, default=25, help="Benchmark iterations per runtime")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    args = parser.parse_args()

    run_comprehensive_edge_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        width=args.width,
        height=args.height,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
