# Benchmarking & Performance

This document describes how to benchmark the EdgeVision AI models and presents the most recent results.

## Benchmark Script

You can benchmark the different runtimes using the provided script:

```bash
python scripts/benchmark.py
```

This script will evaluate:
- PyTorch CPU (FP32)
- ONNX CPU (FP32)
- ONNX CPU (INT8)

It outputs results to `experiments/results/edge_benchmark.csv` and JSON.

## Latest Results (CPU-only Host)

*Device: Host CPU (x86_64), PyTorch CPU build, ONNX Runtime CPU execution provider.*

| Runtime             | Precision | Mean Inference (ms) | P50 Total Latency | Throughput | Model Size |
|---------------------|-----------|---------------------|-------------------|------------|------------|
| **PyTorch**         | FP32      | 52.61 ms           | 52.39 ms          | 19.1 FPS   | 13.60 MB   |
| **ONNX Runtime**    | FP32      | 111.17 ms          | 113.04 ms         | 8.8 FPS    | 13.46 MB   |
| **ONNX Runtime**    | INT8      | 137.12 ms          | 138.82 ms         | 7.2 FPS    | 4.11 MB    |

### Key Takeaways

- **Storage**: INT8 Quantization provides a ~70% size reduction (13.6MB -> 4.1MB), critical for constrained Edge devices.
- **Latency**: On this specific x86 CPU architecture, the PyTorch FP32 runtime is highly optimized, whereas the ONNX Runtime with Dynamic INT8 incurs conversion overheads, resulting in higher latency.
- **Hardware Acceleration**: The host machine contains an NVIDIA GeForce RTX 3050 Laptop GPU, but the current PyTorch/ONNX runtime installation is CPU-only (`torch.cuda.is_available() == False`). To utilize GPU acceleration, CUDA-compatible wheels must be installed in the environment.
