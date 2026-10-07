# Edge AI Benchmark Summary
*Generated on 2026-09-10*

## Hardware Context
- **CPU**: Host CPU
- **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Not active in this run due to PyTorch CPU-only installation)
- **Target OS**: Windows

## Model Details
- **Architecture**: SSDLite320 MobileNetV3 Large
- **Original Format**: PyTorch (FP32)
- **Export Format**: ONNX
- **Quantization**: Dynamic INT8 (ONNX)

## Benchmark Results (CPU)

| Runtime | Precision | Provider | Mean Inference (ms) | P50 Total Latency (ms) | FPS | Model Size (MB) |
|---------|-----------|----------|---------------------|------------------------|-----|-----------------|
| PyTorch | FP32      | CPU      | 52.61              | 52.39                 | 19.1| 13.60           |
| ONNX    | FP32      | CPU (EP) | 111.17             | 113.04                | 8.8 | 13.46           |
| ONNX    | INT8      | CPU (EP) | 137.12             | 138.82                | 7.2 | 4.11            |

## Key Findings & Observations

1. **Quantization Impact**: Dynamic INT8 quantization successfully reduced the model size by ~70% (from 13.6MB down to 4.1MB).
2. **CPU Inference Performance**: The PyTorch runtime is significantly faster on this specific CPU architecture compared to ONNX Runtime. PyTorch achieves ~19.1 FPS, while ONNX FP32 drops to ~8.8 FPS and INT8 drops further to ~7.2 FPS.
3. **ONNX INT8 Overhead**: As often observed with dynamic INT8 quantization on certain x86/x64 CPUs lacking specific native instruction sets or facing heavy operator conversion overheads in ONNX Runtime, the INT8 model is slower than the FP32 model. This demonstrates a common Edge AI tradeoff: smaller memory footprint vs. higher latency.
4. **CUDA Compatibility**: Real GPU acceleration was bypassed in this benchmark run because `torch.cuda.is_available() == False` in the current virtual environment. The PyTorch build is CPU-only. To utilize the RTX 3050, a CUDA-enabled PyTorch wheel is required.

## Next Steps
- For deployment on memory-constrained Edge devices, the INT8 ONNX model provides a substantial reduction in storage/RAM requirements.
- For maximum throughput on the current CPU, PyTorch FP32 is recommended.
- To unlock the NVIDIA RTX 3050 Laptop GPU, the environment must be updated with a CUDA-compatible PyTorch build, after which a GPU-specific benchmark should be executed.
