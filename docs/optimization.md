# Edge Optimization

This document outlines the optimization strategies implemented for Edge AI deployment.

## ONNX Export

The Object Detection model (SSDLite320 MobileNetV3 Large) is exported to ONNX format to decouple the runtime from the PyTorch training framework. This enables lightweight inference engines such as ONNX Runtime to execute the model.

- **Inputs**: (1, 3, 320, 320) FP32 image tensor.
- **Outputs**: Bounding boxes, scores, and class labels.
- **Export Method**: `torch.onnx.export` (dynamo=False due to data-dependent dynamic features).

## Quantization

To reduce the model footprint and memory bandwidth requirements on Edge devices, we employ **Dynamic INT8 Quantization**.

- **Technique**: ONNX Runtime's `quantize_dynamic` (Dynamic quantization of weight matrices for Linear/MatMul operations).
- **Size Reduction**: ~13.6MB (FP32) -> ~4.1MB (INT8), achieving a 70% reduction in storage and memory loading requirements.

*Note*: On certain x86 CPU architectures without native VNNI/INT8 instruction sets, INT8 quantization can introduce operator overhead that increases latency, even while saving memory. Refer to the [Benchmarking](benchmarking.md) docs for performance metrics.

## Supported Runtimes

1. **PyTorchRuntime**: Native execution (FP32).
2. **ONNXRuntime**: ONNX execution using `CPUExecutionProvider` or (if configured) `CUDAExecutionProvider` or `TensorrtExecutionProvider`. Supports both FP32 and INT8 models.
