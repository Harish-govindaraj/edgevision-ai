# Real-Time Pipeline Benchmark Summary

*Measured on Host CPU*

| Runtime | Res | Processed | Cap FPS | Proc FPS | Dropped | P50 (ms) | P95 (ms) | Inf (ms) | Track (ms) |
|---|---|---|---|---|---|---|---|---|---|
| PYTORCH CPU (FP32) | 640x480 | 165 | 30.0 | 16.6 | 135 | 60.38 | 66.47 | 60.02 | 0.01 |
| ONNX CPU (FP32) | 640x480 | 70 | 30.0 | 6.9 | 230 | 145.28 | 154.3 | 144.98 | 0.01 |
| ONNX CPU (INT8) | 640x480 | 67 | 30.0 | 6.6 | 233 | 148.6 | 175.79 | 150.44 | 0.01 |
