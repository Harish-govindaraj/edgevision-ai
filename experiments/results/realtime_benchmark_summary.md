# Real-Time Pipeline Benchmark Summary

*Measured on Host CPU*

| Runtime | Res | Processed | Cap FPS | Proc FPS | Dropped | P50 (ms) | P95 (ms) | Inf (ms) | Track (ms) |
|---|---|---|---|---|---|---|---|---|---|
| PYTORCH CPU (FP32) | 640x480 | 175 | 30.0 | 18.8 | 125 | 54.05 | 61.69 | 53.59 | 0.0 |
| ONNX CPU (FP32) | 640x480 | 70 | 30.0 | 7.1 | 230 | 144.32 | 155.88 | 143.96 | 0.01 |
| ONNX CPU (INT8) | 640x480 | 66 | 29.9 | 6.9 | 234 | 150.66 | 182.77 | 150.92 | 0.0 |
