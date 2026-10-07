"""INT8 post-training quantization utilities for edge inference optimization."""

from pathlib import Path
from typing import Any, Dict, Optional
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_onnx_model(
    input_model_path: str,
    output_model_path: str,
    weight_type: QuantType = QuantType.QUInt8,
) -> Dict[str, Any]:
    """
    Apply dynamic INT8 post-training quantization to an ONNX model.

    Quantizes weights to 8-bit integers while dynamically quantizing activations
    at runtime, significantly reducing memory bandwidth and cache pressure on edge CPUs.

    Args:
        input_model_path: Path to source FP32 ONNX model.
        output_model_path: Path for saving quantized INT8 ONNX model.
        weight_type: Target quantized weight data type (QUInt8 or QInt8).

    Returns:
        Summary dict containing original and quantized file sizes and compression ratio.
    """
    in_path = Path(input_model_path)
    out_path = Path(output_model_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig_size_mb = in_path.stat().st_size / (1024.0 * 1024.0)

    # Perform quantization
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=weight_type,
    )

    quant_size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    compression_ratio = orig_size_mb / max(quant_size_mb, 1e-6)

    # Verify session can be initialized
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    return {
        "source_model": str(in_path),
        "quantized_model": str(out_path),
        "original_size_mb": float(orig_size_mb),
        "quantized_size_mb": float(quant_size_mb),
        "compression_ratio": float(compression_ratio),
        "active_providers": sess.get_providers(),
    }
