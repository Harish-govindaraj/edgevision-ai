"""Edge AI model optimization, ONNX export, and quantization package."""

from cv_engine.optimization.export_onnx import export_detector_to_onnx
from cv_engine.optimization.quantization import quantize_onnx_model

__all__ = ["export_detector_to_onnx", "quantize_onnx_model"]
