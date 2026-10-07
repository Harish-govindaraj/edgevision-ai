"""ONNX model exporter, validator, and numerical verification engine for EdgeVision AI."""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def export_detector_to_onnx(
    output_path: str = "models/ssdlite320_mobilenet_v3_large.onnx",
    input_size: Tuple[int, int] = (320, 320),
    opset_version: int = 14,
    verify_numerical: bool = True,
) -> Dict[str, Any]:
    """
    Export SSDLite320 MobileNetV3-Large detector to an optimized ONNX model.

    Validates graph consistency with onnx.checker and verifies numerical
    parity against native PyTorch on a synthetic input tensor.

    Args:
        output_path: Target filepath for exported .onnx model.
        input_size: (H, W) input resolution.
        opset_version: ONNX operator set version (default: 14).
        verify_numerical: If True, execute inference comparison with PyTorch.

    Returns:
        Dictionary with model metadata, file size, and verification metrics.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading PyTorch detector weights...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.eval()

    h, w = input_size
    dummy_input = [torch.randn(3, h, w)]

    print(f"[*] Exporting model to ONNX format (opset={opset_version})...")
    t0 = time.perf_counter()

    torch.onnx.export(
        model,
        (dummy_input,),
        str(out_file),
        dynamo=False,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
    )
    export_duration = time.perf_counter() - t0
    file_size_mb = out_file.stat().st_size / (1024.0 * 1024.0)
    print(f"    Exported to: {out_file} ({file_size_mb:.2f} MB in {export_duration:.2f}s)")

    # 1. Structural graph validation
    print("[*] Verifying ONNX graph integrity with onnx.checker...")
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    print("    Graph validation PASSED.")

    verification_result: Dict[str, Any] = {
        "output_path": str(out_file),
        "file_size_mb": float(file_size_mb),
        "opset_version": opset_version,
        "input_resolution": f"{w}x{h}",
        "graph_valid": True,
        "export_time_s": float(export_duration),
    }

    # 2. Numerical verification against PyTorch
    if verify_numerical:
        print("[*] Comparing ONNX Runtime outputs against PyTorch baseline...")
        test_tensor = torch.zeros(3, h, w)
        # Create a distinct pattern to produce deterministic boxes
        test_tensor[:, 50:150, 50:150] = 0.8

        with torch.no_grad():
            pt_out = model([test_tensor])[0]
        pt_boxes = pt_out["boxes"].numpy()
        pt_scores = pt_out["scores"].numpy()
        pt_labels = pt_out["labels"].numpy()

        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        ort_outs = session.run(None, {input_name: test_tensor.numpy()})
        ort_boxes, ort_scores, ort_labels = ort_outs[0], ort_outs[1], ort_outs[2]

        box_diff = float(np.max(np.abs(pt_boxes - ort_boxes)))
        score_diff = float(np.max(np.abs(pt_scores - ort_scores)))
        labels_match = bool(np.array_equal(pt_labels, ort_labels))

        print(f"    Max Box Coordinate Delta: {box_diff:.6f}")
        print(f"    Max Confidence Score Delta: {score_diff:.6f}")
        print(f"    Class Labels Exact Match: {labels_match}")

        verification_result["numerical_verification"] = {
            "max_box_delta": box_diff,
            "max_score_delta": score_diff,
            "labels_match": labels_match,
            "verified": box_diff < 1e-2 and score_diff < 1e-3,
        }

    return verification_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Object Detector to ONNX.")
    parser.add_argument("--output", type=str, default="models/ssdlite320_mobilenet_v3_large.onnx", help="Output path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    export_detector_to_onnx(output_path=args.output, opset_version=args.opset)
