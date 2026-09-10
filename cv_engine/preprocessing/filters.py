"""Image preprocessing utilities for the EdgeVision AI pipeline."""

import cv2
import numpy as np


def resize_frame(
    frame: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Resize a frame to the requested dimensions."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def gaussian_blur(
    frame: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply Gaussian noise reduction."""
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd number.")

    return cv2.GaussianBlur(
        frame,
        (kernel_size, kernel_size),
        0,
    )


def canny_edges(
    frame: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """Detect edges using the Canny edge detector."""
    if low_threshold >= high_threshold:
        raise ValueError(
            "low_threshold must be smaller than high_threshold."
        )

    return cv2.Canny(
        frame,
        low_threshold,
        high_threshold,
    )


def preprocess_frame(
    frame: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the complete MVP preprocessing pipeline.

    Returns:
        resized_frame: Resized BGR frame.
        grayscale: Grayscale representation.
        edges: Canny edge map.
    """
    resized = resize_frame(frame, width, height)
    grayscale = to_grayscale(resized)
    blurred = gaussian_blur(grayscale)
    edges = canny_edges(blurred)

    return resized, grayscale, edges