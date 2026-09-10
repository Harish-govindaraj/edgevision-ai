"""HOG (Histogram of Oriented Gradients) feature extraction for EdgeVision AI."""

from typing import Tuple
import cv2
import numpy as np
from skimage.feature import hog as skimage_hog


def extract_hog_features(
    image: np.ndarray,
    image_size: Tuple[int, int] = (128, 128),
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """
    Extract deterministic HOG (Histogram of Oriented Gradients) features.

    Supports configurable image resizing, orientations, cell/block dimensions,
    and block normalization. Uses skimage.feature.hog for robust, deterministic
    extraction across all OpenCV builds.

    Args:
        image: Input BGR or grayscale image as a NumPy array.
        image_size: Target image size as (width, height).
        orientations: Number of gradient orientation bins (default: 9).
        pixels_per_cell: Size (in pixels) of a cell as (y, x) or (height, width).
        cells_per_block: Number of cells in each block as (y, x).
        block_norm: Block normalization method (e.g., 'L2-Hys' or 'L2').

    Returns:
        One-dimensional float32 HOG feature vector.

    Raises:
        ValueError: If input image is None, empty, or has invalid dimensions.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image must not be empty.")

    if image.ndim not in (2, 3):
        raise ValueError(f"Input image must have 2 or 3 dimensions, got {image.ndim}.")

    # Resize to target dimension (width, height)
    resized = cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA,
    )

    # Convert to grayscale if 3-channel
    if resized.ndim == 3:
        grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = resized

    # Compute deterministic HOG features
    features = skimage_hog(
        grayscale,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True,
    )

    return features.flatten().astype(np.float32)
