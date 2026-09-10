"""Tests for visual feature extraction."""

import numpy as np

from cv_engine.inference.features import extract_hog_features


def test_hog_features_from_color_image() -> None:
    """HOG should produce a one-dimensional feature vector."""

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    features = extract_hog_features(image)

    assert features.ndim == 1
    assert features.size > 0
    assert features.dtype == np.float32


def test_hog_features_from_grayscale_image() -> None:
    """HOG should also accept grayscale images."""

    image = np.zeros(
        (128, 128),
        dtype=np.uint8,
    )

    features = extract_hog_features(image)

    assert features.ndim == 1
    assert features.size > 0


def test_hog_rejects_empty_image() -> None:
    """Empty images should raise a clear error."""

    empty_image = np.array([], dtype=np.uint8)

    try:
        extract_hog_features(empty_image)
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty image.")