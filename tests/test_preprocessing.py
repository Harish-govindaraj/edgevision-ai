"""Tests for EdgeVision preprocessing utilities."""

import numpy as np

from cv_engine.preprocessing.filters import (
    canny_edges,
    gaussian_blur,
    preprocess_frame,
    resize_frame,
    to_grayscale,
)


def create_test_frame() -> np.ndarray:
    """Create a synthetic BGR test image."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_resize_frame() -> None:
    frame = create_test_frame()

    result = resize_frame(frame, 640, 480)

    assert result.shape == (480, 640, 3)


def test_to_grayscale() -> None:
    frame = create_test_frame()

    result = to_grayscale(frame)

    assert result.shape == (720, 1280)


def test_gaussian_blur() -> None:
    frame = create_test_frame()

    result = gaussian_blur(frame)

    assert result.shape == frame.shape


def test_canny_edges() -> None:
    frame = create_test_frame()
    grayscale = to_grayscale(frame)

    result = canny_edges(grayscale)

    assert result.shape == grayscale.shape
    assert result.ndim == 2
    assert result.dtype == np.uint8

def test_preprocess_frame() -> None:
    frame = create_test_frame()

    resized, grayscale, edges = preprocess_frame(
        frame,
        width=640,
        height=480,
    )

    assert resized.shape == (480, 640, 3)
    assert grayscale.shape == (480, 640)
    assert edges.shape == (480, 640)