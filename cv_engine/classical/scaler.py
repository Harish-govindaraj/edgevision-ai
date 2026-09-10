"""Feature scaling utilities for classical ML pipelines in EdgeVision AI."""

from pathlib import Path
from typing import Union
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    StandardScaler wrapper with serialization support for edge ML pipelines.

    Normalizes feature vectors to zero mean and unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """Fit the scaler to feature matrix X."""
        X = np.asarray(X)
        if X.size == 0 or X.ndim != 2:
            raise ValueError("Input feature matrix X must be a non-empty 2D array.")
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform feature matrix X using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform.")
        X = np.asarray(X)
        if X.ndim == 1:
            return self.scaler.transform(X.reshape(1, -1)).flatten()
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted scaler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureScaler":
        """Load serialized scaler from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        instance = cls()
        instance.scaler = joblib.load(path)
        instance.is_fitted = True
        return instance
