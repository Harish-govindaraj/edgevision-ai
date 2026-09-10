"""Principal Component Analysis (PCA) dimensionality reduction module."""

from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    """
    Reusable Principal Component Analysis (PCA) module for EdgeVision AI.

    Provides feature compression, variance analysis, inverse reconstruction,
    and model serialization for classical edge computer vision pipelines.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = 50,
        random_state: int = 42,
    ) -> None:
        """
        Initialize PCAReducer.

        Args:
            n_components: Number of components to keep. If float between 0 and 1,
                          represents the target variance ratio to preserve.
            random_state: Deterministic random state for reproducible results.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """Fit PCA model on feature matrix X."""
        X = np.asarray(X)
        if X.size == 0 or X.ndim != 2:
            raise ValueError("Input feature matrix X must be a non-empty 2D array.")
        self.pca.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project feature matrix X into principal component subspace."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer must be fitted before transform.")
        X = np.asarray(X)
        if X.ndim == 1:
            return self.pca.transform(X.reshape(1, -1)).flatten()
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA model and transform X in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct original feature space from reduced representation."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer must be fitted before inverse_transform.")
        X_reduced = np.asarray(X_reduced)
        if X_reduced.ndim == 1:
            return self.pca.inverse_transform(X_reduced.reshape(1, -1)).flatten()
        return self.pca.inverse_transform(X_reduced)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Return percentage of variance explained by each principal component."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer is not fitted yet.")
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_explained_variance_(self) -> np.ndarray:
        """Return cumulative explained variance ratio across components."""
        return np.cumsum(self.explained_variance_ratio_)

    @property
    def n_components_(self) -> int:
        """Return actual number of components selected."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer is not fitted yet.")
        return self.pca.n_components_

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted PCA model to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PCAReducer":
        """Load serialized PCA model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PCA model file not found: {path}")
        instance = cls()
        instance.pca = joblib.load(path)
        instance.is_fitted = True
        instance.n_components = instance.pca.n_components
        return instance
