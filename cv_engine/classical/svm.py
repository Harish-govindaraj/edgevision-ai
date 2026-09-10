"""Support Vector Machine (SVM) classification module for EdgeVision AI."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.svm import SVC


class SVMClassifier:
    """
    Support Vector Classifier wrapper for edge computer vision pipelines.

    Supports linear and non-linear (RBF) kernels with calibrated probability
    estimates and serialization.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        probability: bool = True,
        random_state: int = 42,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        base_svc = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
        )
        if probability:
            self.model = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
        else:
            self.model = base_svc
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Fit SVM model on training features X and labels y."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.size == 0 or X.ndim != 2:
            raise ValueError("Feature matrix X must be a non-empty 2D array.")
        if len(X) != len(y):
            raise ValueError(f"X samples ({len(X)}) does not match y labels ({len(y)}).")

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for feature matrix X."""
        if not self.is_fitted:
            raise RuntimeError("SVMClassifier must be fitted before predict.")
        X = np.asarray(X)
        if X.ndim == 1:
            return self.model.predict(X.reshape(1, -1))
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for feature matrix X."""
        if not self.is_fitted:
            raise RuntimeError("SVMClassifier must be fitted before predict_proba.")
        if not self.probability:
            raise RuntimeError("Probability estimation was disabled at initialization.")
        X = np.asarray(X)
        if X.ndim == 1:
            return self.model.predict_proba(X.reshape(1, -1))[0]
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the given test data and labels."""
        if not self.is_fitted:
            raise RuntimeError("SVMClassifier must be fitted before scoring.")
        return float(self.model.score(X, y))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute standard classification evaluation metrics."""
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted SVM model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SVMClassifier":
        """Load serialized SVM model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SVM model file not found: {path}")
        instance = cls()
        instance.model = joblib.load(path)
        instance.is_fitted = True
        return instance
