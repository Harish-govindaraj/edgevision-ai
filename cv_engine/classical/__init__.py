"""Classical machine learning components for EdgeVision AI."""

from cv_engine.classical.scaler import FeatureScaler
from cv_engine.classical.pca import PCAReducer
from cv_engine.classical.svm import SVMClassifier

__all__ = ["FeatureScaler", "PCAReducer", "SVMClassifier"]
