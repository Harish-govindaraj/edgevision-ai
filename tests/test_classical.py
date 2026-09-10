"""Unit tests for classical ML components: Scaler, PCA, SVM, and pipeline integration."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from cv_engine.classical.scaler import FeatureScaler
from cv_engine.classical.pca import PCAReducer
from cv_engine.classical.svm import SVMClassifier
from cv_engine.features.hog import extract_hog_features


@pytest.fixture
def synthetic_features() -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic synthetic feature dataset for testing."""
    rng = np.random.RandomState(42)
    # 60 samples, 100 features, 3 balanced classes
    X = rng.randn(60, 100).astype(np.float32)
    # Add class-specific signals
    y = np.repeat([0, 1, 2], 20)
    for c in range(3):
        X[y == c, c * 10 : (c + 1) * 10] += 3.0
    return X, y


def test_scaler_fit_transform_persistence(synthetic_features) -> None:
    """Test FeatureScaler normalization and serialization."""
    X, _ = synthetic_features
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-5)

    # Test single 1D vector transform
    vec_scaled = scaler.transform(X[0])
    assert vec_scaled.ndim == 1
    assert len(vec_scaled) == X.shape[1]
    np.testing.assert_allclose(vec_scaled, X_scaled[0], atol=1e-5)

    # Test persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "scaler.joblib"
        scaler.save(path)
        loaded = FeatureScaler.load(path)
        loaded_scaled = loaded.transform(X[0])
        np.testing.assert_allclose(vec_scaled, loaded_scaled, atol=1e-5)


def test_scaler_unfitted_raises() -> None:
    """Unfitted scaler should raise RuntimeError."""
    scaler = FeatureScaler()
    with pytest.raises(RuntimeError, match="must be fitted"):
        scaler.transform(np.zeros((5, 5)))


def test_pca_fit_transform_and_variance(synthetic_features) -> None:
    """Test PCAReducer dimensionality reduction, variance properties, and reconstruction."""
    X, _ = synthetic_features
    n_components = 15
    pca = PCAReducer(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    assert X_reduced.shape == (X.shape[0], n_components)
    assert pca.n_components_ == n_components
    assert len(pca.explained_variance_ratio_) == n_components
    assert np.all(pca.explained_variance_ratio_ >= 0.0)

    # Cumulative variance should be monotonically increasing and <= 1.0
    cum_var = pca.cumulative_explained_variance_
    assert len(cum_var) == n_components
    assert np.all(np.diff(cum_var) >= 0.0)
    assert cum_var[-1] <= 1.00001

    # Reconstruction test
    X_reconstructed = pca.inverse_transform(X_reduced)
    assert X_reconstructed.shape == X.shape

    # Test persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pca.joblib"
        pca.save(path)
        loaded = PCAReducer.load(path)
        loaded_reduced = loaded.transform(X)
        np.testing.assert_allclose(X_reduced, loaded_reduced, atol=1e-5)


def test_pca_variance_ratio_mode(synthetic_features) -> None:
    """Test PCA with target variance ratio (e.g., 0.80)."""
    X, _ = synthetic_features
    pca = PCAReducer(n_components=0.80, random_state=42)
    X_reduced = pca.fit_transform(X)

    assert X_reduced.shape[1] < X.shape[1]
    assert pca.cumulative_explained_variance_[-1] >= 0.80


def test_svm_classifier_fit_predict_proba_evaluate(synthetic_features) -> None:
    """Test SVMClassifier fit, predict, probabilities, and metric reporting."""
    X, y = synthetic_features
    svm = SVMClassifier(kernel="rbf", C=1.0, probability=True, random_state=42)
    svm.fit(X, y)

    # Predictions
    y_pred = svm.predict(X)
    assert len(y_pred) == len(y)

    # Probabilities
    probs = svm.predict_proba(X)
    assert probs.shape == (len(y), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Evaluation metrics
    metrics = svm.evaluate(X, y)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert metrics["accuracy"] > 0.8  # Strong signal in synthetic data

    # Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "svm.joblib"
        svm.save(path)
        loaded = SVMClassifier.load(path)
        assert np.array_equal(loaded.predict(X), y_pred)


def test_end_to_end_hog_pca_svm_pipeline() -> None:
    """Test complete classical vision pipeline: Image -> HOG -> Scaler -> PCA -> SVM."""
    rng = np.random.RandomState(42)
    # Generate 12 small synthetic images (6 per class)
    # Class 0: horizontal stripes; Class 1: vertical stripes
    images = []
    labels = []
    for i in range(12):
        cls = i % 2
        img = np.zeros((64, 64), dtype=np.uint8)
        if cls == 0:
            img[::8, :] = 255
        else:
            img[:, ::8] = 255
        # Add slight noise
        noise = rng.randint(0, 30, (64, 64), dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(cls)

    # Extract HOG features
    hog_feats = np.array([
        extract_hog_features(img, image_size=(64, 64), orientations=8, pixels_per_cell=(16, 16))
        for img in images
    ])
    labels = np.array(labels)

    assert hog_feats.ndim == 2
    orig_dim = hog_feats.shape[1]

    # Scale
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(hog_feats)

    # Reduce with PCA
    pca = PCAReducer(n_components=6, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    assert X_pca.shape == (12, 6)
    assert X_pca.shape[1] < orig_dim

    # Train SVM
    svm = SVMClassifier(kernel="linear", C=1.0, random_state=42)
    svm.fit(X_pca, labels)

    # Evaluate
    metrics = svm.evaluate(X_pca, labels)
    assert metrics["accuracy"] == 1.0
