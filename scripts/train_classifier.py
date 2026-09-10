"""
Reproducible Classical ML & PCA Dimensionality Reduction Experiment for EdgeVision AI.

Compares:
  Baseline: Image -> HOG (High-D) -> StandardScaler -> SVM
  Reduced:  Image -> HOG (High-D) -> StandardScaler -> PCA (Low-D) -> SVM

Measures and records:
  - Original feature dimensions vs PCA dimensions
  - Individual and cumulative explained variance
  - Classification accuracy, macro precision, recall, F1 score
  - Training time (seconds)
  - Inference latency per sample (mean, median/P50, P95 in milliseconds)
  - Serialized model disk footprint (KB)

Results are saved deterministically to:
  experiments/results/pca_experiment.json
  experiments/results/pca_experiment.csv
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cv_engine.classical.pca import PCAReducer
from cv_engine.classical.scaler import FeatureScaler
from cv_engine.classical.svm import SVMClassifier
from cv_engine.features.hog import extract_hog_features


def load_dataset(
    n_samples: int = 1200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real computer vision digit images.

    Args:
        n_samples: Number of samples to use (capped at total available).
        random_state: Random seed for shuffling.

    Returns:
        images: 2D numpy array of images of shape (N, H, W).
        labels: 1D numpy array of class labels.
    """
    data = load_digits()
    images = data.images.astype(np.uint8)
    labels = data.target

    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(images))
    selected = indices[: min(n_samples, len(images))]

    return images[selected], labels[selected]


def extract_dataset_hog_features(
    images: np.ndarray,
    image_size: Tuple[int, int] = (32, 32),
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (4, 4),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Extract deterministic HOG feature vectors for all images."""
    features = [
        extract_hog_features(
            img,
            image_size=image_size,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )
        for img in images
    ]
    return np.array(features, dtype=np.float32)


def measure_inference_latency(
    predict_fn,
    X_test: np.ndarray,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, float]:
    """
    Measure single-sample inference latency with warmup using perf_counter.

    Returns latencies in milliseconds.
    """
    n_test = len(X_test)
    sample_indices = np.random.choice(n_test, size=min(n_test, benchmark_runs), replace=True)

    # Warmup
    for idx in sample_indices[:warmup_runs]:
        _ = predict_fn(X_test[idx : idx + 1])

    # Benchmarking
    latencies_ms: List[float] = []
    for idx in sample_indices:
        t0 = time.perf_counter()
        _ = predict_fn(X_test[idx : idx + 1])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    return {
        "mean_latency_ms": float(np.mean(latencies_arr)),
        "p50_latency_ms": float(np.percentile(latencies_arr, 50)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
    }


def run_pca_experiment(
    n_samples: int = 1200,
    n_components: int = 40,
    test_size: float = 0.25,
    random_state: int = 42,
    output_dir: str = "experiments/results",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """Execute the complete baseline vs PCA-reduced ML experiment."""
    print("=" * 60)
    print("EdgeVision AI — Classical ML & PCA Dimensionality Reduction")
    print("=" * 60)

    # 1. Dataset loading
    print(f"[*] Loading dataset (n_samples={n_samples}, seed={random_state})...")
    images, labels = load_dataset(n_samples=n_samples, random_state=random_state)
    print(f"    Loaded {len(images)} images across {len(np.unique(labels))} classes.")

    # 2. HOG Feature extraction
    print("[*] Extracting HOG features...")
    t0_hog = time.perf_counter()
    X_hog = extract_dataset_hog_features(images)
    hog_time = time.perf_counter() - t0_hog
    orig_dim = X_hog.shape[1]
    print(f"    Extracted {orig_dim}-dimensional HOG features for {len(X_hog)} images in {hog_time:.2f}s.")

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"    Split: {len(X_train)} train samples, {len(X_test)} test samples.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path = Path(models_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # BASELINE PIPELINE: HOG -> StandardScaler -> SVM
    # -------------------------------------------------------------
    print("\n--- Running BASELINE Pipeline (HOG -> StandardScaler -> SVM) ---")
    t0_base_train = time.perf_counter()
    scaler_base = FeatureScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train)
    X_test_base_scaled = scaler_base.transform(X_test)

    svm_base = SVMClassifier(kernel="rbf", C=10.0, random_state=random_state)
    svm_base.fit(X_train_base_scaled, y_train)
    base_train_time = time.perf_counter() - t0_base_train
    print(f"    Baseline Training time: {base_train_time:.3f}s")

    base_eval = svm_base.evaluate(X_test_base_scaled, y_test)
    base_latency = measure_inference_latency(
        lambda x: svm_base.predict(scaler_base.transform(x)),
        X_test,
    )
    print(f"    Baseline Accuracy: {base_eval['accuracy']:.4f} | F1: {base_eval['f1_macro']:.4f}")
    print(f"    Baseline P50 Latency: {base_latency['p50_latency_ms']:.3f} ms | P95: {base_latency['p95_latency_ms']:.3f} ms")

    # Serialize baseline models
    base_scaler_file = mod_path / "baseline_scaler.joblib"
    base_svm_file = mod_path / "baseline_svm.joblib"
    scaler_base.save(base_scaler_file)
    svm_base.save(base_svm_file)
    base_model_size_kb = (base_scaler_file.stat().st_size + base_svm_file.stat().st_size) / 1024.0

    # -------------------------------------------------------------
    # REDUCED PIPELINE: HOG -> StandardScaler -> PCA -> SVM
    # -------------------------------------------------------------
    print(f"\n--- Running REDUCED Pipeline (HOG -> Scaler -> PCA ({n_components} components) -> SVM) ---")
    t0_red_train = time.perf_counter()
    scaler_red = FeatureScaler()
    X_train_red_scaled = scaler_red.fit_transform(X_train)
    X_test_red_scaled = scaler_red.transform(X_test)

    pca = PCAReducer(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_red_scaled)
    X_test_pca = pca.transform(X_test_red_scaled)

    svm_red = SVMClassifier(kernel="rbf", C=10.0, random_state=random_state)
    svm_red.fit(X_train_pca, y_train)
    red_train_time = time.perf_counter() - t0_red_train
    print(f"    Reduced Training time: {red_train_time:.3f}s")

    red_eval = svm_red.evaluate(X_test_pca, y_test)
    red_latency = measure_inference_latency(
        lambda x: svm_red.predict(pca.transform(scaler_red.transform(x))),
        X_test,
    )
    explained_var_ratio = pca.explained_variance_ratio_.tolist()
    cum_explained_var = float(pca.cumulative_explained_variance_[-1])

    print(f"    Reduced Accuracy: {red_eval['accuracy']:.4f} | F1: {red_eval['f1_macro']:.4f}")
    print(f"    Reduced Dimensions: {orig_dim} -> {pca.n_components_} ({100 * (1 - pca.n_components_ / orig_dim):.1f}% reduction)")
    print(f"    Cumulative Explained Variance: {cum_explained_var * 100:.2f}%")
    print(f"    Reduced P50 Latency: {red_latency['p50_latency_ms']:.3f} ms | P95: {red_latency['p95_latency_ms']:.3f} ms")

    # Serialize reduced models
    red_scaler_file = mod_path / "reduced_scaler.joblib"
    red_pca_file = mod_path / "reduced_pca.joblib"
    red_svm_file = mod_path / "reduced_svm.joblib"
    scaler_red.save(red_scaler_file)
    pca.save(red_pca_file)
    svm_red.save(red_svm_file)
    red_model_size_kb = (
        red_scaler_file.stat().st_size + red_pca_file.stat().st_size + red_svm_file.stat().st_size
    ) / 1024.0

    # -------------------------------------------------------------
    # STRUCTURED REPORTING & EXPORT
    # -------------------------------------------------------------
    experiment_results: Dict[str, Any] = {
        "experiment_name": "hog_pca_svm_dimensionality_reduction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": "digits_vision_benchmark",
            "total_samples": len(images),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "classes": int(len(np.unique(labels))),
            "random_seed": random_state,
        },
        "pca_metrics": {
            "original_dimensions": orig_dim,
            "reduced_dimensions": pca.n_components_,
            "dimension_reduction_ratio": float(1.0 - (pca.n_components_ / orig_dim)),
            "cumulative_explained_variance": cum_explained_var,
            "explained_variance_ratio": explained_var_ratio,
        },
        "baseline": {
            "pipeline": "HOG -> StandardScaler -> SVM",
            "feature_dim": orig_dim,
            "training_time_s": float(base_train_time),
            "accuracy": base_eval["accuracy"],
            "precision_macro": base_eval["precision_macro"],
            "recall_macro": base_eval["recall_macro"],
            "f1_macro": base_eval["f1_macro"],
            "precision_weighted": base_eval["precision_weighted"],
            "recall_weighted": base_eval["recall_weighted"],
            "f1_weighted": base_eval["f1_weighted"],
            "latency_ms": base_latency,
            "model_size_kb": float(base_model_size_kb),
        },
        "reduced": {
            "pipeline": f"HOG -> StandardScaler -> PCA({n_components}) -> SVM",
            "feature_dim": pca.n_components_,
            "training_time_s": float(red_train_time),
            "accuracy": red_eval["accuracy"],
            "precision_macro": red_eval["precision_macro"],
            "recall_macro": red_eval["recall_macro"],
            "f1_macro": red_eval["f1_macro"],
            "precision_weighted": red_eval["precision_weighted"],
            "recall_weighted": red_eval["recall_weighted"],
            "f1_weighted": red_eval["f1_weighted"],
            "latency_ms": red_latency,
            "model_size_kb": float(red_model_size_kb),
        },
        "comparison": {
            "accuracy_delta": float(red_eval["accuracy"] - base_eval["accuracy"]),
            "f1_delta": float(red_eval["f1_macro"] - base_eval["f1_macro"]),
            "training_speedup_factor": float(base_train_time / max(red_train_time, 1e-6)),
            "latency_p50_speedup_factor": float(base_latency["p50_latency_ms"] / max(red_latency["p50_latency_ms"], 1e-6)),
            "model_size_ratio": float(red_model_size_kb / max(base_model_size_kb, 1e-6)),
        },
    }

    # Save JSON
    json_path = out_path / "pca_experiment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\n[+] Saved structured experiment JSON: {json_path}")

    # Save CSV comparison
    comparison_df = pd.DataFrame([
        {
            "Pipeline": "Baseline (HOG+SVM)",
            "Features": orig_dim,
            "Explained Var": 1.0,
            "Accuracy": base_eval["accuracy"],
            "Precision": base_eval["precision_macro"],
            "Recall": base_eval["recall_macro"],
            "F1": base_eval["f1_macro"],
            "Train Time (s)": round(base_train_time, 4),
            "Inference P50 (ms)": round(base_latency["p50_latency_ms"], 4),
            "Inference P95 (ms)": round(base_latency["p95_latency_ms"], 4),
            "Model Size (KB)": round(base_model_size_kb, 2),
        },
        {
            "Pipeline": f"Reduced (HOG+PCA+SVM)",
            "Features": pca.n_components_,
            "Explained Var": round(cum_explained_var, 4),
            "Accuracy": red_eval["accuracy"],
            "Precision": red_eval["precision_macro"],
            "Recall": red_eval["recall_macro"],
            "F1": red_eval["f1_macro"],
            "Train Time (s)": round(red_train_time, 4),
            "Inference P50 (ms)": round(red_latency["p50_latency_ms"], 4),
            "Inference P95 (ms)": round(red_latency["p95_latency_ms"], 4),
            "Model Size (KB)": round(red_model_size_kb, 2),
        },
    ])
    csv_path = out_path / "pca_experiment.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"[+] Saved comparison CSV: {csv_path}")
    print("\nExperiment Summary Table:")
    print(comparison_df.to_string(index=False))

    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EdgeVision AI PCA Experiment.")
    parser.add_argument("--samples", type=int, default=1200, help="Number of dataset samples")
    parser.add_argument("--components", type=int, default=40, help="PCA components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    args = parser.parse_args()

    run_pca_experiment(
        n_samples=args.samples,
        n_components=args.components,
        random_state=args.seed,
        output_dir=args.output,
    )
