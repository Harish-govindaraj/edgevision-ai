"""Unit tests for the PCA experiment pipeline execution and result generation."""

import json
from pathlib import Path
import tempfile
import pandas as pd
import pytest
from scripts.train_classifier import run_pca_experiment


def test_pca_experiment_execution_and_schema() -> None:
    """Verify run_pca_experiment executes and produces valid JSON/CSV outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "results"
        models_dir = Path(tmpdir) / "models"

        # Run on a quick subset of 100 samples
        results = run_pca_experiment(
            n_samples=100,
            n_components=10,
            test_size=0.25,
            random_state=42,
            output_dir=str(out_dir),
            models_dir=str(models_dir),
        )

        assert results is not None
        assert "experiment_name" in results
        assert "pca_metrics" in results
        assert "baseline" in results
        assert "reduced" in results
        assert "comparison" in results

        # Check dimension metrics
        assert results["pca_metrics"]["original_dimensions"] > 0
        assert results["pca_metrics"]["reduced_dimensions"] == 10
        assert results["pca_metrics"]["cumulative_explained_variance"] > 0.0
        assert len(results["pca_metrics"]["explained_variance_ratio"]) == 10

        # Check saved JSON file
        json_file = out_dir / "pca_experiment.json"
        assert json_file.exists()
        with open(json_file, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["experiment_name"] == results["experiment_name"]

        # Check saved CSV file
        csv_file = out_dir / "pca_experiment.csv"
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert "Features" in df.columns
        assert "Inference P50 (ms)" in df.columns

        # Check saved models
        assert (models_dir / "baseline_scaler.joblib").exists()
        assert (models_dir / "baseline_svm.joblib").exists()
        assert (models_dir / "reduced_scaler.joblib").exists()
        assert (models_dir / "reduced_pca.joblib").exists()
        assert (models_dir / "reduced_svm.joblib").exists()
