import streamlit as st
import pandas as pd
from pathlib import Path

def render_benchmark() -> None:
    """Reads benchmark CSV and renders a table/chart in Streamlit."""
    csv_path = Path("experiments/results/edge_benchmark.csv")
    
    if not csv_path.exists():
        st.warning("Benchmark results not found. Run `scripts/benchmark.py` first.")
        return
        
    try:
        df = pd.read_csv(csv_path)
        
        # Display raw table
        st.dataframe(df, use_container_width=True)
        
        # Add a bar chart for latency comparison
        st.caption("Total P50 Latency (ms) - Lower is better")
        chart_data = df[["runtime", "precision", "total_p50_ms"]].copy()
        chart_data["label"] = chart_data["runtime"] + " " + chart_data["precision"]
        chart_data = chart_data.set_index("label")
        
        st.bar_chart(chart_data["total_p50_ms"])
        
        st.caption("*Note: Measured on the host CPU. GPU execution requires a CUDA-enabled PyTorch environment.*")
    except Exception as e:
        st.error(f"Error loading benchmark data: {e}")
