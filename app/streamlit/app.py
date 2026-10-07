import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from components.sidebar import render_sidebar
from components.metrics import render_metrics
from components.benchmark import render_benchmark
from utils.video_processor import run_video_stream

st.set_page_config(
    page_title="EdgeVision AI Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main() -> None:
    st.title("EdgeVision AI — Demonstration Dashboard")
    st.markdown("""
    *Real-Time Multi-Task Computer Vision & Edge AI Platform*
    
    This dashboard demonstrates the integration of the EdgeVision AI pipeline, running end-to-end detection, tracking, and analytics.
    """)

    # Render Sidebar
    config = render_sidebar()

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Vision")
        st.caption("Running pipeline in real-time.")
        
        # We will use an empty container to hold the video stream
        video_placeholder = st.empty()
        
        # Start button
        start_btn = st.button("Start Inference Stream", type="primary")
        stop_btn = st.button("Stop Inference Stream")

        if start_btn:
            st.session_state["run_stream"] = True
        if stop_btn:
            st.session_state["run_stream"] = False

    with col2:
        st.subheader("Real-Time Metrics")
        metrics_placeholder = st.empty()
        
        st.divider()
        
        st.subheader("Benchmark Comparison")
        render_benchmark()

    # Execute stream if requested
    if st.session_state.get("run_stream", False):
        try:
            run_video_stream(config, video_placeholder, metrics_placeholder)
        except Exception as e:
            st.error(f"Error running stream: {e}")
            st.session_state["run_stream"] = False
    else:
        video_placeholder.info("Click 'Start Inference Stream' to begin.")
        
        with metrics_placeholder.container():
            st.metric("Status", "Idle")

if __name__ == "__main__":
    main()
