import streamlit as st
from cv_engine.utils.metrics_tracker import PipelineMetrics

def render_metrics(
    metrics_placeholder,
    metrics: PipelineMetrics,
    active_count: int,
    total_unique: int,
    tripwire_in: int,
    tripwire_out: int,
    device_name: str,
) -> None:
    """Updates the metrics placeholder with current telemetry."""
    with metrics_placeholder.container():
        # First row: FPS & Queue
        m1, m2, m3 = st.columns(3)
        m1.metric("Capture FPS", f"{metrics.capture_fps:.1f}")
        m2.metric("Processing FPS", f"{metrics.processing_fps:.1f}")
        m3.metric("Dropped Frames", f"{metrics.dropped_frames}")

        # Second row: Latency
        b1, b2, b3 = st.columns(3)
        b1.metric("P50 Latency", f"{metrics.total_p50_ms:.1f} ms")
        b2.metric("Det Latency", f"{metrics.inference_ms:.1f} ms")
        b3.metric("Device", device_name.upper())

        # Third row: Counts
        st.markdown("#### Analytics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Tracks", active_count)
        c2.metric("Total Seen", total_unique)
        c3.metric("Tripwire IN", tripwire_in)
        c4.metric("Tripwire OUT", tripwire_out)
