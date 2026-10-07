import time
import streamlit as st
from typing import Dict, Any
import cv2

from cv_engine.inference.runtime import create_inference_runtime
from cv_engine.pipeline import AsyncRealtimePipeline
from components.metrics import render_metrics

def run_video_stream(
    config: Dict[str, Any],
    video_placeholder,
    metrics_placeholder
) -> None:
    """
    Executes the video pipeline, yielding annotated frames to Streamlit using the async bounded pipeline.
    """
    source = config["source"]
    if source is None:
        video_placeholder.info("Please upload a video file or select a webcam.")
        return

    # Parse runtime backend/precision
    runtime_choice = config["runtime"]
    if "PyTorch" in runtime_choice:
        backend = "pytorch"
        precision = "FP32"
    elif "ONNX" in runtime_choice:
        backend = "onnx"
        if "INT8" in runtime_choice:
            precision = "INT8"
            model_path = "models/ssdlite320_mobilenet_v3_large_int8.onnx"
        else:
            precision = "FP32"
            model_path = "models/ssdlite320_mobilenet_v3_large.onnx"
    else:
        video_placeholder.error(f"Unknown runtime selected: {runtime_choice}")
        return

    # Cache model initialization
    cache_key = f"runtime_{backend}_{precision}_{config['device']}_{config['confidence']}"
    if cache_key not in st.session_state:
        st.session_state.clear()
        with st.spinner(f"Loading {backend.upper()} ({precision}) model..."):
            kwargs = {
                "backend": backend,
                "device_or_provider": config["device"],
                "confidence_threshold": config["confidence"],
                "precision": precision
            }
            if backend == "onnx":
                kwargs["model_path"] = model_path
                
            rt = create_inference_runtime(**kwargs)
            st.session_state[cache_key] = rt
    
    runtime = st.session_state[cache_key]

    pipeline = AsyncRealtimePipeline(
        source=source,
        runtime=runtime,
        width=640,
        height=480,
        enable_tracking=config["enable_tracking"],
        tripwire_y=0,
        headless=True
    )
    
    pipeline.start()
    
    update_counter = 0
    try:
        while st.session_state.get("run_stream", False):
            res = pipeline.get_output(timeout=0.5)
            if res is None:
                if not pipeline._capture_thread.is_alive() and pipeline.input_queue.empty() and pipeline.output_queue.empty():
                    video_placeholder.warning("End of video stream.")
                    st.session_state["run_stream"] = False
                    break
                # Allow Streamlit to respond to UI clicks
                time.sleep(0.01)
                continue
                
            annotated_frame, metrics, summary = res
            
            # Update stream visually
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
            
            # Update metrics occasionally to avoid UI lag
            update_counter += 1
            if update_counter % 3 == 0:
                render_metrics(
                    metrics_placeholder,
                    metrics=metrics,
                    active_count=summary.get("active_tracks_count", 0),
                    total_unique=summary.get("total_unique_observed", 0),
                    tripwire_in=summary.get("tripwire_in", 0),
                    tripwire_out=summary.get("tripwire_out", 0),
                    device_name=runtime.provider_name
                )
                
            # Allow Streamlit to respond to Stop button
            time.sleep(0.001)

    finally:
        pipeline.stop()
