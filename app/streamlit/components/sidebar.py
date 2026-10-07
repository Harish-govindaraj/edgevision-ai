import streamlit as st
import torch
from pathlib import Path

def get_available_runtimes() -> list:
    """Return a list of available runtimes based on system capabilities."""
    runtimes = ["PyTorch (FP32)"]
    
    # Check if ONNX model exists
    onnx_fp32_path = Path("models/ssdlite320_mobilenet_v3_large.onnx")
    onnx_int8_path = Path("models/ssdlite320_mobilenet_v3_large_int8.onnx")
    
    if onnx_fp32_path.exists():
        runtimes.append("ONNX (FP32)")
    if onnx_int8_path.exists():
        runtimes.append("ONNX (INT8)")
        
    return runtimes

def render_sidebar() -> dict:
    """Renders the sidebar and returns the configuration dict."""
    st.sidebar.header("Pipeline Configuration")
    
    # Source
    st.sidebar.subheader("Input Source")
    source_type = st.sidebar.selectbox("Source Type", ["Webcam", "Video File"])
    
    source = 0
    if source_type == "Video File":
        video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if video_file:
            # We can save it to a temp path for cv2
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            source = tfile.name
        else:
            source = None # Indicate we are waiting
    else:
        cam_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0)
        source = cam_idx

    # Runtime Selection
    st.sidebar.subheader("Runtime & Optimization")
    runtimes = get_available_runtimes()
    selected_runtime = st.sidebar.selectbox("Inference Backend", runtimes)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = st.sidebar.selectbox("Device", ["cuda", "cpu"])
    else:
        st.sidebar.info("CUDA not available. CPU execution forced.")

    # Detection Config
    st.sidebar.subheader("Detection & Tracking")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    
    enable_tracking = st.sidebar.checkbox("Enable Tracking", value=True)
    enable_tripwire = st.sidebar.checkbox("Enable Tripwire", value=True)
    
    config = {
        "source": source,
        "runtime": selected_runtime,
        "device": device,
        "confidence": conf_thresh,
        "enable_tracking": enable_tracking,
        "enable_tripwire": enable_tripwire,
        "source_type": source_type
    }
    
    return config
