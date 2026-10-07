# EdgeVision AI Demonstration Dashboard

## Purpose
The Streamlit dashboard serves as a professional, interactive orchestration layer over the EdgeVision AI pipeline. It demonstrates real-time object detection, multi-object tracking, and edge AI analytics (like virtual tripwires) using the local system's capabilities.

## Architecture
The dashboard architecture is built on Streamlit but strictly isolates the core Computer Vision and Edge AI processing modules to preserve system integrity.
- **`app/streamlit/app.py`**: The main entry point.
- **`app/streamlit/components/`**: Modules handling specific UI sections (Sidebar configuration, Real-time metrics, Benchmark visualization).
- **`app/streamlit/utils/video_processor.py`**: The bridge that iteratively invokes `cv_engine` inference runtimes and trackers over video frames, dispatching annotated results and metrics back to the UI.

## Features
- **Live Vision Stream**: Renders the video feed with bounding boxes, object IDs, trajectories, and tripwires.
- **Runtime Selection**: Dynamically allows switching between native PyTorch and ONNX Runtime execution environments to test different deployments (e.g. FP32 vs dynamic INT8).
- **Real-Time KPI Metrics**: Displays throughput (FPS), component latency, and object tracking counts.
- **Benchmark Comparison**: Loads and displays offline benchmark data to evaluate architectural trade-offs without halting the live feed.

## Installation
Ensure the project requirements and Streamlit are installed:
```bash
pip install -r requirements.txt
pip install streamlit pandas
```

## Launching the Dashboard
Start the dashboard from the repository root:
```bash
streamlit run app/streamlit/app.py
```

## Input Sources
- **Webcam**: Uses your local machine's camera (e.g., `/dev/video0` or `0`).
- **Video File**: You can upload a recorded video clip (MP4/AVI) for reproducible testing.

## Limitations
- **Current Execution**: On this build, the model is executing purely on the CPU, as standard PyTorch wheels are installed. The host's NVIDIA GeForce RTX 3050 GPU is bypassed.
- **Future Scale**: Installing a CUDA-compatible environment (`torch+cu118` or `121`) will automatically unlock GPU acceleration in both the PyTorch and ONNX execution providers.
