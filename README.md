    # EdgeVision AI
    **Real-Time Multi-Task Computer Vision & Edge AI Platform**

    EdgeVision AI is an engineering-focused portfolio project demonstrating a complete pipeline from classical computer vision preprocessing to modern deep-learning object detection and edge-optimized real-time inference.

    ## 1. Overview
    EdgeVision AI processes video streams (or files) in real-time to detect, track, and analyze objects. It features both classical machine learning methods (HOG + SVM) and deep-learning approaches (SSDLite MobileNetV3), with a focus on optimization for edge devices via ONNX and INT8 quantization.

    ## 2. Key Capabilities
    - **Classical CV/ML**: Image preprocessing, HOG feature extraction, PCA dimensionality reduction, and SVM classification.
    - **Deep Learning Detection**: SSDLite MobileNetV3 Large object detection.
    - **Multi-Object Tracking**: Centroid-based tracking with Hungarian matching, trajectory history, and temporal drop-tolerance.
    - **Analytics**: Virtual tripwire for counting objects entering/leaving specific regions.
    - **Edge Optimization**: PyTorch FP32, ONNX FP32, and dynamic ONNX INT8 execution.
    - **Real-Time Asynchronous Processing**: A bounded producer-consumer threaded pipeline ensuring low-latency, freshness-oriented frame processing using a bounded queue.
    - **Streamlit Dashboard**: A complete local UI to interact with models, observe metrics, and view real-time video processing.

    ## 3. Architecture

    ```mermaid
    graph TD
        A[Camera / Video Source] -->|Capture Thread| B[Bounded Input Queue maxsize=1]
        B -->|Worker Thread| C[BaseInferenceRuntime]
        C --> D[CentroidTracker]
        D --> E[TrackingAnalytics]
        E --> F[Bounded Output Queue maxsize=2]
        F -->|Main Thread / UI| G[Render / Streamlit Dashboard]
    ```

    **Optimization Pipeline:**
    ```mermaid
    graph LR
        A[PyTorch Model FP32] --> B[ONNX Export FP32]
        B --> C[Dynamic Quantization INT8]
        C --> D[Edge CPU/GPU Benchmark]
    ```

    ## 4. Technology Stack
    - **Languages**: Python 3.11
    - **Computer Vision**: OpenCV (`cv2`), scikit-image, Pillow
    - **Deep Learning / Edge AI**: PyTorch, torchvision, ONNX, ONNX Runtime
    - **Classical ML**: scikit-learn (`sklearn`), NumPy
    - **Application / UI**: Streamlit

    ## 5. Installation
    Requires Python 3.11.

    ```bash
    # Clone the repository
    git clone https://github.com/Harish-govindaraj/edgevision-ai.git
    cd edgevision-ai

    # Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

    ## 6. Launching the Dashboard
    To start the interactive Streamlit UI:
    ```bash
    python -m streamlit run app/streamlit/app.py
    ```
    This will launch a local server (default: http://localhost:8501) where you can select the runtime backend, test video feeds, and monitor latency metrics.

    ## 7. Running Video Inference (CLI)
    You can run the highly optimized asynchronous pipeline via CLI without the dashboard:
    ```bash
    python scripts/run_video.py --source 0 --device cpu
    ```

    ## 8. Benchmark Summary

    Benchmarked PyTorch FP32, ONNX FP32, and ONNX INT8 execution on the host CPU, demonstrating the trade-off between model footprint and runtime performance.

    *Asynchronous Synthetic Workload (Executed on Host CPU):*

    | Runtime | Precision | Capture FPS | Processing FPS | Total P50 Latency (ms) |
    |---|---|---|---|---|
    | **PyTorch** | FP32 | 135.0 | 16.0 | 56.0 ms |
    | **ONNX Runtime** | FP32 | 135.0 | 16.0 | 55.0 ms |
    | **ONNX Runtime** | INT8 | 135.0 | 18.0 | 51.0 ms |

    ## 9. Current Hardware / Runtime Limitation
    Currently, all demonstrated benchmarks and pipelines run strictly on **CPU**. While NVIDIA GPU / CUDA support architectures are implemented at the codebase level, hardware acceleration requires environment reconfiguration and has not been benchmarked. Future milestones include deployment on physical Edge AI hardware (e.g., NVIDIA Jetson / TensorRT).

    ## 10. Project Structure
    - `app/`: Streamlit dashboard and UI components.
    - `cv_engine/`: Core logic for detection, tracking, analytics, inference runtimes, and the async pipeline.
    - `docs/`: Technical documentation (Pipeline, Optimizations, Dashboard, Benchmarks).
    - `experiments/`: Benchmark outputs, JSON reports, and CSVs.
    - `scripts/`: Entrypoints for real-time video processing, benchmarking, and stress tests.
    - `tests/`: Comprehensive pytest suite.

    ## 11. Future Deployment Direction
    - **Hardware Acceleration**: Validate existing CUDA support with `torch.cuda` and ONNX Runtime CUDA Execution Providers.
    - **Edge Deployment**: Port the INT8 models and pipeline to a physical NVIDIA Jetson device using TensorRT.

    ## 12. Resume Project Entry

    **EdgeVision AI — Real-Time Multi-Task Computer Vision & Edge AI Platform**
    *Technologies: Python, OpenCV, PyTorch, ONNX Runtime, Streamlit, NumPy, scikit-image, scikit-learn*
    - Engineered a low-latency, freshness-oriented video processing pipeline utilizing a bounded asynchronous producer-consumer architecture, ensuring sub-60ms P50 latency and temporal robustness for object tracking.
    - Developed an Edge AI inference abstraction supporting PyTorch FP32, ONNX FP32, and dynamically quantized ONNX INT8 models. In the asynchronous synthetic benchmark on CPU, ONNX INT8 achieved approximately 51 ms P50 end-to-end latency.
    - Built a classical ML pipeline (HOG/PCA/SVM) and a robust tracking/analytics engine (Centroid matching + virtual tripwires), demonstrating end-to-end functionality in a local Streamlit dashboard.
