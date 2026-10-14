"""Streamlit dashboard for EdgeDrive perception pipeline.

Provides a web UI for:
- Uploading/viewing images for detection
- Running the perception pipeline with visualization
- Displaying benchmark results
- Showing model metadata and pipeline stats

Run with: streamlit run src/edgedrive/dashboard/app.py
"""

from __future__ import annotations

import io
import time
from typing import Any, Optional

import numpy as np

from ..common.logging import get_logger
from ..config.settings import BenchmarkConfig, DetectorConfig, PipelineConfig, TrackerConfig
from ..perception.pipeline import PerceptionPipeline
from ..benchmarking.benchmark import BenchmarkRunner


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on a frame.

    Args:
        frame: HxWx3 RGB uint8 array.
        detections: List of Detection objects.

    Returns:
        Frame with drawn annotations (BGR for OpenCV/Streamlit display).
    """
    import cv2

    annotated = frame.copy()
    # Convert RGB to BGR for OpenCV drawing
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    colors = {
        "person": (0, 0, 255),
        "car": (0, 255, 0),
        "truck": (0, 255, 255),
        "bus": (0, 200, 0),
        "motorcycle": (255, 0, 0),
        "bicycle": (255, 255, 0),
        "traffic light": (0, 165, 255),
        "stop sign": (0, 0, 200),
    }
    default_color = (128, 128, 128)

    for det in detections:
        color = colors.get(det.class_name, default_color)
        x1, y1, x2, y2 = (
            int(det.bbox.x1), int(det.bbox.y1),
            int(det.bbox.x2), int(det.bbox.y2),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def run_dashboard() -> None:
    """Run the Streamlit dashboard.

    This function is called when running:
        streamlit run src/edgedrive/dashboard/app.py
    """
    import streamlit as st

    st.set_page_config(
        page_title="EdgeDrive Perception",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 EdgeDrive: Automotive AI Perception")
    st.markdown("Real-time object detection, tracking, and event generation for driving video.")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    enable_tracking = st.sidebar.checkbox("Enable Tracking", value=True)
    enable_events = st.sidebar.checkbox("Enable Events", value=True)
    latency_threshold = st.sidebar.slider("Latency Threshold (ms)", 10.0, 500.0, 100.0, 10.0)

    # --- Model Info ---
    st.sidebar.header("Model")
    model_name = st.sidebar.selectbox(
        "Detection Model",
        ["fasterrcnn_mobilenet_v3_large_320_fpn"],
    )

    # --- Initialize pipeline ---
    @st.cache_resource
    def get_pipeline(conf, track, events, latency):
        config = PipelineConfig(
            detector=DetectorConfig(
                model_name=model_name,
                pretrained=True,
                confidence_threshold=conf,
                device="cpu",
            ),
            tracker=TrackerConfig(min_hits=3, max_age=30),
            enable_tracking=track,
            enable_events=events,
            latency_threshold_ms=latency,
        )
        pipeline = PerceptionPipeline(config)
        pipeline.initialize()
        return pipeline

    try:
        pipeline = get_pipeline(confidence, enable_tracking, enable_events, latency_threshold)
        meta = pipeline.detector.metadata()
        st.sidebar.success(f"✅ {meta.name}\n{meta.num_parameters / 1e6:.1f}M params")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")
        return

    # --- Main content ---
    tab1, tab2, tab3 = st.tabs(["📷 Detection", "📊 Benchmark", "📈 Stats"])

    with tab1:
        st.header("Image Detection")
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            import cv2

            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(frame, caption="Input", use_container_width=True)

            result = pipeline.process_frame(frame, source=uploaded.name)
            annotated = draw_detections(frame, result.detections)

            with col2:
                st.image(annotated, caption=f"{result.num_detections} detections", use_container_width=True)

            st.write(f"**Inference time:** {result.inference_time_ms:.1f}ms")
            if result.detections:
                st.dataframe([
                    {
                        "Class": d.class_name,
                        "Confidence": f"{d.confidence:.3f}",
                        "BBox": f"({d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f})",
                        "Automotive": d.is_automotive_relevant,
                    }
                    for d in result.detections
                ])
            if result.tracks:
                st.write("**Active Tracks:**")
                st.dataframe([
                    {"ID": t.track_id, "Class": t.class_name, "State": t.state.value, "Hits": t.hits}
                    for t in result.tracks
                ])
        else:
            st.info("Upload an image to run detection.")

    with tab2:
        st.header("Benchmark")
        warmup = st.number_input("Warmup Iterations", 1, 20, 3)
        timed = st.number_input("Timed Iterations", 5, 100, 20)

        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                from ..ingestion.fixtures import create_random_frame

                bench = BenchmarkRunner(BenchmarkConfig(
                    warmup_iterations=int(warmup),
                    timed_iterations=int(timed),
                ))
                frames = [create_random_frame(320, 320, seed=42)]
                result = bench.run(
                    detector_config=pipeline._config.detector,
                    input_frames=frames,
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Latency", f"{result.latency_mean_ms:.1f}ms")
                col2.metric("P95 Latency", f"{result.latency_p95_ms:.1f}ms")
                col3.metric("Throughput", f"{result.throughput_fps:.1f} FPS")
                col4.metric("Model Size", f"{result.model_size_mb:.1f}MB")

    with tab3:
        st.header("Pipeline Statistics")
        stats = pipeline.get_stats()
        st.json(stats)


if __name__ == "__main__":
    run_dashboard()