# EdgeDrive — Project Report

## 1. What We Did

EdgeDrive is a production-oriented automotive perception platform that processes driving video to detect road users, track objects over time, evaluate uncertainty, and generate explainable perception events. The project was built across 6 milestones, each adding a new layer of capability:

| Milestone | Focus | Status |
|-----------|-------|--------|
| Milestone 1 | Core perception vertical slice (detection, tracking, events, ingestion, benchmarking) | ✅ Complete |
| Milestone 2 | Model optimization (ONNX export, ONNX Runtime inference, PyTorch-vs-ONNX comparison) | ✅ Complete |
| Milestone 3 | API & dashboard (FastAPI REST server, Streamlit visualization UI) | ✅ Complete |
| Milestone 4 | MLOps (MLflow experiment tracking, Delta Lake storage) | ✅ Complete |
| Milestone 5 | Observability (Prometheus metrics, health checks, Dockerfile) | ✅ Complete |
| Milestone 6 | Advanced integration (CARLA bridge, ROS 2 bridge, sensor fusion) | ✅ Complete |

### Architecture Overview

```
Video/Image Input → Detector (torchvision/ONNX) → IoU Tracker → Event Engine → Results
                      ↕                                ↕              ↕
                   Benchmark Runner               Track State    Perception Events
                      ↕                                ↕
                   Comparison (PyTorch vs ONNX)    Sensor Fusion (camera+radar)
                      ↕
                   API Server / Streamlit Dashboard
                      ↕
                   MLflow Tracking / Delta Storage / Prometheus Metrics
```

### Primary Model

**fasterrcnn_mobilenet_v3_large_320_fpn** (torchvision)
- 19.4M parameters (~74 MB FP32)
- Input: 320×320 RGB
- COCO classes (80 object categories)
- Automotive-relevant: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- License: Apache 2.0
- Runs on CPU (ARM64 compatible)

---

## 2. Technologies Used

| Technology | Version | Purpose |
|------------|---------|--------|
| Python | 3.12 | Primary language |
| PyTorch (torch) | 2.14.1+cpu | Deep learning framework for model inference |
| torchvision | 0.29.1+cpu | Pretrained detection models (Faster R-CNN, RetinaNet) |
| OpenCV (opencv-python-headless) | 4.10.0 | Video/image reading, resizing, BGR↔RGB conversion |
| NumPy | ≥1.24.0 | Array operations, IoU computation, image tensors |
| Pydantic | ≥2.0.0 | Data validation, config models, serialized types |
| PyYAML | ≥6.0 | YAML configuration file loading |
| structlog | ≥23.0.0 | Structured JSON logging |
| ONNX / onnxruntime | ≥1.14.0 / ≥1.15.0 | Model export and cross-runtime inference |
| FastAPI | ≥0.100.0 | REST API server for perception endpoints |
| uvicorn | ≥0.23.0 | ASGI server for FastAPI |
| Streamlit | ≥1.25.0 | Web dashboard for visualizing detections |
| MLflow | ≥2.0.0 | Experiment tracking, metric/parameter logging |
| prometheus-client | ≥0.17.0 | Prometheus-format metrics export |
| pytest | ≥7.0.0 | Unit and integration testing |
| Docker | python:3.12-slim | Containerized deployment (multi-stage Dockerfile) |
| CARLA | (optional) | Driving simulator for synthetic scenarios |
| rclpy / ROS 2 | (optional) | Robot OS bridge for autonomous vehicle stacks |

---

## 3. File-by-File Breakdown

### Root-Level Files

| File | Purpose | Output / Functions |
|------|---------|-------------------|
| `README.md` | Project overview, architecture diagram, model info, roadmap | Markdown documentation |
| `pyproject.toml` | Python package config, dependencies, pytest config, ruff linting rules | Package metadata, optional dependency groups (onnx, api, dashboard, mlops, observability) |
| `requirements.txt` | Pinned dependency list for pip install | Installable dependency list |
| `Dockerfile.py` | Multi-stage Docker build for edge deployment | Docker image — Stage 1: builder (installs deps), Stage 2: runtime (copies app, exposes port 8000, healthcheck) |

### `configs/`

| File | Purpose | Output / Functions |
|------|---------|-------------------|
| `perception.yaml` | Default YAML configuration for the perception pipeline | Loaded by `load_config()` → `AppConfig` Pydantic model; sets detector model, tracker params, benchmark settings, log level |

### `src/edgedrive/common/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `types.py` | Core data types — the contracts between all modules | `BoundingBox` (with `.iou()`, `.area`, `.center`, `.width`, `.height`), `Detection` (with `.is_automotive_relevant`), `Track`, `FrameMetadata`, `FrameResult`, `PerceptionEvent`, `BenchmarkResult`, `ModelMetadata`, `VideoMetadata`; Enums: `BoundingBoxFormat`, `EventType`, `TrackState`, `ProcessingStatus`; Helpers: `utc_now()`, `gen_id()` | Pydantic v2 model instances used throughout the pipeline |
| `exceptions.py` | Exception hierarchy — all errors derive from `EdgeDriveError` | `EdgeDriveError`, `ConfigError`, `IngestionError`, `DetectorError`, `TrackerError`, `PipelineError`, `EventError`, `BenchmarkError`, `ModelLoadError`, `InferenceError`, `ExportError` | Raised exceptions with descriptive messages |
| `logging.py` | Structured logging configuration | `configure_logging(level, json_output)` — configures structlog or falls back to stdlib; `get_logger(name)` — returns a logger instance | Structured log output (JSON or console format) |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/config/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `settings.py` | Configuration models and YAML loader | `DetectorConfig`, `TrackerConfig`, `PipelineConfig`, `BenchmarkConfig`, `AppConfig`; `load_config(config_path, **overrides)` — loads YAML, applies env var overrides (`EDGEDRIVE_DEVICE`, `EDGEDRIVE_LOG_LEVEL`), validates with Pydantic; `_cuda_available()` — graceful CUDA fallback | Validated `AppConfig` instance |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/perception/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `detector.py` | Abstract detector interface + registry | `Detector` (ABC) — `load()`, `detect()`, `metadata()`, `warmup()`, `is_loaded`; `register_detector(name)` — decorator to register detectors; `get_detector(name, config)` — factory by name; `list_detectors()` — list registered names | Detector instances (torchvision, onnx) |
| `torchvision_detector.py` | Real detector using torchvision Faster R-CNN | `TorchvisionDetector` — `load()` loads model weights (supports 4 models: fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn); `detect(frame, confidence_threshold)` — preprocesses (resize, normalize, tensor), runs inference, post-processes (scale boxes, filter by area/confidence); `metadata()` — returns `ModelMetadata`; `COCO_CLASSES` — 91-class name list | `list[Detection]` with bounding boxes, class names, confidence scores |
| `pipeline.py` | End-to-end perception pipeline orchestrator | `PerceptionPipeline` — `initialize()` loads model; `process_frame(frame, frame_index, source)` — detection → tracking → events with timing; `process_frames(frames, source)` — generator yielding `FrameResult` per frame; `get_stats()` — frames processed, total detections, events; `reset()` — clear state | `FrameResult` (detections, tracks, events, timing, status) |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/tracking/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `iou_tracker.py` | IoU-based multi-object tracker (SORT-inspired) | `IoUTracker` — `update(detections)` matches detections to tracks via greedy IoU, creates/updates/ages/removes tracks; `tracks` property (all non-removed); `active_tracks` property (tentative + confirmed); `_match()` — greedy IoU matrix matching; `_create_track()`, `_update_track()` (computes velocity_px), `_age_track()`, `_remove_stale_tracks()`; `reset()` | `list[Track]` with track_id, state (TENTATIVE→CONFIRMED→LOST→REMOVED), velocity, hits, misses |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/events/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `engine.py` | Perception event generation from detection/tracking outputs | `EventEngine` — `process_frame(frame_index, source, detections, tracks, inference_time_ms)` generates: `VEHICLE_DETECTED`, `PEDESTRIAN_DETECTED`, `OBJECT_DETECTED`, `CONFIDENCE_DEGRADED`, `TRACK_NEW`, `TRACK_LOST`, `LATENCY_EXCEEDED`; `reset()` — clears previous track IDs | `list[PerceptionEvent]` with event type, confidence, evidence dict |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/ingestion/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `video.py` | Video and image sequence reading via OpenCV | `VideoReader` — `get_metadata()` returns `VideoMetadata` (width, height, fps, frame_count, duration); `frames(convert_to_rgb)` generator yielding HxWx3 uint8 arrays; context manager support; `ImageSequenceReader` — reads PNG/JPG/BMP/TIFF from a directory | `VideoMetadata`, generator of `np.ndarray` frames |
| `fixtures.py` | Synthetic test fixtures for deterministic testing | `create_synthetic_frame(width, height, shapes, seed)` — colored rectangles with noise; `create_synthetic_video(num_frames, move_shapes, seed)` — sequence with moving shapes for tracking tests; `create_black_frame()`, `create_random_frame()` | `np.ndarray` (HxWx3 uint8) or `list[np.ndarray]` |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/benchmarking/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `benchmark.py` | Reproducible benchmark runner | `BenchmarkRunner` — `run(detector_config, input_frames, dataset_version)` measures mean/p50/p95 latency, throughput FPS, model size, peak memory; `save_results_csv()` and `save_results_json()` static methods | `BenchmarkResult` (latency, throughput, model size, status); CSV/JSON files |
| `comparison.py` | PyTorch vs ONNX Runtime comparison benchmark | `ComparisonBenchmark` — `run(detector_config, onnx_output_path, input_frames)` runs both runtimes, computes speedup/throughput/size ratios; `ComparisonResult` with `summary()` human-readable output | `ComparisonResult` (pytorch vs onnx metrics, speedup ratios) |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/optimization/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `onnx_exporter.py` | Export torchvision models to ONNX format | `OnnxExporter` — `export(output_path, opset_version, simplify)` loads model, wraps with `DetectionWrapper` (flattens dict output to tuple), runs `torch.onnx.export()`, validates with `onnx.checker`, optionally simplifies with onnxsim | `ExportResult` (onnx_path, file_size_mb, export_time_ms, opset_version) |
| `onnx_detector.py` | ONNX Runtime-based detector (registered as "onnx") | `OnnxDetector` — `load()` creates `ort.InferenceSession` with CPU/CUDA providers; `set_onnx_path(path)`; `detect(frame, confidence_threshold)` — same preprocessing as TorchvisionDetector, runs ONNX inference, builds `Detection` objects | `list[Detection]` identical format to torchvision detector |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/api/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `server.py` | FastAPI REST server for perception endpoints | `EdgeDriveServer` — `health()`, `detect(frame)`, `process_frame(frame, source)`, `model_info()`, `stats()`; `decode_image(image_b64)` — base64→numpy; `create_app(config)` — builds FastAPI app with endpoints: `/health`, `/detect`, `/detect/upload`, `/pipeline`, `/model`, `/stats` | JSON responses (detections, tracks, events, timing, model metadata) |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/dashboard/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `app.py` | Streamlit web dashboard for visual perception | `draw_detections(frame, detections)` — draws bounding boxes with class-specific colors; `run_dashboard()` — Streamlit app with 3 tabs: Detection (image upload + annotated output), Benchmark (latency/throughput metrics), Stats (pipeline stats JSON) | Streamlit web UI with annotated images, benchmark metrics, pipeline stats |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/mlops/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `storage.py` | Delta Lake / filesystem result persistence | `ResultStore` — `save_frame_result(result)` saves FrameResult to Delta table (Spark) or JSON file (fallback); `save_events(events)` saves PerceptionEvent list; `save_benchmark(result)` saves BenchmarkResult; `_save_to_delta()`, `_save_to_json()` internal methods | Delta table rows or JSON files in `outputs/results/` |
| `tracking.py` | MLflow experiment tracking | `MLflowTracker` — `log_benchmark(result)` logs params (model, runtime, hardware) and metrics (latency, throughput, size); `log_pipeline_run(frame_result)` logs detections/tracks/timing; `log_comparison(comparison_result)` logs PyTorch-vs-ONNX metrics | MLflow runs with logged params/metrics |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/observability/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `metrics.py` | Prometheus-compatible metrics and health checks | `MetricsCollector` — `increment(name, value)` counters; `observe(name, value)` histograms (keeps last 1000); `set_gauge(name, value)`; `get_histogram_stats()` returns count/mean/min/max/p50/p95; `export_prometheus()` text format; `export_dict()` JSON format; `HealthCheck` — `register()`, `liveness()`, `readiness()` | Prometheus text format metrics string; health status dicts |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/fusion/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `sensor_fusion.py` | Camera-radar late sensor fusion | `SensorFusionEngine` — `fuse(camera_detections, radar_detections, image_width, fov_deg)` associates camera bboxes with radar returns by azimuth proximity, computes weighted fused confidence (camera 0.6 / radar 0.4), handles unmatched camera-only and radar-only detections; `RadarDetection` (range, azimuth, velocity, rcs), `FusedDetection` (fused confidence, velocity, sources) | `list[FusedDetection]` with fused confidence, range, velocity, source list |
| `__init__.py` | Package init | — | — |

### `src/edgedrive/simulation/`

| File | Purpose | Key Functions / Classes | Output |
|------|---------|------------------------|-------|
| `carla_bridge.py` | CARLA simulator bridge for synthetic driving scenarios | `CarlaBridge` — `connect()` connects to CARLA server; `setup_camera(vehicle)` spawns RGB camera sensor; `frames(max_frames, timeout_s)` generator yielding camera frames; `spawn_vehicle(vehicle_type)`; `disconnect()` cleanup; context manager support | `np.ndarray` frames (HxWx3 uint8 RGB) from CARLA simulation |
| `ros_bridge.py` | ROS 2 communication bridge for AV stack integration | `ROS2Bridge` — `start(pipeline)` initializes rclpy node, subscribes to `/edgedrive/camera/image`, publishes detections to `/edgedrive/detections`, events to `/edgedrive/events`; `spin()` blocking; `stop()` cleanup | ROS 2 topic messages (JSON detections, events) |
| `__init__.py` | Package init | — | — |

### `tests/`

| File | Purpose | What It Tests |
|------|---------|-------------|
| `test_types.py` | Core data types | BoundingBox creation/validation, IoU computation, Detection properties, Track state transitions, FrameResult construction |
| `test_config.py` | Configuration loading | Default config, YAML loading, env var overrides, CUDA fallback, validation errors |
| `test_tracker.py` | IoU tracker | Track creation, matching, aging, removal, velocity estimation, reset behavior |
| `test_events.py` | Event engine | Vehicle/pedestrian/object detection events, confidence degraded, track new/lost, latency exceeded |
| `test_ingestion.py` | Video/image ingestion | VideoReader metadata, frame generation, ImageSequenceReader, fixtures (synthetic frames/video) |
| `test_pipeline.py` | Perception pipeline | End-to-end frame processing, tracking integration, event generation, stats, reset, error handling |
| `test_onnx.py` | ONNX optimization | ONNX export, ONNX detector inference, comparison benchmark (PyTorch vs ONNX) |
| `test_api.py` | FastAPI server | Health endpoint, detection, pipeline processing, model info, stats, image decoding |
| `test_mlops.py` | MLOps integration | MLflow tracking (benchmark/pipeline/comparison logging), Delta/JSON storage |
| `test_observability.py` | Metrics & health | Counter/histogram/gauge operations, Prometheus export, health checks |
| `test_fusion.py` | Sensor fusion | Camera-radar association, fused confidence, unmatched handling, radar-only detection |
| `test_simulation.py` | Simulation bridges | CARLA bridge (mock), ROS 2 bridge (mock) — verifies interface contracts |
| `__init__.py` | Package init | — |

### `docs/`

| File | Purpose |
|------|---------|
| `PROJECT_GUIDE.md` | Detailed developer guide with setup instructions and architecture explanations |

---

## 4. Data Flow Summary

```
Input (video/image) 
  → VideoReader.frames() or fixtures.create_synthetic_video()
  → PerceptionPipeline.process_frame()
    → TorchvisionDetector.detect() or OnnxDetector.detect()  →  list[Detection]
    → IoUTracker.update(detections)                           →  list[Track]
    → EventEngine.process_frame(detections, tracks, timing)    →  list[PerceptionEvent]
  → FrameResult (detections + tracks + events + timing + status)
  → Optional: ResultStore.save_frame_result() / MLflowTracker.log_pipeline_run()
  → Optional: EdgeDriveServer.detect() / Streamlit dashboard display
```

## 5. Test Suite

All tests pass using `pytest`. Run with:
```bash
export PYTHONPATH=/path/to/edgedrive/src:$PYTHONPATH
python -m pytest edgedrive/tests/ -v
```

Test modules cover unit tests (types, config, tracker, events, ingestion) and integration tests (pipeline, ONNX, API, MLOps, observability, fusion, simulation).

## 6. Environment

- Python 3.12, ARM64, Databricks serverless (CPU-only)
- torch 2.14.1+cpu, torchvision 0.29.1+cpu
- OpenCV 4.10.0 (headless)
- No GPU, no CARLA, no ROS 2 required for core functionality
- All optional dependencies (onnx, fastapi, streamlit, mlflow, carla, rclpy) are lazily imported

## 7. License

Apache 2.0