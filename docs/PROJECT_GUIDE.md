# EdgeDrive - Complete Project Guide

> Comprehensive reference for the entire EdgeDrive codebase:
> every file, class, function, design decision, and data flow.

---

## 1. Project Overview

EdgeDrive is a production-oriented automotive perception platform built on Databricks.
It processes driving video to detect road users (vehicles, pedestrians, cyclists),
tracks objects across consecutive frames, evaluates uncertainty, and generates
explainable perception events.

Core engineering question:
> How can we deliver reliable automotive perception under strict latency, memory,
> compute, and deployment constraints while preserving measurable model quality?

### Built (Milestone 1)
- Real object detection (torchvision Faster R-CNN MobileNetV3, 19.4M params)
- Multi-object tracking (custom IoU-based tracker)
- Perception event generation (8 event types)
- Reproducible benchmarking with CSV/JSON export
- 60 automated tests (all passing)
- Live demo notebook with real model loading and inference

### Not Yet Built
- ONNX export/quantization (Milestone 2)
- FastAPI server/Streamlit dashboard (Milestone 3)
- MLflow tracking (Milestone 4)
- Docker/CI/observability (Milestone 5)
- CARLA/ROS 2/sensor fusion (Milestone 6)

---

## 2. Environment and Constraints

| Property | Value |
|----------|-------|
| Python | 3.12.3 |
| Architecture | ARM64 (aarch64) |
| Compute | Databricks serverless (CPU-only, no GPU) |
| Runtime | client.6.3 |
| Cloud | AWS |

### Key Packages

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.14.1+cpu | CPU-only build for ARM64 |
| torchvision | 0.29.1+cpu | Pretrained detection models |
| opencv-python-headless | 4.10.0.84 | 5.x crashes on ARM64 - must use 4.x |
| numpy | 2.3.4 | Pre-installed |
| pandas | 2.3.3 | Pre-installed |
| pydantic | 2.13.3 | Pre-installed |
| pytest | 8.4.2 | Pre-installed |
| structlog | 26.1.0 | Installed during setup |
| mlflow | 3.12.0 | Pre-installed |

### Critical Constraints

1. No GPU - all inference runs on CPU (ARM64)
2. No __pycache__ - workspace FS does not support it; pytest needs --assert=plain
3. OpenCV 5.x causes SIGABRT on ARM64 - must pin to 4.10.0.84
4. File edits via API may not immediately persist to disk
5. Serverless compute - no persistent filesystem between sessions

---

## 3. Architecture Summary

```
Video/Image Input (OpenCV VideoReader or Synthetic Fixtures)
         |
         v
  PerceptionPipeline
    |-- Detector (ABC)
    |     |-- TorchvisionDetector (Faster R-CNN MobileNetV3)
    |     |     - detect(frame) -> list[Detection]
    |     |     - load() -> downloads pretrained COCO weights
    |
    |-- IoUTracker
    |     - update(detections) -> list[Track]
    |     - Greedy IoU matching, track lifecycle
    |
    |-- EventEngine
    |     - process_frame(...) -> list[PerceptionEvent]
    |     - 8 event types (vehicle/pedestrian/track/latency)
    |
    v
  Output: FrameResult (detections, tracks, events, timing)

  BenchmarkRunner - reproducible latency/throughput measurement
  Config (Pydantic) - YAML loading + env var overrides
```

### Design Principles
1. Pydantic v2 for all data types (validation, serialization)
2. Abstract interfaces - Detector ABC with registry pattern
3. Modular - each module has clear responsibility
4. Type hints everywhere
5. No mock inference in production (mocks only in tests, labelled)
6. Graceful degradation - CUDA auto-falls-back to CPU
7. Structured logging (structlog with stdlib fallback)

---

## 4. Repository Structure

```
edgedrive/                          <- Root project directory
  src/edgedrive/                  <- Main Python package
    __init__.py                 <- Package metadata (version 0.1.0, Apache-2.0)
    common/                     <- Shared infrastructure
      __init__.py               <- Module docstring
      types.py                  <- ALL core data types (Pydantic models)
      exceptions.py             <- Exception hierarchy
      logging.py                <- Structured logging setup
    config/                     <- Configuration management
      __init__.py               <- Exports AppConfig, load_config
      settings.py               <- Pydantic config models + YAML loader
    perception/                 <- Object detection
      __init__.py               <- Exports Detector, get_detector, etc.
      detector.py               <- Abstract Detector ABC + registry
      torchvision_detector.py   <- Real detector (torchvision models)
      pipeline.py               <- PerceptionPipeline orchestrator
    tracking/                   <- Object tracking
      __init__.py               <- Exports IoUTracker
      iou_tracker.py            <- IoU-based multi-object tracker
    events/                     <- Perception event generation
      __init__.py               <- Exports EventEngine
      engine.py                 <- Event generation logic
    ingestion/                  <- Data input
      __init__.py               <- Exports VideoReader, ImageSequenceReader
      video.py                  <- OpenCV video/image reader
      fixtures.py               <- Synthetic test frames
    benchmarking/               <- Performance measurement
      __init__.py               <- Exports BenchmarkRunner
      benchmark.py              <- Benchmark runner + CSV/JSON export
  tests/                          <- Test suite
    __init__.py                 <- Test package init
    test_types.py              <- Tests for BoundingBox, Detection, Track, etc.
    test_config.py             <- Tests for config loading and validation
    test_tracker.py            <- Tests for IoUTracker
    test_events.py             <- Tests for EventEngine
    test_pipeline.py           <- Tests for PerceptionPipeline (mock detector)
    test_ingestion.py           <- Tests for synthetic fixtures
  configs/                        <- Configuration files
    perception.yaml             <- Default perception configuration
  docs/                           <- Documentation
    PROJECT_GUIDE.md            <- THIS FILE
  README.md                       <- Project README with roadmap
```

---

## 5. File-by-File Reference

### 5.1 src/edgedrive/__init__.py
Package initialization with metadata: version 0.1.0, Apache-2.0 license.

### 5.2 src/edgedrive/common/types.py
Defines ALL core data types. Every module imports from here. Uses Pydantic v2.

Helper Functions:
- utc_now() -> datetime - Returns current UTC time (timezone-aware)
- gen_id(prefix: str) -> str - Generates unique 12-char hex ID with optional prefix

Enums:
- BoundingBoxFormat - XYXY, XYWH
- EventType - 8 types: VEHICLE_DETECTED, PEDESTRIAN_DETECTED, OBJECT_DETECTED,
  TRACK_LOST, TRACK_NEW, CONFIDENCE_DEGRADED, LATENCY_EXCEEDED, SENSOR_TIMESTAMP_MISMATCH
- TrackState - TENTATIVE, CONFIRMED, LOST, REMOVED
- ProcessingStatus - SUCCESS, WARNING, ERROR, SKIPPED

Pydantic Models:

**BoundingBox** (frozen) - Axis-aligned bbox in pixel coordinates (x1,y1,x2,y2)
  - Properties: width, height, area, center
  - Methods: iou(other) -> float (Intersection-over-Union)
  - Validators: x2 >= x1, y2 >= y1

**Detection** - Single object detection
  - Fields: bbox, class_id, class_name, confidence, model_name, model_version
  - Properties: is_automotive_relevant (person/car/bus/truck/motorcycle/bicycle/traffic light/stop sign)

**Track** - Tracked object across frames
  - Fields: track_id, bbox, class_id, class_name, confidence, state, age, hits,
    misses, frame_index, prev_bbox, velocity_px
  - Properties: is_confirmed (state==CONFIRMED), is_active (TENTATIVE or CONFIRMED)

**FrameMetadata** - Single frame metadata (frame_index, timestamp, width, height, source)

**FrameResult** - Complete perception result for one frame
  - Fields: frame_metadata, detections, tracks, inference_time_ms, total_time_ms,
    model_name, model_version, status, error_message
  - Properties: num_detections, num_tracks, automotive_detections

**PerceptionEvent** - Typed perception event
  - Fields: event_id (auto-gen), event_type, timestamp, frame_index, source_frame,
    track_id, class_name, confidence, evidence (dict), model_name, model_version, status

**ModelMetadata** (frozen) - Model info: name, version, architecture, num_parameters,
  input_resolution, num_classes, precision, framework, pretrained_dataset, license

**BenchmarkResult** - Benchmark metrics: run_id, timestamp, model_name, model_version,
  dataset_version, runtime, precision_mode, hardware, input_resolution, batch_size,
  num_iterations, warmup_iterations, latency_mean_ms, latency_p50_ms, latency_p95_ms,
  throughput_fps, model_size_mb, peak_memory_mb, accuracy_metric, status, error_message

**VideoMetadata** (frozen) - Video file info: path, width, height, fps, frame_count,
  duration_seconds, codec, file_size_bytes

### 5.3 src/edgedrive/common/exceptions.py
Exception hierarchy. All derive from EdgeDriveError:
- ConfigError - Config loading/validation failure
- IngestionError - Data ingestion failure
- DetectorError -> ModelLoadError, InferenceError
- TrackerError, PipelineError, EventError, BenchmarkError, ExportError

### 5.4 src/edgedrive/common/logging.py
Structured logging using structlog with stdlib fallback.
- configure_logging(level="INFO", json_output=False) - One-time setup
- get_logger(name="edgedrive") - Returns structlog or stdlib logger
- Global _configured flag prevents re-configuration

### 5.5 src/edgedrive/config/settings.py
Pydantic config models and YAML loader.

**DetectorConfig** - model_name, model_version, pretrained, num_classes,
  input_resolution (320x320), confidence_threshold (0.5), nms_threshold (0.5),
  device (cpu/cuda with auto-fallback), min_detection_area (100.0)

**TrackerConfig** - iou_threshold (0.3), max_age (30), min_hits (3), min_confidence (0.3)

**PipelineConfig** - detector (DetectorConfig), tracker (TrackerConfig),
  enable_tracking, enable_events, latency_threshold_ms (100.0), max_frame_count

**BenchmarkConfig** - warmup_iterations (5), timed_iterations (50), batch_size (1),
  input_resolution (320x320), output_dir, export_onnx

**AppConfig** - Top-level: pipeline, benchmark, log_level, log_json

**load_config(config_path=None, **overrides) -> AppConfig**
  - Loads YAML, applies env var overrides (EDGEDRIVE_DEVICE, EDGEDRIVE_LOG_LEVEL),
    applies programmatic overrides, validates with Pydantic
  - Raises ConfigError on missing file, YAML error, or validation failure

### 5.6 src/edgedrive/perception/detector.py
Abstract Detector interface and registry pattern.

**Detector (ABC)** - Abstract base class for all detectors:
  - load() -> None - Load model weights
  - detect(frame: np.ndarray, confidence_threshold=0.5) -> list[Detection]
  - metadata() -> ModelMetadata
  - is_loaded -> bool (abstract property)
  - warmup(num_iterations=3) -> None - Runs dummy inference (concrete method)

**Registry Pattern:**
- register_detector(name: str) - Decorator to register a Detector subclass
- get_detector(name: str, config: DetectorConfig) -> Detector - Create by name
- list_detectors() -> list[str] - List all registered names
- Global _REGISTRY: dict[str, type[Detector]]

### 5.7 src/edgedrive/perception/torchvision_detector.py
Real object detector using torchvision pretrained models. Registered as "torchvision".

**Constant:** COCO_CLASSES - 91 COCO class names (index 0 = background)

**TorchvisionDetector(Detector):**
  __init__(config: DetectorConfig)

  load() -> None:
    - Lazy imports torch/torchvision (not at module level)
    - Supports 4 models: fasterrcnn_mobilenet_v3_large_320_fpn (primary, 19.4M params),
      fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
    - Downloads pretrained COCO weights if pretrained=True
    - Moves to device, sets eval mode
    - Computes ModelMetadata with parameter count
    - Raises ModelLoadError on failure

  detect(frame, confidence_threshold=0.5) -> list[Detection]:
    - Resize frame to input_resolution (using cv2 or PIL fallback)
    - Normalize to [0,1], convert to tensor (C,H,W), add batch dim
    - Run inference with torch.no_grad()
    - Scale boxes back to original frame coordinates
    - Filter by confidence threshold and min_detection_area
    - Map class IDs to COCO_CLASSES names
    - Returns list[Detection]
    - Raises InferenceError on failure

  metadata() -> ModelMetadata - Returns cached metadata
  is_loaded -> bool

  Key design: Lazy imports allow module import without torch installed.

### 5.8 src/edgedrive/perception/pipeline.py
Orchestrates detection, tracking, and event generation.

**PerceptionPipeline:**
  __init__(config: PipelineConfig, detector: Detector? = None)
    - Creates TorchvisionDetector if detector is None
    - Creates IoUTracker if enable_tracking
    - Creates EventEngine if enable_events

  Properties: detector -> Detector, all_events -> list[PerceptionEvent]

  initialize() -> None - Loads the detector model

  process_frame(frame, frame_index=0, source="unknown") -> FrameResult:
    1. Detection: detector.detect(frame, confidence_threshold)
    2. Tracking: tracker.update(detections) (if enabled)
    3. Events: event_engine.process_frame(...) (if enabled)
    4. Returns FrameResult with timing and status
    5. Catches detection errors, marks status as ERROR

  process_frames(frames, source) -> Generator[FrameResult]
    - Respects max_frame_count config

  get_stats() -> dict: frames_processed, total_detections, avg_detections_per_frame,
    total_events, model_name, tracking_enabled, events_enabled

  reset() -> None - Clears tracker, events, counters

### 5.9 src/edgedrive/tracking/iou_tracker.py
IoU-based multi-object tracker. Inspired by SORT, no external dependencies.

Design: Greedy IoU matching (not Hungarian), track lifecycle
(TENTATIVE -> CONFIRMED -> LOST -> REMOVED), pixel-space velocity (uncalibrated),
no Kalman filtering.

**IoUTracker:**
  __init__(config: TrackerConfig)
  State: _tracks (dict[int, Track]), _next_id (int), _frame_index (int)

  Properties:
  - tracks -> list[Track] - All non-REMOVED tracks
  - active_tracks -> list[Track] - TENTATIVE and CONFIRMED only

  update(detections) -> list[Track]:
    1. Increment frame index
    2. Filter detections by min_confidence
    3. Match via _match()
    4. Update matched tracks (_update_track)
    5. Create new tracks for unmatched (_create_track)
    6. Age unmatched tracks - excludes matched and newly created
    7. Remove stale tracks (_remove_stale_tracks)
    8. Return active_tracks

  _match(detections) -> (matched_pairs, unmatched_det_indices):
    - Computes IoU matrix (active_tracks x detections)
    - Greedy: sorts pairs by IoU descending, picks best non-conflicting matches
    - Pairs below iou_threshold are not matched

  _create_track(detection) -> None:
    - Creates Track with next_id, state=TENTATIVE, misses=0, hits=1

  _update_track(track_id, detection) -> None:
    - Stores prev_bbox, computes pixel velocity (center displacement)
    - Resets misses, increments hits
    - Promotes to CONFIRMED if hits >= min_hits

  _age_track(track_id) -> None:
    - Increments misses and age
    - Transitions to LOST if misses >= max_age // 2

  _remove_stale_tracks() -> None:
    - Marks as REMOVED if misses >= max_age

  reset() -> None - Clears all state

  Bug fixed: Initially aged newly created tracks in same frame. Fixed by
  tracking newly_created IDs and excluding them from aging loop.

### 5.10 src/edgedrive/events/engine.py
Generates typed PerceptionEvent records. Separates measured facts,
heuristic indicators, and system facts. No invented collision probabilities.

**EventEngine:**
  __init__(latency_threshold_ms=100.0, confidence_threshold=0.3,
           model_name="unknown", model_version="unknown")
  State: _prev_track_ids (set[int]) for track_lost detection

  process_frame(frame_index, source, detections, tracks, inference_time_ms)
    -> list[PerceptionEvent]:

    1. Detection events (for automotive-relevant detections):
       - VEHICLE_DETECTED (car/bus/truck/motorcycle)
       - PEDESTRIAN_DETECTED (person)
       - OBJECT_DETECTED (bicycle/traffic light/stop sign)
       - CONFIDENCE_DEGRADED (if confidence < threshold)

    2. Tracking events:
       - TRACK_NEW (track reaches CONFIRMED with 3 hits)
       - TRACK_LOST (track ID from prev frame no longer present)

    3. System events:
       - LATENCY_EXCEEDED (if inference_time_ms > threshold)

  reset() -> None - Clears prev_track_ids

### 5.11 src/edgedrive/ingestion/video.py
Video and image sequence reading using OpenCV.

**VideoReader:**
  __init__(path: str, max_frames: int? = None) - Raises IngestionError if not found
  get_metadata() -> VideoMetadata - width, height, fps, frame_count, duration
  frames(convert_to_rgb=True) -> Generator[np.ndarray] - Yields HxWx3 uint8 frames
  close() -> None - Releases VideoCapture
  Context manager: with VideoReader(...) as reader: ...
  - Lazy open, BGR->RGB conversion, fps=0 defaults to 30.0

**ImageSequenceReader:**
  __init__(directory: str, max_frames: int? = None)
  Supported: .png, .jpg, .jpeg, .bmp, .tiff
  frames(convert_to_rgb=True) -> Generator[np.ndarray] - Sorted alphabetically

### 5.12 src/edgedrive/ingestion/fixtures.py
Deterministic synthetic frames for testing. NOT real inference replacements.

**Functions:**
- create_synthetic_frame(width=320, height=320, shapes=None, seed=42) -> np.ndarray
  - Gray (128) frame with colored rectangles + seed-dependent noise (0-10 per pixel)
  - Default shapes: 3 rectangles simulating road objects
  - Shape dict: {"type":"rect", "x1":..., "y1":..., "x2":..., "y2":..., "color":(R,G,B)}

- create_synthetic_video(num_frames=10, width=320, height=320, move_shapes=True, seed=42)
    -> list[np.ndarray]
  - If move_shapes=True, shapes move 3px/frame (for tracking tests)
  - Uses same seed for all frames (noise consistent for static tests)

- create_black_frame(width=320, height=320) -> np.ndarray - All zeros
- create_random_frame(width=320, height=320, seed=42) -> np.ndarray - Random noise

### 5.13 src/edgedrive/benchmarking/benchmark.py
Reproducible benchmark runner. Real measurements, no invented results.

**BenchmarkRunner:**
  __init__(config: BenchmarkConfig)

  run(detector_config, input_frames=None, dataset_version="synthetic")
    -> BenchmarkResult:
    1. Create + load detector
    2. Generate random frames if input_frames is None
    3. Warm-up: warmup_iterations inferences (discarded)
    4. Timed: timed_iterations inferences (latency recorded)
    5. Compute: mean, p50 (median), p95 (numpy percentile), FPS (1000/mean)
    6. Model size: num_parameters * 4 bytes / 1024^2 (FP32)
    7. Peak memory: resource.getrusage(RUSAGE_SELF).ru_maxrss
    8. Returns BenchmarkResult with all metrics
    9. On failure: returns BenchmarkResult with status="error"

  _get_model_size(detector) -> float? - From parameter count
  _get_peak_memory() -> float? - From resource.getrusage

  Static methods:
  - save_results_csv(results, output_path) -> str - CSV with all fields
  - save_results_json(results, output_path) -> str - JSON with model_dump

---

## 6. Test Suite (60 tests, all passing)

### Running Tests
```bash
export PYTHONPATH=/path/to/edgedrive/src:/path/to/edgedrive:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/edgedrive_pycache

python -m pytest edgedrive/tests/ \
  -v --tb=short --no-header --assert=plain \
  -p no:cacheprovider -o cache_dir=/tmp/pytest_cache
```
The --assert=plain flag is required because workspace FS does not support __pycache__.

### Test Files

**test_types.py (15 tests)** - BoundingBox (7: valid, invalid, zero-area, IoU identical/no-overlap/partial),
Detection (3: valid, invalid confidence, non-automotive), Track (2: creation, confirmed),
FrameResult (1), PerceptionEvent (1), ModelMetadata (1)

**test_config.py (11 tests)** - DetectorConfig (3: defaults, custom, invalid),
TrackerConfig (2: defaults, invalid IoU), PipelineConfig (2: defaults, nested),
load_config (4: default, YAML, missing file, invalid YAML)

**test_tracker.py (10 tests)** - IoUTracker:
- test_single_detection_creates_track
- test_matching_same_object (same track_id across frames)
- test_track_confirmation (3 hits -> CONFIRMED)
- test_no_detections_keeps_track (survives 3 empty frames)
- test_track_removed_after_max_age
- test_multiple_objects (2 tracks with different IDs)
- test_velocity_estimation (pixel displacement computed)
- test_reset, test_low_confidence_filtered

**test_events.py (10 tests)** - EventEngine:
- test_vehicle_detected, test_pedestrian_detected
- test_low_confidence_event (CONFIDENCE_DEGRADED)
- test_latency_exceeded, test_latency_not_exceeded
- test_track_new_event, test_track_lost_event
- test_non_automotive_no_event (cat -> no event)
- test_event_has_metadata

**test_pipeline.py (7 tests)** - PerceptionPipeline with MockDetector:
- MockDetector: Returns single "car" detection, test-only, NOT production
- test_process_single_frame, test_tracking_across_frames
- test_events_generated, test_events_disabled
- test_tracking_disabled, test_get_stats, test_reset

**test_ingestion.py (9 tests)** - Synthetic fixtures:
- test_synthetic_frame_shape, test_synthetic_frame_deterministic
- test_synthetic_frame_different_seeds, test_black_frame, test_random_frame
- test_synthetic_video_sequence, test_synthetic_video_moving_shapes
- test_synthetic_video_static_shapes, test_custom_shapes

### Bugs Found and Fixed
1. Tracker aging bug - newly created tracks aged in same frame; fixed with newly_created set
2. Fixture determinism - RandomState created but never used; fixed by adding seed-dependent noise
3. Static video frames - used seed+i per frame; fixed by using same seed for all frames

---

## 7. Benchmark Results (REAL, ARM64 CPU)

| Metric | Value |
|--------|-------|
| Model | fasterrcnn_mobilenet_v3_large_320_fpn |
| Parameters | 19.4M |
| Model size (FP32) | 74.0 MB |
| Input resolution | 320x320 |
| Warmup | 3 iterations |
| Timed | 20 iterations |
| Mean latency | 127.0 ms |
| P50 latency | 119.7 ms |
| P95 latency | 200.9 ms |
| Throughput | 7.9 FPS |
| Hardware | ARM64 CPU (Databricks serverless) |
| Runtime | PyTorch 2.14.1+cpu |
| Precision | FP32 |

Results vary based on compute load. Measured on shared serverless instance.

---

## 8. Known Issues and Workarounds

1. OpenCV 5.x SIGABRT on ARM64 - Pin to opencv-python-headless==4.10.0.84
2. No __pycache__ on workspace FS - Use --assert=plain, PYTHONDONTWRITEBYTECODE=1
3. File edit propagation delay - Sometimes need direct file I/O as workaround
4. Notebook cell targeting - editAsset can only target first cell nuid
5. Synthetic frames produce 0 detections - Expected (colored rectangles, not road scenes)

---

## 9. Demo Notebook

Name: EdgeDrive Perception Pipeline (ID: 1540073804304780)
Path: /Users/praveenbabugeddada7@gmail.com/Drafts/EdgeDrive Perception Pipeline

Cells:
1. Markdown title - Project description
2. Install Dependencies - %pip install torch/torchvision/opencv/structlog
3. EdgeDrive End-to-End Demo - Complete demo in one cell:
   - Imports all modules, lists detectors and automotive classes
   - Loads pretrained model (downloads ~74MB COCO weights)
   - Runs pipeline on 5 synthetic frames (detection + tracking + events)
   - Runs benchmark (3 warmup + 20 timed, real measurements)
   - Runs full test suite (60 tests)
4-5. Legacy cells (redundant with cell 3)

---

## 10. Data Flow

Frame Processing:
  Input (np.ndarray HxWx3 RGB)
    -> PerceptionPipeline.process_frame()
      -> Detector.detect() -> list[Detection]
      -> IoUTracker.update(detections) -> list[Track]
      -> EventEngine.process_frame() -> list[PerceptionEvent]
    -> FrameResult (detections + tracks + events + timing)

Benchmark:
  BenchmarkRunner.run(detector_config, input_frames)
    -> Create + load detector
    -> Warm-up N iterations (discarded)
    -> Timed M iterations (latency recorded)
    -> Compute mean/p50/p95/FPS
    -> BenchmarkResult

---

## 11. Roadmap

Completed:
- [x] Milestone 1: Working perception vertical slice
  - Real detector, IoU tracker, event engine, pipeline, benchmark, 60 tests, demo

Planned:
- [ ] Milestone 2: ONNX export, quantization, accuracy-vs-latency analysis
- [ ] Milestone 3: FastAPI server, Streamlit dashboard
- [ ] Milestone 4: MLflow tracking, Delta Lake, Databricks notebooks/jobs
- [ ] Milestone 5: Docker, CI, Prometheus, OpenTelemetry, deployment docs
- [ ] Milestone 6: CARLA, ROS 2, camera-radar fusion, TensorRT, Kafka

---

## 12. Dependency Graph

```
common.types <- common.exceptions
     |
     |-> config.settings (uses types, exceptions)
     |-> perception.detector (uses types)
     |      |-> perception.torchvision_detector (uses detector, types, config, exceptions)
     |             |-> perception.pipeline (uses detector, tracker, events, types, config)
     |-> tracking.iou_tracker (uses types, config)
     |-> events.engine (uses types)
     |-> ingestion.video (uses types, exceptions)
     |-> ingestion.fixtures (uses numpy only)
     |-> benchmarking.benchmark (uses types, config, detector, exceptions)

External: torch/torchvision (lazy), cv2 (lazy), pydantic, structlog (optional), pyyaml, numpy
```

---

*Generated as part of EdgeDrive Milestone 1. All info reflects actual codebase.*