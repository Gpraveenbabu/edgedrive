
# EdgeDrive — Project Guide

**An edge-oriented computer vision and perception pipeline for autonomous driving research**

EdgeDrive is a modular Python project for experimenting with object detection, object tracking, perception events, video ingestion, and performance benchmarking. It is designed with a clear separation between perception components, tracking logic, event processing, configuration, and benchmarking.

This guide explains how to install EdgeDrive, run the pipeline, understand its outputs, execute tests, benchmark inference, and explore the codebase.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Run EdgeDrive](#5-run-edgedrive)
6. [Process a Video](#6-process-a-video)
7. [Understand the Outputs](#7-understand-the-outputs)
8. [Object Tracking and Events](#8-object-tracking-and-events)
9. [Benchmark Performance](#9-benchmark-performance)
10. [Configuration](#10-configuration)
11. [Run Tests](#11-run-tests)
12. [Project Structure](#12-project-structure)
13. [Troubleshooting](#13-troubleshooting)
14. [Limitations](#14-limitations)
15. [Roadmap](#15-roadmap)

---

## 1. Project Overview

EdgeDrive provides a foundation for building and evaluating a computer vision perception pipeline.

The pipeline follows this general flow:

```text
Video / Image Frames
        |
        v
Video or Image Ingestion
        |
        v
Frame Preprocessing
        |
        v
Object Detection
        |
        v
Object Tracking
        |
        v
Perception Event Generation
        |
        v
Results and Performance Metrics
```

Each component has a separate responsibility, making the system easier to test, maintain, and extend.

### Main components

| Component | Responsibility |
|---|---|
| Ingestion | Reads video files or image sequences |
| Perception | Detects objects in individual frames |
| Tracking | Associates detections across frames |
| Events | Generates events from detections, tracks, and performance conditions |
| Configuration | Controls pipeline and component settings |
| Benchmarking | Measures inference and pipeline performance |
| Common types | Defines shared data models |

### Who is this project for?

- Developers learning computer vision system architecture.
- Students experimenting with object detection and tracking.
- Engineers interested in edge-oriented inference benchmarking.
- Researchers building prototypes for autonomous driving perception.
- Contributors who want to add new detectors, trackers, or event rules.

**Important:** EdgeDrive is a research and engineering prototype. It is not a validated autonomous-driving system.

---

## 2. Features

### 2.1 Object Detection

EdgeDrive includes a detector interface and a TorchVision-based implementation.

Supported detector configurations include:

- Faster R-CNN MobileNetV3 Large 320 FPN
- Faster R-CNN MobileNetV3 Large FPN
- Faster R-CNN ResNet-50 FPN
- RetinaNet ResNet-50 FPN

The primary baseline is Faster R-CNN MobileNetV3 Large 320 FPN.

The detector supports confidence filtering, frame preprocessing, inference, and scaling predicted coordinates back to the original frame dimensions.

### 2.2 Object Tracking

The IoU tracker associates detections between frames using intersection-over-union matching.

It maintains track identities and supports track lifecycle states:

- `TENTATIVE`
- `CONFIRMED`
- `LOST`
- `REMOVED`

The current implementation uses greedy IoU matching. It does not implement Kalman filtering.

### 2.3 Event Generation

The event engine supports event types related to:

- Object detections
- New tracks
- Lost tracks
- Degraded detection confidence
- Excessive processing latency

Events allow downstream components to react to perception results without coupling that logic directly to the detector.

### 2.4 Video and Image Ingestion

The ingestion layer provides:

- `VideoReader` for video input.
- `ImageSequenceReader` for sequences of images.
- Synthetic fixtures for repeatable tests.

Video and image ingestion returns frames in RGB format.

Synthetic fixtures use artificial backgrounds, colored rectangles, and noise. They are useful for software tests, but they are not realistic road scenes.

### 2.5 Benchmarking

The benchmark runner measures processing performance and supports exporting results to CSV and JSON.

This makes it possible to compare model configurations and track changes across experiments.

---

## 3. Requirements

The recorded development environment includes:

- Python 3.12.3
- PyTorch 2.14.1 CPU build for the documented CPU benchmark
- TorchVision compatible with the installed PyTorch version
- OpenCV headless 4.10.0.84
- NumPy
- pandas
- Pydantic
- pytest
- structlog

The project has also been tested in a CPU-only ARM64 Databricks environment.

Your exact dependency versions may differ. Use the repository's dependency files as the source of truth for installation.

### Hardware

A GPU is not required for the CPU baseline. A compatible GPU may improve inference performance, but the installed PyTorch and TorchVision builds must support the intended device.

The first execution may download pretrained model weights if they are not already cached.

---

## 4. Installation

### Step 1: Clone the repository

```bash
git clone <YOUR_EDGEDRIVE_REPOSITORY_URL>
cd edgedrive
```

Replace the placeholder with your actual repository URL.

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

If the repository provides a requirements file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repository instead uses a project configuration file, follow the installation instructions defined there.

Do not install conflicting PyTorch and TorchVision builds. Choose versions compatible with each other and with your CPU or GPU environment.

### Step 4: Verify the environment

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

If these commands complete successfully, the core imports are available.

---

## 5. Run EdgeDrive

The precise entry point depends on the repository's available scripts or demo notebook. The following examples illustrate how to use the core pipeline classes; adapt imports and arguments to the checked-out version.

### Step 1: Import the pipeline

```python
from edgedrive.perception.pipeline import PerceptionPipeline
```

### Step 2: Load the project configuration

The configuration module provides settings such as `DetectorConfig`, `TrackerConfig`, `PipelineConfig`, and `BenchmarkConfig`.

```python
from edgedrive.config.settings import load_config

config = load_config()
```

Check the actual function signature and default configuration in your checkout before running this example.

### Step 3: Initialize and process frames

The pipeline exposes operations such as initialization, frame processing, batch frame processing, statistics retrieval, and reset.

Conceptually, usage looks like this:

```python
pipeline = PerceptionPipeline(config)
pipeline.initialize()

# Process an RGB frame using the interface supported by your version.
result = pipeline.process_frame(frame)

print(result)
```

Here, `frame` must be an RGB image in the format expected by the implementation.

The exact constructor and `process_frame` signatures should be confirmed in:

```text
src/edgedrive/perception/pipeline.py
```

### What happens internally?

1. The pipeline receives a frame.
2. The detector produces object detections.
3. The tracker associates detections with existing tracks.
4. The event engine evaluates supported event conditions.
5. The pipeline returns structured results and updates its statistics.

---

## 6. Process a Video

EdgeDrive provides a `VideoReader` for reading video frames.

A typical video-processing workflow is:

1. Provide a video file.
2. Read frames through the ingestion layer.
3. Process each frame through the perception pipeline.
4. Collect detections, tracks, events, and timing information.
5. Export or analyze the results.

The following is a conceptual example. Use the actual constructor, iteration interface, and pipeline signatures defined in your checkout.

```python
from edgedrive.ingestion.video import VideoReader

video_path = "data/example_video.mp4"

reader = VideoReader(video_path)

for frame in reader:
    result = pipeline.process_frame(frame)
    print(result)
```

**Before running:** Check `src/edgedrive/ingestion/video.py` for the exact `VideoReader` interface. Some versions may expose a context manager or an explicit frame-reading method rather than direct iteration.

### Input expectations

- Video paths must be accessible to the running process.
- Frames are represented in RGB format by the ingestion layer.
- Frame dimensions and frame rates depend on the source video.
- Large videos can require substantial processing time and memory.

For repeatable tests, use the project's synthetic fixtures rather than assuming synthetic images represent actual road conditions.

---

## 7. Understand the Outputs

EdgeDrive uses structured data models to represent perception results.

The shared types are defined in:

```text
src/edgedrive/common/types.py
```

Important models include:

- `BoundingBox`
- `Detection`
- `Track`
- `FrameMetadata`
- `FrameResult`
- `PerceptionEvent`
- `ModelMetadata`
- `BenchmarkResult`
- `VideoMetadata`

### 7.1 Detection output

A detection represents an object predicted by the model. Depending on the data model, it includes information such as the class, confidence score, and bounding box.

Illustrative example only:

```json
{
  "class_name": "car",
  "confidence": 0.94,
  "bbox": {
    "x1": 120,
    "y1": 85,
    "x2": 340,
    "y2": 260
  }
}
```

This is an example of the output concept, not a recorded prediction from a real EdgeDrive run.

### 7.2 Tracking output

A track represents an object associated across frames.

Illustrative example only:

```json
{
  "track_id": 7,
  "state": "CONFIRMED",
  "class_name": "car"
}
```

Track IDs help associate detections over time. They do not guarantee that an object identity will remain correct through occlusions or difficult scenes.

### 7.3 Event output

The event engine can generate events associated with detections, tracks, confidence, or processing latency.

Illustrative example only:

```json
{
  "event_type": "TRACK_NEW",
  "track_id": 7
}
```

Use the actual `PerceptionEvent` model and event engine implementation to determine the exact serialized fields and supported event values.

### 7.4 Benchmark output

Benchmark results can be exported to CSV or JSON. Depending on the benchmark configuration, the results can include timing statistics and model metadata.

Illustrative example only:

```json
{
  "warmup_iterations": 3,
  "timed_iterations": 20,
  "mean_latency_ms": 127.0,
  "p50_latency_ms": 119.7,
  "p95_latency_ms": 200.9,
  "throughput_fps": 7.9
}
```

These values correspond to a previously recorded CPU benchmark and should not be treated as guaranteed results on other hardware.

---

## 8. Object Tracking and Events

### 8.1 How tracking works

The current IoU tracker compares bounding boxes from new detections with existing tracks.

Intersection over Union measures the overlap between two bounding boxes:

```text
IoU = Area of Intersection / Area of Union
```

The tracker uses overlap to associate detections with tracks, subject to its configured matching logic.

### 8.2 Track lifecycle

The implementation supports these lifecycle states:

| State | Meaning |
|---|---|
| `TENTATIVE` | A newly created track that has not yet met confirmation requirements |
| `CONFIRMED` | A track that has met the confirmation requirements |
| `LOST` | A track that is temporarily not matched |
| `REMOVED` | A track that has been retired |

Exact transition thresholds depend on the tracker configuration and implementation.

### 8.3 Events

The event engine separates event logic from model inference.

Supported event categories include new/lost tracks, detections, degraded confidence, and excessive latency.

The event type enum may contain additional values that are not actively emitted by the current engine. Check `src/edgedrive/events/engine.py` before relying on a particular event.

---

## 9. Benchmark Performance

Benchmarking helps evaluate the latency and throughput of different detector configurations.

### 9.1 Recorded CPU baseline

The following results were recorded in an ARM64 CPU-only Databricks environment.

| Metric | Recorded value |
|---|---:|
| Model | Faster R-CNN MobileNetV3 Large 320 FPN |
| Parameters | 19.4 million |
| Approximate FP32 weights size | 74 MB |
| Input resolution | 320 × 320 |
| Warm-up iterations | 3 |
| Timed iterations | 20 |
| Mean latency | 127.0 ms |
| P50 latency | 119.7 ms |
| P95 latency | 200.9 ms |
| Throughput | 7.9 FPS |
| PyTorch build | 2.14.1+cpu |

These are recorded measurements, not a performance guarantee. Results can vary with hardware, software versions, input dimensions, CPU load, and benchmark settings.

### 9.2 Run a benchmark

The project includes a benchmark runner in:

```text
src/edgedrive/benchmarking/benchmark.py
```

The runner supports benchmarking and CSV/JSON export.

Consult the actual `BenchmarkRunner` constructor and `run` method before using it, since argument names and return types are version-dependent.

### 9.3 Benchmarking recommendations

For meaningful comparisons:

- Use the same input frames and image dimensions.
- Keep warm-up iterations separate from timed iterations.
- Record hardware and software versions.
- Compare latency percentiles, not only mean latency.
- Repeat runs to understand variability.
- Clearly distinguish CPU and GPU measurements.
- Do not compare results from different workloads as if they were equivalent.

---

## 10. Configuration

Configuration is defined in:

```text
src/edgedrive/config/settings.py
```

The available configuration models include:

| Configuration | Purpose |
|---|---|
| `DetectorConfig` | Detector settings |
| `TrackerConfig` | Tracking settings |
| `PipelineConfig` | Pipeline settings |
| `BenchmarkConfig` | Benchmark settings |
| `AppConfig` | Application-level settings |

The module also exposes `load_config`.

Check the source configuration definitions for supported fields, default values, validation rules, and file formats. Avoid assuming a field exists unless it is defined in your version.

---

## 11. Run Tests

Run the test suite from the repository root.

```bash
python -m pytest edgedrive/tests/ -v
```

If the package cannot be imported, configure the Python path according to your repository layout.

For example, in a compatible Unix environment:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
python -m pytest edgedrive/tests/ -v
```

If the repository uses a different test directory, adjust the command accordingly.

### Databricks or restricted environments

For the previously used CPU-only environment, a command that avoids writing bytecode and pytest cache files to restricted locations is:

```bash
export PYTHONPATH=/path/to/edgedrive/src:/path/to/edgedrive:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/edgedrive_pycache

python -m pytest edgedrive/tests/ \
  -v --tb=short --no-header --assert=plain \
  -p no:cacheprovider -o cache_dir=/tmp/pytest_cache
```

Replace `/path/to/edgedrive` with the actual repository path.

### What should you test?

- Data model validation.
- Detector output handling.
- Tracker association and lifecycle transitions.
- Event generation.
- Video and image ingestion.
- Benchmark execution and export.
- Pipeline behavior and configuration.

Always use the latest test run as the source of truth for the test count. Do not publish a hard-coded test total unless it has been verified against the current checkout.

---

## 12. Project Structure

The following is a conceptual map of the main source modules. Confirm the actual files in your checkout before relying on this as an exhaustive file listing.

```text
edgedrive/
├── src/
│   └── edgedrive/
│       ├── common/
│       │   └── types.py
│       ├── config/
│       │   └── settings.py
│       ├── perception/
│       │   ├── detector.py
│       │   ├── torchvision_detector.py
│       │   └── pipeline.py
│       ├── tracking/
│       │   └── iou_tracker.py
│       ├── events/
│       │   └── engine.py
│       ├── ingestion/
│       │   ├── video.py
│       │   └── fixtures.py
│       └── benchmarking/
│           └── benchmark.py
├── edgedrive/
│   └── tests/
├── docs/
│   └── PROJECT_GUIDE.md
├── README.md
└── requirements.txt
```

The exact directory structure may differ by branch or packaging setup.

### Where should you start reading?

| File | Start here when you want to... |
|---|---|
| `common/types.py` | Understand shared data models |
| `config/settings.py` | Understand configuration |
| `perception/detector.py` | Understand the detector abstraction |
| `perception/torchvision_detector.py` | Understand model inference |
| `perception/pipeline.py` | Understand orchestration |
| `tracking/iou_tracker.py` | Understand object association |
| `events/engine.py` | Understand event rules |
| `ingestion/video.py` | Understand video ingestion |
| `ingestion/fixtures.py` | Understand synthetic test inputs |
| `benchmarking/benchmark.py` | Understand benchmarking and exports |

---

## 13. Troubleshooting

### Import errors

**Possible cause:** The package is not installed or Python cannot find the source directory.

**Try:**

```bash
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
python -c "import edgedrive; print('Import successful')"
```

Confirm that the command matches your package layout.

### Pretrained model download fails

**Possible cause:** The environment has no network access or cannot reach the model hosting service.

**Try:**

- Run in an environment with permitted network access.
- Download or cache the required weights in an approved environment.
- Configure the application to use the available local cache, if supported.
- Check that the runtime user can access the cache directory.

Do not assume the detector will work offline before the required model weights are available.

### OpenCV import or runtime errors

**Possible cause:** Incompatible OpenCV wheels or conflicts between OpenCV packages.

**Try:**

- Check the version pinned by the project.
- Avoid installing multiple conflicting OpenCV packages.
- Recreate the environment using the repository's dependency specification.

The documented ARM64 environment used `opencv-python-headless==4.10.0.84` because OpenCV 5.x caused crashes in that environment.

### Zero detections on synthetic frames

**Possible cause:** The synthetic fixtures do not represent realistic road scenes.

**Explanation:** The fixtures use artificial backgrounds and colored rectangles. A pretrained detector may return zero detections or predictions that are not meaningful.

Use suitable real-world images or videos to evaluate detection quality.

### Benchmark results differ

**Possible cause:** Hardware, CPU contention, model warm-up, software versions, input dimensions, or workload differences.

**Try:** Record the environment, preserve the same input workload, repeat the benchmark, and compare percentiles as well as average latency.

### Disk or cache quota exceeded

**Possible cause:** Model weights, package caches, temporary files, or generated outputs are consuming the available storage.

**Try:**

- Inspect disk usage and model cache locations.
- Remove only files you can safely regenerate.
- Redirect temporary and test cache directories to an approved location with available space.
- Avoid downloading duplicate copies of large model weights.

---

## 14. Limitations

EdgeDrive is an engineering prototype, and its current limitations matter when interpreting results.

### Perception limitations

- The pretrained detectors are general-purpose models, not validated automotive perception systems.
- Synthetic fixtures are suitable for software testing, not road-scene evaluation.
- No autonomous-driving safety certification is implied.
- Detection quality depends on input conditions and model capabilities.

### Tracking limitations

- The current tracker uses greedy IoU matching.
- Kalman filtering is not implemented.
- Track IDs can change or become incorrect under occlusion, missed detections, or difficult motion.
- Velocity derived in pixel space is not calibrated to real-world physical units.

### Deployment limitations

The following capabilities should not be described as completed unless they have since been implemented and tested:

- Production REST API
- Streamlit dashboard
- ONNX export and quantization
- Full MLflow experiment-tracking integration
- Delta Lake integration
- Complete Docker and CI/CD deployment
- Production observability
- CARLA or ROS 2 integration
- Multi-sensor fusion

EdgeDrive should not be used as the sole perception system for a real vehicle or any safety-critical application.

---

## 15. Roadmap

The checklist below separates existing capabilities from planned extensions.

### Core functionality

- [x] Modular perception pipeline
- [x] TorchVision detector implementation
- [x] IoU-based object tracking
- [x] Track lifecycle management
- [x] Perception event engine
- [x] Video and image-sequence ingestion
- [x] Synthetic fixtures for testing
- [x] Benchmark runner
- [x] CSV and JSON benchmark export
- [x] Structured data models
- [x] Automated tests

### Planned extensions

- [ ] Production REST API
- [ ] Interactive Streamlit dashboard
- [ ] ONNX export and model quantization
- [ ] MLflow experiment tracking
- [ ] Delta Lake integration
- [ ] Docker packaging and CI/CD
- [ ] Production monitoring and observability
- [ ] CARLA simulation integration
- [ ] ROS 2 integration
- [ ] Camera and additional sensor fusion
- [ ] Evaluation on realistic driving datasets

**Roadmap note:** The checked items describe the core capabilities identified in the current project guide. The unchecked items are future work, not existing functionality.

---

## Contributing

Contributions are welcome.

Before submitting a change:

1. Read the relevant source module.
2. Add or update tests for the behavior.
3. Run the applicable test suite.
4. Document configuration changes and new outputs.
5. Report benchmark conditions when making performance claims.
6. Keep unsupported features clearly marked as planned.

---

## License

Add the project's actual license and attribution details here. Do not assume that pretrained model weights or third-party dependencies use the same license as EdgeDrive.

---

## Summary

EdgeDrive provides a modular foundation for experimenting with object detection, tracking, event generation, video ingestion, and inference benchmarking.

Start with the shared data models, detector implementation, perception pipeline, and test suite. Then extend one component at a time, validating each change with tests and reproducible benchmarks.

The goal is to keep the system modular, measurable, testable, and ready for carefully validated future extensions.
