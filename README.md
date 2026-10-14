# EdgeDrive: Automotive AI & Edge Intelligence Platform

A production-oriented automotive perception platform that processes driving
driving video to detect road users, track objects over time, evaluate
uncertainty, and generate explainable perception events.

## Architecture

```
Video/Image Input → Detector (torchvision) → IoU Tracker → Event Engine → Results
                     ↕                        ↕              ↕
                  Benchmark Runner      Track State      Perception Events
```

### Components

| Module | Status | Description |
|--------|--------|-------------|
| `common/types.py` | ✅ Tested | Core data types (Detection, Track, Event, etc.) |
| `config/settings.py` | ✅ Tested | Pydantic configuration with YAML loading |
| `perception/detector.py` | ✅ Tested | Abstract detector interface + registry |
| `perception/torchvision_detector.py` | ✅ Tested | Real detector using torchvision Faster R-CNN |
| `perception/pipeline.py` | ✅ Tested | End-to-end perception pipeline |
| `tracking/iou_tracker.py` | ✅ Tested | IoU-based multi-object tracker |
| `events/engine.py` | ✅ Tested | Perception event generation |
| `ingestion/video.py` | ✅ Tested | Video/image sequence reader (OpenCV) |
| `ingestion/fixtures.py` | ✅ Tested | Synthetic test fixtures |
| `benchmarking/benchmark.py` | ✅ Tested | Reproducible benchmark runner |

### Environment

- Python 3.12, ARM64, Databricks serverless (CPU-only)
- torch 2.14.1+cpu, torchvision 0.29.1+cpu
- OpenCV 4.10.0 (headless)
- No GPU, no CARLA, no ROS 2 required for core functionality

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless==4.10.0.84 structlog pyyaml pydantic
```

## Primary Model

**fasterrcnn_mobilenet_v3_large_320_fpn** (torchvision)
- 19.4M parameters (~74 MB FP32)
- Input: 320×320 RGB
- COCO classes (80 object categories)
- Automotive-relevant: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- License: Apache 2.0

## Running Tests

```bash
# Set PYTHONPATH to include src directory
export PYTHONPATH=/path/to/edgedrive/src:$PYTHONPATH

# Run all unit tests
python -m pytest edgedrive/tests/ -v

# Run specific test module
python -m pytest edgedrive/tests/test_tracker.py -v
```

## Limitations

- CPU-only inference (no GPU on serverless compute)
- No real-world velocity estimation (pixel-space only, uncalibrated)
- No ONNX/TensorRT optimization (planned for Milestone 2)
- No FastAPI server or dashboard (planned for Milestone 3)
- No MLflow integration (planned for Milestone 4)
- Not validated for safety-critical vehicle control
- Pixel displacement ≠ calibrated real-world velocity

## Roadmap

- [x] Milestone 1: Working perception vertical slice
- [ ] Milestone 2: Benchmarking and model optimization (ONNX, quantization)
- [ ] Milestone 3: API and dashboard (FastAPI + Streamlit)
- [ ] Milestone 4: MLOps and Databricks (MLflow, Delta Lake)
- [ ] Milestone 5: Production engineering (Docker, CI, observability)
- [ ] Milestone 6: Advanced automotive integration (CARLA, ROS 2, fusion)

## License

Apache 2.0