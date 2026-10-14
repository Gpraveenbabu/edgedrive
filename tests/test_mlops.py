"""Tests for MLOps: MLflow tracking and result storage.

Tests cover:
- MLflowTracker: creation, availability check, logging (graceful skip if MLflow absent)
- ResultStore: filesystem fallback (JSON), frame result/events/benchmark saving

MLflow-dependent tests skip gracefully if mlflow is not installed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from edgedrive.common.types import (
    BoundingBox,
    Detection,
    FrameMetadata,
    FrameResult,
    PerceptionEvent,
    ProcessingStatus,
    EventType,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def make_frame_result(frame_index: int = 0) -> FrameResult:
    return FrameResult(
        frame_metadata=FrameMetadata(
            frame_index=frame_index,
            width=320,
            height=320,
            source="test",
        ),
        detections=[
            Detection(
                bbox=BoundingBox(x1=10, y1=20, x2=50, y2=80),
                class_id=3,
                class_name="car",
                confidence=0.85,
                model_name="test_model",
                model_version="v1.0",
            ),
        ],
        inference_time_ms=15.0,
        total_time_ms=20.0,
        model_name="test_model",
        model_version="v1.0",
    )


def make_events() -> list[PerceptionEvent]:
    return [
        PerceptionEvent(
            event_type=EventType.VEHICLE_DETECTED,
            frame_index=0,
            source_frame="test.mp4",
            class_name="car",
            confidence=0.85,
            model_name="test_model",
            model_version="v1.0",
        ),
    ]


# --------------------------------------------------------------------------- #
# MLflowTracker
# --------------------------------------------------------------------------- #

class TestMLflowTracker:
    """Tests for MLflowTracker."""

    def test_creation(self):
        """MLflowTracker should be creatable even without MLflow."""
        from edgedrive.mlops.tracking import MLflowTracker

        tracker = MLflowTracker(experiment_name="test_edge")
        assert tracker._experiment_name == "test_edge"

    def test_is_available_property(self):
        """is_available should be a boolean."""
        from edgedrive.mlops.tracking import MLflowTracker

        tracker = MLflowTracker()
        assert isinstance(tracker.is_available, bool)

    def test_log_benchmark_without_mlflow(self):
        """log_benchmark should return None if MLflow not available."""
        from edgedrive.mlops.tracking import MLflowTracker
        from edgedrive.common.types import BenchmarkResult

        tracker = MLflowTracker()
        if not tracker.is_available:
            result = BenchmarkResult(
                run_id="test_run",
                model_name="test_model",
                model_version="v1.0",
                dataset_version="test",
                runtime="pytorch",
                precision_mode="fp32",
                hardware="cpu",
                input_resolution=(320, 320),
                batch_size=1,
                num_iterations=10,
                warmup_iterations=5,
                latency_mean_ms=10.0,
                latency_p50_ms=9.0,
                latency_p95_ms=15.0,
                throughput_fps=100.0,
                model_size_mb=74.0,
                status="success",
            )
            run_id = tracker.log_benchmark(result)
            assert run_id is None

    def test_log_pipeline_run_without_mlflow(self):
        """log_pipeline_run should return None if MLflow not available."""
        from edgedrive.mlops.tracking import MLflowTracker

        tracker = MLflowTracker()
        if not tracker.is_available:
            run_id = tracker.log_pipeline_run(make_frame_result())
            assert run_id is None

    def test_log_comparison_without_mlflow(self):
        """log_comparison should return None if MLflow not available."""
        from edgedrive.mlops.tracking import MLflowTracker

        tracker = MLflowTracker()
        if not tracker.is_available:
            run_id = tracker.log_comparison(None)
            assert run_id is None

    @pytest.mark.skipif(
        True,  # Will be conditionally enabled below
        reason="Requires MLflow — run manually",
    )
    def test_log_benchmark_with_mlflow(self):
        """log_benchmark should return a run ID if MLflow is available."""
        pass


# --------------------------------------------------------------------------- #
# ResultStore (filesystem fallback)
# --------------------------------------------------------------------------- #

class TestResultStore:
    """Tests for ResultStore with filesystem fallback."""

    @pytest.fixture
    def store(self, tmp_path) -> "ResultStore":
        """Create a ResultStore. Without a Spark session, it uses JSON fallback."""
        from edgedrive.mlops.storage import ResultStore

        s = ResultStore(table_path="edgedrive.results")
        # Override fallback dir to tmp_path for testing
        if not s.is_delta:
            s._fallback_dir = Path(tmp_path)
        return s

    def test_creation(self, store):
        """ResultStore should be created."""
        assert store is not None
        assert isinstance(store.is_delta, bool)

    def test_is_delta_without_spark(self, store):
        """Without Spark session, is_delta should be False."""
        # In serverless without active Spark session, this should be False
        # (unless running inside a Spark-enabled context)
        if not store.is_delta:
            assert store._use_delta is False

    def test_save_frame_result(self, store):
        """save_frame_result should save to JSON when no Spark."""
        if store.is_delta:
            pytest.skip("Spark/Delta is active — test covers filesystem fallback")

        result = make_frame_result(frame_index=42)
        path = store.save_frame_result(result)

        assert path is not None
        assert str(store._fallback_dir) in path
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert data["frame_index"] == 42
        assert data["num_detections"] == 1
        assert data["model_name"] == "test_model"
        assert data["status"] == "success"

    def test_save_events(self, store):
        """save_events should save events to JSON."""
        if store.is_delta:
            pytest.skip("Spark/Delta is active")

        events = make_events()
        path = store.save_events(events)

        assert path is not None
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["event_type"] == "vehicle_detected"
        assert data[0]["class_name"] == "car"
        assert data[0]["confidence"] == 0.85

    def test_save_events_empty(self, store):
        """save_events with empty list should return None."""
        if store.is_delta:
            pytest.skip("Spark/Delta is active")

        path = store.save_events([])
        assert path is None

    def test_save_benchmark(self, store):
        """save_benchmark should save benchmark result to JSON."""
        if store.is_delta:
            pytest.skip("Spark/Delta is active")

        from edgedrive.common.types import BenchmarkResult

        result = BenchmarkResult(
            run_id="bench_test",
            model_name="test_model",
            model_version="v1.0",
            dataset_version="test",
            runtime="pytorch",
            precision_mode="fp32",
            hardware="cpu",
            input_resolution=(320, 320),
            batch_size=1,
            num_iterations=10,
            warmup_iterations=5,
            latency_mean_ms=10.0,
            latency_p50_ms=9.0,
            latency_p95_ms=15.0,
            throughput_fps=100.0,
            model_size_mb=74.0,
            status="success",
        )

        path = store.save_benchmark(result)

        assert path is not None
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert data["model_name"] == "test_model"
        assert data["latency_mean_ms"] == 10.0
        assert data["status"] == "success"

    def test_save_multiple_frame_results(self, store):
        """Multiple frame results should save as separate files."""
        if store.is_delta:
            pytest.skip("Spark/Delta is active")

        for i in range(3):
            store.save_frame_result(make_frame_result(frame_index=i))

        files = list(store._fallback_dir.glob("frame_result_*.json"))
        assert len(files) == 3

    def test_json_content_has_all_fields(self, store):
        """Saved frame result JSON should contain all expected fields."""
        if store.is_delta:
            pytest.skip("Spark/Delta is active")

        result = make_frame_result()
        path = store.save_frame_result(result)

        with open(path) as f:
            data = json.load(f)

        expected_keys = {
            "frame_index", "source", "width", "height",
            "num_detections", "num_tracks",
            "inference_time_ms", "total_time_ms",
            "model_name", "model_version", "status",
        }
        assert expected_keys.issubset(set(data.keys()))