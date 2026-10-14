"""Tests for the perception pipeline.

Uses a mock detector to test pipeline logic without loading a real model.
Mock detections are deterministic and clearly labelled — they are NOT
used in production code paths.
"""

import numpy as np
import pytest

from edgedrive.common.types import (
    BoundingBox,
    Detection,
    FrameResult,
    ModelMetadata,
    ProcessingStatus,
)
from edgedrive.config.settings import PipelineConfig, DetectorConfig, TrackerConfig
from edgedrive.perception.detector import Detector
from edgedrive.perception.pipeline import PerceptionPipeline


class MockDetector(Detector):
    """Deterministic mock detector for unit tests.

    Returns predefined detections for testing pipeline logic.
    NOT used in production code — only in tests.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._loaded = False
        self._call_count = 0

    def load(self) -> None:
        self._loaded = True

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list[Detection]:
        self._call_count += 1
        if not self._loaded:
            self.load()
        # Return a single car detection
        h, w = frame.shape[:2]
        return [
            Detection(
                bbox=BoundingBox(x1=10, y1=10, x2=min(100, w), y2=min(100, h)),
                class_id=3,
                class_name="car",
                confidence=0.95,
                model_name="mock",
                model_version="test",
            )
        ]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="mock",
            version="test",
            input_resolution=self._config.input_resolution,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class TestPerceptionPipeline:
    def test_process_single_frame(self):
        config = PipelineConfig(
            detector=DetectorConfig(),
            tracker=TrackerConfig(),
        )
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_index=0, source="test")

        assert isinstance(result, FrameResult)
        assert result.num_detections == 1
        assert result.detections[0].class_name == "car"
        assert result.inference_time_ms > 0
        assert result.status == ProcessingStatus.SUCCESS

    def test_tracking_across_frames(self):
        config = PipelineConfig(
            detector=DetectorConfig(),
            tracker=TrackerConfig(min_hits=1),
        )
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        result1 = pipeline.process_frame(frame, frame_index=0)
        result2 = pipeline.process_frame(frame, frame_index=1)

        # Both frames should have detections
        assert result1.num_detections == 1
        assert result2.num_detections == 1

        # Tracks should maintain the same ID across frames
        assert len(result2.tracks) == 1
        assert result2.tracks[0].track_id == result1.tracks[0].track_id
        assert result2.tracks[0].hits == 2

    def test_events_generated(self):
        config = PipelineConfig(
            detector=DetectorConfig(),
            tracker=TrackerConfig(min_hits=1),
            enable_events=True,
        )
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        pipeline.process_frame(frame, frame_index=0)

        # Should have generated events
        assert len(pipeline.all_events) > 0

    def test_events_disabled(self):
        config = PipelineConfig(
            detector=DetectorConfig(),
            enable_events=False,
        )
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        pipeline.process_frame(frame, frame_index=0)

        assert len(pipeline.all_events) == 0

    def test_tracking_disabled(self):
        config = PipelineConfig(
            detector=DetectorConfig(),
            enable_tracking=False,
        )
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, frame_index=0)

        assert result.num_detections == 1
        assert result.num_tracks == 0

    def test_get_stats(self):
        config = PipelineConfig()
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        pipeline.process_frame(frame, frame_index=0)
        pipeline.process_frame(frame, frame_index=1)

        stats = pipeline.get_stats()
        assert stats["frames_processed"] == 2
        assert stats["total_detections"] == 2
        assert stats["avg_detections_per_frame"] == 1.0

    def test_reset(self):
        config = PipelineConfig()
        detector = MockDetector(config.detector)
        pipeline = PerceptionPipeline(config, detector=detector)
        pipeline.initialize()

        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        pipeline.process_frame(frame, frame_index=0)
        assert pipeline.get_stats()["frames_processed"] == 1

        pipeline.reset()
        assert pipeline.get_stats()["frames_processed"] == 0
        assert len(pipeline.all_events) == 0