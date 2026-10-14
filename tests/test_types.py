"""Tests for core data types."""

import pytest
from pydantic import ValidationError

from edgedrive.common.types import (
    BoundingBox,
    Detection,
    FrameMetadata,
    FrameResult,
    ModelMetadata,
    PerceptionEvent,
    ProcessingStatus,
    Track,
    TrackState,
    EventType,
)


class TestBoundingBox:
    def test_valid_bbox(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.area == 90 * 180
        cx, cy = bbox.center
        assert cx == 55.0
        assert cy == 110.0

    def test_invalid_x2_lt_x1(self):
        with pytest.raises(ValidationError, match="x2"):
            BoundingBox(x1=100, y1=10, x2=50, y2=100)

    def test_invalid_y2_lt_y1(self):
        with pytest.raises(ValidationError, match="y2"):
            BoundingBox(x1=10, y1=100, x2=50, y2=50)

    def test_zero_area_bbox(self):
        bbox = BoundingBox(x1=50, y1=50, x2=50, y2=50)
        assert bbox.area == 0

    def test_iou_identical(self):
        b1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b2 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert b1.iou(b2) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        b1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b2 = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert b1.iou(b2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        b1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        # intersection = 50x50 = 2500, union = 10000 + 10000 - 2500 = 17500
        assert b1.iou(b2) == pytest.approx(2500 / 17500)


class TestDetection:
    def test_valid_detection(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        det = Detection(
            bbox=bbox, class_id=3, class_name="car", confidence=0.95,
        )
        assert det.class_name == "car"
        assert det.is_automotive_relevant is True
        assert det.confidence == 0.95

    def test_invalid_confidence(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        with pytest.raises(ValidationError):
            Detection(bbox=bbox, class_id=1, class_name="person", confidence=1.5)

    def test_non_automotive(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        det = Detection(
            bbox=bbox, class_id=17, class_name="cat", confidence=0.8,
        )
        assert det.is_automotive_relevant is False


class TestTrack:
    def test_track_creation(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        track = Track(
            track_id=0, bbox=bbox, class_id=3, class_name="car",
            confidence=0.9, frame_index=0,
        )
        assert track.state == TrackState.TENTATIVE
        assert track.is_confirmed is False
        assert track.is_active is True
        assert track.hits == 1
        assert track.misses == 0

    def test_track_confirmed(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        track = Track(
            track_id=0, bbox=bbox, class_id=3, class_name="car",
            confidence=0.9, state=TrackState.CONFIRMED, frame_index=5,
        )
        assert track.is_confirmed is True
        assert track.is_active is True


class TestFrameResult:
    def test_frame_result_creation(self):
        meta = FrameMetadata(frame_index=0, width=320, height=320, source="test.mp4")
        result = FrameResult(
            frame_metadata=meta,
            detections=[],
            tracks=[],
            inference_time_ms=15.5,
            total_time_ms=20.0,
        )
        assert result.num_detections == 0
        assert result.num_tracks == 0
        assert result.status == ProcessingStatus.SUCCESS
        assert result.error_message is None


class TestPerceptionEvent:
    def test_event_creation(self):
        event = PerceptionEvent(
            event_type=EventType.VEHICLE_DETECTED,
            frame_index=0,
            class_name="car",
            confidence=0.9,
        )
        assert event.event_type == EventType.VEHICLE_DETECTED
        assert event.event_id.startswith("evt_")
        assert event.model_name == "unknown"


class TestModelMetadata:
    def test_metadata_creation(self):
        meta = ModelMetadata(name="test_model", version="v1.0")
        assert meta.name == "test_model"
        assert meta.input_resolution == (320, 320)
        assert meta.precision == "fp32"