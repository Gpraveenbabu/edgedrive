"""Tests for the event engine."""

import pytest

from edgedrive.common.types import BoundingBox, Detection, EventType, Track, TrackState
from edgedrive.events.engine import EventEngine


def make_detection(class_name="car", confidence=0.9, class_id=3):
    return Detection(
        bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
    )


def make_track(track_id=0, class_name="car", confidence=0.9, hits=1, state=TrackState.TENTATIVE):
    return Track(
        track_id=track_id,
        bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
        class_id=3,
        class_name=class_name,
        confidence=confidence,
        state=state,
        hits=hits,
        frame_index=0,
    )


class TestEventEngine:
    def test_vehicle_detected(self):
        engine = EventEngine()
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[make_detection(class_name="car")],
            tracks=[],
            inference_time_ms=15.0,
        )
        types = [e.event_type for e in events]
        assert EventType.VEHICLE_DETECTED in types

    def test_pedestrian_detected(self):
        engine = EventEngine()
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[make_detection(class_name="person", class_id=1)],
            tracks=[],
            inference_time_ms=15.0,
        )
        types = [e.event_type for e in events]
        assert EventType.PEDESTRIAN_DETECTED in types

    def test_low_confidence_event(self):
        engine = EventEngine(confidence_threshold=0.5)
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[make_detection(confidence=0.3)],
            tracks=[],
            inference_time_ms=15.0,
        )
        types = [e.event_type for e in events]
        assert EventType.CONFIDENCE_DEGRADED in types

    def test_latency_exceeded(self):
        engine = EventEngine(latency_threshold_ms=50.0)
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[],
            tracks=[],
            inference_time_ms=75.0,
        )
        types = [e.event_type for e in events]
        assert EventType.LATENCY_EXCEEDED in types

    def test_latency_not_exceeded(self):
        engine = EventEngine(latency_threshold_ms=100.0)
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[],
            tracks=[],
            inference_time_ms=50.0,
        )
        types = [e.event_type for e in events]
        assert EventType.LATENCY_EXCEEDED not in types

    def test_track_new_event(self):
        engine = EventEngine()
        track = make_track(hits=3, state=TrackState.CONFIRMED)
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[],
            tracks=[track],
            inference_time_ms=15.0,
        )
        types = [e.event_type for e in events]
        assert EventType.TRACK_NEW in types

    def test_track_lost_event(self):
        engine = EventEngine()
        # Frame 0: track present
        track = make_track(track_id=5)
        engine.process_frame(0, "test.mp4", [], [track], 15.0)
        # Frame 1: track gone
        events = engine.process_frame(1, "test.mp4", [], [], 15.0)
        types = [e.event_type for e in events]
        assert EventType.TRACK_LOST in types

    def test_non_automotive_no_event(self):
        engine = EventEngine()
        events = engine.process_frame(
            frame_index=0,
            source="test.mp4",
            detections=[make_detection(class_name="cat", class_id=17)],
            tracks=[],
            inference_time_ms=15.0,
        )
        types = [e.event_type for e in events]
        # Cat is not automotive-relevant, so no detection event
        assert EventType.VEHICLE_DETECTED not in types
        assert EventType.PEDESTRIAN_DETECTED not in types
        assert EventType.OBJECT_DETECTED not in types

    def test_event_has_metadata(self):
        engine = EventEngine(model_name="test_model", model_version="v1.0")
        events = engine.process_frame(
            frame_index=5,
            source="test.mp4",
            detections=[make_detection()],
            tracks=[],
            inference_time_ms=15.0,
        )
        assert len(events) > 0
        assert events[0].model_name == "test_model"
        assert events[0].model_version == "v1.0"
        assert events[0].frame_index == 5
        assert events[0].event_id.startswith("evt_")