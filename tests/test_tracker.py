"""Tests for the IoU tracker."""

import pytest

from edgedrive.common.types import BoundingBox, Detection, TrackState
from edgedrive.config.settings import TrackerConfig
from edgedrive.tracking.iou_tracker import IoUTracker


def make_detection(x1, y1, x2, y2, class_name="car", confidence=0.9, class_id=3):
    """Helper to create a detection."""
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
    )


class TestIoUTracker:
    def test_single_detection_creates_track(self):
        tracker = IoUTracker(TrackerConfig())
        dets = [make_detection(50, 50, 150, 150)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TENTATIVE
        assert tracks[0].hits == 1
        assert tracks[0].track_id == 0

    def test_matching_same_object(self):
        tracker = IoUTracker(TrackerConfig())
        # Frame 1: object at position A
        dets1 = [make_detection(50, 50, 150, 150)]
        tracks1 = tracker.update(dets1)
        track_id = tracks1[0].track_id

        # Frame 2: same object slightly moved
        dets2 = [make_detection(53, 50, 153, 150)]
        tracks2 = tracker.update(dets2)

        # Should have same track_id, now with 2 hits
        assert len(tracks2) == 1
        assert tracks2[0].track_id == track_id
        assert tracks2[0].hits == 2

    def test_track_confirmation(self):
        tracker = IoUTracker(TrackerConfig(min_hits=3))
        # Need 3 consecutive matches to confirm
        dets = [make_detection(50, 50, 150, 150)]
        for _ in range(3):
            tracks = tracker.update(dets)
        assert tracks[0].state == TrackState.CONFIRMED
        assert tracks[0].hits >= 3

    def test_no_detections_keeps_track(self):
        tracker = IoUTracker(TrackerConfig(max_age=30, min_hits=1))
        tracker.update([make_detection(50, 50, 150, 150)])
        # No detections for a few frames
        for _ in range(3):
            tracks = tracker.update([])
        # Track should still exist in tracks list (within max_age)
        all_tracks = tracker.tracks
        assert len(all_tracks) == 1
        assert all_tracks[0].misses == 3

    def test_track_removed_after_max_age(self):
        tracker = IoUTracker(TrackerConfig(max_age=3, min_hits=1))
        tracker.update([make_detection(50, 50, 150, 150)])
        # No detections for max_age frames
        for _ in range(3):
            tracker.update([])
        # After max_age misses, track should be removed
        active = tracker.active_tracks
        assert len(active) == 0

    def test_multiple_objects(self):
        tracker = IoUTracker(TrackerConfig())
        dets = [
            make_detection(50, 50, 150, 150),
            make_detection(200, 200, 300, 300),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_velocity_estimation(self):
        tracker = IoUTracker(TrackerConfig())
        tracker.update([make_detection(50, 50, 150, 150)])
        tracks = tracker.update([make_detection(60, 50, 160, 150)])
        assert tracks[0].velocity_px is not None
        dx, dy = tracks[0].velocity_px
        assert dx == pytest.approx(10.0)  # moved 10 pixels right
        assert dy == pytest.approx(0.0)

    def test_reset(self):
        tracker = IoUTracker(TrackerConfig())
        tracker.update([make_detection(50, 50, 150, 150)])
        assert len(tracker.active_tracks) == 1
        tracker.reset()
        assert len(tracker.active_tracks) == 0
        assert tracker._next_id == 0

    def test_low_confidence_filtered(self):
        tracker = IoUTracker(TrackerConfig(min_confidence=0.5))
        dets = [make_detection(50, 50, 150, 150, confidence=0.3)]
        tracks = tracker.update(dets)
        assert len(tracks) == 0