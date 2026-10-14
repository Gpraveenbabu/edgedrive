"""Tests for data ingestion and fixtures."""

import numpy as np
import pytest

from edgedrive.common.exceptions import IngestionError
from edgedrive.ingestion.fixtures import (
    create_black_frame,
    create_random_frame,
    create_synthetic_frame,
    create_synthetic_video,
)


class TestSyntheticFixtures:
    def test_synthetic_frame_shape(self):
        frame = create_synthetic_frame(width=320, height=240)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_synthetic_frame_deterministic(self):
        f1 = create_synthetic_frame(seed=42)
        f2 = create_synthetic_frame(seed=42)
        assert np.array_equal(f1, f2)

    def test_synthetic_frame_different_seeds(self):
        f1 = create_synthetic_frame(seed=42)
        f2 = create_synthetic_frame(seed=99)
        assert not np.array_equal(f1, f2)

    def test_black_frame(self):
        frame = create_black_frame(320, 240)
        assert frame.shape == (240, 320, 3)
        assert frame.sum() == 0  # All zeros

    def test_random_frame(self):
        frame = create_random_frame(320, 240, seed=42)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        # Should have non-zero values
        assert frame.sum() > 0

    def test_synthetic_video_sequence(self):
        frames = create_synthetic_video(num_frames=5, width=320, height=240)
        assert len(frames) == 5
        for f in frames:
            assert f.shape == (240, 320, 3)
            assert f.dtype == np.uint8

    def test_synthetic_video_moving_shapes(self):
        frames = create_synthetic_video(num_frames=10, move_shapes=True)
        # Frames should differ (shapes move)
        assert not np.array_equal(frames[0], frames[9])

    def test_synthetic_video_static_shapes(self):
        frames = create_synthetic_video(num_frames=5, move_shapes=False)
        # All frames should be identical (no movement)
        for i in range(1, 5):
            assert np.array_equal(frames[0], frames[i])

    def test_custom_shapes(self):
        shapes = [
            {"type": "rect", "x1": 0, "y1": 0, "x2": 100, "y2": 100, "color": (255, 0, 0)},
        ]
        frame = create_synthetic_frame(320, 320, shapes=shapes)
        # Top-left corner should be approximately red (noise may shift values slightly)
        pixel = frame[50, 50]
        assert pixel[0] > 245  # Red channel dominant
        assert pixel[1] < 15   # Green channel low
        assert pixel[2] < 15   # Blue channel low