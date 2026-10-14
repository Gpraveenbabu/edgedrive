"""Tests for camera-radar sensor fusion.

Tests cover:
- RadarDetection: creation, polar→Cartesian conversion
- FusedDetection: creation, field validation
- SensorFusionEngine: association, fused confidence, unmatched handling
"""

from __future__ import annotations

import math

import pytest

from edgedrive.common.types import BoundingBox, Detection
from edgedrive.fusion.sensor_fusion import (
    SensorType,
    RadarDetection,
    FusedDetection,
    SensorFusionEngine,
)


# --------------------------------------------------------------------------- #
# RadarDetection
# --------------------------------------------------------------------------- #

class TestRadarDetection:
    """Tests for RadarDetection model."""

    def test_creation(self):
        """RadarDetection should be created with required fields."""
        rd = RadarDetection(
            range_m=50.0,
            azimuth_deg=10.0,
            velocity_ms=-5.0,
        )
        assert rd.range_m == 50.0
        assert rd.azimuth_deg == 10.0
        assert rd.velocity_ms == -5.0
        assert rd.confidence == 0.5  # default
        assert rd.sensor_id == "radar_front"  # default

    def test_position_m(self):
        """position_m should convert polar to Cartesian."""
        rd = RadarDetection(range_m=10.0, azimuth_deg=0.0, velocity_ms=0.0)
        x, y = rd.position_m
        assert x == pytest.approx(10.0, abs=0.01)  # cos(0) = 1
        assert y == pytest.approx(0.0, abs=0.01)  # sin(0) = 0

    def test_position_m_90_degrees(self):
        """position_m at 90° should give (0, range)."""
        rd = RadarDetection(range_m=10.0, azimuth_deg=90.0, velocity_ms=0.0)
        x, y = rd.position_m
        assert x == pytest.approx(0.0, abs=0.01)
        assert y == pytest.approx(10.0, abs=0.01)

    def test_position_m_45_degrees(self):
        """position_m at 45° should give equal x and y."""
        rd = RadarDetection(range_m=10.0, azimuth_deg=45.0, velocity_ms=0.0)
        x, y = rd.position_m
        assert x == pytest.approx(y, abs=0.01)
        assert x == pytest.approx(10.0 * math.cos(math.radians(45)), abs=0.01)

    def test_negative_range_rejected(self):
        """RadarDetection should reject negative range."""
        with pytest.raises(Exception):
            RadarDetection(range_m=-1.0, azimuth_deg=0.0, velocity_ms=0.0)

    def test_confidence_bounds(self):
        """Confidence should be clamped to [0, 1]."""
        with pytest.raises(Exception):
            RadarDetection(range_m=10.0, azimuth_deg=0.0, velocity_ms=0.0, confidence=1.5)

    def test_with_rcs(self):
        """RadarDetection should accept rcs_db field."""
        rd = RadarDetection(
            range_m=30.0,
            azimuth_deg=-15.0,
            velocity_ms=2.0,
            rcs_db=-10.0,
        )
        assert rd.rcs_db == -10.0


# --------------------------------------------------------------------------- #
# FusedDetection
# --------------------------------------------------------------------------- #

class TestFusedDetection:
    """Tests for FusedDetection model."""

    def test_creation(self):
        """FusedDetection should be created with required fields."""
        fd = FusedDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=50, y2=80),
            class_name="car",
            fused_confidence=0.8,
        )
        assert fd.class_name == "car"
        assert fd.fused_confidence == 0.8
        assert fd.sources == []  # default

    def test_with_all_fields(self):
        """FusedDetection should accept all optional fields."""
        fd = FusedDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            class_name="truck",
            fused_confidence=0.9,
            camera_confidence=0.85,
            radar_confidence=0.95,
            range_m=40.0,
            velocity_ms=-3.0,
            azimuth_deg=5.0,
            sources=[SensorType.CAMERA, SensorType.RADAR],
        )
        assert fd.camera_confidence == 0.85
        assert fd.radar_confidence == 0.95
        assert fd.range_m == 40.0
        assert fd.velocity_ms == -3.0
        assert SensorType.CAMERA in fd.sources
        assert SensorType.RADAR in fd.sources

    def test_confidence_bounds(self):
        """FusedDetection should reject confidence > 1."""
        with pytest.raises(Exception):
            FusedDetection(
                bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                class_name="car",
                fused_confidence=1.5,
            )


# --------------------------------------------------------------------------- #
# SensorFusionEngine
# --------------------------------------------------------------------------- #

class TestSensorFusionEngine:
    """Tests for SensorFusionEngine."""

    @pytest.fixture
    def engine(self) -> SensorFusionEngine:
        return SensorFusionEngine(
            max_assoc_distance_m=5.0,
            camera_weight=0.6,
            radar_weight=0.4,
        )

    @pytest.fixture
    def camera_detection(self) -> Detection:
        """A camera detection centered horizontally (azimuth ~0°)."""
        return Detection(
            bbox=BoundingBox(x1=140, y1=100, x2=180, y2=200),
            class_id=3,
            class_name="car",
            confidence=0.8,
            model_name="torchvision",
            model_version="v1",
        )

    @pytest.fixture
    def radar_detection(self) -> RadarDetection:
        """A radar detection at 0° azimuth (centered)."""
        return RadarDetection(
            range_m=50.0,
            azimuth_deg=0.0,
            velocity_ms=-5.0,
            confidence=0.7,
        )

    def test_fuse_empty_inputs(self, engine):
        """Fusing empty lists should return empty list."""
        result = engine.fuse([], [])
        assert result == []

    def test_fuse_camera_only(self, engine, camera_detection):
        """Camera-only (no radar) should return camera-only FusedDetection."""
        result = engine.fuse([camera_detection], [])

        assert len(result) == 1
        assert result[0].class_name == "car"
        assert result[0].fused_confidence == 0.8
        assert result[0].camera_confidence == 0.8
        assert result[0].radar_confidence is None
        assert SensorType.CAMERA in result[0].sources
        assert SensorType.RADAR not in result[0].sources

    def test_fuse_radar_only(self, engine, radar_detection):
        """Radar-only (no camera) should return radar-only detection with approximate bbox."""
        result = engine.fuse([], [radar_detection])

        assert len(result) == 1
        assert result[0].class_name == "unknown_radar"
        assert result[0].fused_confidence < radar_detection.confidence
        assert result[0].radar_confidence == 0.7
        assert result[0].range_m == 50.0
        assert result[0].velocity_ms == -5.0
        assert SensorType.RADAR in result[0].sources

    def test_fuse_matched_pair(self, engine, camera_detection, radar_detection):
        """Camera + radar at similar azimuth should produce a fused detection."""
        result = engine.fuse([camera_detection], [radar_detection], image_width=320, fov_deg=90.0)

        assert len(result) == 1
        assert result[0].class_name == "car"
        assert SensorType.CAMERA in result[0].sources
        assert SensorType.RADAR in result[0].sources
        assert result[0].camera_confidence == 0.8
        assert result[0].radar_confidence == 0.7
        assert result[0].range_m == 50.0
        assert result[0].velocity_ms == -5.0

    def test_fused_confidence_weighted(self, engine, camera_detection, radar_detection):
        """Fused confidence should be weighted average: 0.6*0.8 + 0.4*0.7 = 0.76."""
        result = engine.fuse([camera_detection], [radar_detection], image_width=320, fov_deg=90.0)

        expected = 0.6 * 0.8 + 0.4 * 0.7
        assert result[0].fused_confidence == pytest.approx(expected, abs=0.01)

    def test_fuse_mismatched_azimuth(self, engine, camera_detection):
        """Radar at very different azimuth should not match camera detection."""
        radar_far = RadarDetection(
            range_m=50.0,
            azimuth_deg=25.0,  # Different enough from camera's ~0° azimuth
            velocity_ms=-3.0,
            confidence=0.6,
        )

        result = engine.fuse([camera_detection], [radar_far], image_width=320, fov_deg=90.0)

        # Should produce 2 detections: camera-only and radar-only
        assert len(result) == 2
        cam_results = [r for r in result if SensorType.CAMERA in r.sources and SensorType.RADAR not in r.sources]
        rad_results = [r for r in result if SensorType.RADAR in r.sources and SensorType.CAMERA not in r.sources]
        assert len(cam_results) == 1
        assert len(rad_results) == 1

    def test_fuse_multiple_cameras_one_radar(self, engine, radar_detection):
        """Multiple camera detections with one radar — only closest should match."""
        cam1 = Detection(
            bbox=BoundingBox(x1=140, y1=100, x2=180, y2=200),  # center ~0°
            class_id=3, class_name="car", confidence=0.8,
        )
        cam2 = Detection(
            bbox=BoundingBox(x1=280, y1=100, x2=310, y2=200),  # center far right
            class_id=3, class_name="car", confidence=0.6,
        )

        result = engine.fuse([cam1, cam2], [radar_detection], image_width=320, fov_deg=90.0)

        # Should have: 1 fused (cam1+radar), 1 camera-only (cam2)
        fused = [r for r in result if SensorType.RADAR in r.sources and SensorType.CAMERA in r.sources]
        camera_only = [r for r in result if SensorType.CAMERA in r.sources and SensorType.RADAR not in r.sources]

        assert len(fused) == 1
        assert len(camera_only) == 1
        assert fused[0].fused_confidence > camera_only[0].fused_confidence

    def test_fuse_multiple_radar_one_camera(self, engine, camera_detection):
        """Multiple radar detections with one camera — only closest should match."""
        radar1 = RadarDetection(range_m=50.0, azimuth_deg=0.0, velocity_ms=-5.0, confidence=0.7)
        radar2 = RadarDetection(range_m=60.0, azimuth_deg=45.0, velocity_ms=-3.0, confidence=0.5)

        result = engine.fuse([camera_detection], [radar1, radar2], image_width=320, fov_deg=90.0)

        # Should have: 1 fused (camera+radar1), 1 radar-only (radar2)
        assert len(result) == 2
        fused = [r for r in result if SensorType.CAMERA in r.sources and SensorType.RADAR in r.sources]
        radar_only = [r for r in result if SensorType.RADAR in r.sources and SensorType.CAMERA not in r.sources]

        assert len(fused) == 1
        assert len(radar_only) == 1

    def test_custom_weights(self):
        """Custom camera/radar weights should affect fused confidence."""
        engine = SensorFusionEngine(camera_weight=0.3, radar_weight=0.7)
        cam = Detection(
            bbox=BoundingBox(x1=140, y1=100, x2=180, y2=200),
            class_id=3, class_name="car", confidence=0.9,
        )
        rad = RadarDetection(range_m=50.0, azimuth_deg=0.0, velocity_ms=0.0, confidence=0.5)

        result = engine.fuse([cam], [rad], image_width=320, fov_deg=90.0)

        expected = 0.3 * 0.9 + 0.7 * 0.5  # 0.62
        assert result[0].fused_confidence == pytest.approx(expected, abs=0.01)

    def test_fused_confidence_capped_at_1(self):
        """Fused confidence should not exceed 1.0."""
        engine = SensorFusionEngine(camera_weight=0.6, radar_weight=0.4)
        cam = Detection(
            bbox=BoundingBox(x1=140, y1=100, x2=180, y2=200),
            class_id=3, class_name="car", confidence=1.0,
        )
        rad = RadarDetection(range_m=50.0, azimuth_deg=0.0, velocity_ms=0.0, confidence=1.0)

        result = engine.fuse([cam], [rad], image_width=320, fov_deg=90.0)

        assert result[0].fused_confidence <= 1.0

    def test_radar_only_has_bbox(self, engine, radar_detection):
        """Radar-only detection should still have an approximate bounding box."""
        result = engine.fuse([], [radar_detection], image_width=320, fov_deg=90.0)

        assert result[0].bbox is not None
        assert result[0].bbox.area > 0

    def test_sensor_type_enum(self):
        """SensorType enum should have CAMERA, RADAR, LIDAR."""
        assert SensorType.CAMERA == "camera"
        assert SensorType.RADAR == "radar"
        assert SensorType.LIDAR == "lidar"