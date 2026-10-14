"""Camera-radar sensor fusion for EdgeDrive.

Fuses 2D camera detections (bounding boxes, class, confidence)
with radar detections (position, velocity, range) to produce
fused detections with improved confidence and velocity estimates.

The fusion uses a simple distance-based association: camera detections
are matched to radar returns based on bounding-box center proximity.
Fused confidence combines camera and radar confidence using a
weighted average.

This is a late-fusion architecture — each sensor runs its own detection,
and fusion happens at the object level.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from ..common.logging import get_logger
from ..common.types import BoundingBox, Detection


class SensorType(str, Enum):
    """Supported sensor types for fusion."""
    CAMERA = "camera"
    RADAR = "radar"
    LIDAR = "lidar"


class RadarDetection(BaseModel):
    """A detection from a radar sensor."""
    range_m: float = Field(..., ge=0, description="Distance to target in meters")
    azimuth_deg: float = Field(..., description="Azimuth angle in degrees")
    velocity_ms: float = Field(..., description="Radial velocity in m/s")
    rcs_db: float = Field(default=0.0, description="Radar cross section in dB")
    confidence: float = Field(default=0.5, ge=0, le=1)
    sensor_id: str = Field(default="radar_front")

    @property
    def position_m(self) -> tuple[float, float]:
        """Convert polar to Cartesian coordinates."""
        import math
        rad = math.radians(self.azimuth_deg)
        return (self.range_m * math.cos(rad), self.range_m * math.sin(rad))


class FusedDetection(BaseModel):
    """A detection produced by fusing multiple sensor sources."""
    bbox: BoundingBox
    class_name: str
    fused_confidence: float = Field(..., ge=0, le=1)
    camera_confidence: Optional[float] = None
    radar_confidence: Optional[float] = None
    range_m: Optional[float] = None
    velocity_ms: Optional[float] = None
    azimuth_deg: Optional[float] = None
    sources: list[SensorType] = Field(default_factory=list)


class SensorFusionEngine:
    """Late-fusion engine combining camera and radar detections.

    Associates camera bounding boxes with radar returns based on
    horizontal position proximity. Unmatched camera detections are
    kept with camera-only confidence. Unmatched radar returns are
    kept as radar-only detections (no bounding box).
    """

    def __init__(
        self,
        max_assoc_distance_m: float = 5.0,
        camera_weight: float = 0.6,
        radar_weight: float = 0.4,
    ) -> None:
        self._max_distance = max_assoc_distance_m
        self._cam_w = camera_weight
        self._rad_w = radar_weight
        self._logger = get_logger("edgedrive.fusion")

    def fuse(
        self,
        camera_detections: list[Detection],
        radar_detections: list[RadarDetection],
        image_width: int = 320,
        fov_deg: float = 90.0,
    ) -> list[FusedDetection]:
        """Fuse camera and radar detections.

        Args:
            camera_detections: 2D bounding box detections from camera.
            radar_detections: Radar returns with range/azimuth/velocity.
            image_width: Image width in pixels (for azimuth mapping).
            fov_deg: Camera horizontal field of view in degrees.

        Returns:
            List of FusedDetection objects.
        """
        fused: list[FusedDetection] = []
        matched_radar: set[int] = set()

        for cam_det in camera_detections:
            # Estimate azimuth from horizontal pixel position
            cx = (cam_det.bbox.x1 + cam_det.bbox.x2) / 2
            azimuth = (cx / image_width - 0.5) * fov_deg

            best_radar_idx = None
            best_dist = float("inf")

            for i, rad in enumerate(radar_detections):
                if i in matched_radar:
                    continue
                azimuth_diff = abs(rad.azimuth_deg - azimuth)
                range_diff = abs(rad.range_m - 50)  # rough distance check
                dist = azimuth_diff + range_diff * 0.1
                if dist < best_dist and azimuth_diff < 15:
                    best_dist = dist
                    best_radar_idx = i

            if best_radar_idx is not None:
                rad = radar_detections[best_radar_idx]
                matched_radar.add(best_radar_idx)

                fused_conf = self._cam_w * cam_det.confidence + self._rad_w * rad.confidence
                fused_conf = min(fused_conf, 1.0)

                fused.append(FusedDetection(
                    bbox=cam_det.bbox,
                    class_name=cam_det.class_name,
                    fused_confidence=fused_conf,
                    camera_confidence=cam_det.confidence,
                    radar_confidence=rad.confidence,
                    range_m=rad.range_m,
                    velocity_ms=rad.velocity_ms,
                    azimuth_deg=rad.azimuth_deg,
                    sources=[SensorType.CAMERA, SensorType.RADAR],
                ))
            else:
                # Camera-only detection
                fused.append(FusedDetection(
                    bbox=cam_det.bbox,
                    class_name=cam_det.class_name,
                    fused_confidence=cam_det.confidence,
                    camera_confidence=cam_det.confidence,
                    azimuth_deg=azimuth,
                    sources=[SensorType.CAMERA],
                ))

        # Radar-only detections (no bounding box — create approximate box)
        for i, rad in enumerate(radar_detections):
            if i not in matched_radar:
                # Create an approximate bounding box from radar position
                cx_norm = (rad.azimuth_deg / fov_deg + 0.5) * image_width
                box_w = 30  # approximate width in pixels
                fused.append(FusedDetection(
                    bbox=BoundingBox(
                        x1=max(0, cx_norm - box_w / 2),
                        y1=100,
                        x2=min(image_width, cx_norm + box_w / 2),
                        y2=200,
                    ),
                    class_name="unknown_radar",
                    fused_confidence=rad.confidence * 0.7,  # lower without camera
                    radar_confidence=rad.confidence,
                    range_m=rad.range_m,
                    velocity_ms=rad.velocity_ms,
                    azimuth_deg=rad.azimuth_deg,
                    sources=[SensorType.RADAR],
                ))

        self._logger.info(
            "fusion_complete",
            camera_count=len(camera_detections),
            radar_count=len(radar_detections),
            fused_count=len(fused),
        )

        return fused