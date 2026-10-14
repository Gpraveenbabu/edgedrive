"""Core data types for EdgeDrive perception pipeline.

All types use Pydantic v2 for validation and serialization.
These are the contracts between every module: ingestion, perception,
tracking, events, and benchmarking.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def gen_id(prefix: str = "") -> str:
    """Generate a unique identifier with an optional prefix."""
    short = uuid.uuid4().hex[:12]
    return f"{prefix}{short}" if prefix else short


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class BoundingBoxFormat(str, Enum):
    """Supported bounding box formats."""
    XYXY = "xyxy"  # left, top, right, bottom
    XYWH = "xywh"  # left, top, width, height


class EventType(str, Enum):
    """Perception event types emitted by the event engine."""
    VEHICLE_DETECTED = "vehicle_detected"
    PEDESTRIAN_DETECTED = "pedestrian_detected"
    OBJECT_DETECTED = "object_detected"
    TRACK_LOST = "track_lost"
    TRACK_NEW = "track_new"
    CONFIDENCE_DEGRADED = "confidence_degraded"
    LATENCY_EXCEEDED = "latency_exceeded"
    SENSOR_TIMESTAMP_MISMATCH = "sensor_timestamp_mismatch"


class TrackState(str, Enum):
    """Lifecycle state of a tracked object."""
    TENTATIVE = "tentative"    # newly created, not yet confirmed
    CONFIRMED = "confirmed"     # tracked across enough frames
    LOST = "lost"              # not matched recently, may recover
    REMOVED = "removed"        # permanently removed


class ProcessingStatus(str, Enum):
    """Status of a perception processing step."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Detection-level types
# --------------------------------------------------------------------------- #

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (xyxy format)."""
    model_config = ConfigDict(frozen=True)

    x1: float = Field(..., ge=0, description="Left edge (pixels)")
    y1: float = Field(..., ge=0, description="Top edge (pixels)")
    x2: float = Field(..., ge=0, description="Right edge (pixels)")
    y2: float = Field(..., ge=0, description="Bottom edge (pixels)")

    @field_validator("x2")
    @classmethod
    def x2_gt_x1(cls, v: float, info) -> float:
        x1 = info.data.get("x1")
        if x1 is not None and v < x1:
            raise ValueError(f"x2 ({v}) must be >= x1 ({x1})")
        return v

    @field_validator("y2")
    @classmethod
    def y2_gt_y1(cls, v: float, info) -> float:
        y1 = info.data.get("y1")
        if y1 is not None and v < y1:
            raise ValueError(f"y2 ({v}) must be >= y1 ({y1})")
        return v

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-Union with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class Detection(BaseModel):
    """A single object detection from a detector."""
    model_config = ConfigDict(frozen=False)

    bbox: BoundingBox
    class_id: int = Field(..., ge=0, description="COCO class ID")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    model_name: str = Field(default="unknown", description="Model that produced this detection")
    model_version: str = Field(default="unknown", description="Model version")

    @property
    def is_automotive_relevant(self) -> bool:
        """Whether this detection is relevant for driving perception."""
        return self.class_name in {
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "traffic light", "stop sign",
        }


# --------------------------------------------------------------------------- #
# Track-level types
# --------------------------------------------------------------------------- #

class Track(BaseModel):
    """A tracked object across consecutive frames."""
    model_config = ConfigDict(frozen=False)

    track_id: int = Field(..., ge=0, description="Unique track identifier")
    bbox: BoundingBox
    class_id: int = Field(..., ge=0)
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    state: TrackState = Field(default=TrackState.TENTATIVE)
    age: int = Field(default=0, ge=0, description="Number of frames since track creation")
    hits: int = Field(default=1, ge=0, description="Number of detection matches")
    misses: int = Field(default=0, ge=0, description="Consecutive frames without a match")
    frame_index: int = Field(..., ge=0, description="Index of the current frame")
    prev_bbox: Optional[BoundingBox] = Field(default=None, description="Previous frame bbox")
    velocity_px: Optional[tuple[float, float]] = Field(
        default=None, description="Pixel-space velocity (dx, dy) per frame"
    )

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)


# --------------------------------------------------------------------------- #
# Frame-level types
# --------------------------------------------------------------------------- #

class FrameMetadata(BaseModel):
    """Metadata associated with a single frame."""
    frame_index: int = Field(..., ge=0)
    timestamp: Optional[datetime] = Field(default=None)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    source: str = Field(default="unknown", description="Source file or stream name")


class FrameResult(BaseModel):
    """Complete perception result for a single frame."""
    frame_metadata: FrameMetadata
    detections: list[Detection] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    inference_time_ms: float = Field(..., ge=0.0, description="Model inference time in ms")
    total_time_ms: float = Field(..., ge=0.0, description="Total processing time in ms")
    model_name: str = Field(default="unknown")
    model_version: str = Field(default="unknown")
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS)
    error_message: Optional[str] = Field(default=None)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    @property
    def automotive_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.is_automotive_relevant]


# --------------------------------------------------------------------------- #
# Event-level types
# --------------------------------------------------------------------------- #

class PerceptionEvent(BaseModel):
    """A perception event emitted by the event engine."""
    model_config = ConfigDict(frozen=False)

    event_id: str = Field(default_factory=lambda: gen_id("evt_"))
    event_type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    frame_index: int = Field(..., ge=0)
    source_frame: Optional[str] = Field(default=None)
    track_id: Optional[int] = Field(default=None)
    class_name: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    evidence: dict[str, Any] = Field(default_factory=dict)
    model_name: str = Field(default="unknown")
    model_version: str = Field(default="unknown")
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS)


# --------------------------------------------------------------------------- #
# Model metadata
# --------------------------------------------------------------------------- #

class ModelMetadata(BaseModel):
    """Metadata about a detection model."""
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    architecture: str = Field(default="unknown")
    num_parameters: Optional[int] = Field(default=None)
    input_resolution: tuple[int, int] = Field(default=(320, 320))
    num_classes: int = Field(default=91)
    precision: str = Field(default="fp32")
    framework: str = Field(default="pytorch")
    pretrained_dataset: str = Field(default="coco")
    license: str = Field(default="apache-2.0")


# --------------------------------------------------------------------------- #
# Benchmark types
# --------------------------------------------------------------------------- #

class BenchmarkResult(BaseModel):
    """A single benchmark run result."""
    run_id: str = Field(default_factory=lambda: gen_id("bench_"))
    timestamp: datetime = Field(default_factory=utc_now)
    model_name: str
    model_version: str
    dataset_version: str = Field(default="unknown")
    runtime: str = Field(default="pytorch")
    precision_mode: str = Field(default="fp32")
    hardware: str = Field(default="cpu")
    input_resolution: tuple[int, int] = Field(default=(320, 320))
    batch_size: int = Field(default=1, ge=1)
    num_iterations: int = Field(..., ge=1)
    warmup_iterations: int = Field(..., ge=0)
    latency_mean_ms: float = Field(..., ge=0.0)
    latency_p50_ms: float = Field(..., ge=0.0)
    latency_p95_ms: float = Field(..., ge=0.0)
    throughput_fps: float = Field(..., ge=0.0)
    model_size_mb: Optional[float] = Field(default=None)
    peak_memory_mb: Optional[float] = Field(default=None)
    accuracy_metric: Optional[float] = Field(default=None)
    accuracy_metric_name: Optional[str] = Field(default=None)
    status: str = Field(default="success")
    error_message: Optional[str] = Field(default=None)


class VideoMetadata(BaseModel):
    """Metadata for a video file."""
    model_config = ConfigDict(frozen=True)

    path: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    fps: float = Field(..., ge=0.0)
    frame_count: int = Field(..., ge=0)
    duration_seconds: float = Field(..., ge=0.0)
    codec: str = Field(default="unknown")
    file_size_bytes: int = Field(..., ge=0)