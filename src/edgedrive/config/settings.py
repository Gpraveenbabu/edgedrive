"""Configuration models and loader for EdgeDrive.

Uses Pydantic v2 for validation. Config can be loaded from YAML files,
environment variables, or Python dicts. The core data model is
independent of Databricks-specific APIs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectorConfig(BaseModel):
    """Configuration for the detection model."""
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="fasterrcnn_mobilenet_v3_large_320_fpn")
    model_version: str = Field(default="v1.0.0")
    pretrained: bool = Field(default=True, description="Use pretrained COCO weights")
    num_classes: int = Field(default=91, ge=1)
    input_resolution: tuple[int, int] = Field(default=(320, 320))
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = Field(default="cpu", description="Device: cpu or cuda")
    min_detection_area: float = Field(
        default=100.0, ge=0.0,
        description="Minimum bounding box area in pixels to keep"
    )

    @model_validator(mode="after")
    def validate_device(self) -> "DetectorConfig":
        if self.device == "cuda" and not _cuda_available():
            # Gracefully fall back to CPU if CUDA not available
            self.device = "cpu"
        return self


class TrackerConfig(BaseModel):
    """Configuration for the object tracker."""
    model_config = ConfigDict(extra="forbid")

    iou_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="IoU threshold for matching detections to tracks"
    )
    max_age: int = Field(
        default=30, ge=1,
        description="Max frames a track can survive without a match"
    )
    min_hits: int = Field(
        default=3, ge=1,
        description="Min hits before a track is confirmed"
    )
    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Min detection confidence to start tracking"
    )


class PipelineConfig(BaseModel):
    """Configuration for the perception pipeline."""
    model_config = ConfigDict(extra="forbid")

    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    enable_tracking: bool = Field(default=True)
    enable_events: bool = Field(default=True)
    latency_threshold_ms: float = Field(
        default=100.0, ge=0.0,
        description="Threshold for latency-exceeded events"
    )
    max_frame_count: Optional[int] = Field(
        default=None, ge=1,
        description="Max frames to process (None = all)"
    )


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking."""
    model_config = ConfigDict(extra="forbid")

    warmup_iterations: int = Field(default=5, ge=0)
    timed_iterations: int = Field(default=50, ge=1)
    batch_size: int = Field(default=1, ge=1)
    input_resolution: tuple[int, int] = Field(default=(320, 320))
    output_dir: str = Field(default="outputs/benchmark")
    export_onnx: bool = Field(default=False, description="Try ONNX export during benchmark")


class AppConfig(BaseModel):
    """Top-level application configuration."""
    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)


def _cuda_available() -> bool:
    """Check if CUDA is available without importing torch at module level."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_config(config_path: Optional[str] = None, **overrides: Any) -> AppConfig:
    """Load configuration from a YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.
        **overrides: Keyword overrides applied on top of the loaded config.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If the config file is invalid or cannot be loaded.
    """
    from ..common.exceptions import ConfigError

    data: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        try:
            with open(path) as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    data = loaded
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parse error: {e}") from e

    # Apply environment variable overrides for specific keys
    if os.environ.get("EDGEDRIVE_DEVICE"):
        data.setdefault("pipeline", {}).setdefault("detector", {})["device"] = \
            os.environ["EDGEDRIVE_DEVICE"]
    if os.environ.get("EDGEDRIVE_LOG_LEVEL"):
        data["log_level"] = os.environ["EDGEDRIVE_LOG_LEVEL"]

    # Apply programmatic overrides
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    try:
        return AppConfig(**data)
    except Exception as e:
        raise ConfigError(f"Config validation error: {e}") from e