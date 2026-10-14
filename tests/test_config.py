"""Tests for configuration loading and validation."""

import os
from pathlib import Path

import pytest

from edgedrive.common.exceptions import ConfigError
from edgedrive.config.settings import (
    AppConfig,
    BenchmarkConfig,
    DetectorConfig,
    PipelineConfig,
    TrackerConfig,
    load_config,
)


class TestDetectorConfig:
    def test_defaults(self):
        config = DetectorConfig()
        assert config.model_name == "fasterrcnn_mobilenet_v3_large_320_fpn"
        assert config.device == "cpu"
        assert config.confidence_threshold == 0.5
        assert config.input_resolution == (320, 320)

    def test_custom_values(self):
        config = DetectorConfig(
            model_name="retinanet_resnet50_fpn",
            confidence_threshold=0.7,
            device="cpu",
        )
        assert config.model_name == "retinanet_resnet50_fpn"
        assert config.confidence_threshold == 0.7

    def test_invalid_confidence(self):
        with pytest.raises(Exception):
            DetectorConfig(confidence_threshold=1.5)


class TestTrackerConfig:
    def test_defaults(self):
        config = TrackerConfig()
        assert config.iou_threshold == 0.3
        assert config.max_age == 30
        assert config.min_hits == 3

    def test_invalid_iou(self):
        with pytest.raises(Exception):
            TrackerConfig(iou_threshold=1.5)


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.enable_tracking is True
        assert config.enable_events is True
        assert config.latency_threshold_ms == 100.0

    def test_nested_config(self):
        config = PipelineConfig(
            detector=DetectorConfig(confidence_threshold=0.8),
            tracker=TrackerConfig(max_age=50),
        )
        assert config.detector.confidence_threshold == 0.8
        assert config.tracker.max_age == 50


class TestLoadConfig:
    def test_default_config(self):
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.pipeline.detector.model_name == "fasterrcnn_mobilenet_v3_large_320_fpn"

    def test_yaml_config(self, tmp_path):
        yaml_content = """
pipeline:
  detector:
    model_name: retinanet_resnet50_fpn
    confidence_threshold: 0.7
  tracker:
    max_age: 50
log_level: DEBUG
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = load_config(str(config_path))
        assert config.pipeline.detector.model_name == "retinanet_resnet50_fpn"
        assert config.pipeline.detector.confidence_threshold == 0.7
        assert config.pipeline.tracker.max_age == 50
        assert config.log_level == "DEBUG"

    def test_missing_file(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(": : invalid yaml : :")
        with pytest.raises(ConfigError):
            load_config(str(config_path))