"""Tests for the FastAPI server and EdgeDriveServer.

Tests cover:
- EdgeDriveServer health, detect, process_frame, model_info, stats
- decode_image base64 decoding
- create_app FastAPI app creation
- HTTP endpoint integration via TestClient

Heavy dependencies (torch, torchvision) are imported lazily so the test
module itself loads without them. Model-dependent tests skip gracefully.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
import pytest

from edgedrive.api.server import EdgeDriveServer, decode_image, create_app
from edgedrive.common.types import ProcessingStatus
from edgedrive.config.settings import PipelineConfig, DetectorConfig, TrackerConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_dummy_image_b64(width: int = 64, height: int = 64) -> str:
    """Create a small base64-encoded PNG for API tests."""
    try:
        import cv2
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("encode failed")
        return base64.b64encode(buf.tobytes()).decode()
    except ImportError:
        raw = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
        return base64.b64encode(raw).decode()


# --------------------------------------------------------------------------- #
# EdgeDriveServer (no HTTP)
# --------------------------------------------------------------------------- #

class TestEdgeDriveServer:
    """Unit tests for EdgeDriveServer without HTTP."""

    @pytest.fixture
    def config(self) -> PipelineConfig:
        return PipelineConfig(
            detector=DetectorConfig(
                model_name="fasterrcnn_mobilenet_v3_large_320_fpn",
                pretrained=True,
                input_resolution=(320, 320),
                device="cpu",
                confidence_threshold=0.5,
            ),
            tracker=TrackerConfig(),
            enable_tracking=True,
            enable_events=True,
        )

    def test_server_creation(self, config):
        """Server should be created with a config."""
        server = EdgeDriveServer(config)
        assert server._config is not None
        assert server._pipeline is None

    def test_server_creation_no_config(self):
        """Server should work with default config."""
        server = EdgeDriveServer()
        assert server._config is not None

    def test_health_before_init(self, config):
        """Health check should work before pipeline init."""
        server = EdgeDriveServer(config)
        h = server.health()
        assert h["status"] == "healthy"
        assert h["model_loaded"] is False
        assert h["request_count"] == 0
        assert "detectors_available" in h

    def test_stats_before_init(self, config):
        """Stats should report not_initialized before pipeline is created."""
        server = EdgeDriveServer(config)
        s = server.stats()
        assert s["status"] == "not_initialized"

    def test_pipeline_lazy_init(self, config):
        """Accessing .pipeline should lazily initialize (requires torch)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

        server = EdgeDriveServer(config)
        pipe = server.pipeline
        assert pipe is not None
        h = server.health()
        assert h["model_loaded"] is True

    def test_detect_returns_dict(self, config):
        """detect() should return a dict with expected keys (requires torch)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

        server = EdgeDriveServer(config)
        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        result = server.detect(frame)

        assert isinstance(result, dict)
        assert "detections" in result
        assert "num_detections" in result
        assert "inference_time_ms" in result
        assert "model_name" in result
        assert isinstance(result["detections"], list)
        assert result["num_detections"] == len(result["detections"])

    def test_process_frame_returns_dict(self, config):
        """process_frame() should return a dict with pipeline results (requires torch)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

        server = EdgeDriveServer(config)
        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        result = server.process_frame(frame, source="test")

        assert isinstance(result, dict)
        assert "frame_index" in result
        assert "num_detections" in result
        assert "num_tracks" in result
        assert "inference_time_ms" in result
        assert "total_time_ms" in result
        assert "status" in result

    def test_model_info_returns_dict(self, config):
        """model_info() should return model metadata (requires torch)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

        server = EdgeDriveServer(config)
        info = server.model_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert "architecture" in info
        assert "num_parameters" in info
        assert "input_resolution" in info
        assert "framework" in info
        assert info["name"] == "fasterrcnn_mobilenet_v3_large_320_fpn"

    def test_request_count_increments(self, config):
        """Request count should increment on detect/process_frame."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

        server = EdgeDriveServer(config)
        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        server.detect(frame)
        assert server._request_count == 1

        server.process_frame(frame)
        assert server._request_count == 2


# --------------------------------------------------------------------------- #
# decode_image
# --------------------------------------------------------------------------- #

class TestDecodeImage:
    """Tests for decode_image utility."""

    def test_decode_valid_png(self):
        """decode_image should decode a base64 PNG to numpy array."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("cv2 not installed")

        b64 = _make_dummy_image_b64(64, 48)
        frame = decode_image(b64)

        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    def test_decode_dimensions(self):
        """Decoded image should match requested dimensions."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("cv2 not installed")

        b64 = _make_dummy_image_b64(100, 80)
        frame = decode_image(b64)

        assert frame.shape[0] == 80
        assert frame.shape[1] == 100


# --------------------------------------------------------------------------- #
# create_app (FastAPI)
# --------------------------------------------------------------------------- #

class TestCreateApp:
    """Tests for FastAPI app creation and endpoints."""

    @pytest.fixture
    def app(self):
        try:
            from fastapi import FastAPI  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed")
        try:
            return create_app()
        except RuntimeError as e:
            if "python-multipart" in str(e):
                pytest.skip("python-multipart not installed")
            raise

    def test_app_is_fastapi(self, app):
        """create_app should return a FastAPI instance."""
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_app_title(self, app):
        """App title should be EdgeDrive Perception API."""
        assert "EdgeDrive" in app.title

    def test_app_has_routes(self, app):
        """App should have the expected routes registered."""
        paths = {route.path for route in app.routes}
        assert "/health" in paths
        assert "/detect" in paths
        assert "/detect/upload" in paths
        assert "/pipeline" in paths
        assert "/model" in paths
        assert "/stats" in paths

    def test_health_endpoint(self, app):
        """GET /health should return 200 with status healthy."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx/TestClient not installed")

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_stats_endpoint(self, app):
        """GET /stats should return 200."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("httpx/TestClient not installed")

        client = TestClient(app)
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_detect_endpoint(self, app):
        """POST /detect should return 200 with detection results (requires torch)."""
        try:
            import torch  # noqa: F401
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("torch/fastapi not installed")

        client = TestClient(app)
        b64 = _make_dummy_image_b64(320, 320)
        resp = client.post("/detect", json={"image_b64": b64})
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert "num_detections" in data
        assert "inference_time_ms" in data

    def test_pipeline_endpoint(self, app):
        """POST /pipeline should return 200 with full pipeline results (requires torch)."""
        try:
            import torch  # noqa: F401
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("torch/fastapi not installed")

        client = TestClient(app)
        b64 = _make_dummy_image_b64(320, 320)
        resp = client.post("/pipeline", json={"image_b64": b64})
        assert resp.status_code == 200
        data = resp.json()
        assert "num_detections" in data
        assert "num_tracks" in data
        assert "status" in data

    def test_model_endpoint(self, app):
        """GET /model should return 200 with model info (requires torch)."""
        try:
            import torch  # noqa: F401
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("torch/fastapi not installed")

        client = TestClient(app)
        resp = client.get("/model")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "framework" in data

    def test_detect_upload_endpoint(self, app):
        """POST /detect/upload should accept a file upload (requires torch)."""
        try:
            import torch  # noqa: F401
            import cv2
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("torch/cv2/fastapi not installed")

        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", frame)
        assert ok

        client = TestClient(app)
        resp = client.post(
            "/detect/upload",
            files={"file": ("test.png", buf.tobytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data