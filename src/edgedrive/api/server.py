"""FastAPI server for EdgeDrive perception pipeline.

Exposes REST endpoints for:
- Health checks
- Object detection on uploaded images
- Pipeline processing (detection + tracking + events)
- Benchmarking
- Model metadata

The server lazily loads the detection model on first request.
Designed for edge deployment — minimal dependencies, CPU inference.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Optional

import numpy as np

from ..common.logging import get_logger
from ..common.types import FrameResult, ProcessingStatus
from ..config.settings import (
    BenchmarkConfig,
    DetectorConfig,
    PipelineConfig,
    TrackerConfig,
    load_config,
)
from ..perception.detector import get_detector, list_detectors
from ..perception.pipeline import PerceptionPipeline


class EdgeDriveServer:
    """EdgeDrive perception server with lazy model loading.

    Wraps a PerceptionPipeline and exposes it via methods that
    a FastAPI app can call. Separating the server logic from the
    FastAPI app makes it testable without HTTP.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._config = config or PipelineConfig()
        self._pipeline: Optional[PerceptionPipeline] = None
        self._logger = get_logger("edgedrive.api.server")
        self._request_count = 0
        self._total_inference_ms = 0.0

    @property
    def pipeline(self) -> PerceptionPipeline:
        """Lazily initialize the perception pipeline."""
        if self._pipeline is None:
            self._pipeline = PerceptionPipeline(self._config)
            self._pipeline.initialize()
            self._logger.info(
                "server_ready",
                model=self._pipeline.detector.metadata().name,
            )
        return self._pipeline

    def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": self._pipeline is not None,
            "detectors_available": list_detectors(),
            "request_count": self._request_count,
            "total_inference_ms": round(self._total_inference_ms, 2),
        }

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Run detection on a single frame.

        Args:
            frame: HxWx3 uint8 RGB array.

        Returns:
            Dict with detections, timing, and metadata.
        """
        self._request_count += 1
        start = time.perf_counter()

        detections = self.pipeline.detector.detect(
            frame,
            confidence_threshold=self._config.detector.confidence_threshold,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._total_inference_ms += elapsed_ms

        return {
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": {
                        "x1": d.bbox.x1,
                        "y1": d.bbox.y1,
                        "x2": d.bbox.x2,
                        "y2": d.bbox.y2,
                    },
                    "is_automotive_relevant": d.is_automotive_relevant,
                }
                for d in detections
            ],
            "num_detections": len(detections),
            "inference_time_ms": round(elapsed_ms, 2),
            "model_name": self.pipeline.detector.metadata().name,
        }

    def process_frame(self, frame: np.ndarray, source: str = "api") -> dict[str, Any]:
        """Process a frame through the full pipeline.

        Args:
            frame: HxWx3 uint8 RGB array.
            source: Source identifier.

        Returns:
            Dict with detections, tracks, events, and timing.
        """
        self._request_count += 1
        result = self.pipeline.process_frame(frame, source=source)

        return {
            "frame_index": result.frame_metadata.frame_index,
            "num_detections": result.num_detections,
            "num_tracks": result.num_tracks,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": {
                        "x1": d.bbox.x1,
                        "y1": d.bbox.y1,
                        "x2": d.bbox.x2,
                        "y2": d.bbox.y2,
                    },
                }
                for d in result.detections
            ],
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "state": t.state.value,
                    "hits": t.hits,
                    "misses": t.misses,
                }
                for t in result.tracks
            ],
            "inference_time_ms": round(result.inference_time_ms, 2),
            "total_time_ms": round(result.total_time_ms, 2),
            "status": result.status.value,
        }

    def model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        meta = self.pipeline.detector.metadata()
        return {
            "name": meta.name,
            "version": meta.version,
            "architecture": meta.architecture,
            "num_parameters": meta.num_parameters,
            "input_resolution": list(meta.input_resolution),
            "num_classes": meta.num_classes,
            "precision": meta.precision,
            "framework": meta.framework,
            "license": meta.license,
        }

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        if self._pipeline is None:
            return {"status": "not_initialized"}
        return self._pipeline.get_stats()


def decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64-encoded image to a numpy array.

    Args:
        image_b64: Base64-encoded image bytes.

    Returns:
        HxWx3 uint8 RGB numpy array.
    """
    import cv2

    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def create_app(config: Optional[PipelineConfig] = None) -> Any:
    """Create a FastAPI application for EdgeDrive.

    Args:
        config: Optional pipeline configuration. Uses defaults if None.

    Returns:
        FastAPI app instance.
    """
    from fastapi import FastAPI, UploadFile, File
    from pydantic import BaseModel

    server = EdgeDriveServer(config)
    app = FastAPI(
        title="EdgeDrive Perception API",
        description="Automotive AI perception pipeline",
        version="1.0.0",
    )

    class DetectRequest(BaseModel):
        image_b64: str

    @app.get("/health")
    async def health():
        return server.health()

    @app.post("/detect")
    async def detect(req: DetectRequest):
        frame = decode_image(req.image_b64)
        return server.detect(frame)

    @app.post("/detect/upload")
    async def detect_upload(file: UploadFile = File(...)):
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode()
        frame = decode_image(image_b64)
        return server.detect(frame)

    @app.post("/pipeline")
    async def pipeline(req: DetectRequest):
        frame = decode_image(req.image_b64)
        return server.process_frame(frame)

    @app.get("/model")
    async def model_info():
        return server.model_info()

    @app.get("/stats")
    async def stats():
        return server.stats()

    return app