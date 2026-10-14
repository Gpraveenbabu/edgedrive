"""Perception pipeline for EdgeDrive.

Orchestrates detection, tracking, and event generation for each frame.
The pipeline is the central processing unit that ties together all
perception components.
"""

from __future__ import annotations

import time
from typing import Any, Generator, Optional

import numpy as np

from ..common.logging import get_logger
from ..common.types import FrameMetadata, FrameResult, ProcessingStatus
from ..config.settings import PipelineConfig
from .detector import Detector, get_detector


class PerceptionPipeline:
    """End-to-end perception pipeline.

    Processes frames through:
    1. Object detection (via Detector)
    2. Multi-object tracking (via IoUTracker)
    3. Event generation (via EventEngine)

    Each step is timed and results are returned as FrameResult objects.
    """

    def __init__(
        self,
        config: PipelineConfig,
        detector: Optional[Detector] = None,
    ) -> None:
        self._config = config
        self._logger = get_logger("edgedrive.perception.pipeline")

        # Initialize detector
        if detector is not None:
            self._detector = detector
        else:
            self._detector = get_detector("torchvision", config.detector)

        # Initialize tracker
        self._tracker: Any = None
        if config.enable_tracking:
            from ..tracking.iou_tracker import IoUTracker
            self._tracker = IoUTracker(config.tracker)

        # Initialize event engine
        self._event_engine: Any = None
        if config.enable_events:
            from ..events.engine import EventEngine
            self._event_engine = EventEngine(
                latency_threshold_ms=config.latency_threshold_ms,
                model_name=config.detector.model_name,
                model_version=config.detector.model_version,
            )

        self._frame_count = 0
        self._total_detections = 0
        self._all_events: list[Any] = []

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def all_events(self) -> list[Any]:
        """All events collected across all processed frames."""
        return self._all_events

    def initialize(self) -> None:
        """Load the detector model. Call before processing frames."""
        self._logger.info("initializing_pipeline")
        self._detector.load()
        self._logger.info(
            "pipeline_ready",
            model=self._detector.metadata().name,
            tracking=self._config.enable_tracking,
            events=self._config.enable_events,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        source: str = "unknown",
    ) -> FrameResult:
        """Process a single frame through the full pipeline.

        Args:
            frame: Input image (HxWx3, uint8, RGB format).
            frame_index: Frame sequence number.
            source: Source identifier (file path, stream name).

        Returns:
            FrameResult with detections, tracks, timing, and status.
        """
        total_start = time.perf_counter()
        h, w = frame.shape[:2]
        status = ProcessingStatus.SUCCESS
        error_msg: Optional[str] = None

        # --- Detection ---
        det_start = time.perf_counter()
        try:
            detections = self._detector.detect(
                frame,
                confidence_threshold=self._config.detector.confidence_threshold,
            )
        except Exception as e:
            self._logger.error("detection_failed", error=str(e), frame=frame_index)
            detections = []
            status = ProcessingStatus.ERROR
            error_msg = f"Detection failed: {e}"
        inference_time_ms = (time.perf_counter() - det_start) * 1000.0

        # --- Tracking ---
        tracks = []
        if self._tracker is not None and status != ProcessingStatus.ERROR:
            tracks = self._tracker.update(detections)

        # --- Events ---
        if self._event_engine is not None:
            events = self._event_engine.process_frame(
                frame_index=frame_index,
                source=source,
                detections=detections,
                tracks=tracks,
                inference_time_ms=inference_time_ms,
            )
            self._all_events.extend(events)

        total_time_ms = (time.perf_counter() - total_start) * 1000.0

        self._frame_count += 1
        self._total_detections += len(detections)

        return FrameResult(
            frame_metadata=FrameMetadata(
                frame_index=frame_index,
                width=w,
                height=h,
                source=source,
            ),
            detections=detections,
            tracks=tracks,
            inference_time_ms=inference_time_ms,
            total_time_ms=total_time_ms,
            model_name=self._config.detector.model_name,
            model_version=self._config.detector.model_version,
            status=status,
            error_message=error_msg,
        )

    def process_frames(
        self,
        frames: list[np.ndarray],
        source: str = "unknown",
    ) -> Generator[FrameResult, None, None]:
        """Process a sequence of frames, yielding results one at a time.

        Args:
            frames: List of input images (HxWx3, uint8, RGB).
            source: Source identifier.

        Yields:
            FrameResult for each frame.
        """
        max_frames = self._config.max_frame_count
        for i, frame in enumerate(frames):
            if max_frames is not None and i >= max_frames:
                self._logger.info("max_frame_count_reached", count=max_frames)
                break
            yield self.process_frame(frame, frame_index=i, source=source)

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._total_detections,
            "avg_detections_per_frame": (
                self._total_detections / self._frame_count
                if self._frame_count > 0 else 0.0
            ),
            "total_events": len(self._all_events),
            "model_name": self._config.detector.model_name,
            "tracking_enabled": self._config.enable_tracking,
            "events_enabled": self._config.enable_events,
        }

    def reset(self) -> None:
        """Reset pipeline state for processing a new video."""
        if self._tracker is not None:
            self._tracker.reset()
        self._all_events.clear()
        self._frame_count = 0
        self._total_detections = 0