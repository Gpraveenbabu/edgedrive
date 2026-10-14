"""Perception event engine for EdgeDrive.

Consumes validated perception outputs (detections, tracks, timing) and
generates typed PerceptionEvent records. Events are facts, not predictions:
- "vehicle_detected" is a measured fact from the detector.
- "confidence_degraded" is a heuristic indicator based on a threshold.
- "latency_exceeded" is a measured system fact.

The engine deliberately separates facts, heuristic indicators, and
validated predictions. No collision probabilities or risk scores are
invented from arbitrary thresholds.
"""

from __future__ import annotations

from typing import Any, Optional

from ..common.logging import get_logger
from ..common.types import Detection, EventType, PerceptionEvent, Track, TrackState


class EventEngine:
    """Generates perception events from detection and tracking outputs.

    Configurable thresholds determine when heuristic events fire:
    - latency_threshold_ms: When inference_time_ms exceeds this, emit LATENCY_EXCEEDED.
    - confidence_threshold: Below this, emit CONFIDENCE_DEGRADED.
    """

    def __init__(
        self,
        latency_threshold_ms: float = 100.0,
        confidence_threshold: float = 0.3,
        model_name: str = "unknown",
        model_version: str = "unknown",
    ) -> None:
        self._latency_threshold_ms = latency_threshold_ms
        self._confidence_threshold = confidence_threshold
        self._model_name = model_name
        self._model_version = model_version
        self._logger = get_logger("edgedrive.events.engine")

        # Track previous frame's track IDs for track_lost detection
        self._prev_track_ids: set[int] = set()

    def process_frame(
        self,
        frame_index: int,
        source: str,
        detections: list[Detection],
        tracks: list[Track],
        inference_time_ms: float,
    ) -> list[PerceptionEvent]:
        """Generate events for a single frame.

        Args:
            frame_index: Current frame number.
            source: Source identifier.
            detections: Detections from the detector.
            tracks: Active tracks from the tracker.
            inference_time_ms: Model inference time in milliseconds.

        Returns:
            List of PerceptionEvent objects for this frame.
        """
        events: list[PerceptionEvent] = []

        # --- Detection events ---
        for det in detections:
            if not det.is_automotive_relevant:
                continue

            if det.class_name in ("car", "bus", "truck", "motorcycle"):
                event_type = EventType.VEHICLE_DETECTED
            elif det.class_name == "person":
                event_type = EventType.PEDESTRIAN_DETECTED
            else:
                event_type = EventType.OBJECT_DETECTED

            events.append(PerceptionEvent(
                event_type=event_type,
                frame_index=frame_index,
                source_frame=source,
                class_name=det.class_name,
                confidence=det.confidence,
                evidence={
                    "bbox": det.bbox.model_dump(),
                    "class_id": det.class_id,
                },
                model_name=self._model_name,
                model_version=self._model_version,
            ))

            # Low-confidence detection event
            if det.confidence < self._confidence_threshold:
                events.append(PerceptionEvent(
                    event_type=EventType.CONFIDENCE_DEGRADED,
                    frame_index=frame_index,
                    source_frame=source,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    evidence={
                        "threshold": self._confidence_threshold,
                        "message": (
                            f"Detection confidence {det.confidence:.3f} "
                            f"below threshold {self._confidence_threshold:.3f}"
                        ),
                    },
                    model_name=self._model_name,
                    model_version=self._model_version,
                ))

        # --- Tracking events ---
        current_track_ids = set()
        for track in tracks:
            current_track_ids.add(track.track_id)

            # New track confirmed
            if track.state == TrackState.CONFIRMED and track.hits == 3:
                events.append(PerceptionEvent(
                    event_type=EventType.TRACK_NEW,
                    frame_index=frame_index,
                    source_frame=source,
                    track_id=track.track_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    evidence={
                        "hits": track.hits,
                        "age": track.age,
                    },
                    model_name=self._model_name,
                    model_version=self._model_version,
                ))

        # Tracks that were present before but are now lost
        lost_ids = self._prev_track_ids - current_track_ids
        for track_id in lost_ids:
            events.append(PerceptionEvent(
                event_type=EventType.TRACK_LOST,
                frame_index=frame_index,
                source_frame=source,
                track_id=track_id,
                evidence={
                    "message": f"Track {track_id} not matched in frame {frame_index}",
                },
                model_name=self._model_name,
                model_version=self._model_version,
            ))

        self._prev_track_ids = current_track_ids

        # --- Latency event ---
        if inference_time_ms > self._latency_threshold_ms:
            events.append(PerceptionEvent(
                event_type=EventType.LATENCY_EXCEEDED,
                frame_index=frame_index,
                source_frame=source,
                evidence={
                    "inference_time_ms": inference_time_ms,
                    "threshold_ms": self._latency_threshold_ms,
                    "message": (
                        f"Inference time {inference_time_ms:.1f}ms exceeded "
                        f"threshold {self._latency_threshold_ms:.1f}ms"
                    ),
                },
                model_name=self._model_name,
                model_version=self._model_version,
            ))

        return events

    def reset(self) -> None:
        """Reset event engine state."""
        self._prev_track_ids = set()