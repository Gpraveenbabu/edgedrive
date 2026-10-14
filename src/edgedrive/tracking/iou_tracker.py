"""IoU-based multi-object tracker for EdgeDrive.

A lightweight, deterministic tracker that matches detections to existing
tracks using Intersection-over-Union (IoU). Inspired by the SORT (Simple
Online and Realtime Tracking) algorithm but with no external dependencies.

Design decisions:
- Greedy IoU matching (not Hungarian assignment) for simplicity and determinism.
- Track lifecycle: TENTATIVE -> CONFIRMED -> LOST -> REMOVED.
- Pixel-space velocity estimation (uncalibrated, not real-world velocity).
- No Kalman filtering — simple bounding box carry-over for unmatched tracks.

This is intentionally simple for the initial release. ByteTrack or a
Kalman-filter-based tracker can be added as a drop-in replacement via the
Tracker interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.logging import get_logger
from ..common.types import BoundingBox, Detection, Track, TrackState
from ..config.settings import TrackerConfig


class IoUTracker:
    """IoU-based multi-object tracker.

    Maintains track state across frames. Call update() with each frame's
detections to get the current track list.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 0
        self._frame_index: int = 0
        self._logger = get_logger("edgedrive.tracking.iou_tracker")

    @property
    def tracks(self) -> list[Track]:
        """Return all active (non-removed) tracks."""
        return [t for t in self._tracks.values() if t.state != TrackState.REMOVED]

    @property
    def active_tracks(self) -> list[Track]:
        """Return tracks that are tentative or confirmed."""
        return [t for t in self._tracks.values() if t.is_active]

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections for the current frame.

        Args:
            detections: Detections from the current frame.

        Returns:
            List of all active tracks after this update.
        """
        self._frame_index += 1

        # Filter detections by min confidence
        valid_dets = [
            d for d in detections
            if d.confidence >= self._config.min_confidence
        ]

        # Match detections to existing tracks via IoU
        matched_track_ids, unmatched_det_indices = self._match(valid_dets)

        # Update matched tracks
        for track_id, det_idx in matched_track_ids:
            self._update_track(track_id, valid_dets[det_idx])

        # Create new tracks for unmatched detections
        newly_created: set[int] = set()
        for det_idx in unmatched_det_indices:
            self._create_track(valid_dets[det_idx])
            newly_created.add(self._next_id - 1)

        # Age unmatched tracks (exclude matched and newly created)
        matched_ids = {tid for tid, _ in matched_track_ids}
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_ids and track_id not in newly_created:
                self._age_track(track_id)

        # Remove tracks that exceeded max_age
        self._remove_stale_tracks()

        return self.active_tracks

    def _match(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Match detections to existing tracks using greedy IoU.

        Returns:
            (matched_pairs, unmatched_detection_indices)
            where matched_pairs is a list of (track_id, det_index).
        """
        if not self._tracks or not detections:
            return [], list(range(len(detections)))

        # Compute IoU matrix: (num_tracks, num_detections)
        active = [t for t in self._tracks.values() if t.is_active]
        if not active:
            return [], list(range(len(detections)))

        iou_matrix = np.zeros((len(active), len(detections)))
        for i, track in enumerate(active):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = track.bbox.iou(det.bbox)

        # Greedy matching: pick highest IoU pairs first
        matched_pairs: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        # Flatten and sort by IoU descending
        indices = np.dstack(np.unravel_index(
            np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape
        ))[0]

        for track_idx, det_idx in indices:
            if iou_matrix[track_idx, det_idx] < self._config.iou_threshold:
                break
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            matched_pairs.append((active[track_idx].track_id, int(det_idx)))
            used_tracks.add(track_idx)
            used_dets.add(int(det_idx))

        unmatched_dets = [
            j for j in range(len(detections)) if j not in used_dets
        ]
        return matched_pairs, unmatched_dets

    def _create_track(self, detection: Detection) -> None:
        """Create a new track from a detection."""
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = Track(
            track_id=track_id,
            bbox=detection.bbox,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            state=TrackState.TENTATIVE,
            age=0,
            hits=1,
            misses=0,
            frame_index=self._frame_index,
        )
        self._logger.debug(
            "track_created",
            track_id=track_id,
            class_name=detection.class_name,
            frame=self._frame_index,
        )

    def _update_track(self, track_id: int, detection: Detection) -> None:
        """Update an existing track with a new detection."""
        track = self._tracks[track_id]
        prev_bbox = track.bbox
        cx_old, cy_old = prev_bbox.center
        cx_new, cy_new = detection.bbox.center
        velocity = (cx_new - cx_old, cy_new - cy_old)

        self._tracks[track_id] = Track(
            track_id=track_id,
            bbox=detection.bbox,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            state=TrackState.CONFIRMED if track.hits >= self._config.min_hits - 1 else TrackState.TENTATIVE,
            age=track.age + 1,
            hits=track.hits + 1,
            misses=0,
            frame_index=self._frame_index,
            prev_bbox=prev_bbox,
            velocity_px=velocity,
        )

    def _age_track(self, track_id: int) -> None:
        """Age a track that was not matched in this frame."""
        track = self._tracks[track_id]
        self._tracks[track_id] = Track(
            track_id=track_id,
            bbox=track.bbox,
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            state=TrackState.LOST if track.misses + 1 >= self._config.max_age // 2 else track.state,
            age=track.age + 1,
            hits=track.hits,
            misses=track.misses + 1,
            frame_index=self._frame_index,
            prev_bbox=track.prev_bbox,
            velocity_px=track.velocity_px,
        )

    def _remove_stale_tracks(self) -> None:
        """Remove tracks that exceeded the max age."""
        for track_id in list(self._tracks.keys()):
            track = self._tracks[track_id]
            if track.misses >= self._config.max_age:
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=track.bbox,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    state=TrackState.REMOVED,
                    age=track.age,
                    hits=track.hits,
                    misses=track.misses,
                    frame_index=track.frame_index,
                )
                self._logger.debug(
                    "track_removed",
                    track_id=track_id,
                    age=track.age,
                    hits=track.hits,
                )

    def reset(self) -> None:
        """Reset tracker state (for reprocessing a new video)."""
        self._tracks.clear()
        self._next_id = 0
        self._frame_index = 0