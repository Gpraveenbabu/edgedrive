"""Video ingestion for EdgeDrive.

Reads video files using OpenCV and yields frames as numpy arrays.
Supports any format OpenCV can decode. Also provides frame extraction
from image sequences and synthetic video generation for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

import numpy as np

from ..common.exceptions import IngestionError
from ..common.logging import get_logger
from ..common.types import VideoMetadata


class VideoReader:
    """Reads frames from a video file using OpenCV.

    Usage:
        reader = VideoReader("path/to/video.mp4")
        metadata = reader.get_metadata()
        for frame in reader.frames():
            # frame is HxWx3 uint8 BGR
            ...
        reader.close()

    Or as a context manager:
        with VideoReader("path/to/video.mp4") as reader:
            for frame in reader.frames():
                ...
    """

    def __init__(self, path: str, max_frames: Optional[int] = None) -> None:
        self._path = Path(path)
        self._max_frames = max_frames
        self._cap = None
        self._logger = get_logger("edgedrive.ingestion.video")

        if not self._path.exists():
            raise IngestionError(f"Video file not found: {path}")

    def _open(self) -> None:
        """Open the video capture."""
        if self._cap is not None:
            return
        try:
            import cv2
        except ImportError:
            raise IngestionError("OpenCV (cv2) is required for video reading")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise IngestionError(f"Failed to open video: {self._path}")

    def get_metadata(self) -> VideoMetadata:
        """Return metadata about the video file."""
        self._open()
        assert self._cap is not None

        import cv2
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        file_size = self._path.stat().st_size

        if fps <= 0:
            fps = 30.0  # Default assumption
        if frame_count <= 0:
            frame_count = 0

        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=str(self._path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            file_size_bytes=file_size,
        )

    def frames(self, convert_to_rgb: bool = True) -> Generator[np.ndarray, None, None]:
        """Yield frames from the video as numpy arrays.

        Args:
            convert_to_rgb: If True (default), convert BGR to RGB.

        Yields:
            Frame as HxWx3 uint8 numpy array.
        """
        self._open()
        assert self._cap is not None

        import cv2

        frame_idx = 0
        while True:
            if self._max_frames is not None and frame_idx >= self._max_frames:
                break

            ret, frame = self._cap.read()
            if not ret:
                break

            if convert_to_rgb and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame
            frame_idx += 1

    def close(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ImageSequenceReader:
    """Reads frames from a directory of images (PNG, JPG, etc.)."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, directory: str, max_frames: Optional[int] = None) -> None:
        self._directory = Path(directory)
        self._max_frames = max_frames

        if not self._directory.exists():
            raise IngestionError(f"Directory not found: {directory}")

    def frames(self, convert_to_rgb: bool = True) -> Generator[np.ndarray, None, None]:
        """Yield frames from image files in the directory."""
        try:
            import cv2
        except ImportError:
            raise IngestionError("OpenCV (cv2) is required for image reading")

        files = sorted(
            f for f in self._directory.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if not files:
            raise IngestionError(f"No images found in {self._directory}")

        for i, fpath in enumerate(files):
            if self._max_frames is not None and i >= self._max_frames:
                break
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue
            if convert_to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame