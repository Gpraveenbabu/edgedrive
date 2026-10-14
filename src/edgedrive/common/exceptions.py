"""Exception hierarchy for EdgeDrive.

All EdgeDrive-specific errors derive from EdgeDriveError so callers can
catch the entire family with a single except clause.
"""

from __future__ import annotations


class EdgeDriveError(Exception):
    """Base exception for all EdgeDrive errors."""


class ConfigError(EdgeDriveError):
    """Raised when configuration loading or validation fails."""


class IngestionError(EdgeDriveError):
    """Raised when data ingestion fails (file not found, corrupted, etc.)."""


class DetectorError(EdgeDriveError):
    """Raised when the detector fails to load or run inference."""


class TrackerError(EdgeDriveError):
    """Raised when the tracker encounters an unrecoverable error."""


class PipelineError(EdgeDriveError):
    """Raised when the perception pipeline fails."""


class EventError(EdgeDriveError):
    """Raised when event generation fails."""


class BenchmarkError(EdgeDriveError):
    """Raised when benchmarking fails."""


class ModelLoadError(DetectorError):
    """Raised when a model cannot be loaded."""


class InferenceError(DetectorError):
    """Raised when model inference fails."""


class ExportError(EdgeDriveError):
    """Raised when model export (ONNX, etc.) fails."""
