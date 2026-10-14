"""Perception engine: detection, tracking, and pipeline."""

from .detector import Detector, get_detector, list_detectors, register_detector
from .pipeline import PerceptionPipeline

# Import torchvision_detector so the @register_detector("torchvision")
# decorator runs and populates the registry at import time.
from . import torchvision_detector  # noqa: F401

__all__ = ["Detector", "get_detector", "list_detectors", "register_detector", "PerceptionPipeline"]