"""Detector interface and registry for EdgeDrive.

Defines the abstract Detector contract that all model adapters implement.
The registry allows looking up detector implementations by name, enabling
the benchmarking system to compare different models interchangeably.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from ..common.types import Detection, ModelMetadata

if TYPE_CHECKING:
    from ..config.settings import DetectorConfig


class Detector(abc.ABC):
    """Abstract base class for object detectors.

    A detector takes a single frame (numpy array, HxWx3, BGR or RGB) and
    returns a list of Detection objects. Implementations include:
    - TorchvisionDetector (PyTorch / torchvision models)
    - ONNXDetector (ONNX Runtime inference) — future
    - TensorRTDetector (NVIDIA TensorRT) — future
    """

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: Input image as numpy array (HxWx3, uint8, RGB format).
            confidence_threshold: Minimum confidence to keep.

        Returns:
            List of Detection objects above the threshold.
        """

    @abc.abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return metadata about this detector's model."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model has been loaded and is ready for inference."""

    def warmup(self, num_iterations: int = 3) -> None:
        """Run dummy inference to warm up the model.

        Default implementation creates a random input and runs detect().
        Subclasses can override for more efficient warmup.
        """
        if not self.is_loaded:
            self.load()
        h, w = self.metadata().input_resolution
        for _ in range(num_iterations):
            dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            self.detect(dummy, confidence_threshold=0.0)


# --------------------------------------------------------------------------- #
# Detector Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[Detector]] = {}


def register_detector(name: str) -> type:
    """Decorator to register a detector class in the global registry.

    Usage:
        @register_detector("torchvision")
        class TorchvisionDetector(Detector):
            ...
    """
    def decorator(cls: type[Detector]) -> type[Detector]:
        if name in _REGISTRY:
            raise ValueError(f"Detector '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_detector(name: str, config: "DetectorConfig") -> Detector:
    """Create a detector by registry name.

    Args:
        name: Registered detector name (e.g. "torchvision").
        config: Detector configuration.

    Returns:
        Detector instance (not yet loaded).

    Raises:
        KeyError: If name is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Detector '{name}' not found. Available: {available}")
    return _REGISTRY[name](config)


def list_detectors() -> list[str]:
    """Return sorted list of registered detector names."""
    return sorted(_REGISTRY.keys())