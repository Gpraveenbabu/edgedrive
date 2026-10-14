"""ONNX Runtime-based detector for EdgeDrive.

Runs inference using ONNX Runtime instead of PyTorch, enabling
performance comparison between native PyTorch and ONNX execution.
Registered as 'onnx' in the detector registry.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..common.exceptions import InferenceError, ModelLoadError
from ..common.logging import get_logger
from ..common.types import BoundingBox, Detection, ModelMetadata
from ..config.settings import DetectorConfig
from ..perception.detector import Detector, register_detector
from ..perception.torchvision_detector import COCO_CLASSES


@register_detector("onnx")
class OnnxDetector(Detector):
    """Object detector using ONNX Runtime.

    Loads an exported ONNX model and runs inference via onnxruntime.
    Produces the same Detection objects as TorchvisionDetector.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._session = None
        self._loaded = False
        self._logger = get_logger("edgedrive.optimization.onnx_detector")
        self._metadata: Optional[ModelMetadata] = None
        self._onnx_path: Optional[str] = None

    def load(self) -> None:
        """Load the ONNX model into an ONNX Runtime session."""
        if self._loaded:
            return

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModelLoadError(f"onnxruntime not installed: {e}") from e

        # The ONNX path must be set via config or set_onnx_path()
        onnx_path = self._onnx_path or getattr(
            self._config, "onnx_path", None
        )
        if not onnx_path:
            raise ModelLoadError(
                "No ONNX model path provided. Use set_onnx_path() or "
                "set config.onnx_path"
            )

        import os
        if not os.path.exists(onnx_path):
            raise ModelLoadError(f"ONNX file not found: {onnx_path}")

        self._logger.info("loading_onnx_model", path=onnx_path)

        try:
            # Use CPUExecutionProvider (GPU providers are optional)
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")

            self._session = ort.InferenceSession(
                onnx_path, providers=providers
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {e}") from e

        self._loaded = True

        # Estimate model size from file
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        # Rough param estimate: size_mb * 1024^2 / 4 bytes (FP32)
        num_params_est = int(file_size_mb * 1024 * 1024 / 4)

        self._metadata = ModelMetadata(
            name=self._config.model_name,
            version=self._config.model_version,
            architecture=f"{self._config.model_name}_onnx",
            num_parameters=num_params_est,
            input_resolution=self._config.input_resolution,
            num_classes=self._config.num_classes,
            precision="fp32",
            framework="onnxruntime",
            pretrained_dataset="coco" if self._config.pretrained else "none",
            license="apache-2.0",
        )

        self._logger.info(
            "onnx_model_loaded",
            path=onnx_path,
            size_mb=round(file_size_mb, 2),
            providers=providers,
        )

    def set_onnx_path(self, path: str) -> None:
        """Set the path to the ONNX model file."""
        self._onnx_path = path

    def detect(
        self, frame: np.ndarray, confidence_threshold: float = 0.5
    ) -> list[Detection]:
        """Run detection using ONNX Runtime.

        Args:
            frame: Input image (HxWx3, uint8, RGB format).
            confidence_threshold: Minimum confidence to keep.

        Returns:
            List of Detection objects above the threshold.
        """
        if not self._loaded:
            self.load()
        if self._session is None:
            raise InferenceError("ONNX session not loaded")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Resize if needed (same preprocessing as TorchvisionDetector)
        target_h, target_w = self._config.input_resolution
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            try:
                import cv2

                frame_resized = cv2.resize(frame, (target_w, target_h))
            except ImportError:
                from PIL import Image

                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                frame_resized = np.array(pil_img)
        else:
            frame_resized = frame

        # Preprocess: (H, W, C) -> (1, C, H, W), float32, [0, 1]
        input_tensor = (
            frame_resized.astype(np.float32) / 255.0
        )
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor)

        # Run inference
        try:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(
                None, {input_name: input_tensor}
            )
        except Exception as e:
            raise InferenceError(f"ONNX inference failed: {e}") from e

        # Parse outputs: (boxes, scores, labels)
        boxes = outputs[0]
        scores = outputs[1]
        labels = outputs[2]

        # Squeeze batch dim if present
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]
        if labels.ndim == 2:
            labels = labels[0]

        # Scale boxes back to original frame size
        orig_h, orig_w = frame.shape[:2]
        if frame_resized.shape[0] != orig_h or frame_resized.shape[1] != orig_w:
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        # Build Detection objects
        detections: list[Detection] = []
        for i in range(len(scores)):
            if scores[i] < confidence_threshold:
                continue

            x1, y1, x2, y2 = boxes[i]
            bbox = BoundingBox(
                x1=float(max(0, x1)),
                y1=float(max(0, y1)),
                x2=float(min(orig_w, x2)),
                y2=float(min(orig_h, y2)),
            )

            if bbox.area < self._config.min_detection_area:
                continue

            class_id = int(labels[i])
            class_name = (
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES)
                else f"class_{class_id}"
            )

            detections.append(
                Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(scores[i]),
                    model_name=f"{self._config.model_name}_onnx",
                    model_version=self._config.model_version,
                )
            )

        return detections

    def metadata(self) -> ModelMetadata:
        """Return model metadata."""
        if self._metadata is None:
            self._metadata = ModelMetadata(
                name=self._config.model_name,
                version=self._config.model_version,
                input_resolution=self._config.input_resolution,
                num_classes=self._config.num_classes,
                precision="fp32",
                framework="onnxruntime",
                pretrained_dataset="coco" if self._config.pretrained else "none",
                license="apache-2.0",
            )
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._loaded