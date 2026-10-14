"""Torchvision-based object detector for EdgeDrive.

Uses torchvision's pretrained detection models (Faster R-CNN with MobileNetV3)
for real inference on CPU. Supports COCO classes with automotive-relevant
categories: person, bicycle, car, motorcycle, bus, truck, traffic light,
stop sign.

License: torchvision models are released under the Apache 2.0 License.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ..common.exceptions import InferenceError, ModelLoadError
from ..common.logging import get_logger
from ..common.types import BoundingBox, Detection, ModelMetadata
from ..config.settings import DetectorConfig
from .detector import Detector, register_detector


# COCO class names (91 categories including background)
COCO_CLASSES: list[str] = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@register_detector("torchvision")
class TorchvisionDetector(Detector):
    """Real object detector using torchvision pretrained models.

    Primary model: fasterrcnn_mobilenet_v3_large_320_fpn
    - 19.4M parameters (~74 MB FP32)
    - Input: 320x320 RGB
    - Trained on COCO dataset (80 object classes)
    - Runs on CPU (ARM64 compatible)
    - License: Apache 2.0
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._model: Any = None
        self._loaded = False
        self._logger = get_logger("edgedrive.perception.torchvision_detector")
        self._metadata: Optional[ModelMetadata] = None

    def load(self) -> None:
        """Load the torchvision detection model."""
        if self._loaded:
            return

        import torch
        import torchvision

        model_name = self._config.model_name
        self._logger.info("loading_model", model=model_name, device=self._config.device)

        try:
            weights_arg = "DEFAULT" if self._config.pretrained else None
            if model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
                self._model = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=weights_arg, num_classes=self._config.num_classes
                )
            elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
                from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
                self._model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=weights_arg, num_classes=self._config.num_classes
                )
            elif model_name == "fasterrcnn_resnet50_fpn":
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                self._model = fasterrcnn_resnet50_fpn(
                    weights=weights_arg, num_classes=self._config.num_classes
                )
            elif model_name == "retinanet_resnet50_fpn":
                from torchvision.models.detection import retinanet_resnet50_fpn
                self._model = retinanet_resnet50_fpn(
                    weights=weights_arg, num_classes=self._config.num_classes
                )
            else:
                raise ModelLoadError(f"Unknown model: {model_name}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_name}: {e}") from e

        self._model.to(self._config.device)
        self._model.eval()
        self._loaded = True

        # Compute metadata
        num_params = sum(p.numel() for p in self._model.parameters())
        self._metadata = ModelMetadata(
            name=model_name,
            version=self._config.model_version,
            architecture=model_name,
            num_parameters=num_params,
            input_resolution=self._config.input_resolution,
            num_classes=self._config.num_classes,
            precision="fp32",
            framework="pytorch",
            pretrained_dataset="coco" if self._config.pretrained else "none",
            license="apache-2.0",
        )

        self._logger.info(
            "model_loaded",
            model=model_name,
            params_m=round(num_params / 1e6, 2),
            device=self._config.device,
        )

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: Input image (HxWx3, uint8, RGB format).
            confidence_threshold: Minimum confidence to keep.

        Returns:
            List of Detection objects above the threshold.
        """
        if not self._loaded:
            self.load()
        if self._model is None:
            raise InferenceError("Model not loaded")

        import torch

        # Preprocess: normalize to [0, 1] and convert to tensor
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Resize if needed
        target_h, target_w = self._config.input_resolution
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            try:
                import cv2
                frame_resized = cv2.resize(frame, (target_w, target_h))
            except ImportError:
                # Fallback: simple numpy resize
                from PIL import Image
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                frame_resized = np.array(pil_img)
        else:
            frame_resized = frame

        # Convert to tensor: (C, H, W) float32 normalized to [0, 1]
        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self._config.device)

        # Run inference
        try:
            with torch.no_grad():
                outputs = self._model(tensor)
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

        # Parse outputs
        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

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

            # Filter by min area
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
                    model_name=self._config.model_name,
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
            )
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self._loaded