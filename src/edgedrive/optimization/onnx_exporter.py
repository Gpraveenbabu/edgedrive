"""ONNX model export for EdgeDrive.

Exports torchvision detection models to ONNX format for cross-runtime
benchmarking and edge deployment.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..common.exceptions import ExportError
from ..common.logging import get_logger
from ..config.settings import DetectorConfig


class ExportResult(BaseModel):
    """Result of an ONNX export operation."""

    onnx_path: str
    model_name: str
    model_version: str
    input_resolution: tuple[int, int]
    opset_version: int
    file_size_mb: float
    export_time_ms: float
    status: str = "success"
    error_message: Optional[str] = None
    num_outputs: int = 3
    dynamic_axes: bool = True


class OnnxExporter:
    """Export a torchvision detection model to ONNX format."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._logger = get_logger("edgedrive.optimization.onnx_exporter")

    def export(
        self,
        output_path: str,
        opset_version: int = 17,
        simplify: bool = False,
    ) -> ExportResult:
        """Export the model to ONNX format.

        Args:
            output_path: Path to save the .onnx file.
            opset_version: ONNX opset version (17 recommended).
            simplify: If True, try onnx-simplifier (optional dependency).

        Returns:
            ExportResult with metadata about the export.
        """
        start = time.perf_counter()

        try:
            import torch
        except ImportError as e:
            raise ExportError(f"PyTorch not available for export: {e}") from e

        model_name = self._config.model_name
        self._logger.info("export_start", model=model_name, opset=opset_version)

        # Load the source model
        try:
            weights_arg = "DEFAULT" if self._config.pretrained else None

            if model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
                from torchvision.models.detection import (
                    fasterrcnn_mobilenet_v3_large_320_fpn,
                )
                model = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=weights_arg,
                    num_classes=self._config.num_classes,
                )
            elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
                from torchvision.models.detection import (
                    fasterrcnn_mobilenet_v3_large_fpn,
                )
                model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=weights_arg,
                    num_classes=self._config.num_classes,
                )
            else:
                raise ExportError(
                    f"ONNX export not supported for model: {model_name}"
                )
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(f"Failed to load model for export: {e}") from e

        model.eval()
        model.to(self._config.device)

        # Wrapper to flatten dict output to tuple for ONNX
        class DetectionWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                output = self.model(x)
                result = output[0]
                return result["boxes"], result["scores"], result["labels"]

        wrapper = DetectionWrapper(model)
        wrapper.eval()
        wrapper.to(self._config.device)

        # Dummy input
        h, w = self._config.input_resolution
        dummy_input = torch.randn(1, 3, h, w, device=self._config.device)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX (use dynamo=False for legacy exporter that handles NMS)
        try:
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["boxes", "scores", "labels"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "boxes": {0: "num_detections"},
                    "scores": {0: "num_detections"},
                    "labels": {0: "num_detections"},
                },
                dynamo=False,
            )
        except Exception as e:
            raise ExportError(f"ONNX export failed: {e}") from e

        export_time_ms = (time.perf_counter() - start) * 1000
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        # Validate the ONNX model
        try:
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            self._logger.info(
                "export_validated",
                path=output_path,
                size_mb=round(file_size_mb, 2),
            )
        except ImportError:
            self._logger.warning("onnx_package_missing_validation")
        except Exception as e:
            self._logger.warning("onnx_validation_failed", error=str(e))

        # Optional: simplify the model
        if simplify:
            try:
                import onnxsim

                onnx_model = onnx.load(output_path)
                onnx_model_simplified, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(onnx_model_simplified, output_path)
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    self._logger.info(
                        "model_simplified",
                        new_size_mb=round(file_size_mb, 2),
                    )
            except ImportError:
                self._logger.warning("onnxsim_not_available")

        self._logger.info(
            "export_complete",
            path=output_path,
            size_mb=round(file_size_mb, 2),
            time_ms=round(export_time_ms, 1),
        )

        return ExportResult(
            onnx_path=output_path,
            model_name=model_name,
            model_version=self._config.model_version,
            input_resolution=self._config.input_resolution,
            opset_version=opset_version,
            file_size_mb=round(file_size_mb, 2),
            export_time_ms=round(export_time_ms, 1),
        )