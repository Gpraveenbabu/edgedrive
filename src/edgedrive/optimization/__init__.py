"""Model optimization and export module for EdgeDrive.

Provides ONNX export from PyTorch models and ONNX Runtime-based inference.
This enables the benchmarking system to compare PyTorch vs ONNX performance.
"""

from .onnx_exporter import OnnxExporter, ExportResult
from .onnx_detector import OnnxDetector

__all__ = ["OnnxExporter", "ExportResult", "OnnxDetector"]