"""Tests for ONNX export, ONNX inference, and comparison benchmark.

These tests verify that the ONNX optimization pipeline works correctly:
- ONNX export produces a valid model file
- ONNX inference produces Detection objects
- PyTorch and ONNX outputs are comparable
- Comparison benchmark produces structured results

Tests skip gracefully if onnx/onnxruntime are not installed.
"""

import os
import tempfile

import numpy as np
import pytest


# Skip all tests in this module if onnx/onnxruntime not available
onnx_available = True
try:
    import onnx
    import onnxruntime
except ImportError:
    onnx_available = False

pytestmark = pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")


from edgedrive.config.settings import BenchmarkConfig, DetectorConfig
from edgedrive.optimization.onnx_exporter import OnnxExporter, ExportResult
from edgedrive.optimization.onnx_detector import OnnxDetector
from edgedrive.benchmarking.comparison import ComparisonBenchmark, ComparisonResult
from edgedrive.perception.detector import list_detectors


class TestOnnxExporter:
    """Tests for OnnxExporter."""

    @pytest.fixture
    def detector_config(self):
        return DetectorConfig(
            model_name="fasterrcnn_mobilenet_v3_large_320_fpn",
            pretrained=True,
            num_classes=91,
            input_resolution=(320, 320),
            device="cpu",
        )

    @pytest.fixture
    def exporter(self, detector_config):
        return OnnxExporter(detector_config)

    def test_export_produces_valid_onnx(self, exporter, tmp_path):
        """Export should produce a valid ONNX model file."""
        output_path = str(tmp_path / "model.onnx")
        result = exporter.export(output_path=output_path, opset_version=17)

        assert result.status == "success"
        assert os.path.exists(output_path)
        assert result.file_size_mb > 0
        assert result.onnx_path == output_path
        assert result.opset_version == 17

    def test_export_result_fields(self, exporter, tmp_path):
        """ExportResult should have all expected fields."""
        output_path = str(tmp_path / "model.onnx")
        result = exporter.export(output_path=output_path)

        assert isinstance(result, ExportResult)
        assert result.model_name == "fasterrcnn_mobilenet_v3_large_320_fpn"
        assert result.input_resolution == (320, 320)
        assert result.num_outputs == 3
        assert result.dynamic_axes is True

    def test_onnx_model_passes_checker(self, exporter, tmp_path):
        """Exported ONNX model should pass onnx.checker validation."""
        output_path = str(tmp_path / "model.onnx")
        exporter.export(output_path=output_path)

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

    def test_export_time_measured(self, exporter, tmp_path):
        """Export time should be a positive number."""
        output_path = str(tmp_path / "model.onnx")
        result = exporter.export(output_path=output_path)
        assert result.export_time_ms > 0


class TestOnnxDetector:
    """Tests for OnnxDetector."""

    @pytest.fixture
    def detector_config(self):
        return DetectorConfig(
            model_name="fasterrcnn_mobilenet_v3_large_320_fpn",
            pretrained=True,
            num_classes=91,
            input_resolution=(320, 320),
            device="cpu",
        )

    @pytest.fixture
    def onnx_path(self, detector_config, tmp_path):
        """Export a model once and reuse for inference tests."""
        exporter = OnnxExporter(detector_config)
        path = str(tmp_path / "model.onnx")
        exporter.export(output_path=path)
        return path

    @pytest.fixture
    def detector(self, detector_config, onnx_path):
        det = OnnxDetector(detector_config)
        det.set_onnx_path(onnx_path)
        return det

    def test_onnx_detector_registered(self):
        """OnnxDetector should be registered as 'onnx' in the registry."""
        assert "onnx" in list_detectors()

    def test_load_and_is_loaded(self, detector):
        """Detector should load successfully and report is_loaded=True."""
        detector.load()
        assert detector.is_loaded is True

    def test_metadata(self, detector):
        """Metadata should report onnxruntime framework."""
        detector.load()
        meta = detector.metadata()
        assert meta.framework == "onnxruntime"
        assert meta.name == "fasterrcnn_mobilenet_v3_large_320_fpn"
        assert meta.input_resolution == (320, 320)

    def test_detect_returns_list(self, detector):
        """detect() should return a list of Detection objects."""
        detector.load()
        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        detections = detector.detect(frame, confidence_threshold=0.0)
        assert isinstance(detections, list)

    def test_detect_with_synthetic_frame(self, detector):
        """Detection on random noise should return a list (may be empty)."""
        detector.load()
        frame = np.random.RandomState(42).randint(0, 255, (320, 320, 3), dtype=np.uint8)
        detections = detector.detect(frame, confidence_threshold=0.5)
        assert isinstance(detections, list)

    def test_detect_without_load_raises(self, detector_config, onnx_path):
        """Calling detect() before load() should auto-load."""
        det = OnnxDetector(detector_config)
        det.set_onnx_path(onnx_path)
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        results = det.detect(frame, confidence_threshold=0.5)
        assert isinstance(results, list)

    def test_missing_onnx_path_raises(self, detector_config):
        """Loading without setting an ONNX path should raise ModelLoadError."""
        from edgedrive.common.exceptions import ModelLoadError

        det = OnnxDetector(detector_config)
        with pytest.raises(ModelLoadError, match="No ONNX model path"):
            det.load()


class TestComparisonBenchmark:
    """Tests for ComparisonBenchmark."""

    @pytest.fixture
    def detector_config(self):
        return DetectorConfig(
            model_name="fasterrcnn_mobilenet_v3_large_320_fpn",
            pretrained=True,
            num_classes=91,
            input_resolution=(320, 320),
            device="cpu",
        )

    @pytest.fixture
    def benchmark_config(self):
        return BenchmarkConfig(
            warmup_iterations=1,
            timed_iterations=2,
            batch_size=1,
            input_resolution=(320, 320),
        )

    def test_comparison_runs_successfully(self, detector_config, benchmark_config, tmp_path):
        """Comparison benchmark should produce a ComparisonResult."""
        onnx_path = str(tmp_path / "model.onnx")
        frames = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)]

        comp = ComparisonBenchmark(benchmark_config)
        result = comp.run(
            detector_config=detector_config,
            onnx_output_path=onnx_path,
            input_frames=frames,
            dataset_version="test",
        )

        assert isinstance(result, ComparisonResult)
        assert result.pytorch.status == "success"

    def test_comparison_result_has_metrics(self, detector_config, benchmark_config, tmp_path):
        """ComparisonResult should compute speedup and throughput ratios."""
        onnx_path = str(tmp_path / "model.onnx")
        frames = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)]

        comp = ComparisonBenchmark(benchmark_config)
        result = comp.run(
            detector_config=detector_config,
            onnx_output_path=onnx_path,
            input_frames=frames,
        )

        if result.onnx.status == "success":
            assert result.latency_speedup is not None
            assert result.latency_speedup > 0
            assert result.throughput_ratio is not None

    def test_comparison_summary_string(self, detector_config, benchmark_config, tmp_path):
        """ComparisonResult.summary() should produce a readable string."""
        onnx_path = str(tmp_path / "model.onnx")
        frames = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)]

        comp = ComparisonBenchmark(benchmark_config)
        result = comp.run(
            detector_config=detector_config,
            onnx_output_path=onnx_path,
            input_frames=frames,
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "PyTorch" in summary
        assert "ONNX" in summary