"""Comparison benchmark for PyTorch vs ONNX Runtime inference.

Runs the same model under both PyTorch and ONNX Runtime, measures
latency/throughput/model-size, and returns a structured comparison.
All measurements are real - no results are invented.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from pydantic import BaseModel

from ..common.logging import get_logger
from ..common.types import BenchmarkResult
from ..config.settings import BenchmarkConfig, DetectorConfig
from .benchmark import BenchmarkRunner


class ComparisonResult(BaseModel):
    """Side-by-side comparison of PyTorch vs ONNX benchmark results."""

    pytorch: BenchmarkResult
    onnx: BenchmarkResult
    onnx_export_path: Optional[str] = None
    latency_speedup: Optional[float] = None
    throughput_ratio: Optional[float] = None
    size_ratio: Optional[float] = None
    notes: str = ""

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "PyTorch vs ONNX Comparison",
            "=" * 50,
            f"PyTorch: mean={self.pytorch.latency_mean_ms:.1f}ms "
            f"p50={self.pytorch.latency_p50_ms:.1f}ms "
            f"p95={self.pytorch.latency_p95_ms:.1f}ms "
            f"fps={self.pytorch.throughput_fps:.1f} "
            f"size={self.pytorch.model_size_mb:.1f}MB",
            f"ONNX:    mean={self.onnx.latency_mean_ms:.1f}ms "
            f"p50={self.onnx.latency_p50_ms:.1f}ms "
            f"p95={self.onnx.latency_p95_ms:.1f}ms "
            f"fps={self.onnx.throughput_fps:.1f} "
            f"size={self.onnx.model_size_mb:.1f}MB",
        ]

        if self.latency_speedup is not None:
            lines.append(
                f"Speedup: {self.latency_speedup:.2f}x "
                f"({'ONNX faster' if self.latency_speedup > 1 else 'PyTorch faster'})"
            )
        if self.throughput_ratio is not None:
            lines.append(
                f"Throughput ratio: {self.throughput_ratio:.2f}x"
            )
        if self.size_ratio is not None:
            lines.append(
                f"Size ratio: {self.size_ratio:.2f}x"
            )

        return "\n".join(lines)


class ComparisonBenchmark:
    """Run PyTorch and ONNX benchmarks side by side."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._logger = get_logger("edgedrive.benchmarking.comparison")

    def run(
        self,
        detector_config: DetectorConfig,
        onnx_output_path: str = "/tmp/edgedrive_model.onnx",
        input_frames: Optional[list[np.ndarray]] = None,
        dataset_version: str = "synthetic",
    ) -> ComparisonResult:
        """Run comparison benchmark."""
        # Step 1: Benchmark PyTorch
        self._logger.info("benchmark_pytorch_start")
        runner = BenchmarkRunner(self._config)
        pytorch_result = runner.run(
            detector_config=detector_config,
            input_frames=input_frames,
            dataset_version=dataset_version,
        )

        # Step 2: Export to ONNX
        onnx_path = None
        try:
            from ..optimization.onnx_exporter import OnnxExporter

            exporter = OnnxExporter(detector_config)
            export_result = exporter.export(
                output_path=onnx_output_path, opset_version=17,
            )
            onnx_path = export_result.onnx_path
        except Exception as e:
            self._logger.error("onnx_export_failed", error=str(e))
            onnx_result = BenchmarkResult(
                run_id=f"onnx_failed_{int(time.time())}",
                timestamp=time.time(),
                model_name=detector_config.model_name,
                model_version=detector_config.model_version,
                dataset_version=dataset_version,
                runtime="onnxruntime",
                precision_mode="fp32",
                hardware="cpu",
                input_resolution=detector_config.input_resolution,
                batch_size=self._config.batch_size,
                num_iterations=0, warmup_iterations=0,
                latency_mean_ms=0, latency_p50_ms=0, latency_p95_ms=0,
                throughput_fps=0, model_size_mb=0,
                status="error", error_message=str(e),
            )
            return ComparisonResult(
                pytorch=pytorch_result, onnx=onnx_result,
                notes=f"ONNX export failed: {e}",
            )

        # Step 3: Benchmark ONNX Runtime
        from ..optimization.onnx_detector import OnnxDetector

        onnx_config = DetectorConfig(
            model_name=detector_config.model_name,
            model_version=detector_config.model_version,
            pretrained=detector_config.pretrained,
            num_classes=detector_config.num_classes,
            input_resolution=detector_config.input_resolution,
            confidence_threshold=detector_config.confidence_threshold,
            nms_threshold=detector_config.nms_threshold,
            device="cpu",
            min_detection_area=detector_config.min_detection_area,
        )
        onnx_detector = OnnxDetector(onnx_config)
        onnx_detector.set_onnx_path(onnx_path)

        onnx_result = self._benchmark_onnx(
            onnx_detector=onnx_detector,
            detector_config=onnx_config,
            input_frames=input_frames,
            dataset_version=dataset_version,
            onnx_file_size_mb=export_result.file_size_mb,
        )

        # Step 4: Compute comparison metrics
        latency_speedup = None
        throughput_ratio = None
        size_ratio = None
        if onnx_result.latency_mean_ms > 0:
            latency_speedup = pytorch_result.latency_mean_ms / onnx_result.latency_mean_ms
        if onnx_result.throughput_fps > 0:
            throughput_ratio = onnx_result.throughput_fps / pytorch_result.throughput_fps
        if onnx_result.model_size_mb > 0:
            size_ratio = pytorch_result.model_size_mb / onnx_result.model_size_mb

        return ComparisonResult(
            pytorch=pytorch_result, onnx=onnx_result,
            onnx_export_path=onnx_path,
            latency_speedup=latency_speedup,
            throughput_ratio=throughput_ratio,
            size_ratio=size_ratio,
            notes="Real measurements on ARM64 CPU",
        )

    def _benchmark_onnx(
        self, onnx_detector, detector_config, input_frames,
        dataset_version, onnx_file_size_mb,
    ) -> BenchmarkResult:
        """Run ONNX Runtime benchmark manually."""
        import uuid

        run_id = f"onnx_{uuid.uuid4().hex[:8]}"
        onnx_detector.load()

        if input_frames is None:
            h, w = detector_config.input_resolution
            input_frames = [
                np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                for _ in range(self._config.timed_iterations)
            ]

        # Warmup
        for i in range(self._config.warmup_iterations):
            onnx_detector.detect(input_frames[i % len(input_frames)], confidence_threshold=0.0)

        # Timed
        latencies = []
        for i in range(self._config.timed_iterations):
            start = time.perf_counter()
            onnx_detector.detect(input_frames[i % len(input_frames)], confidence_threshold=0.0)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies_arr = np.array(latencies)
        mean_ms = float(np.mean(latencies_arr))
        p50_ms = float(np.median(latencies_arr))
        p95_ms = float(np.percentile(latencies_arr, 95))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        peak_mem = None
        try:
            import resource
            peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass

        return BenchmarkResult(
            run_id=run_id, timestamp=time.time(),
            model_name=detector_config.model_name,
            model_version=detector_config.model_version,
            dataset_version=dataset_version,
            runtime="onnxruntime", precision_mode="fp32",
            hardware="cpu",
            input_resolution=detector_config.input_resolution,
            batch_size=self._config.batch_size,
            num_iterations=self._config.timed_iterations,
            warmup_iterations=self._config.warmup_iterations,
            latency_mean_ms=round(mean_ms, 2),
            latency_p50_ms=round(p50_ms, 2),
            latency_p95_ms=round(p95_ms, 2),
            throughput_fps=round(fps, 2),
            model_size_mb=round(onnx_file_size_mb, 2),
            peak_memory_mb=round(peak_mem, 2) if peak_mem else None,
            status="success",
        )


class ComparisonBenchmark:
    """Run PyTorch and ONNX benchmarks side by side.

    Exports the model to ONNX, benchmarks both runtimes with the same
    inputs and configuration, and returns a ComparisonResult.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._logger = get_logger("edgedrive.benchmarking.comparison")

    def run(
        self,
        detector_config: DetectorConfig,
        onnx_output_path: str = "/tmp/edgedrive_model.onnx",
        input_frames: Optional[list[np.ndarray]] = None,
        dataset_version: str = "synthetic",
    ) -> ComparisonResult:
        """Run comparison benchmark.

        Args:
            detector_config: Detector configuration.
            onnx_output_path: Where to save the exported ONNX model.
            input_frames: Input frames (None = auto-generate random).
            dataset_version: Dataset identifier.

        Returns:
            ComparisonResult with both benchmark results.
        """
        # Step 1: Benchmark PyTorch
        self._logger.info("benchmark_pytorch_start")
        runner = BenchmarkRunner(self._config)
        pytorch_result = runner.run(
            detector_config=detector_config,
            input_frames=input_frames,
            dataset_version=dataset_version,
        )
        self._logger.info(
            "benchmark_pytorch_done",
            mean_ms=round(pytorch_result.latency_mean_ms, 1),
            fps=round(pytorch_result.throughput_fps, 1),
        )

        # Step 2: Export to ONNX
        onnx_path = None
        onnx_result = None

        try:
            from ..optimization.onnx_exporter import OnnxExporter

            exporter = OnnxExporter(detector_config)
            export_result = exporter.export(
                output_path=onnx_output_path,
                opset_version=17,
            )
            onnx_path = export_result.onnx_path
            self._logger.info(
                "onnx_exported",
                path=onnx_path,
                size_mb=export_result.file_size_mb,
            )
        except Exception as e:
            self._logger.error("onnx_export_failed", error=str(e))
            # Create a failed benchmark result
            onnx_result = BenchmarkResult(
                run_id=f"onnx_failed_{int(time.time())}",
                timestamp=time.time(),
                model_name=detector_config.model_name,
                model_version=detector_config.model_version,
                dataset_version=dataset_version,
                runtime="onnxruntime",
                precision_mode="fp32",
                hardware="cpu",
                input_resolution=detector_config.input_resolution,
                batch_size=self._config.batch_size,
                num_iterations=0,
                warmup_iterations=0,
                latency_mean_ms=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                throughput_fps=0.0,
                model_size_mb=0.0,
                status="error",
                error_message=str(e),
            )

            # Return with failed ONNX result
            return ComparisonResult(
                pytorch=pytorch_result,
                onnx=onnx_result,
                onnx_export_path=None,
                notes=f"ONNX export failed: {e}",
            )

        # Step 3: Benchmark ONNX Runtime
        self._logger.info("benchmark_onnx_start")
        onnx_config = DetectorConfig(
            model_name=detector_config.model_name,
            model_version=detector_config.model_version,
            pretrained=detector_config.pretrained,
            num_classes=detector_config.num_classes,
            input_resolution=detector_config.input_resolution,
            confidence_threshold=detector_config.confidence_threshold,
            nms_threshold=detector_config.nms_threshold,
            device="cpu",
            min_detection_area=detector_config.min_detection_area,
        )

        from ..optimization.onnx_detector import OnnxDetector

        onnx_detector = OnnxDetector(onnx_config)
        onnx_detector.set_onnx_path(onnx_path)

        # Benchmark ONNX manually (BenchmarkRunner uses registry)
        onnx_result = self._benchmark_onnx(
            onnx_detector=onnx_detector,
            detector_config=onnx_config,
            input_frames=input_frames,
            dataset_version=dataset_version,
            onnx_file_size_mb=export_result.file_size_mb,
        )

        self._logger.info(
            "benchmark_onnx_done",
            mean_ms=round(onnx_result.latency_mean_ms, 1),
            fps=round(onnx_result.throughput_fps, 1),
        )

        # Step 4: Compute comparison metrics
        latency_speedup = None
        throughput_ratio = None
        size_ratio = None

        if onnx_result.latency_mean_ms > 0:
            latency_speedup = (
                pytorch_result.latency_mean_ms / onnx_result.latency_mean_ms
            )
        if onnx_result.throughput_fps > 0:
            throughput_ratio = (
                onnx_result.throughput_fps / pytorch_result.throughput_fps
            )
        if onnx_result.model_size_mb > 0:
            size_ratio = (
                pytorch_result.model_size_mb / onnx_result.model_size_mb
            )

        return ComparisonResult(
            pytorch=pytorch_result,
            onnx=onnx_result,
            onnx_export_path=onnx_path,
            latency_speedup=latency_speedup,
            throughput_ratio=throughput_ratio,
            size_ratio=size_ratio,
            notes="Real measurements on ARM64 CPU (Databricks serverless)",
        )

    def _benchmark_onnx(
        self,
        onnx_detector,
        detector_config,
        input_frames,
        dataset_version,
        onnx_file_size_mb,
    ) -> BenchmarkResult:
        """Run ONNX Runtime benchmark manually."""
        import time as _time
        import uuid

        run_id = f"onnx_{uuid.uuid4().hex[:8]}"

        onnx_detector.load()

        # Generate input frames if not provided
        if input_frames is None:
            h, w = detector_config.input_resolution
            input_frames = [
                np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                for _ in range(self._config.timed_iterations)
            ]

        # Warmup
        for i in range(self._config.warmup_iterations):
            idx = i % len(input_frames)
            onnx_detector.detect(input_frames[idx], confidence_threshold=0.0)

        # Timed
        latencies = []
        for i in range(self._config.timed_iterations):
            idx = i % len(input_frames)
            start = _time.perf_counter()
            onnx_detector.detect(input_frames[idx], confidence_threshold=0.0)
            latencies.append((_time.perf_counter() - start) * 1000)

        latencies_arr = np.array(latencies)
        mean_ms = float(np.mean(latencies_arr))
        p50_ms = float(np.median(latencies_arr))
        p95_ms = float(np.percentile(latencies_arr, 95))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        # Peak memory
        peak_mem = None
        try:
            import resource

            peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass

        return BenchmarkResult(
            run_id=run_id,
            timestamp=_time.time(),
            model_name=detector_config.model_name,
            model_version=detector_config.model_version,
            dataset_version=dataset_version,
            runtime="onnxruntime",
            precision_mode="fp32",
            hardware="cpu",
            input_resolution=detector_config.input_resolution,
            batch_size=self._config.batch_size,
            num_iterations=self._config.timed_iterations,
            warmup_iterations=self._config.warmup_iterations,
            latency_mean_ms=round(mean_ms, 2),
            latency_p50_ms=round(p50_ms, 2),
            latency_p95_ms=round(p95_ms, 2),
            throughput_fps=round(fps, 2),
            model_size_mb=round(onnx_file_size_mb, 2),
            peak_memory_mb=round(peak_mem, 2) if peak_mem else None,
            status="success",
        )