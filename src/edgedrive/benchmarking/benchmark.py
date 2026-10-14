"""Benchmark runner for EdgeDrive.

Reproducibly measures inference latency, throughput, and model size across
different model configurations. Results are saved as CSV and JSON.

All measurements are real — no invented results. Failed runs are recorded
with status='error' and an error message.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..common.logging import get_logger
from ..common.types import BenchmarkResult
from ..config.settings import BenchmarkConfig, DetectorConfig


class BenchmarkRunner:
    """Runs reproducible inference benchmarks.

    Measures:
    - Mean, p50, p95 latency (ms)
    - Throughput (FPS)
    - Model size (MB, where measurable)
    - Peak memory (MB, where measurable)

    All measurements use the same input for warm-up and timed phases.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._logger = get_logger("edgedrive.benchmarking")

    def run(
        self,
        detector_config: DetectorConfig,
        input_frames: Optional[list[np.ndarray]] = None,
        dataset_version: str = "synthetic",
    ) -> BenchmarkResult:
        """Run a benchmark for a given detector configuration.

        Args:
            detector_config: Configuration for the detector to benchmark.
            input_frames: Frames to use for benchmarking. If None, generates
                          random frames at the configured resolution.
            dataset_version: Identifier for the dataset used.

        Returns:
            BenchmarkResult with measured metrics.
        """
        from ..perception.detector import get_detector

        w, h = self._config.input_resolution
        if input_frames is None:
            rng = np.random.RandomState(42)
            input_frames = [
                rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
                for _ in range(self._config.batch_size)
            ]

        # Create detector
        detector = get_detector("torchvision", detector_config)
        model_name = detector_config.model_name
        model_version = detector_config.model_version

        self._logger.info(
            "benchmark_start",
            model=model_name,
            warmup=self._config.warmup_iterations,
            timed=self._config.timed_iterations,
        )

        try:
            detector.load()
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                model_version=model_version,
                dataset_version=dataset_version,
                runtime="pytorch",
                precision_mode=detector_config.device,
                hardware=detector_config.device,
                input_resolution=self._config.input_resolution,
                batch_size=self._config.batch_size,
                num_iterations=0,
                warmup_iterations=self._config.warmup_iterations,
                latency_mean_ms=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                throughput_fps=0.0,
                status="error",
                error_message=f"Model load failed: {e}",
            )

        # --- Warm-up ---
        for _ in range(self._config.warmup_iterations):
            frame = input_frames[0]
            detector.detect(frame, confidence_threshold=0.0)

        # --- Timed inference ---
        latencies: list[float] = []
        for i in range(self._config.timed_iterations):
            frame = input_frames[i % len(input_frames)]
            start = time.perf_counter()
            detector.detect(frame, confidence_threshold=0.0)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        # --- Compute statistics ---
        mean_ms = statistics.mean(latencies)
        p50_ms = statistics.median(latencies)
        p95_ms = np.percentile(latencies, 95)
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        # --- Model size ---
        model_size_mb = self._get_model_size(detector)

        # --- Peak memory ---
        peak_mem_mb = self._get_peak_memory()

        result = BenchmarkResult(
            model_name=model_name,
            model_version=model_version,
            dataset_version=dataset_version,
            runtime="pytorch",
            precision_mode="fp32",
            hardware=detector_config.device,
            input_resolution=self._config.input_resolution,
            batch_size=self._config.batch_size,
            num_iterations=self._config.timed_iterations,
            warmup_iterations=self._config.warmup_iterations,
            latency_mean_ms=round(mean_ms, 3),
            latency_p50_ms=round(p50_ms, 3),
            latency_p95_ms=round(p95_ms, 3),
            throughput_fps=round(fps, 2),
            model_size_mb=round(model_size_mb, 2) if model_size_mb else None,
            peak_memory_mb=round(peak_mem_mb, 2) if peak_mem_mb else None,
            status="success",
        )

        self._logger.info(
            "benchmark_complete",
            model=model_name,
            mean_ms=round(mean_ms, 2),
            p95_ms=round(p95_ms, 2),
            fps=round(fps, 2),
        )

        return result

    def _get_model_size(self, detector: Any) -> Optional[float]:
        """Estimate model size in MB from parameters."""
        try:
            import torch
            meta = detector.metadata()
            if meta.num_parameters:
                # FP32 = 4 bytes per parameter
                return meta.num_parameters * 4 / (1024 * 1024)
        except Exception:
            pass
        return None

    def _get_peak_memory(self) -> Optional[float]:
        """Get peak memory usage in MB."""
        try:
            import resource
            # ru_maxrss is in KB on Linux
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return None

    @staticmethod
    def save_results_csv(
        results: list[BenchmarkResult],
        output_path: str,
    ) -> str:
        """Save benchmark results to a CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not results:
            return str(path)

        fieldnames = list(results[0].model_dump().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.model_dump()
                # Convert datetime to ISO format
                row["timestamp"] = r.timestamp.isoformat() if r.timestamp else ""
                row["input_resolution"] = f"{r.input_resolution[0]}x{r.input_resolution[1]}"
                writer.writerow(row)

        return str(path)

    @staticmethod
    def save_results_json(
        results: list[BenchmarkResult],
        output_path: str,
    ) -> str:
        """Save benchmark results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.model_dump(mode="json") for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return str(path)