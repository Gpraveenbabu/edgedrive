"""MLflow experiment tracking for EdgeDrive.

Logs benchmark results, pipeline metrics, and model metadata
to MLflow. Works with both Databricks-managed MLflow and local MLflow.

Usage:
    tracker = MLflowTracker(experiment_name="edgedrive_benchmarks")
    tracker.log_benchmark(benchmark_result)
    tracker.log_pipeline_run(frame_result)
"""

from __future__ import annotations

from typing import Any, Optional

from ..common.logging import get_logger
from ..common.types import BenchmarkResult, FrameResult


class MLflowTracker:
    """MLflow experiment tracking wrapper for EdgeDrive.

    Logs metrics, parameters, and artifacts from benchmark runs
    and pipeline executions. Gracefully handles environments where
    MLflow is not configured (logs a warning and skips).
    """

    def __init__(
        self,
        experiment_name: str = "edgedrive_benchmarks",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._logger = get_logger("edgedrive.mlops.tracking")
        self._mlflow = None
        self._available = False

        try:
            import mlflow
            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._available = True
            self._logger.info("mlflow_ready", experiment=experiment_name)
        except ImportError:
            self._logger.warning("mlflow_not_installed")
        except Exception as e:
            self._logger.warning("mlflow_setup_failed", error=str(e))

    @property
    def is_available(self) -> bool:
        """Whether MLflow tracking is available."""
        return self._available

    def log_benchmark(self, result: BenchmarkResult) -> Optional[str]:
        """Log a benchmark result to MLflow.

        Args:
            result: BenchmarkResult from BenchmarkRunner.

        Returns:
            Run ID if logged, None if MLflow not available.
        """
        if not self._available or self._mlflow is None:
            self._logger.warning("mlflow_unavailable_skip_log")
            return None

        with self._mlflow.start_run(run_name=f"benchmark_{result.model_name}") as run:
            # Log parameters
            self._mlflow.log_param("model_name", result.model_name)
            self._mlflow.log_param("model_version", result.model_version)
            self._mlflow.log_param("runtime", result.runtime)
            self._mlflow.log_param("precision", result.precision_mode)
            self._mlflow.log_param("hardware", result.hardware)
            self._mlflow.log_param("input_resolution", f"{result.input_resolution[0]}x{result.input_resolution[1]}")
            self._mlflow.log_param("batch_size", result.batch_size)
            self._mlflow.log_param("num_iterations", result.num_iterations)
            self._mlflow.log_param("warmup_iterations", result.warmup_iterations)

            # Log metrics
            self._mlflow.log_metric("latency_mean_ms", result.latency_mean_ms)
            self._mlflow.log_metric("latency_p50_ms", result.latency_p50_ms)
            self._mlflow.log_metric("latency_p95_ms", result.latency_p95_ms)
            self._mlflow.log_metric("throughput_fps", result.throughput_fps)

            if result.model_size_mb is not None:
                self._mlflow.log_metric("model_size_mb", result.model_size_mb)
            if result.peak_memory_mb is not None:
                self._mlflow.log_metric("peak_memory_mb", result.peak_memory_mb)
            if result.accuracy_metric is not None:
                self._mlflow.log_metric(result.accuracy_metric_name or "accuracy", result.accuracy_metric)

            self._logger.info("benchmark_logged", run_id=run.info.run_id)
            return run.info.run_id

    def log_pipeline_run(
        self,
        frame_result: FrameResult,
        run_name: str = "pipeline_run",
    ) -> Optional[str]:
        """Log a single pipeline frame result to MLflow.

        Args:
            frame_result: FrameResult from PerceptionPipeline.process_frame().
            run_name: MLflow run name.

        Returns:
            Run ID if logged, None if MLflow not available.
        """
        if not self._available or self._mlflow is None:
            return None

        with self._mlflow.start_run(run_name=run_name) as run:
            self._mlflow.log_param("model_name", frame_result.model_name)
            self._mlflow.log_param("model_version", frame_result.model_version)

            self._mlflow.log_metric("num_detections", frame_result.num_detections)
            self._mlflow.log_metric("num_tracks", frame_result.num_tracks)
            self._mlflow.log_metric("inference_time_ms", frame_result.inference_time_ms)
            self._mlflow.log_metric("total_time_ms", frame_result.total_time_ms)

            self._logger.info("pipeline_run_logged", run_id=run.info.run_id)
            return run.info.run_id

    def log_comparison(self, comparison_result: Any) -> Optional[str]:
        """Log a PyTorch vs ONNX comparison result to MLflow.

        Args:
            comparison_result: ComparisonResult from ComparisonBenchmark.

        Returns:
            Run ID if logged, None if MLflow not available.
        """
        if not self._available or self._mlflow is None:
            return None

        with self._mlflow.start_run(run_name="pytorch_vs_onnx") as run:
            self._mlflow.log_param("comparison_type", "pytorch_vs_onnx")

            self._mlflow.log_metric("pytorch_latency_mean_ms", comparison_result.pytorch.latency_mean_ms)
            self._mlflow.log_metric("pytorch_throughput_fps", comparison_result.pytorch.throughput_fps)
            self._mlflow.log_metric("onnx_latency_mean_ms", comparison_result.onnx.latency_mean_ms)
            self._mlflow.log_metric("onnx_throughput_fps", comparison_result.onnx.throughput_fps)

            if comparison_result.latency_speedup is not None:
                self._mlflow.log_metric("latency_speedup", comparison_result.latency_speedup)
            if comparison_result.throughput_ratio is not None:
                self._mlflow.log_metric("throughput_ratio", comparison_result.throughput_ratio)
            if comparison_result.size_ratio is not None:
                self._mlflow.log_metric("size_ratio", comparison_result.size_ratio)

            return run.info.run_id