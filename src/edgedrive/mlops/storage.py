"""Delta Lake storage for EdgeDrive results.

Stores FrameResult and PerceptionEvent records to Delta tables
for persistent, queryable storage. Works within Databricks
(spark session) or locally (writes JSON/CSV as fallback).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..common.logging import get_logger
from ..common.types import BenchmarkResult, FrameResult, PerceptionEvent


class ResultStore:
    """Persist EdgeDrive results to Delta Lake or filesystem.

    On Databricks with Spark: writes to Delta tables.
    Without Spark: writes JSON files as fallback.
    """

    def __init__(self, table_path: str = "edgedrive.results") -> None:
        self._table_path = table_path
        self._logger = get_logger("edgedrive.mlops.storage")
        self._spark = None
        self._use_delta = False

        # Always set up fallback dir so _save_to_json works even if Delta write fails
        self._fallback_dir = Path("outputs/results")
        self._fallback_dir.mkdir(parents=True, exist_ok=True)

        try:
            from pyspark.sql import SparkSession
            self._spark = SparkSession.getActiveSession()
            if self._spark is not None:
                self._use_delta = True
                self._logger.info("delta_storage_ready", table=table_path)
        except Exception:
            pass

        if not self._use_delta:
            self._logger.info("filesystem_storage_ready", path=str(self._fallback_dir))

    @property
    def is_delta(self) -> bool:
        """Whether Delta Lake storage is available."""
        return self._use_delta

    def save_frame_result(self, result: FrameResult) -> Optional[str]:
        """Save a single FrameResult.

        Returns:
            Storage path or table name, None on failure.
        """
        data = {
            "frame_index": result.frame_metadata.frame_index,
            "source": result.frame_metadata.source,
            "width": result.frame_metadata.width,
            "height": result.frame_metadata.height,
            "num_detections": result.num_detections,
            "num_tracks": result.num_tracks,
            "inference_time_ms": result.inference_time_ms,
            "total_time_ms": result.total_time_ms,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "status": result.status.value,
        }

        if self._use_delta and self._spark is not None:
            return self._save_to_delta(data, f"{self._table_path}.frame_results")
        else:
            return self._save_to_json(data, f"frame_result_{result.frame_metadata.frame_index}")

    def save_events(self, events: list[PerceptionEvent]) -> Optional[str]:
        """Save perception events.

        Returns:
            Storage path or table name, None on failure.
        """
        if not events:
            return None

        data = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "frame_index": e.frame_index,
                "source_frame": e.source_frame,
                "track_id": e.track_id,
                "class_name": e.class_name,
                "confidence": e.confidence,
                "model_name": e.model_name,
                "model_version": e.model_version,
                "status": e.status.value,
            }
            for e in events
        ]

        if self._use_delta and self._spark is not None:
            return self._save_to_delta(data, f"{self._table_path}.events")
        else:
            return self._save_to_json(data, f"events_{events[0].frame_index}")

    def save_benchmark(self, result: BenchmarkResult) -> Optional[str]:
        """Save a benchmark result.

        Returns:
            Storage path or table name, None on failure.
        """
        data = result.model_dump(mode="json")

        if self._use_delta and self._spark is not None:
            return self._save_to_delta(data, f"{self._table_path}.benchmarks")
        else:
            return self._save_to_json(data, f"benchmark_{result.run_id}")

    def _save_to_delta(self, data: Any, table: str) -> str:
        """Save data to a Delta table via Spark."""
        try:
            assert self._spark is not None
            if isinstance(data, list):
                df = self._spark.createDataFrame(data)
            else:
                df = self._spark.createDataFrame([data])
            df.write.format("delta").mode("append").saveAsTable(table)
            self._logger.info("delta_saved", table=table)
            return table
        except Exception as e:
            self._logger.error("delta_save_failed", error=str(e))
            return self._save_to_json(data, table.replace(".", "_"))

    def _save_to_json(self, data: Any, name: str) -> str:
        """Save data to a JSON file as fallback."""
        path = self._fallback_dir / f"{name}.json"
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._logger.info("json_saved", path=str(path))
            return str(path)
        except Exception as e:
            self._logger.error("json_save_failed", error=str(e))
            return ""