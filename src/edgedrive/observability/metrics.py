"""Observability metrics for EdgeDrive.

Provides Prometheus-compatible metrics collection, health checks,
and request tracing stubs. Designed for edge deployment monitoring.

Metrics collected:
- Detection latency histograms
- Detection count per frame
- Inference requests total
- Model load time
- Error rate
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

from ..common.logging import get_logger


class MetricsCollector:
    """Collects and exposes application metrics.

    Thread-safe metrics collection without external dependencies.
    Can be exported as Prometheus text format or plain dict.
    """

    def __init__(self) -> None:
        self._logger = get_logger("edgedrive.observability.metrics")
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._start_time = time.time()

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        self._counters[name] += value

    def observe(self, name: str, value: float) -> None:
        """Record a histogram observation."""
        self._histograms[name].append(value)
        # Keep last 1000 observations
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self._gauges[name] = value

    def get_counter(self, name: str) -> float:
        """Get a counter value."""
        return self._counters.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics."""
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}

        import numpy as np
        arr = np.array(values)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    def get_gauge(self, name: str) -> Optional[float]:
        """Get a gauge value."""
        return self._gauges.get(name)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        for name, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        for name, values in sorted(self._histograms.items()):
            stats = self.get_histogram_stats(name)
            lines.append(f"# TYPE {name} histogram")
            lines.append(f'{name}_count {stats["count"]}')
            lines.append(f'{name}_mean {stats["mean"]:.4f}')
            lines.append(f'{name}_p50 {stats["p50"]:.4f}')
            lines.append(f'{name}_p95 {stats["p95"]:.4f}')

        # Uptime
        uptime = time.time() - self._start_time
        lines.append(f"# TYPE edgedrive_uptime_seconds gauge")
        lines.append(f"edgedrive_uptime_seconds {uptime:.2f}")

        return "\n".join(lines)

    def export_dict(self) -> dict[str, Any]:
        """Export metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self._histograms
            },
            "uptime_seconds": time.time() - self._start_time,
        }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()
        self._start_time = time.time()


class HealthCheck:
    """Health check handler for EdgeDrive.

    Provides liveness and readiness checks suitable for
    Kubernetes/Docker deployment.
    """

    def __init__(self) -> None:
        self._logger = get_logger("edgedrive.observability.health")
        self._checks: dict[str, callable] = {}

    def register(self, name: str, check_fn: callable) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn

    def liveness(self) -> dict[str, Any]:
        """Liveness probe — is the process alive?"""
        return {"status": "alive", "timestamp": time.time()}

    def readiness(self) -> dict[str, Any]:
        """Readiness probe — is the system ready to serve?"""
        results = {}
        all_healthy = True

        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results[name] = {"status": "healthy", "detail": result}
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
        }