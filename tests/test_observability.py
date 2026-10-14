"""Tests for observability metrics and health checks.

Tests cover:
- MetricsCollector: counters, histograms, gauges, stats, export formats, reset
- HealthCheck: liveness, readiness, registered checks
"""

from __future__ import annotations

import time

import pytest

from edgedrive.observability.metrics import MetricsCollector, HealthCheck


# --------------------------------------------------------------------------- #
# MetricsCollector
# --------------------------------------------------------------------------- #

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_creation(self):
        """Collector should be created with empty metrics."""
        mc = MetricsCollector()
        assert mc.get_counter("nonexistent") == 0.0

    def test_increment_counter(self):
        """increment() should add to counter."""
        mc = MetricsCollector()
        mc.increment("requests_total")
        assert mc.get_counter("requests_total") == 1.0

        mc.increment("requests_total", 4)
        assert mc.get_counter("requests_total") == 5.0

    def test_increment_default_value(self):
        """increment() with no value should add 1."""
        mc = MetricsCollector()
        mc.increment("errors")
        assert mc.get_counter("errors") == 1.0

    def test_multiple_counters(self):
        """Multiple counters should be independent."""
        mc = MetricsCollector()
        mc.increment("requests_total")
        mc.increment("errors_total", 2)

        assert mc.get_counter("requests_total") == 1.0
        assert mc.get_counter("errors_total") == 2.0

    def test_observe_histogram(self):
        """observe() should record values in histogram."""
        mc = MetricsCollector()
        mc.observe("latency_ms", 10.0)
        mc.observe("latency_ms", 20.0)
        mc.observe("latency_ms", 30.0)

        stats = mc.get_histogram_stats("latency_ms")
        assert stats["count"] == 3
        assert stats["mean"] == pytest.approx(20.0)
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0

    def test_histogram_p50_p95(self):
        """Histogram stats should compute p50 and p95."""
        mc = MetricsCollector()
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for v in values:
            mc.observe("latency", float(v))

        stats = mc.get_histogram_stats("latency")
        assert stats["count"] == 10
        assert stats["p50"] == pytest.approx(55.0, abs=5.0)
        assert stats["p95"] >= 90.0

    def test_histogram_empty(self):
        """Histogram stats for nonexistent metric should return zeros."""
        mc = MetricsCollector()
        stats = mc.get_histogram_stats("nonexistent")
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["p50"] == 0.0
        assert stats["p95"] == 0.0

    def test_set_gauge(self):
        """set_gauge() should set a gauge value."""
        mc = MetricsCollector()
        mc.set_gauge("active_connections", 42)

        assert mc.get_gauge("active_connections") == 42.0

    def test_gauge_overwrite(self):
        """Setting a gauge again should overwrite the value."""
        mc = MetricsCollector()
        mc.set_gauge("queue_size", 10)
        mc.set_gauge("queue_size", 20)

        assert mc.get_gauge("queue_size") == 20.0

    def test_gauge_nonexistent(self):
        """get_gauge() for nonexistent should return None."""
        mc = MetricsCollector()
        assert mc.get_gauge("nonexistent") is None

    def test_export_prometheus(self):
        """export_prometheus() should produce Prometheus text format."""
        mc = MetricsCollector()
        mc.increment("requests_total", 5)
        mc.observe("latency_ms", 15.0)
        mc.set_gauge("active_connections", 3)

        text = mc.export_prometheus()

        assert isinstance(text, str)
        assert "requests_total 5.0" in text
        assert "active_connections" in text
        assert "latency_ms_count" in text
        assert "edgedrive_uptime_seconds" in text
        assert "# TYPE" in text

    def test_export_dict(self):
        """export_dict() should return a dictionary with all metrics."""
        mc = MetricsCollector()
        mc.increment("requests")
        mc.observe("latency", 10.0)
        mc.set_gauge("gauge1", 5.0)

        d = mc.export_dict()

        assert isinstance(d, dict)
        assert "counters" in d
        assert "gauges" in d
        assert "histograms" in d
        assert "uptime_seconds" in d
        assert d["counters"]["requests"] == 1.0
        assert d["gauges"]["gauge1"] == 5.0
        assert d["histograms"]["latency"]["count"] == 1

    def test_reset(self):
        """reset() should clear all metrics."""
        mc = MetricsCollector()
        mc.increment("counter1")
        mc.observe("hist1", 10.0)
        mc.set_gauge("gauge1", 5.0)

        mc.reset()

        assert mc.get_counter("counter1") == 0.0
        assert mc.get_gauge("gauge1") is None
        assert mc.get_histogram_stats("hist1")["count"] == 0

    def test_histogram_max_1000(self):
        """Histogram should keep at most 1000 observations."""
        mc = MetricsCollector()
        for i in range(1200):
            mc.observe("latency", float(i))

        stats = mc.get_histogram_stats("latency")
        assert stats["count"] == 1000

    def test_prometheus_has_type_annotations(self):
        """Prometheus export should include TYPE lines."""
        mc = MetricsCollector()
        mc.increment("reqs")
        text = mc.export_prometheus()
        assert "# TYPE reqs counter" in text
        assert "# TYPE edgedrive_uptime_seconds gauge" in text


# --------------------------------------------------------------------------- #
# HealthCheck
# --------------------------------------------------------------------------- #

class TestHealthCheck:
    """Tests for HealthCheck."""

    def test_liveness(self):
        """Liveness probe should always return alive."""
        hc = HealthCheck()
        result = hc.liveness()

        assert result["status"] == "alive"
        assert "timestamp" in result

    def test_readiness_no_checks(self):
        """Readiness with no registered checks should be healthy."""
        hc = HealthCheck()
        result = hc.readiness()

        assert result["status"] == "healthy"
        assert result["checks"] == {}

    def test_readiness_with_passing_check(self):
        """Readiness with a passing check should be healthy."""
        hc = HealthCheck()
        hc.register("database", lambda: {"conn": "ok"})

        result = hc.readiness()

        assert result["status"] == "healthy"
        assert "database" in result["checks"]
        assert result["checks"]["database"]["status"] == "healthy"
        assert result["checks"]["database"]["detail"] == {"conn": "ok"}

    def test_readiness_with_failing_check(self):
        """Readiness with a failing check should be unhealthy."""
        hc = HealthCheck()

        def bad_check():
            raise RuntimeError("connection failed")

        hc.register("database", bad_check)

        result = hc.readiness()

        assert result["status"] == "unhealthy"
        assert result["checks"]["database"]["status"] == "unhealthy"
        assert "connection failed" in result["checks"]["database"]["error"]

    def test_readiness_mixed_checks(self):
        """Readiness with mixed pass/fail should be unhealthy."""
        hc = HealthCheck()
        hc.register("good", lambda: "ok")
        hc.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        result = hc.readiness()

        assert result["status"] == "unhealthy"
        assert result["checks"]["good"]["status"] == "healthy"
        assert result["checks"]["bad"]["status"] == "unhealthy"

    def test_register_multiple_checks(self):
        """Multiple registered checks should all appear in readiness."""
        hc = HealthCheck()
        hc.register("db", lambda: "ok")
        hc.register("cache", lambda: "ok")
        hc.register("queue", lambda: "ok")

        result = hc.readiness()

        assert len(result["checks"]) == 3
        assert "db" in result["checks"]
        assert "cache" in result["checks"]
        assert "queue" in result["checks"]

    def test_liveness_has_timestamp(self):
        """Liveness should include a numeric timestamp."""
        hc = HealthCheck()
        result = hc.liveness()
        assert isinstance(result["timestamp"], float)
        assert result["timestamp"] > 0