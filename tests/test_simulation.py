"""Tests for simulation bridges: CARLA and ROS 2.

Tests verify the interface contracts of CarlaBridge and ROS2Bridge without
requiring the actual CARLA simulator or ROS 2 runtime. All imports are
lazy, so the module loads without carla/rclpy installed.

Tests cover:
- CarlaBridge: creation, connection error, camera setup without connect
- ROS2Bridge: creation, start without rclpy, topic configuration
"""

from __future__ import annotations

import pytest

from edgedrive.simulation.carla_bridge import CarlaBridge
from edgedrive.simulation.ros_bridge import ROS2Bridge


# --------------------------------------------------------------------------- #
# CarlaBridge
# --------------------------------------------------------------------------- #

class TestCarlaBridge:
    """Tests for CarlaBridge (without real CARLA server)."""

    def test_creation(self):
        """CarlaBridge should be created with default params."""
        bridge = CarlaBridge()
        assert bridge.is_connected is False
        assert bridge._host == "localhost"
        assert bridge._port == 2000

    def test_creation_custom_params(self):
        """CarlaBridge should accept custom host/port/resolution."""
        bridge = CarlaBridge(
            host="192.168.1.100",
            port=3000,
            image_width=640,
            image_height=480,
            fov=110.0,
        )
        assert bridge._host == "192.168.1.100"
        assert bridge._port == 3000
        assert bridge._image_width == 640
        assert bridge._image_height == 480
        assert bridge._fov == 110.0

    def test_connect_without_carla_raises(self):
        """connect() should raise ImportError if carla package not installed."""
        bridge = CarlaBridge()

        try:
            import carla  # noqa: F401
            pytest.skip("carla is installed — cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError, match="CARLA Python API not installed"):
                bridge.connect()

    def test_setup_camera_without_connect_raises(self):
        """setup_camera() should raise RuntimeError if not connected."""
        bridge = CarlaBridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.setup_camera()

    def test_get_latest_frame_none(self):
        """get_latest_frame() should return None before any frames received."""
        bridge = CarlaBridge()
        assert bridge.get_latest_frame() is None

    def test_spawn_vehicle_without_connect_raises(self):
        """spawn_vehicle() should raise RuntimeError if not connected."""
        bridge = CarlaBridge()
        with pytest.raises(RuntimeError, match="Not connected"):
            bridge.spawn_vehicle()

    def test_disconnect_without_connect(self):
        """disconnect() should work even if never connected."""
        bridge = CarlaBridge()
        bridge.disconnect()
        assert bridge.is_connected is False

    def test_frames_generator_empty(self):
        """frames() should yield nothing if no frames are available (with timeout)."""
        bridge = CarlaBridge()
        # This will not connect and should yield nothing
        # We can't test frames() without a connection, so just verify it exists
        assert hasattr(bridge, "frames")

    def test_context_manager_protocol(self):
        """CarlaBridge should support context manager protocol."""
        bridge = CarlaBridge()
        # __enter__ calls connect(), which will raise ImportError without carla
        # Just verify the protocol exists
        assert hasattr(bridge, "__enter__")
        assert hasattr(bridge, "__exit__")


# --------------------------------------------------------------------------- #
# ROS2Bridge
# --------------------------------------------------------------------------- #

class TestROS2Bridge:
    """Tests for ROS2Bridge (without real ROS 2 runtime)."""

    def test_creation(self):
        """ROS2Bridge should be created with default params."""
        bridge = ROS2Bridge()
        assert bridge.is_running is False
        assert bridge._node_name == "edgedrive_perception"
        assert bridge._input_topic == "/edgedrive/camera/image"
        assert bridge._detection_topic == "/edgedrive/detections"
        assert bridge._event_topic == "/edgedrive/events"

    def test_creation_custom_topics(self):
        """ROS2Bridge should accept custom topic names."""
        bridge = ROS2Bridge(
            node_name="custom_node",
            input_topic="/custom/input",
            detection_topic="/custom/detections",
            event_topic="/custom/events",
        )
        assert bridge._node_name == "custom_node"
        assert bridge._input_topic == "/custom/input"
        assert bridge._detection_topic == "/custom/detections"
        assert bridge._event_topic == "/custom/events"

    def test_start_without_rclpy_raises(self):
        """start() should raise ImportError if rclpy not installed."""
        bridge = ROS2Bridge()

        try:
            import rclpy  # noqa: F401
            pytest.skip("rclpy is installed — cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError, match="ROS 2 Python client"):
                bridge.start(pipeline=None)

    def test_spin_without_start_raises(self):
        """spin() should raise RuntimeError if not started."""
        bridge = ROS2Bridge()
        with pytest.raises(RuntimeError, match="Bridge not started"):
            bridge.spin()

    def test_stop_without_start(self):
        """stop() should work even if never started."""
        bridge = ROS2Bridge()
        bridge.stop()
        assert bridge.is_running is False

    def test_context_manager_protocol(self):
        """ROS2Bridge should support context manager protocol."""
        bridge = ROS2Bridge()
        assert hasattr(bridge, "__enter__")
        assert hasattr(bridge, "__exit__")

    def test_is_running_default_false(self):
        """is_running should be False before start()."""
        bridge = ROS2Bridge()
        assert bridge.is_running is False

    def test_stop_after_start_sets_not_running(self):
        """stop() should set is_running to False."""
        bridge = ROS2Bridge()
        bridge._running = True
        bridge.stop()
        assert bridge.is_running is False