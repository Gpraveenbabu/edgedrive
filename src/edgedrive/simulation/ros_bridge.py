"""ROS 2 communication bridge for EdgeDrive.

Provides a bridge between EdgeDrive's perception pipeline and
ROS 2 topics for integration with autonomous vehicle stacks.

Requires rclpy (ROS 2 Python client) installed:
    pip install rclpy  # from ROS 2 distribution

The bridge lazily imports rclpy — module can be imported
without ROS 2 installed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..common.logging import get_logger
from ..common.types import FrameResult


class ROS2Bridge:
    """Bridge between EdgeDrive and ROS 2.

    Subscribes to camera image topics and publishes perception
    results (detections, events) to ROS 2 topics.

    Topics:
    - Input:  /edgedrive/camera/image (sensor_msgs/Image)
    - Output: /edgedrive/detections (vision_msgs/Detection2DArray)
    - Output: /edgedrive/events (std_msgs/String JSON)
    - Output: /edgedrive/metrics (std_msgs/Float32MultiArray)

    Usage:
        bridge = ROS2Bridge(node_name="edgedrive_perception")
        bridge.start(pipeline)
        # ... runs until shutdown ...
        bridge.stop()
    """

    def __init__(
        self,
        node_name: str = "edgedrive_perception",
        input_topic: str = "/edgedrive/camera/image",
        detection_topic: str = "/edgedrive/detections",
        event_topic: str = "/edgedrive/events",
    ) -> None:
        self._node_name = node_name
        self._input_topic = input_topic
        self._detection_topic = detection_topic
        self._event_topic = event_topic
        self._logger = get_logger("edgedrive.simulation.ros2")
        self._node: Any = None
        self._running = False
        self._pipeline: Any = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, pipeline: Any) -> None:
        """Start the ROS 2 node and subscribe to topics.

        Args:
            pipeline: PerceptionPipeline instance to process frames.

        Raises:
            ImportError: If rclpy not installed.
        """
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
        except ImportError as e:
            raise ImportError(
                "ROS 2 Python client (rclpy) not installed. "
                "Install from your ROS 2 distribution."
            ) from e

        self._pipeline = pipeline
        rclpy.init()

        self._node = rclpy.create_node(self._node_name)

        # Publishers
        self._det_pub = self._node.create_publisher(String, self._detection_topic, 10)
        self._event_pub = self._node.create_publisher(String, self._event_topic, 10)

        # Subscriber
        self._node.create_subscription(
            Image,
            self._input_topic,
            self._on_image,
            10,
        )

        self._running = True
        self._logger.info("ros2_started", node=self._node_name)

    def _on_image(self, msg: Any) -> None:
        """Callback for incoming camera images."""
        from std_msgs.msg import String
        import json

        # Convert ROS Image to numpy array
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )

        # Process through pipeline
        result = self._pipeline.process_frame(frame, source=msg.header.frame_id)

        # Publish detections
        det_msg = String()
        det_msg.data = json.dumps({
            "frame_index": result.frame_metadata.frame_index,
            "num_detections": result.num_detections,
            "inference_time_ms": result.inference_time_ms,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                }
                for d in result.detections
            ],
        })
        self._det_pub.publish(det_msg)

        # Publish events
        for event in self._pipeline.all_events[-10:]:  # Last 10 events
            event_msg = String()
            event_msg.data = json.dumps({
                "event_type": event.event_type.value,
                "frame_index": event.frame_index,
                "class_name": event.class_name,
            })
            self._event_pub.publish(event_msg)

    def spin(self) -> None:
        """Spin the ROS 2 node (blocking)."""
        if not self._running or self._node is None:
            raise RuntimeError("Bridge not started")

        import rclpy
        rclpy.spin(self._node)

    def stop(self) -> None:
        """Stop the ROS 2 node and clean up."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        try:
            import rclpy
            rclpy.shutdown()
        except Exception:
            pass

        self._running = False
        self._logger.info("ros2_stopped")

    def __enter__(self) -> "ROS2Bridge":
        return self

    def __exit__(self, *args) -> None:
        self.stop()