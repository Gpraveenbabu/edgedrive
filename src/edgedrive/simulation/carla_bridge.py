"""CARLA simulation bridge for EdgeDrive.

Connects to a CARLA simulator server to retrieve synthetic
camera frames for testing the perception pipeline in a
controlled simulation environment.

Requires CARLA Python API (carla package) installed separately:
    pip install carla  # from CARLA release

The bridge lazily imports carla — module can be imported
without the package installed.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

import numpy as np

from ..common.logging import get_logger


class CarlaBridge:
    """Bridge to CARLA simulator for synthetic driving scenarios.

    Provides:
    - Connection to CARLA server
    - Camera sensor setup (RGB)
    - Frame retrieval as numpy arrays
    - Vehicle spawn and control stubs
    - Cleanup on disconnect

    Usage:
        bridge = CarlaBridge(host="localhost", port=2000)
        bridge.connect()
        for frame in bridge.frames(max_frames=100):
            # frame is HxWx3 uint8 RGB
            process(frame)
        bridge.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        image_width: int = 320,
        image_height: int = 320,
        fov: float = 90.0,
    ) -> None:
        self._host = host
        self._port = port
        self._image_width = image_width
        self._image_height = image_height
        self._fov = fov
        self._logger = get_logger("edgedrive.simulation.carla")
        self._client: Any = None
        self._world: Any = None
        self._camera: Any = None
        self._connected = False
        self._latest_frame: Optional[np.ndarray] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Connect to the CARLA server.

        Raises:
            ImportError: If carla package not installed.
            RuntimeError: If connection fails.
        """
        try:
            import carla
        except ImportError as e:
            raise ImportError(
                "CARLA Python API not installed. Install with: pip install carla"
            ) from e

        self._logger.info("connecting_to_carla", host=self._host, port=self._port)

        try:
            self._client = carla.Client(self._host, self._port)
            self._client.set_timeout(10.0)
            self._world = self._client.get_world()
            self._connected = True
            self._logger.info("carla_connected")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CARLA: {e}") from e

    def setup_camera(self, vehicle: Any = None) -> None:
        """Set up an RGB camera sensor.

        Args:
            vehicle: CARLA actor to attach camera to. If None, uses
                     a fixed spectator position.
        """
        if not self._connected:
            raise RuntimeError("Not connected to CARLA")

        import carla

        blueprint_lib = self._world.get_blueprint_library()
        camera_bp = blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self._image_width))
        camera_bp.set_attribute("image_size_y", str(self._image_height))
        camera_bp.set_attribute("fov", str(self._fov))

        transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15),
        )

        if vehicle is not None:
            self._camera = self._world.spawn_actor(
                camera_bp, transform, attach_to=vehicle
            )
        else:
            self._camera = self._world.spawn_actor(camera_bp, transform)

        self._camera.listen(self._on_camera_data)
        self._logger.info("camera_setup")

    def _on_camera_data(self, image: Any) -> None:
        """Callback for camera data."""
        self._latest_frame = np.frombuffer(
            image.raw_data, dtype=np.uint8
        ).reshape(
            self._image_height, self._image_width, 4
        )[:, :, :3].copy()  # Drop alpha, keep RGB

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest camera frame."""
        return self._latest_frame

    def frames(
        self,
        max_frames: int = 100,
        timeout_s: float = 1.0,
    ) -> Generator[np.ndarray, None, None]:
        """Generate frames from the CARLA camera.

        Args:
            max_frames: Maximum number of frames to yield.
            timeout_s: Max wait time per frame.

        Yields:
            HxWx3 uint8 RGB numpy arrays.
        """
        import time

        for _ in range(max_frames):
            start = time.time()
            while self._latest_frame is None and time.time() - start < timeout_s:
                time.sleep(0.01)
            if self._latest_frame is not None:
                yield self._latest_frame
                self._latest_frame = None
            else:
                self._logger.warning("frame_timeout")

    def spawn_vehicle(
        self,
        vehicle_type: str = "vehicle.tesla.model3",
    ) -> Any:
        """Spawn a vehicle in the simulation.

        Returns:
            CARLA vehicle actor.
        """
        if not self._connected:
            raise RuntimeError("Not connected to CARLA")

        import carla

        blueprint_lib = self._world.get_blueprint_library()
        vehicle_bp = blueprint_lib.find(vehicle_type)

        spawn_point = self._world.get_map().get_spawn_points()[0]
        vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)

        self._logger.info("vehicle_spawned", type=vehicle_type, id=vehicle.id)
        return vehicle

    def disconnect(self) -> None:
        """Disconnect and clean up all actors."""
        if self._camera is not None:
            self._camera.destroy()
            self._camera = None
        self._connected = False
        self._client = None
        self._world = None
        self._logger.info("carla_disconnected")

    def __enter__(self) -> "CarlaBridge":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()