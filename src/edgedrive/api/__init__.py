"""FastAPI server for EdgeDrive perception pipeline.

Provides REST endpoints for object detection, pipeline processing,
and benchmarking. Designed for edge deployment and integration
with automotive systems.
"""

from .server import create_app, EdgeDriveServer

__all__ = ["create_app", "EdgeDriveServer"]