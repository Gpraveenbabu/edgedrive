"""Synthetic test fixtures for EdgeDrive.

Generates deterministic synthetic frames with known shapes for testing
the perception pipeline without requiring real video files or model downloads.

Fixtures are used by unit tests and the demo mode. They do NOT replace
real inference — they provide deterministic inputs for testing pipeline
logic, tracking, and event generation.
"""

from __future__ import annotations

from typing import Generator

import numpy as np


def create_synthetic_frame(
    width: int = 320,
    height: int = 320,
    shapes: list[dict] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic frame with colored rectangles.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        shapes: List of shape dicts with keys: type ("rect"), x1, y1, x2, y2,
                color (BGR tuple). If None, generates default shapes.
        seed: Random seed for deterministic output.

    Returns:
        HxWx3 uint8 RGB numpy array.
    """
    rng = np.random.RandomState(seed)
    frame = np.full((height, width, 3), 128, dtype=np.uint8)

    if shapes is None:
        # Default: a few rectangles simulating road objects
        shapes = [
            {"type": "rect", "x1": 50, "y1": 100, "x2": 150, "y2": 200, "color": (255, 100, 50)},
            {"type": "rect", "x1": 200, "y1": 150, "x2": 280, "y2": 250, "color": (50, 200, 50)},
            {"type": "rect", "x1": 100, "y1": 50, "x2": 180, "y2": 90, "color": (50, 50, 255)},
        ]

    for shape in shapes:
        if shape["type"] == "rect":
            x1, y1 = int(shape["x1"]), int(shape["y1"])
            x2, y2 = int(shape["x2"]), int(shape["y2"])
            color = shape.get("color", (255, 255, 255))
            frame[y1:y2, x1:x2] = color

    # Add small seed-dependent noise so different seeds produce different frames
    noise = rng.randint(0, 10, (height, width, 3), dtype=np.uint8)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def create_synthetic_video(
    num_frames: int = 10,
    width: int = 320,
    height: int = 320,
    move_shapes: bool = True,
    seed: int = 42,
) -> list[np.ndarray]:
    """Create a sequence of synthetic frames for testing.

    Args:
        num_frames: Number of frames to generate.
        width: Frame width.
        height: Frame height.
        move_shapes: If True, shapes move slightly between frames (for tracking tests).
        seed: Random seed.

    Returns:
        List of HxWx3 uint8 RGB numpy arrays.
    """
    frames: list[np.ndarray] = []
    base_shapes = [
        {"type": "rect", "x1": 50, "y1": 100, "x2": 150, "y2": 200, "color": (255, 100, 50)},
        {"type": "rect", "x1": 200, "y1": 150, "x2": 280, "y2": 250, "color": (50, 200, 50)},
    ]

    for i in range(num_frames):
        if move_shapes:
            offset = i * 3  # Move shapes 3 pixels per frame
            shapes = [
                {
                    "type": "rect",
                    "x1": min(s["x1"] + offset, width - 10),
                    "y1": s["y1"],
                    "x2": min(s["x2"] + offset, width),
                    "y2": s["y2"],
                    "color": s["color"],
                }
                for s in base_shapes
            ]
        else:
            shapes = base_shapes

        frames.append(create_synthetic_frame(width, height, shapes, seed=seed))

    return frames


def create_black_frame(width: int = 320, height: int = 320) -> np.ndarray:
    """Create a completely black frame (no detections expected)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def create_random_frame(width: int = 320, height: int = 320, seed: int = 42) -> np.ndarray:
    """Create a random noise frame for benchmarking."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (height, width, 3), dtype=np.uint8)