"""
Isaac-VLA Shared Utilities
===========================

Common utilities used across the isaac-vla system: configuration loading,
logging setup, image processing, action conversion, and math helpers.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


# ─── Configuration ───────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep-merge two dictionaries. Override takes precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(
    name: str = "isaac-vla",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ─── Image Processing ────────────────────────────────────────────────────────

def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Resize an image to target (width, height) using PIL."""
    from PIL import Image

    pil_img = Image.fromarray(image.astype(np.uint8))
    resample = Image.BILINEAR if interpolation == "bilinear" else Image.NEAREST
    pil_img = pil_img.resize(target_size, resample)
    return np.array(pil_img)


def crop_and_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    center_crop: bool = True,
) -> np.ndarray:
    """Center-crop to square then resize to target size (OpenVLA-OFT preprocessing)."""
    from PIL import Image

    h, w = image.shape[:2]
    if center_crop:
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        image = image[top : top + size, left : left + size]

    pil_img = Image.fromarray(image.astype(np.uint8))
    pil_img = pil_img.resize(target_size, Image.BILINEAR)
    return np.array(pil_img)


def encode_image_base64(image: np.ndarray) -> str:
    """Encode a numpy image array as base64 JPEG string."""
    import base64
    from io import BytesIO

    from PIL import Image

    pil_img = Image.fromarray(image.astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_image_base64(b64_string: str) -> np.ndarray:
    """Decode a base64 JPEG string to numpy image array."""
    import base64
    from io import BytesIO

    from PIL import Image

    buf = BytesIO(base64.b64decode(b64_string))
    pil_img = Image.open(buf).convert("RGB")
    return np.array(pil_img)


# ─── Action Math ─────────────────────────────────────────────────────────────

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (XYZ convention) to a 3×3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to Euler angles (XYZ convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.array([roll, pitch, yaw])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] or [x, y, z, w] to 3×3 rotation matrix.

    Isaac Sim uses scalar-first convention: [w, x, y, z].
    """
    if q.shape[0] == 4:
        # Detect convention: if |q[0]| <= 1, assume scalar-first
        if abs(q[0]) <= 1.0 and abs(q[3]) <= 1.0:
            w, x, y, z = q  # scalar-first
        else:
            x, y, z, w = q  # scalar-last
    else:
        raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z] (scalar-first)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def delta_ee_to_pose(
    current_position: np.ndarray,
    current_orientation_quat: np.ndarray,
    delta_ee: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a delta EE action to current EE pose.

    Args:
        current_position: [x, y, z] current EE position
        current_orientation_quat: [w, x, y, z] current EE orientation
        delta_ee: [dx, dy, dz, droll, dpitch, dyaw, gripper] delta action

    Returns:
        Tuple of (new_position, new_orientation_quat)
    """
    new_position = current_position + delta_ee[:3]

    # Convert delta Euler to rotation matrix
    delta_rot = euler_to_rotation_matrix(delta_ee[3], delta_ee[4], delta_ee[5])

    # Current orientation as rotation matrix
    current_rot = quaternion_to_rotation_matrix(current_orientation_quat)

    # Apply delta rotation
    new_rot = current_rot @ delta_rot
    new_quat = rotation_matrix_to_quaternion(new_rot)

    return new_position, new_quat


# ─── Safety ──────────────────────────────────────────────────────────────────

def clamp_to_workspace(
    position: np.ndarray,
    bounds: Dict[str, List[float]],
) -> np.ndarray:
    """Clamp a position to the workspace bounds."""
    clamped = position.copy()
    for i, axis in enumerate(["x", "y", "z"]):
        lo, hi = bounds[axis]
        clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped


def clip_action_magnitude(
    action: np.ndarray,
    max_position: float = 0.05,
    max_rotation: float = 0.1,
) -> np.ndarray:
    """Clip action delta magnitudes for safety."""
    clipped = action.copy()
    # Position deltas
    clipped[:3] = np.clip(clipped[:3], -max_position, max_position)
    # Rotation deltas
    clipped[3:6] = np.clip(clipped[3:6], -max_rotation, max_rotation)
    return clipped


# ─── Timing ──────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager timer."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.4f}s")


# ─── Franka Constants ────────────────────────────────────────────────────────

FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FRANKA_JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FRANKA_JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

FRANKA_HOME_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

FRANKA_GRIPPER_OPEN = 0.04    # meters
FRANKA_GRIPPER_CLOSED = 0.0   # meters
FRANKA_GRIPPER_THRESHOLD = 0.5  # VLA output threshold

# Franka EE link name
FRANKA_EE_LINK = "panda_hand"