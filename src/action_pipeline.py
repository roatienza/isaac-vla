"""
Action Pipeline: VLA Output → Isaac Sim Actions
=================================================

Converts OpenVLA-OFT action outputs into Isaac Sim-compatible joint
position commands. This is the critical bridge between the VLA model
and the simulated robot.

Pipeline:
    VLA raw action (7D delta EE pose + gripper)
        → Denormalize (using dataset statistics)
        → Clip (safety bounds)
        → Apply delta to current EE pose
        → IK solve (EE pose → joint positions)
        → Smooth (interpolate if needed)
        → Joint position targets for Isaac Sim

Action formats:
    - OpenVLA-OFT (L1 regression): 7D continuous [dx, dy, dz, droll, dpitch, dyaw, gripper]
    - OpenVLA-OFT (tokenized): Discretized bins → same 7D after decoding
    - Gripper: Binary (open/close) based on threshold

For the Franka in Isaac Sim, we need to convert delta EE actions to
joint position targets using inverse kinematics.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.ik_solver import IKSolver, create_ik_solver
from src.utils import (
    FRANKA_GRIPPER_CLOSED,
    FRANKA_GRIPPER_OPEN,
    FRANKA_GRIPPER_THRESHOLD,
    FRANKA_HOME_JOINTS,
    FRANKA_JOINT_LIMITS_LOWER,
    FRANKA_JOINT_LIMITS_UPPER,
    clamp_to_workspace,
    clip_action_magnitude,
    delta_ee_to_pose,
)

logger = logging.getLogger(__name__)


# ─── Dataset Statistics ──────────────────────────────────────────────────────

# Default normalization statistics for LIBERO spatial (OpenVLA-OFT pretrained)
# These will be overridden by custom statistics when fine-tuned on kitchen data
DEFAULT_NORM_STATS = {
    "libero_spatial_no_noops": {
        "action": {
            "min": np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0]),
            "max": np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0]),
            "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            "std": np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3]),
        }
    }
}


class ActionPipeline:
    """Converts VLA action outputs to Isaac Sim joint position targets.

    This pipeline handles:
    1. Denormalization of VLA outputs using dataset statistics
    2. Safety clipping of action magnitudes
    3. Delta EE pose → absolute EE pose conversion
    4. IK solving (EE pose → joint positions)
    5. Joint limit clamping
    6. Gripper action interpretation
    7. Action chunk management (open-loop execution)

    Args:
        config: Action pipeline configuration dict
        ik_solver: IK solver instance (or None to create from config)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ik_solver: Optional[IKSolver] = None,
    ):
        self.config = config
        self.action_type = config.get("action_type", "delta_ee")
        self.workspace_bounds = config.get("workspace_bounds", {
            "x": [0.2, 0.8],
            "y": [-0.4, 0.4],
            "z": [0.0, 1.2],
        })

        # IK solver
        if ik_solver is not None:
            self.ik_solver = ik_solver
        else:
            ik_config = config.get("ik", {})
            # Extract the solver method and pass remaining config as kwargs
            solver_method = ik_config.get("solver", "lula")
            ik_kwargs = {k: v for k, v in ik_config.items() if k != "solver"}
            self.ik_solver = create_ik_solver(method=solver_method, **ik_kwargs)

        # Gripper settings
        self.gripper_threshold = config.get("gripper_threshold", FRANKA_GRIPPER_THRESHOLD)
        self.gripper_open = config.get("gripper_action_scale", FRANKA_GRIPPER_OPEN)
        self.gripper_closed = FRANKA_GRIPPER_CLOSED

        # Orientation delta threshold for deciding whether to use orientation IK.
        # Rotation deltas below this magnitude trigger position-only IK (much easier to solve).
        self.orientation_delta_threshold = config.get("orientation_delta_threshold", 0.05)

        # Safety settings
        self.max_ee_velocity = config.get("max_ee_velocity", 0.2)
        self.max_joint_velocity = config.get("max_joint_velocity", 1.0)

        # Action chunk state
        self._action_queue: List[np.ndarray] = []
        self._chunk_index = 0

        # State tracking
        self._last_joint_positions = FRANKA_HOME_JOINTS.copy()
        self._last_ee_position = np.array([0.5, 0.0, 0.8])  # Approximate home EE pos
        self._last_ee_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quat

        # Dataset statistics (for denormalization)
        self._norm_stats = DEFAULT_NORM_STATS
        self._unnorm_key = "libero_spatial_no_noops"

    def set_norm_stats(self, norm_stats: Dict, unnorm_key: str = None):
        """Set custom dataset normalization statistics."""
        self._norm_stats = norm_stats
        if unnorm_key is not None:
            self._unnorm_key = unnorm_key

    def process_action_chunk(
        self,
        actions: List[np.ndarray],
        current_ee_position: np.ndarray,
        current_ee_orientation: np.ndarray,
        current_joint_positions: np.ndarray,
    ) -> Tuple[List[Dict[str, np.ndarray]], bool]:
        """Process a complete action chunk from VLA output.

        Args:
            actions: List of 7D action arrays [dx, dy, dz, droll, dpitch, dyaw, gripper]
            current_ee_position: Current EE position [x, y, z]
            current_ee_orientation: Current EE orientation [w, x, y, z]
            current_joint_positions: Current joint positions (7,)

        Returns:
            Tuple of (list of action dicts, success flag)
            Each action dict contains:
                - "joint_positions": (7,) target joint positions
                - "gripper": float gripper target (0.0 = closed, 0.04 = open)
                - "ee_position": (3,) target EE position
                - "ee_orientation": (4,) target EE orientation
        """
        # Ensure inputs are numpy arrays — VLA server returns JSON lists
        # which need to be converted before numpy operations like .sum()
        current_ee_position = np.asarray(current_ee_position, dtype=np.float64)
        current_ee_orientation = np.asarray(current_ee_orientation, dtype=np.float64)
        current_joint_positions = np.asarray(current_joint_positions, dtype=np.float64)

        processed_actions = []
        ee_pos = current_ee_position.copy()
        ee_quat = current_ee_orientation.copy()
        joint_pos = current_joint_positions.copy()
        all_success = True

        for i, action in enumerate(actions):
            # Convert each action to numpy array (VLA server may return lists)
            action = np.asarray(action, dtype=np.float64)
            result = self.process_single_action(
                action=action,
                current_ee_position=ee_pos,
                current_ee_orientation=ee_quat,
                current_joint_positions=joint_pos,
            )

            if result is None:
                logger.warning(f"Action {i} in chunk failed IK, stopping chunk execution")
                all_success = False
                break

            processed_actions.append(result)
            # Update state for next action in chunk.
            # If IK failed but returned current position (graceful degradation),
            # we still update state to avoid drift.
            ee_pos = result["ee_position"]
            ee_quat = result["ee_orientation"]
            joint_pos = result["joint_positions"]

        # Store chunk for open-loop execution
        self._action_queue = processed_actions
        self._chunk_index = 0

        return processed_actions, all_success

    def process_single_action(
        self,
        action: np.ndarray,
        current_ee_position: np.ndarray,
        current_ee_orientation: np.ndarray,
        current_joint_positions: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Process a single VLA action into a sim action.

        Pipeline:
            1. Clip action magnitude for safety
            2. Convert delta EE to absolute EE pose
            3. Clamp to workspace bounds
            4. Solve IK for joint positions
            5. Clamp joint positions to limits
            6. Interpret gripper command

        Args:
            action: 7D action [dx, dy, dz, droll, dpitch, dyaw, gripper]
            current_ee_position: Current EE position [x, y, z]
            current_ee_orientation: Current EE orientation [w, x, y, z]
            current_joint_positions: Current joint positions (7,)

        Returns:
            Dict with joint_positions, gripper, ee_position, ee_orientation
            or None if IK fails
        """
        # Ensure all inputs are numpy arrays — VLA server returns JSON lists
        # which need to be converted before numpy operations like .sum()
        action = np.asarray(action, dtype=np.float64)
        current_ee_position = np.asarray(current_ee_position, dtype=np.float64)
        current_ee_orientation = np.asarray(current_ee_orientation, dtype=np.float64)
        current_joint_positions = np.asarray(current_joint_positions, dtype=np.float64)

        # 1. Clip action magnitude
        safe_action = clip_action_magnitude(action)

        # 2. Convert delta EE to absolute EE pose
        new_position, new_quat = delta_ee_to_pose(
            current_ee_position, current_ee_orientation, safe_action
        )

        # 3. Clamp to workspace bounds
        new_position = clamp_to_workspace(new_position, self.workspace_bounds)

        # 4. Solve IK
        # Only pass orientation when rotation deltas are significant.
        # A higher threshold avoids triggering hard 6-DOF IK solves for
        # small/negligible rotation deltas, which often cause LULA IK failures.
        rotation_delta_mag = np.linalg.norm(safe_action[3:6])
        target_orientation = new_quat if rotation_delta_mag > self.orientation_delta_threshold else None
        joint_positions, ik_success = self.ik_solver.solve(
            target_position=new_position,
            target_orientation=target_orientation,
            current_joint_positions=current_joint_positions,
        )

        # 6. Interpret gripper command (compute before IK check so it's
        # available in the IK failure fallback path)
        gripper_action = safe_action[6]
        gripper_target = (
            self.gripper_open if gripper_action > self.gripper_threshold
            else self.gripper_closed
        )

        if not ik_success:
            logger.warning("IK solve failed, returning current position with target gripper")
            # Instead of returning None (which stops the entire chunk),
            # return the current joint positions with the target gripper.
            # This allows the robot to at least maintain its position and
            # continue executing subsequent actions in the chunk.
            return {
                "joint_positions": current_joint_positions,
                "gripper": gripper_target,
                "ee_position": current_ee_position,
                "ee_orientation": current_ee_orientation,
            }

        # 5. Clamp joint positions to limits
        joint_positions = np.clip(
            joint_positions,
            FRANKA_JOINT_LIMITS_LOWER + 0.01,
            FRANKA_JOINT_LIMITS_UPPER - 0.01,
        )

        return {
            "joint_positions": joint_positions,
            "gripper": gripper_target,
            "ee_position": new_position,
            "ee_orientation": new_quat,
        }

    def get_next_action(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the next action from the current action chunk queue.

        Returns:
            Next action dict, or None if chunk is exhausted
        """
        if self._chunk_index >= len(self._action_queue):
            return None

        action = self._action_queue[self._chunk_index]
        self._chunk_index += 1
        return action

    @property
    def chunk_remaining(self) -> int:
        """Number of actions remaining in current chunk."""
        return max(0, len(self._action_queue) - self._chunk_index)

    @property
    def chunk_exhausted(self) -> bool:
        """Whether the current action chunk has been fully executed."""
        return self._chunk_index >= len(self._action_queue)


class GripperController:
    """Simple gripper controller for Franka.

    Manages smooth gripper open/close transitions.
    """

    def __init__(
        self,
        open_position: float = FRANKA_GRIPPER_OPEN,
        closed_position: float = FRANKA_GRIPPER_CLOSED,
        max_width: float = 0.08,
        speed: float = 0.1,  # m/s
    ):
        self.open_position = open_position
        self.closed_position = closed_position
        self.max_width = max_width
        self.speed = speed
        self._current = open_position
        self._target = open_position

    def set_target(self, target: float):
        """Set gripper target position."""
        self._target = np.clip(target, self.closed_position, self.open_position)

    def step(self, dt: float) -> float:
        """Step the gripper towards target.

        Args:
            dt: Time step in seconds

        Returns:
            Current gripper position after step
        """
        diff = self._target - self._current
        max_step = self.speed * dt

        if abs(diff) < max_step:
            self._current = self._target
        else:
            self._current += np.sign(diff) * max_step

        return self._current

    @property
    def is_open(self) -> bool:
        return self._current > (self.open_position - self.closed_position) / 2

    @property
    def current_position(self) -> float:
        return self._current