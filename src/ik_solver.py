"""
IK Solver for Franka Emika Panda
==================================

Provides inverse kinematics solvers for converting end-effector pose targets
to joint position commands. Supports multiple IK methods:

1. LULA (NVIDIA's Lumped Articulation IK) — recommended for Isaac Sim
2. Damped Least Squares (Jacobian-based) — pure numpy fallback
3. Differential IK — Isaac Sim's built-in controller

The primary method for Isaac Sim integration is the LULA-based solver, which
uses NVIDIA's kinematics library for accurate and stable IK solutions.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.utils import (
    FRANKA_EE_LINK,
    FRANKA_HOME_JOINTS,
    FRANKA_JOINT_LIMITS_LOWER,
    FRANKA_JOINT_LIMITS_UPPER,
    FRANKA_JOINT_NAMES,
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
)

logger = logging.getLogger(__name__)


class IKSolver:
    """Base IK solver interface."""

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        current_joint_positions: np.ndarray,
        num_iterations: int = 100,
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK for target EE pose.

        Args:
            target_position: [x, y, z] target EE position
            target_orientation: [w, x, y, z] target EE quaternion (or None)
            current_joint_positions: Current 7-DOF joint positions
            num_iterations: Max IK iterations

        Returns:
            Tuple of (joint_positions, success)
        """
        raise NotImplementedError


class DampedLeastSquaresSolver(IKSolver):
    """Damped Least Squares (Levenberg-Marquardt) IK solver.

    Uses the Jacobian pseudo-inverse with damping for numerical stability.
    This is a pure-numpy implementation that works without Isaac Sim.
    """

    def __init__(
        self,
        damping: float = 0.01,
        position_tolerance: float = 0.001,
        orientation_tolerance: float = 0.01,
        max_iterations: int = 100,
        joint_limits_margin: float = 0.01,
    ):
        self.damping = damping
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_iterations = max_iterations
        self.joint_limits_margin = joint_limits_margin

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        current_joint_positions: np.ndarray,
        num_iterations: int = 100,
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK using damped least squares.

        NOTE: This requires a Jacobian function from Isaac Sim. When running
        inside Isaac Sim, use the LULA solver instead. This solver is provided
        as a reference/fallback for testing outside of simulation.
        """
        # This is a placeholder — actual Jacobian computation requires
        # Isaac Sim's articulation API. In practice, use LULA or
        # Isaac Sim's DifferentialIKController.
        logger.warning(
            "DampedLeastSquaresSolver requires Jacobian from Isaac Sim. "
            "Use LULA solver or DifferentialIKController for real IK."
        )
        return current_joint_positions, False


def _find_franka_lula_config_paths():
    """Auto-discover the Franka LULA config files from Isaac Sim's extension path.

    Searches for the robot_description_path (robot_descriptor.yaml) and
    urdf_path (lula_franka_gen.urdf) that LulaKinematicsSolver requires.

    Returns:
        Tuple of (robot_description_path, urdf_path) or (None, None) if not found.
    """
    # Strategy 1: Use Isaac Sim's interface_config_loader if available
    # (requires running inside Isaac Sim with extension manager)
    try:
        from isaacsim.robot_motion.motion_generation import (
            interface_config_loader,
        )
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        if kinematics_config is not None:
            robot_description_path = kinematics_config.get("robot_description_path")
            urdf_path = kinematics_config.get("urdf_path")
            if robot_description_path and urdf_path:
                if os.path.exists(robot_description_path) and os.path.exists(urdf_path):
                    logger.info(
                        f"Found Franka LULA config via interface_config_loader: "
                        f"robot_description={robot_description_path}, urdf={urdf_path}"
                    )
                    return robot_description_path, urdf_path
    except Exception:
        pass

    # Strategy 2: Search for the extension path using get_extension_path_from_name
    try:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        ext_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        if ext_path:
            robot_description_path = os.path.join(
                ext_path, "motion_policy_configs", "franka", "rmpflow", "robot_descriptor.yaml"
            )
            urdf_path = os.path.join(
                ext_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf"
            )
            if os.path.exists(robot_description_path) and os.path.exists(urdf_path):
                logger.info(
                    f"Found Franka LULA config via extension path: "
                    f"robot_description={robot_description_path}, urdf={urdf_path}"
                )
                return robot_description_path, urdf_path
    except Exception:
        pass

    # Strategy 3: Search common Isaac Sim installation paths
    search_paths = []

    # From ISAACSIM_HOME environment variable
    isaacsim_home = os.environ.get("ISAACSIM_HOME")
    if isaacsim_home:
        search_paths.append(isaacsim_home)

    # Common installation locations
    home = os.path.expanduser("~")
    search_paths.extend([
        os.path.join(home, "isaacsim"),
        os.path.join(home, ".isaacsim"),
        "/isaac-sim",
        "/opt/nvidia/isaac-sim",
    ])

    # Also check the parent directory of the isaacsim package
    try:
        import isaacsim
        isaacsim_pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(isaacsim.__file__)))
        search_paths.append(isaacsim_pkg_path)
    except Exception:
        pass

    for base_path in search_paths:
        ext_path = os.path.join(base_path, "exts", "isaacsim.robot_motion.motion_generation")
        if os.path.isdir(ext_path):
            robot_description_path = os.path.join(
                ext_path, "motion_policy_configs", "franka", "rmpflow", "robot_descriptor.yaml"
            )
            urdf_path = os.path.join(
                ext_path, "motion_policy_configs", "franka", "lula_franka_gen.urdf"
            )
            if os.path.exists(robot_description_path) and os.path.exists(urdf_path):
                logger.info(
                    f"Found Franka LULA config via filesystem search: "
                    f"robot_description={robot_description_path}, urdf={urdf_path}"
                )
                return robot_description_path, urdf_path

    logger.error("Could not find Franka LULA config files (robot_descriptor.yaml, lula_franka_gen.urdf)")
    return None, None


class LULA_IKSolver(IKSolver):
    """NVIDIA LULA-based IK solver for Franka.

    Uses Lumped Articulation (LULA) for fast and stable IK solutions.
    This is the recommended solver for Isaac Sim integration.

    LulaKinematicsSolver requires two config files:
        - robot_description_path: YAML file describing the robot's cspace
          (e.g., robot_descriptor.yaml)
        - urdf_path: URDF file describing the robot kinematics
          (e.g., lula_franka_gen.urdf)

    These are provided by Isaac Sim's isaacsim.robot_motion.motion_generation
    extension under motion_policy_configs/franka/.

    If robot_description_path and urdf_path are not provided, the solver
    will auto-discover them from the Isaac Sim installation.

    Requires:
        - Isaac Sim installed with LULA support
        - Franka URDF and robot description YAML files
    """

    def __init__(
        self,
        robot_description_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        robot_urdf_path: Optional[str] = None,
        robot_srdf_path: Optional[str] = None,
        robot_urdf: Optional[str] = None,
        end_effector_frame: str = "right_gripper",
        joint_limits_margin: float = 0.01,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.1,
        ccd_max_iterations: int = 30,
        bfgs_max_iterations: int = 50,
        max_num_descents: int = 10,
        fallback_to_position_only: bool = True,
        **kwargs,
    ):
        # Accept and ignore extra kwargs (e.g., damping, max_iterations)
        # that are passed from config but not used by LULA
        self.robot_urdf_path = robot_urdf_path
        self.robot_srdf_path = robot_srdf_path
        self.robot_urdf = robot_urdf
        self.end_effector_frame = end_effector_frame
        self.joint_limits_margin = joint_limits_margin
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.ccd_max_iterations = ccd_max_iterations
        self.bfgs_max_iterations = bfgs_max_iterations
        self.max_num_descents = max_num_descents
        self.fallback_to_position_only = fallback_to_position_only

        # LULA config paths — these are the paths that LulaKinematicsSolver needs
        # robot_description_path = YAML robot descriptor (cspace definition)
        # urdf_path = URDF file for the robot
        self.robot_description_path = robot_description_path
        self.urdf_path = urdf_path

        self._lula_kinematics = None
        self._initialized = False

    def _initialize(self):
        """Initialize LULA kinematics solver.

        LulaKinematicsSolver.__init__() requires two positional arguments:
            - robot_description_path: path to the robot description YAML file
            - urdf_path: path to the URDF file

        These define the robot's configuration space and kinematic structure
        that LULA uses for IK solving.
        """
        if self._initialized:
            return

        try:
            # Import LulaKinematicsSolver
            # Isaac Sim 4.5+ uses isaacsim.robot_motion.motion_generation
            # Older versions use omni.isaac.motion_generation
            try:
                from isaacsim.robot_motion.motion_generation.lula.kinematics import (
                    LulaKinematicsSolver as IsaacLulaKinematicsSolver,
                )
                logger.info("Using Isaac Sim 4.5+ LULA IK solver (isaacsim namespace)")
            except ImportError:
                from omni.isaac.motion_generation.lula.kinematics import (
                    LulaKinematicsSolver as IsaacLulaKinematicsSolver,
                )
                logger.info("Using legacy LULA IK solver (omni namespace)")

            # Resolve config paths: use provided paths or auto-discover
            robot_description_path = self.robot_description_path
            urdf_path = self.urdf_path

            if not robot_description_path or not urdf_path:
                # Auto-discover Franka LULA config files from Isaac Sim
                auto_desc, auto_urdf = _find_franka_lula_config_paths()
                if auto_desc:
                    robot_description_path = robot_description_path or auto_desc
                if auto_urdf:
                    urdf_path = urdf_path or auto_urdf

            if not robot_description_path or not urdf_path:
                raise FileNotFoundError(
                    "LulaKinematicsSolver requires robot_description_path and urdf_path. "
                    "Could not auto-discover them from the Isaac Sim installation. "
                    "Please provide them explicitly in the config under action_pipeline.ik, e.g.:\n"
                    "  ik:\n"
                    "    solver: lula\n"
                    "    robot_description_path: /path/to/robot_descriptor.yaml\n"
                    "    urdf_path: /path/to/lula_franka_gen.urdf\n"
                    "The files are typically found in:\n"
                    "  <isaacsim>/exts/isaacsim.robot_motion.motion_generation/"
                    "motion_policy_configs/franka/rmpflow/robot_descriptor.yaml\n"
                    "  <isaacsim>/exts/isaacsim.robot_motion.motion_generation/"
                    "motion_policy_configs/franka/lula_franka_gen.urdf"
                )

            # LulaKinematicsSolver.__init__(robot_description_path, urdf_path)
            # These are REQUIRED positional arguments.
            self._lula_kinematics = IsaacLulaKinematicsSolver(
                robot_description_path=robot_description_path,
                urdf_path=urdf_path,
            )

            # Configure LULA IK solver parameters for better convergence
            self._lula_kinematics.set_default_position_tolerance(self.position_tolerance)
            self._lula_kinematics.set_default_orientation_tolerance(self.orientation_tolerance)
            self._lula_kinematics.ccd_max_iterations = self.ccd_max_iterations
            self._lula_kinematics.bfgs_max_iterations = self.bfgs_max_iterations
            self._lula_kinematics.max_num_descents = self.max_num_descents

            # Add cspace seeds for better warm starting — use LULA default_q
            # and Franka home position as seeds
            try:
                lula_default = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75])
                franka_home = FRANKA_HOME_JOINTS.copy()
                self._lula_kinematics.set_default_cspace_seeds(
                    np.array([lula_default, franka_home])
                )
            except Exception as e:
                logger.debug(f"Could not set cspace seeds (non-fatal): {e}")

            self._initialized = True
            logger.info(
                f"LULA IK solver initialized successfully "
                f"(robot_description={robot_description_path}, urdf={urdf_path}, "
                f"end_effector={self.end_effector_frame}, "
                f"pos_tol={self.position_tolerance}, ori_tol={self.orientation_tolerance}, "
                f"ccd_iter={self.ccd_max_iterations}, bfgs_iter={self.bfgs_max_iterations})"
            )
        except ImportError:
            logger.error(
                "LULA requires Isaac Sim. Install Isaac Sim or use "
                "DifferentialIKController instead. "
                "Tried both isaacsim.robot_motion.motion_generation and "
                "omni.isaac.motion_generation."
            )
            raise

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        current_joint_positions: np.ndarray,
        num_iterations: int = 100,
    ) -> Tuple[np.ndarray, bool]:
        """Solve IK using LULA kinematics.

        Implements a two-stage approach:
        1. First attempt with full 6-DOF IK (position + orientation)
        2. If that fails, fall back to position-only IK (no orientation constraint)

        Position-only IK is much easier for the CCD solver because it
        disables the orientation constraint entirely (sets orientation_weight=0
        and orientation_tolerance=2.0), allowing the solver to focus only on
        reaching the target position.

        Args:
            target_position: [x, y, z] target EE position (in meters)
            target_orientation: [w, x, y, z] target EE quaternion (or None)
            current_joint_positions: Current 7-DOF joint positions
            num_iterations: Max IK iterations (not used by LULA CCD solver)

        Returns:
            Tuple of (joint_positions, success)
        """
        self._initialize()

        if self._lula_kinematics is None:
            logger.error("LULA kinematics not initialized")
            return current_joint_positions, False

        # Ensure inputs are numpy arrays
        target_position = np.asarray(target_position, dtype=np.float64)
        if target_orientation is not None:
            target_orientation = np.asarray(target_orientation, dtype=np.float64)
        current_joint_positions = np.asarray(current_joint_positions, dtype=np.float64)

        # LulaKinematicsSolver.compute_inverse_kinematics() signature:
        #   compute_inverse_kinematics(
        #       frame_name: str,
        #       target_position: np.array,
        #       target_orientation: np.array = None,
        #       warm_start: np.array = None,
        #       position_tolerance: float = None,
        #       orientation_tolerance: float = None,
        #   ) -> Tuple[np.array, bool]

        # Stage 1: Try full IK (position + orientation) if orientation is provided
        if target_orientation is not None:
            try:
                ik_result, success = self._lula_kinematics.compute_inverse_kinematics(
                    frame_name=self.end_effector_frame,
                    target_position=target_position,
                    target_orientation=target_orientation,
                    warm_start=current_joint_positions,
                    position_tolerance=self.position_tolerance,
                    orientation_tolerance=self.orientation_tolerance,
                )
                if success and ik_result is not None:
                    joint_positions = np.array(ik_result).flatten()
                    joint_positions = np.clip(
                        joint_positions,
                        FRANKA_JOINT_LIMITS_LOWER + self.joint_limits_margin,
                        FRANKA_JOINT_LIMITS_UPPER - self.joint_limits_margin,
                    )
                    return joint_positions, True
            except TypeError as e:
                # Fallback: try positional args in case of API differences
                logger.warning(f"compute_inverse_kinematics TypeError: {e}, trying positional args")
                try:
                    ik_result, success = self._lula_kinematics.compute_inverse_kinematics(
                        self.end_effector_frame,
                        target_position,
                        target_orientation,
                        current_joint_positions,
                        self.position_tolerance,
                        self.orientation_tolerance,
                    )
                    if success and ik_result is not None:
                        joint_positions = np.array(ik_result).flatten()
                        joint_positions = np.clip(
                            joint_positions,
                            FRANKA_JOINT_LIMITS_LOWER + self.joint_limits_margin,
                            FRANKA_JOINT_LIMITS_UPPER - self.joint_limits_margin,
                        )
                        return joint_positions, True
                except Exception:
                    pass

            # Stage 2: Full IK failed — fall back to position-only IK
            if self.fallback_to_position_only:
                logger.debug("Full IK failed, falling back to position-only IK")
                target_orientation = None  # This tells LULA to disable orientation constraint

        # Position-only IK (no orientation constraint)
        try:
            ik_result, success = self._lula_kinematics.compute_inverse_kinematics(
                frame_name=self.end_effector_frame,
                target_position=target_position,
                target_orientation=None,  # None = position-only, disables orientation constraint
                warm_start=current_joint_positions,
                position_tolerance=self.position_tolerance,
            )
        except TypeError as e:
            logger.warning(f"Position-only IK TypeError: {e}, trying positional args")
            try:
                ik_result, success = self._lula_kinematics.compute_inverse_kinematics(
                    self.end_effector_frame,
                    target_position,
                    None,
                    current_joint_positions,
                    self.position_tolerance,
                )
            except Exception as e2:
                logger.error(f"LULA position-only IK failed: {e2}")
                return current_joint_positions, False

        if success and ik_result is not None:
            joint_positions = np.array(ik_result).flatten()
            # Clamp to joint limits
            joint_positions = np.clip(
                joint_positions,
                FRANKA_JOINT_LIMITS_LOWER + self.joint_limits_margin,
                FRANKA_JOINT_LIMITS_UPPER - self.joint_limits_margin,
            )
            return joint_positions, True
        else:
            logger.warning("LULA IK failed to find solution (position-only)")
            return current_joint_positions, False


class DifferentialIKControllerWrapper:
    """Wrapper around Isaac Sim's DifferentialIKController.

    This uses Isaac Sim's built-in differential IK controller,
    which is the most reliable method for controlling the Franka
    within the simulation environment.

    Usage:
        # Inside Isaac Sim simulation loop
        ik_controller = DifferentialIKControllerWrapper(robot, config)
        ik_controller.set_target(position, orientation)
        action = ik_controller.compute(current_observations)
    """

    def __init__(
        self,
        robot_prim_path: str = "/World/Franka",
        end_effector_prim_path: str = "/World/Franka/panda_hand",
        damping: float = 0.01,
    ):
        self.robot_prim_path = robot_prim_path
        self.end_effector_prim_path = end_effector_prim_path
        self.damping = damping
        self._controller = None
        self._initialized = False

    def initialize(self, world):
        """Initialize the differential IK controller within Isaac Sim.

        Args:
            world: Isaac Sim World object
        """
        try:
            from isaacsim.core.utils.types import ArticulationAction
            from isaacsim.robot.manipulators.examples.franka import Franka

            # The Franka class provides built-in IK control
            # We'll use it through the ArticulationController
            self._world = world
            self._initialized = True
            logger.info("DifferentialIK controller wrapper initialized")

        except ImportError as e:
            logger.error(f"Isaac Sim not available: {e}")
            raise

    def compute(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        current_joint_positions: np.ndarray,
    ) -> np.ndarray:
        """Compute joint positions for target EE pose using differential IK.

        This is meant to be called within the Isaac Sim simulation loop.
        The actual IK computation is done by Isaac Sim's articulation controller.
        """
        # In practice, this delegates to Isaac Sim's ArticulationController
        # See sim_bridge.py for the actual implementation
        raise NotImplementedError(
            "Use sim_bridge.py's _compute_ik method instead, which "
            "integrates with Isaac Sim's simulation loop."
        )


def create_ik_solver(
    method: str = "lula",
    **kwargs,
) -> IKSolver:
    """Factory function to create an IK solver by name.

    Args:
        method: One of "lula", "damped_least_squares"
        **kwargs: Additional arguments for the solver

    Returns:
        IKSolver instance
    """
    solvers = {
        "lula": LULA_IKSolver,
        "damped_least_squares": DampedLeastSquaresSolver,
    }

    if method not in solvers:
        raise ValueError(f"Unknown IK solver: {method}. Available: {list(solvers.keys())}")

    return solvers[method](**kwargs)