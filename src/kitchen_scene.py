"""
Kitchen Scene Builder for Isaac Sim
=====================================

Constructs a kitchen manipulation environment in Isaac Sim with:
- Franka Emika Panda robot arm
- Kitchen counter/table
- Manipulation objects (blocks, plates, mugs)
- Third-person and wrist-mounted cameras
- Ground plane and lighting

This module provides both a programmatic scene builder (no USD files needed)
and support for loading pre-built USD kitchen scenes (e.g., Lightwheel Kitchen).

Usage:
    # Programmatic scene (default)
    scene = KitchenScene(world, config)
    scene.build()

    # Load from USD file
    scene = KitchenScene(world, config, usd_path="/path/to/kitchen.usd")
    scene.build()
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KitchenScene:
    """Builds and manages a kitchen manipulation scene in Isaac Sim.

    Args:
        world: Isaac Sim World object
        config: Scene configuration dict (from default.yaml)
        usd_path: Optional path to a USD scene file
    """

    def __init__(
        self,
        world,
        config: Dict[str, Any],
        usd_path: Optional[str] = None,
    ):
        self.world = world
        self.config = config
        self.usd_path = usd_path

        # Scene objects (populated after build)
        self.robot = None
        self.objects: Dict[str, Any] = {}
        self.cameras: Dict[str, Any] = {}
        self.gripper = None
        self.end_effector = None

        # Object positions for reset
        self._initial_positions: Dict[str, np.ndarray] = {}

        # Wrist camera config (stored for deferred creation after world.reset())
        self._wrist_camera_config: Dict[str, Any] = {}

    def build(self):
        """Build the complete kitchen scene.

        This must be called after world initialization and before
        the simulation loop starts.
        """
        logger.info("Building kitchen scene...")

        if self.usd_path:
            self._load_usd_scene()
        else:
            self._build_programmatic_scene()

        logger.info("Kitchen scene built successfully")

    def _load_usd_scene(self):
        """Load a pre-built USD scene file (e.g., Lightwheel Kitchen)."""
        from isaacsim.core.utils.stage import add_reference_to_stage

        logger.info(f"Loading USD scene: {self.usd_path}")
        add_reference_to_stage(usd_path=self.usd_path, prim_path="/World/Kitchen")

        # Still need to add Franka and cameras
        self._add_robot()
        self._add_cameras()

    def _build_programmatic_scene(self):
        """Build the scene programmatically from config."""
        scene_config = self.config.get("kitchen_scene", {})

        # 1. Ground plane
        self._add_ground_plane()

        # 2. Kitchen counter/table
        self._add_kitchen_table(scene_config)

        # 3. Franka robot
        self._add_robot()

        # 4. Manipulation objects
        self._add_objects(scene_config)

        # 5. Cameras
        self._add_cameras()

        # 6. Lighting
        self._add_lighting()

    def _add_ground_plane(self):
        """Add ground plane to the scene."""
        from isaacsim.core.api.objects import GroundPlane

        self.world.scene.add(GroundPlane(
            prim_path="/World/defaultGroundPlane",
            name="ground_plane",
        ))
        logger.debug("Added ground plane")

    def _add_kitchen_table(self, scene_config: Dict):
        """Add a kitchen counter/table."""
        from isaacsim.core.api.objects import FixedCuboid

        table_height = scene_config.get("table_height", 0.8)
        table_width = scene_config.get("table_width", 3.0)
        table_depth = scene_config.get("table_depth", 3.0)

        # Table center — middle of the 3m×3m table spanning x=[0, 3], y=[-1.5, 1.5]
        table_center_x = table_depth / 2.0

        # Main table surface
        self.world.scene.add(FixedCuboid(
            prim_path="/World/Kitchen/Table",
            name="kitchen_table",
            position=np.array([table_center_x, 0.0, table_height / 2]),
            scale=np.array([table_depth, table_width, table_height]),
            color=np.array([0.6, 0.4, 0.2]),  # Wood color
        ))

        # Table legs (4 corners)
        leg_offset_x = table_depth / 2 - 0.05
        leg_offset_y = table_width / 2 - 0.05
        leg_height = table_height

        for i, (dx, dy) in enumerate([
            (leg_offset_x, leg_offset_y),
            (leg_offset_x, -leg_offset_y),
            (-leg_offset_x, leg_offset_y),
            (-leg_offset_x, -leg_offset_y),
        ]):
            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Kitchen/TableLeg_{i}",
                name=f"table_leg_{i}",
                position=np.array([table_center_x + dx, dy, leg_height / 2]),
                scale=np.array([0.05, 0.05, leg_height]),
                color=np.array([0.5, 0.3, 0.1]),
            ))

        logger.debug(f"Added kitchen table (height={table_height}m)")

    def _add_robot(self):
        """Add Franka Emika Panda robot to the scene.

        The Franka robot's base (panda_link0) is placed so that its base
        is at the same height as the table surface. The Franka URDF defines
        panda_joint1 at z=0.333 relative to panda_link0, meaning the
        shoulder joint is 0.333m above the base. By placing the robot base
        at z=table_height, the robot sits on top of the table with its
        base flush with the table surface — matching how a real Franka is
        typically mounted on a table or pedestal.
        """
        from isaacsim.robot.manipulators.examples.franka import Franka

        robot_config = self.config.get("robot", {})
        prim_path = robot_config.get("prim_path", "/World/Franka")

        # Place the Franka base at table height so the robot sits on the
        # table surface. The Franka's panda_joint1 (shoulder) is at
        # z=0.333 relative to panda_link0 (the base), so the arm
        # naturally extends upward from the table surface.
        scene_config = self.config.get("kitchen_scene", {})
        table_height = scene_config.get("table_height", 0.8)
        table_depth = scene_config.get("table_depth", 3.0)

        # Use explicit position from config if provided, otherwise
        # auto-set position to align with the table center and table_height
        # so the base is flush with the table surface.
        robot_position_cfg = robot_config.get("position", None)
        if robot_position_cfg is not None:
            robot_position = np.array(robot_position_cfg)
        else:
            # Table is centered at x=table_depth/2 (see _add_kitchen_table).
            # Place the robot base at the same x so it sits on the table.
            table_center_x = table_depth / 2.0
            robot_position = np.array([table_center_x, 0.0, table_height])

        self.robot = self.world.scene.add(Franka(
            prim_path=prim_path,
            name="franka_robot",
            position=robot_position,
        ))

        # Store for re-application after world.reset()
        self._robot_prim_path = prim_path
        self._robot_position = robot_position

        # Store references to gripper and end effector
        # These are available after world.reset()
        self.gripper = self.robot.gripper
        self.end_effector = self.robot.end_effector

        logger.debug(f"Added Franka robot at {prim_path}, "
                     f"position={robot_position.tolist()} (base at table height={table_height}m)")

    def enforce_robot_position(self):
        """Re-apply the robot base position in the physics simulation.

        This MUST be called AFTER world.reset() and AFTER settle steps,
        because:

        1. ``set_default_state()`` + ``post_reset()`` only update the
           XFormPrimView (USD prim transform / visual), NOT the physics
           Articulation root position.
        2. ``set_world_pose()`` also only updates the XFormPrimView
           (USD prim transform / visual), NOT the physics state.
        3. After physics sim steps, the physics state (which may still
           be at [0,0,0]) takes over both visually and physically.

        Strategy:
        1. Use set_default_state() for future post_reset() visual resets.
        2. Use set_world_pose() for USD prim visual consistency.
        3. **ALWAYS** write to the physics buffer directly to set the
           Articulation root position — this is the only way to update
           the actual physics state.
        """
        if not hasattr(self, "_robot_position"):
            return

        if self.robot is None:
            logger.warning("Robot not initialized — cannot enforce position.")
            return

        position = self._robot_position
        orientation = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (w, x, y, z)

        # 1. Set default state for future post_reset() visual resets.
        try:
            self.robot.set_default_state(
                position=position,
                orientation=orientation,
            )
            logger.debug(f"Set Franka default state to position={position.tolist()}")
        except Exception as e:
            logger.warning(f"Failed to set Franka default state: {e}")

        # 2. Set USD prim transform for visual consistency.
        try:
            self.robot.set_world_pose(
                position=position,
                orientation=orientation,
            )
            logger.debug("Set Franka USD prim transform via set_world_pose")
        except Exception as e:
            logger.warning(f"set_world_pose failed (non-fatal): {e}")

        # 3. Write to the physics buffer directly to update the Articulation
        # root position. In Isaac Sim 4.5+, ArticulationView has
        # write_root_pose_to_sim(). In Isaac Sim 5.1, the internal
        # _articulation_view may be an Articulation object without this method,
        # so we fall back to set_world_pose() which handles both USD and physics.
        try:
            articulation_view = self.robot._articulation_view

            # Check if write_root_pose_to_sim exists (ArticulationView in 4.5+)
            if hasattr(articulation_view, "write_root_pose_to_sim"):
                root_pos = np.array(position, dtype=np.float64).reshape(1, 3)
                root_quat = np.array(orientation, dtype=np.float64).reshape(1, 4)  # wxyz
                articulation_view.write_root_pose_to_sim(root_pos, root_quat)

                root_lin_vel = np.zeros((1, 3), dtype=np.float64)
                root_ang_vel = np.zeros((1, 3), dtype=np.float64)
                articulation_view.write_root_velocity_to_sim(root_lin_vel, root_ang_vel)

                logger.info(
                    f"Franka base position enforced to "
                    f"{position.tolist()} via physics buffer write"
                )
            else:
                # Isaac Sim 5.1: Articulation object doesn't have
                # write_root_pose_to_sim. set_world_pose() handles both
                # USD prim and physics state.
                logger.debug(
                    "ArticulationView.write_root_pose_to_sim not available "
                    "(Isaac Sim 5.1+) — relying on set_world_pose()"
                )
        except Exception as e:
            logger.warning(f"Failed to write Franka physics root position: {e}")

    def _add_objects(self, scene_config: Dict):
        """Add manipulation objects to the kitchen table."""
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder

        objects_config = scene_config.get("objects", [])
        table_height = scene_config.get("table_height", 0.8)

        for obj_cfg in objects_config:
            name = obj_cfg["name"]
            obj_type = obj_cfg.get("type", "cube")
            position = np.array(obj_cfg["position"])
            color = np.array(obj_cfg.get("color", [0.5, 0.5, 0.5]))
            prim_path = f"/World/Objects/{name}"

            if obj_type == "cube":
                size = obj_cfg.get("size", 0.04)
                self.objects[name] = self.world.scene.add(DynamicCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    scale=np.array([size, size, size]),
                    color=color,
                ))
            elif obj_type == "cylinder":
                radius = obj_cfg.get("radius", 0.03)
                height = obj_cfg.get("height", 0.08)
                self.objects[name] = self.world.scene.add(DynamicCylinder(
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    radius=radius,
                    height=height,
                    color=color,
                ))
            elif obj_type == "sphere":
                radius = obj_cfg.get("radius", 0.02)
                from isaacsim.core.api.objects import DynamicSphere
                self.objects[name] = self.world.scene.add(DynamicSphere(
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    radius=radius,
                    color=color,
                ))

            # Store initial position for reset
            self._initial_positions[name] = position.copy()

            logger.debug(f"Added object '{name}' at {position}")

    @staticmethod
    def _look_at_to_orientation(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Convert camera position + look-at target to quaternion orientation.

        Computes the orientation quaternion (scalar-first: w, x, y, z) for an
        Isaac Sim Camera so that it looks from ``camera_position`` toward
        ``target``.  Uses the Isaac Sim world-camera convention where
        +X is forward, +Z is up.

        Args:
            camera_position: [x, y, z] camera location in world frame
            target: [x, y, z] point the camera should look at

        Returns:
            np.ndarray of shape (4,) — quaternion [w, x, y, z]
        """
        forward = np.array(target) - np.array(camera_position)
        forward = forward / np.linalg.norm(forward)

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # forward nearly aligned with Z — pick alternative up
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Isaac Sim Camera world convention: X=forward, Y=-right, Z=up
        R = np.column_stack([forward, -right, up])

        # Rotation matrix → quaternion (scalar-first)
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

    def _add_cameras(self):
        """Add third-person and wrist cameras to the scene.

        The third-person camera provides the primary observation image.
        The wrist camera provides the secondary observation image.
        Both are required by OpenVLA-OFT (num_images_in_input=2).

        Camera positions and orientations match LIBERO's MuJoCo setup:
        - Third-person: LIBERO agentview camera (pos + quat from bddl_base_domain.py)
        - Wrist: Panda eye_in_hand camera (pos + quat + fovy from robot.xml)

        NOTE: Isaac Sim 4.5+ Camera.__init__() does NOT accept a ``target``
        keyword.  We use either an explicit ``orientation_quat`` from config
        or convert a ``target`` (look-at point) into an orientation quaternion.
        """
        from isaacsim.sensors.camera import Camera

        cameras_config = self.config.get("cameras", {})

        # Third-person camera (LIBERO agentview)
        tp_config = cameras_config.get("third_person", {})
        tp_resolution = tuple(tp_config.get("resolution", [256, 256]))
        tp_position = np.array(tp_config.get("position", [1.0, 0.0, 1.5]))
        tp_fovy = tp_config.get("fovy", 75)

        # Use explicit quaternion if provided, otherwise compute from look-at target
        if "orientation_quat" in tp_config:
            tp_orientation = np.array(tp_config["orientation_quat"])  # [w, x, y, z]
        else:
            tp_target = np.array(tp_config.get("target", [0.5, 0.0, 0.8]))
            tp_orientation = self._look_at_to_orientation(tp_position, tp_target)

        self.cameras["third_person"] = Camera(
            prim_path=tp_config.get("prim_path", "/World/Franka/camera_third"),
            name="third_person_camera",
            resolution=tp_resolution,
            position=tp_position,
            orientation=tp_orientation,
        )
        self.world.scene.add(self.cameras["third_person"])

        # Set fovy on the USD prim after creation (Camera.__init__ doesn't accept fovy)
        if tp_fovy != 75:  # 75 is the USD default, skip if unchanged
            self._set_camera_fovy(self.cameras["third_person"], tp_fovy)

        # Wrist camera is deferred until AFTER world.reset() because:
        # 1. Its prim_path is under panda_hand which doesn't have correct
        #    world-space transforms until after reset
        # 2. Camera() constructor interprets position as WORLD coordinates,
        #    but we need LOCAL coordinates relative to the moving hand
        # We store the config and create it in initialize_cameras() instead.
        self._wrist_camera_config = cameras_config.get("wrist", {})

        logger.debug("Added third-person camera (wrist deferred until after world.reset)")

    @staticmethod
    def _set_camera_fovy(camera, fovy: float):
        """Set the vertical field of view on a Camera's USD prim.

        Args:
            camera: Isaac Sim Camera instance
            fovy: Vertical field of view in degrees
        """
        import math
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim_path = camera.prim_path
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            logger.warning(f"Could not find prim at {prim_path} to set fovy")
            return

        try:
            from pxr import UsdGeom

            geom_camera = UsdGeom.Camera(prim)
            # Get current focal length (USD stores in tenths of stage units)
            focal_length = geom_camera.GetFocalLength() / 10.0  # Convert to stage units
            # Get vertical aperture
            vert_aperture = geom_camera.GetVerticalAperture() / 10.0

            # Compute new focal length for desired fovy:
            # fovy = 2 * arctan(vert_aperture / (2 * focal_length))
            # => focal_length = vert_aperture / (2 * tan(fovy/2))
            new_focal_length = vert_aperture / (2.0 * math.tan(math.radians(fovy / 2.0)))
            geom_camera.GetFocalLengthAttr().Set(new_focal_length * 10.0)  # Convert back to tenths
            logger.debug(f"Set fovy={fovy} on camera '{prim_path}' (focal_length={new_focal_length})")
        except Exception as e:
            logger.warning(f"Failed to set fovy on camera '{prim_path}': {e}")

    @staticmethod
    def _set_camera_clipping_range(camera, t_min: float, t_max: float):
        """Set clipping range on a Camera's USD prim after construction.

        Isaac Sim 4.5+ ``Camera.__init__()`` does not accept
        ``clipping_range_t_min`` / ``clipping_range_t_max`` kwargs.
        These must be set on the underlying USD prim directly.

        Args:
            camera: Isaac Sim Camera instance
            t_min: Near clip plane distance (m)
            t_max: Far clip plane distance (m)
        """
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim_path = camera.prim_path
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            logger.warning(
                f"Could not find prim at {prim_path} to set clipping range"
            )
            return

        # The clipping range is stored on the Camera2 API schema attribute
        # "clippingRange" as a VtVec2f (t_min, t_max).
        try:
            from pxr import Gf, Sdf

            # Set the clippingRange attribute on the camera prim
            clipping_attr = prim.GetAttribute("clippingRange")
            if clipping_attr.IsValid():
                clipping_attr.Set(Gf.Vec2f(t_min, t_max))
            else:
                # Fallback: create the attribute
                clipping_attr = prim.CreateAttribute(
                    "clippingRange", Sdf.ValueTypeNames.Vec2f
                )
                clipping_attr.Set(Gf.Vec2f(t_min, t_max))
            logger.debug(
                f"Set clipping range [{t_min}, {t_max}] on camera '{prim_path}'"
            )
        except Exception as e:
            logger.warning(
                f"Failed to set clipping range on camera '{prim_path}': {e}"
            )

    def initialize_cameras(self):
        """Initialize camera sensors after world.reset().

        Isaac Sim 4.5+ requires ``Camera.initialize()`` to be called after
        ``world.reset()`` so that render products and annotators are set up.
        Without this, ``get_rgba()`` will return ``None`` and the
        ``__del__`` / ``destroy()`` method will raise
        ``AttributeError: 'Camera' object has no attribute '_custom_annotators'``.

        The wrist camera is also created here (deferred from _add_cameras)
        because it needs to be a child of panda_hand with LOCAL coordinates,
        which only makes sense after the robot articulation is initialized.
        """
        # Initialize third-person camera
        if "third_person" in self.cameras:
            try:
                self.cameras["third_person"].initialize()
                logger.debug("Initialized third_person camera")
            except Exception as e:
                logger.warning(f"Failed to initialize third_person camera: {e}")

        # Create wrist camera as a child of panda_hand using USD APIs directly.
        # This ensures the camera moves with the end-effector and uses local coordinates.
        self._create_wrist_camera()

    def _create_wrist_camera(self):
        """Create the wrist-mounted camera as a child of panda_hand.

        Must be called AFTER world.reset() so that the panda_hand prim
        exists with correct transforms. Uses USD APIs directly to create
        a UsdGeom.Camera prim as a child of the TCP link, ensuring it moves
        with the robot and uses local (not world) coordinates.
        """
        import omni.usd

        wr_config = self._wrist_camera_config
        if not wr_config:
            logger.warning("No wrist camera config found")
            return

        wr_resolution = tuple(wr_config.get("resolution", [256, 256]))
        wr_position = np.array(wr_config.get("position", [0.0, 0.0, 0.0]))
        wr_fovy = wr_config.get("fovy", 75)

        if "orientation_quat" in wr_config:
            wr_orientation = np.array(wr_config["orientation_quat"])  # [w, x, y, z]
        else:
            wr_target = np.array(wr_config.get("target", [0.1, 0.0, 0.0]))
            wr_orientation = self._look_at_to_orientation(wr_position, wr_target)

        wr_clip_min = wr_config.get("clipping_range_t_min", 0.01)
        wr_clip_max = wr_config.get("clipping_range_t_max", 2.0)

        stage = omni.usd.get_context().get_stage()
        hand_prim_path = "/World/Franka/panda_hand"
        camera_prim_path = wr_config.get("prim_path", f"{hand_prim_path}/camera_wrist")

        # Find the panda_hand prim (the actual TCP link that moves with the arm)
        hand_prim = stage.GetPrimAtPath(hand_prim_path)
        if not hand_prim.IsValid():
            logger.warning(
                f"Cannot create wrist camera: panda_hand prim not found at "
                f"{hand_prim_path}. Ensure world.reset() was called first."
            )
            return

        # Define the camera prim as a child of panda_hand
        camera_prim = stage.DefinePrim(camera_prim_path, "Camera")

        from pxr import Gf, UsdGeom, Sdf, Vt

        # Set camera properties
        geom_camera = UsdGeom.Camera(camera_prim)
        geom_camera.GetHorizontalApertureAttr().Set(wr_resolution[1] / 10.0)  # tenths
        geom_camera.GetVerticalApertureAttr().Set(wr_resolution[0] / 10.0)

        # Set focal length — use explicit value from config if provided, otherwise derive from fovy
        wr_focal_length = wr_config.get("focal_length", None)
        if wr_focal_length is not None:
            geom_camera.GetFocalLengthAttr().Set(wr_focal_length * 10.0)  # USD stores in tenths
        else:
            import math
            vert_aperture = wr_resolution[0] / 10.0
            new_focal_length = (vert_aperture / (2.0 * math.tan(math.radians(wr_fovy / 2.0))))
            geom_camera.GetFocalLengthAttr().Set(new_focal_length * 10.0)

        # Set clipping range
        clipping_attr = camera_prim.GetAttribute("clippingRange")
        if clipping_attr.IsValid():
            clipping_attr.Set(Gf.Vec2f(wr_clip_min, wr_clip_max))
        else:
            clipping_attr = camera_prim.CreateAttribute(
                "clippingRange", Sdf.ValueTypeNames.Vec2f
            )
            clipping_attr.Set(Gf.Vec2f(wr_clip_min, wr_clip_max))

        # Now wrap it in Isaac Sim Camera class for get_rgba() support.
        # The Camera constructor interprets position/orientation as WORLD-space,
        # so we pass identity here and set the LOCAL transform directly via USD
        # XformOps afterward. This ensures the camera stays fixed relative to
        # panda_hand regardless of the robot's configuration.
        from isaacsim.sensors.camera import Camera

        self.cameras["wrist"] = Camera(
            prim_path=camera_prim_path,
            name="wrist_camera",
            resolution=wr_resolution,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.world.scene.add(self.cameras["wrist"])

        # Override with LOCAL coordinates relative to panda_hand.
        # The Camera constructor already created default xform ops (translate,
        # orient, scale) even with identity values. Use Get*Op() to retrieve
        # the existing ops and overwrite their values, rather than Add*Op()
        # which would fail because the ops already exist.
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.GetTranslateOp().Set(
            Gf.Vec3d(float(wr_position[0]), float(wr_position[1]), float(wr_position[2]))
        )
        xformable.GetOrientOp().Set(
            Gf.Quatd(
                float(wr_orientation[0]),  # w
                float(wr_orientation[1]),  # x
                float(wr_orientation[2]),  # y
                float(wr_orientation[3]),  # z
            )
        )

        # Initialize the camera sensor
        try:
            self.cameras["wrist"].initialize()
            logger.info(
                f"Created and initialized wrist camera at {camera_prim_path} "
                f"(resolution={wr_resolution}, fovy={wr_fovy}, "
                f"clipping=[{wr_clip_min}, {wr_clip_max}])"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wrist camera: {e}")

    def _add_lighting(self):
        """Add scene lighting for better camera images.

        Uses ``isaacsim.core.utils.stage.add_reference_to_stage`` which
        accepts ``(usd_path, prim_path)`` positional arguments in Isaac Sim 4.5+.
        The ``isaacsim.core.experimental.utils`` variant uses ``path=`` instead
        of ``prim_path=``, but the stable API uses ``prim_path``.
        """
        from isaacsim.core.utils.stage import add_reference_to_stage

        # Try multiple import paths for get_assets_root_path
        # (moved from isaacsim.core.utils.nucleus to isaacsim.storage.native in 4.5+)
        assets_root_path = None
        for _import_path in [
            "isaacsim.storage.native",
            "isaacsim.core.utils.nucleus",
        ]:
            try:
                _mod = __import__(_import_path, fromlist=["get_assets_root_path"])
                assets_root_path = _mod.get_assets_root_path()
                if assets_root_path:
                    break
            except Exception:
                continue

        if assets_root_path:
            # Add a simple environment lighting
            add_reference_to_stage(
                usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
                prim_path="/World/ground",
            )
        else:
            logger.warning(
                "Could not find Isaac Sim assets root path — "
                "skipping environment lighting. Ensure Nucleus server is available "
                "or set the asset root path manually."
            )

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        """Capture images from all cameras.

        IMPORTANT: Images are rotated 180 degrees to match LIBERO/OpenVLA-OFT
        training preprocessing. See libero_utils.py:
            img = img[::-1, ::-1]  # rotate 180 degrees to match train preprocessing

        Returns:
            Dict with "third_person" and "wrist" images as numpy arrays (H, W, 3) uint8
        """
        images = {}

        for name, camera in self.cameras.items():
            try:
                rgba = camera.get_rgba()
                if rgba is not None:
                    rgba = np.asarray(rgba)
                    # Isaac Sim Camera.get_rgba() may return different shapes:
                    #   - (H, W, 4) — standard numpy array
                    #   - (H*W, 4)  — flat array that needs reshaping
                    # Handle both cases robustly.
                    if rgba.ndim == 3:
                        # Already (H, W, 4) — just drop alpha
                        img = rgba[:, :, :3].astype(np.uint8)
                    elif rgba.ndim == 2:
                        # Flat (H*W, 4) — need to determine H, W
                        width, height = camera.get_resolution()
                        img = rgba.reshape(height, width, 4)[:, :, :3].astype(np.uint8)
                    else:
                        logger.warning(f"Camera '{name}' returned unexpected array shape {rgba.shape}")
                        continue

                    # CRITICAL: Rotate 180 degrees to match LIBERO/OpenVLA-OFT training preprocessing
                    # Without this rotation, the model sees upside-down images and performance degrades
                    img = img[::-1, ::-1]

                    images[name] = img
                else:
                    logger.warning(f"Camera '{name}' returned None — "
                                   f"ensure initialize_cameras() was called after world.reset()")
            except Exception as e:
                logger.warning(f"Failed to get image from camera '{name}': {e}")

        return images

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Get current robot state (joint positions + gripper).

        Returns:
            Dict with "joint_positions" (7,), "gripper_width" (1,),
            "ee_position" (3,), "ee_orientation" (4,)
        """
        if self.robot is None:
            return {}

        joint_positions = self.robot.get_joint_positions()
        ee_position, ee_orientation = self.robot.end_effector.get_world_pose()

        # Gripper joint positions: returns shape (1, 2) for ParallelGripper
        # (two finger joints). We average them to get a single gripper width.
        if self.gripper is not None:
            gripper_positions = np.asarray(self.gripper.get_joint_positions()).flatten()
            # Average the two finger positions for a single width value
            gripper_width = float(np.mean(gripper_positions))
        else:
            gripper_width = 0.04

        return {
            "joint_positions": np.array(joint_positions[:7]),  # 7 arm joints
            "gripper_width": np.array([gripper_width]),
            "ee_position": np.array(ee_position),
            "ee_orientation": np.array(ee_orientation),
        }

    def get_proprioception(self) -> np.ndarray:
        """Get proprioceptive state for VLA input (8D).

        Returns:
            np.ndarray of shape (8,): 7 joint angles + 1 gripper width
        """
        state = self.get_robot_state()
        if not state:
            return np.zeros(8, dtype=np.float32)

        joint_pos = state["joint_positions"]  # (7,)
        gripper = state["gripper_width"]      # (1,)
        # Both are 1-D — safe to concatenate
        return np.concatenate([joint_pos, gripper]).astype(np.float32)

    def get_object_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all manipulation objects.

        Returns:
            Dict mapping object names to their current positions
        """
        positions = {}
        for name, obj in self.objects.items():
            pos, _ = obj.get_world_pose()
            positions[name] = np.array(pos)
        return positions

    def reset_objects(self):
        """Reset all manipulation objects to their initial positions."""
        for name, obj in self.objects.items():
            if name in self._initial_positions:
                pos = self._initial_positions[name]
                obj.set_world_pose(position=pos)
                # Reset velocity
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))

        logger.debug("Reset all objects to initial positions")

    def reset_robot(self, home_joints: Optional[np.ndarray] = None):
        """Reset robot to home configuration.

        The Franka has 9 DOFs (7 arm + 2 gripper), so we must provide
        all 9 joint positions when calling set_joint_positions without
        joint_indices. This method always resets all 9 DOFs.

        Args:
            home_joints: Joint positions for home pose (7 arm joints, default: FRANKA_HOME_JOINTS)
        """
        from src.utils import FRANKA_HOME_JOINTS, FRANKA_GRIPPER_OPEN

        if home_joints is None:
            home_joints = FRANKA_HOME_JOINTS

        if self.robot is not None:
            # Build full 9-DOF position vector: 7 arm joints + 2 gripper joints (open)
            # The Franka Panda has 9 DOFs total:
            #   [panda_joint1..7, panda_finger_joint1, panda_finger_joint2]
            full_positions = np.zeros(9, dtype=np.float64)
            full_positions[:7] = home_joints
            full_positions[7:] = FRANKA_GRIPPER_OPEN  # Open gripper (0.04 m)

            self.robot.set_joint_positions(full_positions)

            # Also open the gripper via the gripper API
            if self.gripper is not None:
                try:
                    self.gripper.open()
                except Exception as e:
                    logger.debug(f"Gripper open() failed (non-fatal): {e}")

        logger.debug("Reset robot to home position")
