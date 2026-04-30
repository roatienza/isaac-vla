"""
Isaac Sim Bridge: VLA + Franka Control Loop
=============================================

The main simulation loop that connects the VLA server to Isaac Sim.
This module handles:

1. Isaac Sim initialization and scene management
2. Observation capture (images + proprioception)
3. VLA server communication (HTTP requests)
4. Action execution (VLA output → sim actions)
5. Episode management (reset, task loading, success detection)
6. HTTP API for external control
7. Video recording and visualization
8. Interactive command mode

Architecture:
    ┌──────────────┐    HTTP     ┌──────────────────┐
    │  VLA Server  │◄──────────►│  Sim Bridge       │
    │  (GPU:5090)  │  :8777/act │  (Isaac Sim)      │
    └──────────────┘             └────────┬─────────┘
                                         │
                                  ┌───────▼────────┐
                                  │ Franka +       │
                                  │ Kitchen Scene  │
                                  └────────────────┘

The bridge can run in two modes:
1. Server mode: Runs an HTTP API alongside the sim loop
2. Embedded mode: Direct Python API (no HTTP overhead)

Visualization modes:
- GUI mode (headless=False): Opens Isaac Sim window — you can see the robot
- Headless + video recording: Saves MP4 video of the episode
- Interactive mode: Type instructions and watch the robot execute them
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

from src.action_pipeline import ActionPipeline, GripperController
from src.kitchen_scene import KitchenScene
from src.utils import (
    FRANKA_HOME_JOINTS,
    crop_and_resize,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


class SimBridge:
    """Bridge between VLA server and Isaac Sim.

    Manages the simulation loop, observation capture, VLA queries,
    action execution, and visualization.

    Args:
        config_path: Path to configuration YAML file
        vla_server_url: URL of the VLA server (e.g., "http://localhost:8777")
    """

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        vla_server_url: str = "http://localhost:8777",
    ):
        # Load configuration
        self.config = load_config(config_path)
        self.vla_server_url = vla_server_url

        # Simulation settings
        sim_config = self.config.get("sim_bridge", {}).get("isaac_sim", {})
        self.headless = sim_config.get("headless", False)
        self.physics_dt = sim_config.get("physics_dt", 1.0 / 120.0)
        self.render_dt = sim_config.get("render_dt", 1.0 / 30.0)
        self.render_interval = sim_config.get("render_interval", 4)

        # Episode settings
        ep_config = self.config.get("episode", {})
        self.max_steps = ep_config.get("max_steps", 500)
        self.action_chunk_size = ep_config.get("action_chunk_size", 8)
        self.vla_query_frequency = ep_config.get("vla_query_frequency", 8)

        # Action pipeline
        pipeline_config = self.config.get("action_pipeline", {})
        self.action_pipeline = ActionPipeline(pipeline_config)

        # Gripper controller
        self.gripper_controller = GripperController()

        # Kitchen scene
        self.scene = None

        # State
        self._world = None
        self._simulation_app = None
        self._step_count = 0
        self._episode_count = 0
        self._running = False

        # VLA client
        self._vla_client = VLAClient(vla_server_url)

        # Video recording
        self._video_writer = None
        self._video_dir = None
        self._video_frames = []
        self._recording = False

        # Note: No background event loop pump thread. Calling
        # SimulationApp.update() from a separate thread races with
        # World.step(render=True) on the main thread and causes a
        # RecursiveSharedMutex assertion crash. Instead, the interactive
        # input loop in run_sim_bridge.py uses non-blocking I/O with
        # select.select() and periodically steps the simulation on the
        # main thread to keep the GUI responsive.

    def initialize(self):
        """Initialize Isaac Sim and build the scene.

        This must be called before run(). When headless=False (default),
        this opens the Isaac Sim window where you can see the robot.
        """
        logger.info("Initializing Isaac Sim...")

        from isaacsim import SimulationApp

        launch_config = {"headless": self.headless}

        # Set window dimensions for better visibility
        sim_config = self.config.get("sim_bridge", {}).get("isaac_sim", {})
        if "width" in sim_config:
            launch_config["width"] = str(sim_config["width"])
        if "height" in sim_config:
            launch_config["height"] = str(sim_config["height"])

        self._simulation_app = SimulationApp(launch_config)

        if not self.headless:
            logger.info("Isaac Sim window opened — you should see the simulation viewport.")
            logger.info("The window will show the Franka robot in the kitchen scene.")
        else:
            logger.info("Running in headless mode (no GUI window).")

        # Import Isaac Sim modules after SimulationApp is created
        from isaacsim.core.api import World

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=self.render_dt,
        )

        # Build kitchen scene
        scene_config = self.config.get("sim_bridge", {})
        kitchen_config = scene_config.get("kitchen_scene", {})
        usd_path = kitchen_config.get("usd_path")

        self.scene = KitchenScene(self._world, scene_config, usd_path=usd_path)
        self.scene.build()

        # Set robot default state BEFORE world.reset() so that post_reset()
        # uses the correct position. The Franka constructor's position=
        # argument sets the USD prim transform, but the physics default state
        # stored in the XFormPrimView remains [0,0,0] until explicitly set.
        # If we only enforce AFTER reset, the first reset already placed the
        # robot at [0,0,0] and set_world_pose() alone may not persist across
        # subsequent resets or sim steps.
        self.scene.enforce_robot_position()

        # Reset world to initialize physics — post_reset() will now use the
        # correct default state we just set.
        self._world.reset()

        # Re-enforce after reset — physics buffer write is critical because
        # set_world_pose() and set_default_state() only update the USD prim
        # (visual), not the physics Articulation root position.
        self.scene.enforce_robot_position()

        # Initialize camera sensors (must happen after world.reset())
        # Without this, Camera.get_rgba() returns None and Camera.__del__
        # raises AttributeError: 'Camera' object has no attribute '_custom_annotators'
        self.scene.initialize_cameras()

        # Step a few times to let the scene settle
        for _ in range(10):
            self._world.step(render=True)
            # Pump the event loop to keep the GUI responsive during init
            if self._simulation_app is not None:
                self._simulation_app.update()

        # Final enforcement AFTER settle steps — physics state can drift
        #
        self.scene.enforce_robot_position()

        logger.info("Isaac Sim initialized successfully")

    def enable_video_recording(self, video_dir: str = "./data/evaluation_videos"):
        """Enable video recording for episodes.

        Each frame will be captured from the third-person camera and
        saved as an MP4 video file when stop_video_recording() is called.

        Args:
            video_dir: Directory to save video files
        """
        self._video_dir = video_dir
        self._recording = True
        self._video_frames = []
        logger.info(f"Video recording enabled, saving to: {video_dir}")

    def stop_video_recording(self):
        """Stop recording and save the video file."""
        if self._video_frames and self._video_dir:
            self._save_video_frames()
        self._recording = False
        logger.info("Video recording stopped")

    def _capture_video_frame(self):
        """Capture a single frame for video recording."""
        if not self._recording:
            return

        try:
            images = self.scene.get_camera_images()
            tp_image = images.get("third_person")
            if tp_image is not None:
                self._video_frames.append(tp_image.copy())
        except Exception as e:
            logger.debug(f"Failed to capture video frame: {e}")

    def _save_video_frames(self):
        """Save accumulated video frames to an MP4 file."""
        if not self._video_frames:
            return

        import cv2

        os.makedirs(self._video_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self._video_dir, f"episode_{timestamp}.mp4")

        h, w = self._video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30.0
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for frame in self._video_frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()
        logger.info(f"Video saved: {video_path} ({len(self._video_frames)} frames)")

        # Clear frames for next episode
        self._video_frames = []

    def step_simulation(self, num_steps: int = 1):
        """Step the simulation forward (useful for keeping the window responsive).

        Also pumps the SimulationApp event loop to prevent the
        "Isaac Sim is Not Responding" dialog that appears on Linux when
        the Omniverse Kit window does not receive rendering updates for
        several seconds (e.g., during VLA inference, waiting for user
        input, or long pauses between simulation steps).

        Args:
            num_steps: Number of simulation steps to take
        """
        if self._world is not None:
            for _ in range(num_steps):
                self._world.step(render=True)

        # Pump the SimulationApp event loop to keep the GUI responsive.
        # Without this call, the Omniverse Kit window freezes and the
        # desktop environment (GNOME/KDE on Linux) shows a "Not Responding"
        # dialog after ~5 seconds of no event-loop updates.
        if self._simulation_app is not None:
            self._simulation_app.update()

    def get_status(self) -> Dict[str, Any]:
        """Get current bridge status for display.

        Returns:
            Dict with current state information
        """
        status = {
            "initialized": self._world is not None,
            "headless": self.headless,
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "recording": self._recording,
        }

        if self.scene and self.scene.robot is not None:
            try:
                robot_state = self.scene.get_robot_state()
                status["robot"] = {
                    "ee_position": robot_state.get("ee_position", [0, 0, 0]).tolist()
                    if hasattr(robot_state.get("ee_position"), "tolist")
                    else robot_state.get("ee_position", [0, 0, 0]),
                    "gripper_width": float(robot_state.get("gripper_width", 0)),
                }
            except Exception:
                status["robot"] = "unavailable"

        return status

    def run_episode(
        self,
        instruction: str,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run a single episode with a natural language instruction.

        When headless=False, the Isaac Sim window will show the robot
        executing the task in real-time.

        Args:
            instruction: Natural language task description
            max_steps: Maximum steps (default from config)
            callback: Optional callback called each step with (step, obs, action)

        Returns:
            Episode result dict with success, steps, actions, etc.
        """
        if self._world is None:
            raise RuntimeError("Call initialize() before run_episode()")

        max_steps = max_steps or self.max_steps
        self._step_count = 0

        # Reset scene
        self.scene.reset_robot()
        self.scene.reset_objects()

        # Step a few times to let physics settle after reset
        for _ in range(5):
            self._world.step(render=True)
            # Pump event loop to prevent "Not Responding" dialog
            if self._simulation_app is not None:
                self._simulation_app.update()

        logger.info(f"Starting episode: '{instruction}' (max {max_steps} steps)")

        if not self.headless:
            logger.info(">>> Watch the Isaac Sim window to see the robot! <<<")

        actions_log = []
        inference_times = []

        for step in range(max_steps):
            # 1. Capture observation (with error recovery)
            try:
                obs = self._capture_observation()
            except Exception as e:
                logger.error(f"Observation capture failed at step {step}: {e}")
                # Try to recover by stepping the simulation and continuing
                self._world.step(render=True)
                if self._simulation_app is not None:
                    self._simulation_app.update()
                continue

            # 2. Capture video frame if recording
            self._capture_video_frame()

            # 3. Query VLA for action chunk (every N steps)
            try:
                if step % self.vla_query_frequency == 0:
                    # Pump the event loop before the (potentially slow) VLA query
                    # to prevent the "Not Responding" dialog during inference
                    if self._simulation_app is not None:
                        self._simulation_app.update()

                    vla_result = self._query_vla(obs, instruction)

                    if vla_result is None:
                        logger.error("VLA query failed, stopping episode")
                        break

                    inference_times.append(vla_result.get("inference_time_s", 0))

                    # 4. Process action chunk
                    robot_state = self.scene.get_robot_state()

                    # Convert VLA actions from lists to numpy arrays.
                    # The VLA server returns JSON, so actions are plain Python lists.
                    # The action pipeline expects numpy arrays for .sum(), etc.
                    vla_actions = [np.asarray(a, dtype=np.float64) for a in vla_result["actions"]]

                    processed, success = self.action_pipeline.process_action_chunk(
                        actions=vla_actions,
                        current_ee_position=robot_state["ee_position"],
                        current_ee_orientation=robot_state["ee_orientation"],
                        current_joint_positions=robot_state["joint_positions"],
                    )

                    if not success:
                        logger.warning(
                            f"Action chunk processing partially failed at step {step} "
                            f"(IK fallback used — robot will maintain position)"
                        )
            except Exception as e:
                logger.error(f"Action chunk processing error at step {step}: {e}")
                # Fall through to execute any remaining actions from previous chunk

            # 5. Execute next action from chunk
            action = self.action_pipeline.get_next_action()

            if action is not None:
                try:
                    self._execute_action(action)
                    actions_log.append(action)
                except Exception as e:
                    logger.error(f"Action execution failed at step {step}: {e}")
                    # Continue to next step rather than crashing the episode

            # 6. Step simulation (render=True keeps the window visible)
            self._world.step(render=True)

            # Pump event loop to prevent "Not Responding" dialog
            if self._simulation_app is not None:
                self._simulation_app.update()
            self._step_count += 1

            # 8. Callback
            if callback:
                callback(step, obs, action)

            # 8. Log progress periodically
            if step % 50 == 0 and step > 0:
                logger.info(f"  Step {step}/{max_steps}")

        self._episode_count += 1

        # Final video frame
        self._capture_video_frame()

        avg_inference = (
            sum(inference_times) / len(inference_times) if inference_times else 0
        )

        result = {
            "instruction": instruction,
            "steps": self._step_count,
            "episode": self._episode_count,
            "actions_executed": len(actions_log),
            "avg_inference_time": avg_inference,
            "success": False,  # Requires task-specific success detection
        }

        logger.info(
            f"Episode complete: {self._step_count} steps, "
            f"{len(actions_log)} actions, "
            f"avg inference: {avg_inference:.3f}s"
        )

        return result

    def _capture_observation(self) -> Dict[str, Any]:
        """Capture current observation from the scene.

        Returns:
            Dict with:
                - "full_image": (H, W, 3) uint8 third-person image
                - "wrist_image": (H, W, 3) uint8 wrist camera image
                - "state": (8,) float32 proprioceptive state
                - "task_description": str
        """
        # Get camera images
        try:
            images = self.scene.get_camera_images()
        except Exception as e:
            logger.warning(f"Failed to get camera images: {e}")
            images = {}

        # Get proprioception
        try:
            proprio = self.scene.get_proprioception()
        except Exception as e:
            logger.warning(f"Failed to get proprioception: {e}")
            proprio = np.zeros(8, dtype=np.float32)

        # Preprocess images for VLA
        tp_image = crop_and_resize(
            images.get("third_person", np.zeros((480, 640, 3), dtype=np.uint8)),
            target_size=(256, 256),
            center_crop=True,
        )
        wr_image = crop_and_resize(
            images.get("wrist", np.zeros((256, 256, 3), dtype=np.uint8)),
            target_size=(128, 128),
            center_crop=True,
        )

        return {
            "full_image": tp_image,
            "wrist_image": wr_image,
            "state": proprio,
        }

    def _query_vla(
        self,
        observation: Dict[str, Any],
        instruction: str,
        unnorm_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query the VLA server for an action chunk.

        Args:
            observation: Observation dict from _capture_observation
            instruction: Natural language instruction
            unnorm_key: Dataset statistics key for denormalization

        Returns:
            VLA result dict with actions, or None on failure
        """
        try:
            result = self._vla_client.predict_action(
                image=observation["full_image"],
                instruction=instruction,
                wrist_image=observation.get("wrist_image"),
                state=observation.get("state"),
                unnorm_key=unnorm_key,
            )
            return result
        except Exception as e:
            logger.error(f"VLA query failed: {e}")
            return None

    def _execute_action(self, action: Dict[str, np.ndarray]):
        """Execute a single action in the simulation.

        The Franka has 9 DOFs (7 arm + 2 gripper). We apply position
        targets for all 9 joints via ArticulationAction.

        Args:
            action: Dict with "joint_positions" (7,) and "gripper" (scalar)
        """
        if self.scene is None or self.scene.robot is None:
            return

        # Set joint positions
        joint_positions = action["joint_positions"]
        gripper_target = action["gripper"]

        # Apply to robot using ArticulationAction with all 9 DOFs
        from isaacsim.core.utils.types import ArticulationAction

        # Build full 9-DOF position vector
        full_joint_positions = np.zeros(9, dtype=np.float64)
        full_joint_positions[:7] = joint_positions
        full_joint_positions[7:] = gripper_target  # Both finger joints

        robot_action = ArticulationAction(joint_positions=full_joint_positions)
        self.scene.robot.apply_action(robot_action)

        # Update gripper controller
        self.gripper_controller.set_target(gripper_target)

    def reset(self):
        """Reset the scene to initial state."""
        if self.scene is not None:
            self.scene.reset_robot()
            self.scene.reset_objects()

            # Re-enforce robot base position after reset — physics buffer
            # write is critical to update the Articulation root position.
            self.scene.enforce_robot_position()

            # Step a few times to settle
            if self._world is not None:
                for _ in range(5):
                    self._world.step(render=True)
                    # Pump event loop to prevent "Not Responding" dialog
                    if self._simulation_app is not None:
                        self._simulation_app.update()

            # Final enforcement AFTER settle steps
            self.scene.enforce_robot_position()

        self._step_count = 0
        self.action_pipeline._action_queue = []
        self.action_pipeline._chunk_index = 0

        logger.info("Scene reset to initial state")

    def close(self):
        """Clean up and close the simulation."""
        # Save any remaining video frames
        if self._recording and self._video_frames:
            self._save_video_frames()

        if self._simulation_app is not None:
            # Pump the event loop a few times before closing to allow
            # pending UI events to be processed. This prevents the
            # "Not Responding" dialog from appearing during shutdown.
            for _ in range(5):
                self._simulation_app.update()
            self._simulation_app.close()
        logger.info("Simulation closed")


class VLAClient:
    """HTTP client for the VLA server.

    Args:
        server_url: Base URL of the VLA server
    """

    def __init__(self, server_url: str = "http://localhost:8777"):
        self.server_url = server_url.rstrip("/")
        self._session = requests.Session()

    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        wrist_image: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query the VLA server for an action prediction.

        Args:
            image: Third-person camera image (H, W, 3) uint8
            instruction: Natural language task description
            wrist_image: Wrist camera image (H, W, 3) uint8 (optional)
            state: Proprioceptive state (8,) float32 (optional)
            unnorm_key: Dataset statistics key for denormalization

        Returns:
            Dict with "actions" (list of 7D arrays), "inference_time_s", etc.
        """
        payload = {
            "image": image.tolist(),
            "instruction": instruction,
        }

        if wrist_image is not None:
            payload["wrist_image"] = wrist_image.tolist()

        if state is not None:
            payload["state"] = state.tolist()

        if unnorm_key is not None:
            payload["unnorm_key"] = unnorm_key

        response = self._session.post(
            f"{self.server_url}/act",
            json=payload,
            timeout=30.0,
        )

        response.raise_for_status()
        return response.json()

    def health(self) -> Dict:
        """Check VLA server health."""
        response = self._session.get(f"{self.server_url}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()

    def model_info(self) -> Dict:
        """Get VLA model info."""
        response = self._session.get(f"{self.server_url}/model_info", timeout=5.0)
        response.raise_for_status()
        return response.json()

    def warmup(self) -> Dict:
        """Trigger VLA server warmup."""
        response = self._session.post(f"{self.server_url}/warmup", timeout=120.0)
        response.raise_for_status()
        return response.json()


# ─── Embedded Mode (No HTTP) ────────────────────────────────────────────────

class EmbeddedVLABridge:
    """Embedded VLA bridge that loads the model directly (no HTTP server).

    This avoids HTTP overhead by importing the VLA model directly.
    Useful for single-machine setups where the GPU running Isaac Sim
    also runs the VLA model.

    Args:
        config_path: Path to configuration YAML file
    """

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self._vla_server = None
        self._initialized = False

    def initialize(self):
        """Load the VLA model directly."""
        from src.vla_server import VLAServer, VLAConfig

        vla_config = self.config.get("vla_server", {})
        self._vla_server = VLAServer(
            config=VLAConfig(**vla_config.get("model", {}))
        )
        self._vla_server.load_model()
        self._vla_server.warmup()
        self._initialized = True
        logger.info("Embedded VLA model loaded")

    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: str,
    ) -> Any:
        """Predict action using embedded VLA model."""
        if not self._initialized:
            raise RuntimeError("Call initialize() first")
        return self._vla_server.predict_action(observation, instruction)

