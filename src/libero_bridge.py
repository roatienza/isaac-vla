"""
LIBERO Bridge: VLA + MuJoCo Control Loop
=========================================

The main simulation loop that connects the VLA server to the LIBERO
MuJoCo environment. This module provides a drop-in replacement for
the Isaac Sim bridge, using the same VLA server but bypassing IK
solving since LIBERO accepts delta EE actions directly.

Architecture:
    ┌───────────────┐    HTTP     ┌──────────────────┐
    │  VLA Server   │◄───────────►│  LIBERO Bridge   │
    │  (GPU:5090)   │  :8777/act  │  (MuJoCo)        │
    └───────────────┘             └──────────────────┘
                                     │
                                ┌────┴────┐
                                │ LIBERO  │
                                │ MuJoCo  │
                                │ Env     │
                                └─────────┘

Key differences from Isaac Sim:
- Uses MuJoCo physics (same as LIBERO training)
- No IK solving needed — LIBERO accepts delta EE actions directly
- No camera rendering overhead (MuJoCo is fast)
- Direct access to LIBERO task suite and evaluation
- Zero visual domain gap with OpenVLA-OFT training data

This implementation matches the reference OpenVLA-OFT evaluation code:
https://github.com/roatienza/openvla-oft/experiments/robot/libero/
"""

import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import (
    crop_and_resize,
    load_config,
    setup_logging,
    clip_action_magnitude,
)

logger = logging.getLogger(__name__)

# ──── Task Suite Constants ─────────────────────────────────────────────────────

# Max steps per task suite (from reference implementation)
TASK_MAX_STEPS = {
    "libero_spatial": 220,   # longest training demo has 193 steps
    "libero_object": 280,    # longest training demo has 254 steps
    "libero_goal": 300,      # longest training demo has 270 steps
    "libero_10": 520,        # longest training demo has 505 steps
    "libero_90": 400,        # longest training demo has 373 steps
}

# Default normalization statistics for LIBERO (OpenVLA-OFT pretrained)
# These match the training distribution for proper denormalization
DEFAULT_NORM_STATS = {
    "libero_spatial_no_noops": {
        "action": {
            "min": np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0]),
            "max": np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0]),
            "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            "std": np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3]),
        }
    },
    "libero_object_no_noops": {
        "action": {
            "min": np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0]),
            "max": np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0]),
            "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            "std": np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3]),
        }
    },
    "libero_goal_no_noops": {
        "action": {
            "min": np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0]),
            "max": np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0]),
            "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            "std": np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3]),
        }
    },
    "libero_10_no_noops": {
        "action": {
            "min": np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi, 0.0]),
            "max": np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi, 1.0]),
            "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            "std": np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.3]),
        }
    },
}


class LIBEROBridge:
    """Bridge between VLA server and LIBERO MuJoCo environment.

    Manages the simulation loop, observation capture, VLA queries,
    action execution, and evaluation. Unlike the Isaac Sim bridge,
    this bypasses IK solving since LIBERO accepts delta EE actions
    directly.

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

        # LIBERO settings
        libero_config = self.config.get("libero", {})
        self.task_suite_name = libero_config.get("task_suite", "libero_spatial")
        self.task_id = libero_config.get("task_id", 0)
        self.num_episodes = libero_config.get("num_episodes", 10)
        self.camera_heights = libero_config.get("camera_heights", 256)
        self.camera_widths = libero_config.get("camera_widths", 256)
        self.center_crop = libero_config.get("center_crop", True)
        self.target_size = tuple(libero_config.get("target_size", [224, 224]))

        # Episode settings
        ep_config = self.config.get("episode", {})
        self.max_steps = TASK_MAX_STEPS.get(self.task_suite_name, 500)
        self.action_chunk_size = ep_config.get("action_chunk_size", 8)
        self.vla_query_frequency = ep_config.get("vla_query_frequency", 8)
        self.num_steps_wait = libero_config.get("num_steps_wait", 10)

        # Action clipping settings (safety bounds)
        self.max_position_delta = libero_config.get("max_position_delta", 0.05)
        self.max_rotation_delta = libero_config.get("max_rotation_delta", 0.1)

        # LIBERO environment
        self.env = None
        self.task_suite = None
        self.task = None
        self.init_states = None

        # State
        self._step_count = 0
        self._episode_count = 0
        self._running = False
        self._success_count = 0

        # VLA client
        self._vla_client = VLAClient(vla_server_url)

        # Video recording
        self._video_frames = []
        self._recording = False

        # Action chunk state
        self._action_queue: deque = deque(maxlen=self.action_chunk_size)
        self._chunk_index = 0

        # Dataset statistics (for denormalization)
        self._norm_stats = DEFAULT_NORM_STATS
        self._unnorm_key = "libero_spatial_no_noops"

    def initialize(self):
        """Initialize the LIBERO environment and task suite."""
        logger.info("Initializing LIBERO environment...")

        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as e:
            raise ImportError(
                "LIBERO package not found. Install with: "
                "pip install libero && python benchmark_scripts/download_libero_datasets.py"
            ) from e

        # Get task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        if self.task_suite_name not in benchmark_dict:
            raise ValueError(
                f"Unknown LIBERO suite '{self.task_suite_name}'. "
                f"Available: {', '.join(sorted(benchmark_dict.keys()))}"
            )

        self.task_suite = benchmark_dict[self.task_suite_name]()
        self.task = self.task_suite.get_task(self.task_id)

        logger.info(
            f"Loaded task {self.task_id} from suite '{self.task_suite_name}': "
            f"{self.task.language}"
        )

        # Initialize environment
        import os
        from libero.libero import get_libero_path

        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            self.task.problem_folder,
            self.task.bddl_file,
        )

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths,
        }

        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(0)  # IMPORTANT: seed seems to affect object positions

        # Get initial states from task suite (takes int index, not Task object)
        # OffScreenRenderEnv does NOT have get_all_initial_states()
        self.init_states = self.task_suite.get_task_init_states(self.task_id)

        # Update unnorm_key based on task suite
        self._update_unnorm_key()

        logger.info(f"LIBERO environment initialized with {len(self.init_states)} initial states")

    def _update_unnorm_key(self):
        """Update the unnorm_key based on task suite."""
        # Try different key variations
        base_key = self.task_suite_name
        possible_keys = [
            base_key,
            f"{base_key}_no_noops",
        ]

        for key in possible_keys:
            if key in self._norm_stats:
                self._unnorm_key = key
                break
        else:
            logger.warning(
                f"Could not find unnorm_key for suite '{self.task_suite_name}'. "
                f"Using default: {self._unnorm_key}"
            )

    def _capture_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process observation from LIBERO environment.

        In robosuite/LIBERO, camera images are part of the observation dict
        returned by env.reset() and env.step(). We do NOT call sim.render()
        manually — that API has changed in modern MuJoCo.

        Args:
            obs: Raw observation dict from env.reset() or env.step()

        Returns:
            Dict with 'full_image', 'wrist_image', and 'state'
        """
        # Extract images from observation dict
        full_image = obs["agentview_image"]
        wrist_image = obs["robot0_eye_in_hand_image"]

        # CRITICAL: Rotate 180 degrees to match LIBERO/OpenVLA-OFT training preprocessing
        # Reference: https://github.com/roatienza/openvla-oft/experiments/robot/libero/libero_utils.py
        full_image = full_image[::-1, ::-1]
        wrist_image = wrist_image[::-1, ::-1]

        # Ensure images are uint8 (LIBERO may return float32 in [0,1] or [0,255])
        if full_image.dtype != np.uint8:
            if full_image.max() <= 1.0:
                full_image = (full_image * 255).astype(np.uint8)
            else:
                full_image = full_image.astype(np.uint8)
        if wrist_image.dtype != np.uint8:
            if wrist_image.max() <= 1.0:
                wrist_image = (wrist_image * 255).astype(np.uint8)
            else:
                wrist_image = wrist_image.astype(np.uint8)

        # Get proprioception state (8D: 7 joint positions + 1 gripper width)
        # CRITICAL: OpenVLA-OFT expects exactly 8D state
        # robot0_gripper_qpos may return 4 values in some robosuite versions
        # We only need the first value (gripper width)
        state = np.concatenate([
            obs["robot0_joint_pos"][:7],       # 7D joint positions
            obs["robot0_gripper_qpos"][:1],    # 1D gripper width (first value only)
        ]).astype(np.float32)

        return {
            "full_image": full_image,
            "wrist_image": wrist_image,
            "state": state,
        }

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process action for LIBERO environment.

        Args:
            action: 7D action [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            Processed action ready for LIBERO environment
        """
        # Clip action magnitude for safety
        clipped = clip_action_magnitude(
            action,
            max_position=self.max_position_delta,
            max_rotation=self.max_rotation_delta,
        )

        # Normalize gripper action from [0,1] to [-1,1] for LIBERO
        # Reference: https://github.com/roatienza/openvla-oft/experiments/robot/robot_utils.py
        orig_low, orig_high = 0.0, 1.0
        clipped[6] = 2 * (clipped[6] - orig_low) / (orig_high - orig_low) - 1

        # Ensure gripper is in valid range [-1, 1] for LIBERO
        clipped[6] = np.clip(clipped[6], -1.0, 1.0)

        return clipped

    def _get_dummy_action(self) -> np.ndarray:
        """Get dummy/no-op action for stabilization."""
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)

    def run_episode(
        self,
        instruction: Optional[str] = None,
        initial_state: Optional[np.ndarray] = None,
        record_video: bool = False,
    ) -> Dict[str, Any]:
        """Run a single episode.

        Args:
            instruction: Task instruction (defaults to task language)
            initial_state: Initial state to reset to
            record_video: Whether to record video frames

        Returns:
            Dict with episode results
        """
        if instruction is None:
            instruction = self.task.language

        # Reset environment
        if self.env is not None:
            self.env.reset()
            if initial_state is not None:
                self.env.set_init_state(initial_state)
            else:
                self.env.set_init_state(self.init_states[0])

        self._step_count = 0
        self._action_queue.clear()
        self._chunk_index = 0
        self._video_frames = []
        self._recording = record_video

        logger.info(f"Running episode with instruction: {instruction}")
        logger.info(f"Max steps: {self.max_steps}, Wait steps: {self.num_steps_wait}")

        # Run episode
        success = False
        t = 0
        max_steps = self.max_steps + self.num_steps_wait

        try:
            while t < max_steps:
                # Do nothing for the first few timesteps to let objects stabilize
                if t < self.num_steps_wait:
                    obs, reward, done, info = self.env.step(self._get_dummy_action())
                    t += 1
                    continue

                # If action queue is empty, requery VLA
                if len(self._action_queue) == 0:
                    # Capture observation before querying VLA
                    observation = self._capture_observation(obs)

                    # Record video frame if enabled
                    if self._recording:
                        self._video_frames.append(observation["full_image"])

                    result = self._query_vla(observation, instruction)
                    if result is not None and "actions" in result:
                        actions = result["actions"]
                        for action in actions:
                            processed = self._process_action(np.array(action))
                            self._action_queue.append(processed)
                        logger.debug(
                            f"Queried VLA, got {len(actions)} actions. "
                            f"Inference time: {result.get('inference_time_s', 'N/A'):.3f}s"
                        )
                    else:
                        logger.warning("VLA query returned no actions, using dummy action")
                        self._action_queue.append(self._get_dummy_action())

                # Get action from queue and execute
                if len(self._action_queue) > 0:
                    action = self._action_queue.popleft()
                else:
                    action = self._get_dummy_action()

                obs, reward, done, info = self.env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

        except Exception as e:
            logger.error(f"Episode error: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Save video if recorded
        if self._recording and self._video_frames:
            self._save_video_frames(instruction)

        episode_result = {
            "success": success,
            "steps": t,
            "instruction": instruction,
            "task_id": self.task_id,
            "task_suite": self.task_suite_name,
        }

        logger.info(
            f"Episode complete: success={success}, steps={t}, "
            f"instruction={instruction}"
        )

        return episode_result

    def run_evaluation(
        self,
        num_episodes: Optional[int] = None,
        record_video: bool = False,
    ) -> Dict[str, Any]:
        """Run evaluation across multiple episodes.

        Args:
            num_episodes: Number of episodes to run (defaults to config value)
            record_video: Whether to record video frames

        Returns:
            Dict with evaluation results
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        logger.info(
            f"Starting evaluation: {num_episodes} episodes, "
            f"task {self.task_id} from {self.task_suite_name}"
        )

        results = []
        success_count = 0
        total_steps = 0

        for episode_idx in range(num_episodes):
            self._episode_count = episode_idx

            # Get initial state for this episode
            if episode_idx < len(self.init_states):
                initial_state = self.init_states[episode_idx]
            else:
                initial_state = self.init_states[0]

            # Run episode
            result = self.run_episode(
                instruction=self.task.language,
                initial_state=initial_state,
                record_video=record_video,
            )

            results.append(result)
            if result["success"]:
                success_count += 1
            total_steps += result["steps"]

            # Log progress
            logger.info(
                f"Episode {episode_idx + 1}/{num_episodes}: "
                f"success={result['success']}, steps={result['steps']}"
            )

        # Calculate metrics
        success_rate = success_count / num_episodes if num_episodes > 0 else 0.0
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0.0

        eval_result = {
            "task_suite": self.task_suite_name,
            "task_id": self.task_id,
            "task_name": self.task.language,
            "num_episodes": num_episodes,
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_episode_length": avg_steps,
            "episodes": results,
        }

        logger.info(
            f"Evaluation complete: {success_count}/{num_episodes} "
            f"({success_rate:.1%} success rate)"
        )

        return eval_result

    def _query_vla(
        self,
        observation: Dict[str, np.ndarray],
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        """Query the VLA server for an action chunk.

        Args:
            observation: Observation dict from _capture_observation
            instruction: Natural language instruction

        Returns:
            VLA result dict with actions, or None on failure
        """
        try:
            result = self._vla_client.predict_action(
                image=observation["full_image"],
                instruction=instruction,
                wrist_image=observation.get("wrist_image"),
                state=observation.get("state"),
                unnorm_key=self._unnorm_key,
            )
            return result
        except Exception as e:
            logger.error(f"VLA query failed: {e}")
            return None

    def _save_video_frames(self, instruction: str):
        """Save recorded video frames to disk."""
        if not self._video_frames:
            return

        try:
            import imageio
        except ImportError:
            logger.warning("imageio not installed, skipping video save")
            return

        video_dir = Path("data/libero_videos")
        video_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize instruction for filename
        safe_name = instruction.replace(" ", "_").replace("/", "_")[:50]
        video_path = video_dir / f"episode_{self._episode_count}_{safe_name}.mp4"

        imageio.mimwrite(
            str(video_path),
            self._video_frames,
            fps=10,
            quality=8,
        )

        logger.info(f"Saved video to {video_path}")

    def reset(self):
        """Reset the environment to initial state."""
        if self.env is not None:
            self.env.reset()
            self.env.set_init_state(self.init_states[0])

        self._step_count = 0
        self._action_queue.clear()
        self._chunk_index = 0

        logger.info("LIBERO environment reset")

    def close(self):
        """Clean up and close the environment."""
        if self.env is not None:
            self.env.close()
        logger.info("LIBERO environment closed")


class VLAClient:
    """HTTP client for the VLA server.

    Args:
        server_url: Base URL of the VLA server
    """

    def __init__(self, server_url: str = "http://localhost:8777"):
        self.server_url = server_url.rstrip("/")
        self._session = None

    def _get_session(self):
        import requests

        if self._session is None:
            self._session = requests.Session()
        return self._session

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
            "wrist_image": wrist_image.tolist() if wrist_image is not None else None,
            "instruction": instruction,
        }

        if state is not None:
            payload["state"] = state.tolist()

        if unnorm_key is not None:
            payload["unnorm_key"] = unnorm_key

        response = self._get_session().post(
            f"{self.server_url}/act",
            json=payload,
            timeout=30.0,
        )

        response.raise_for_status()
        return response.json()

    def health(self) -> Dict:
        """Check VLA server health."""
        response = self._get_session().get(
            f"{self.server_url}/health", timeout=5.0
        )
        response.raise_for_status()
        return response.json()

    def model_info(self) -> Dict:
        """Get VLA model info."""
        response = self._get_session().get(
            f"{self.server_url}/model_info", timeout=5.0
        )
        response.raise_for_status()
        return response.json()

    def warmup(self) -> Dict:
        """Trigger VLA server warmup."""
        response = self._get_session().post(
            f"{self.server_url}/warmup", timeout=120.0
        )
        response.raise_for_status()
        return response.json()


# ──── Embedded Mode (No HTTP) ─────────────────────────────────────────────────

class EmbeddedLIBEROBridge:
    """Embedded LIBERO bridge that loads the model directly (no HTTP server).

    This avoids HTTP overhead by importing the VLA model directly.
    Useful for single-machine setups where the GPU running LIBERO
    also runs the VLA model.

    Args:
        config_path: Path to configuration YAML file
    """

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self._bridge = None
        self._initialized = False

    def initialize(self):
        """Initialize the LIBERO environment and load the VLA model."""
        self._bridge = LIBEROBridge(
            config_path="config/default.yaml",
            vla_server_url="http://localhost:8777",  # Placeholder
        )
        self._bridge.initialize()
        self._initialized = True
        logger.info("LIBERO bridge initialized")

    def run_episode(self, **kwargs) -> Dict[str, Any]:
        """Run a single episode."""
        if not self._initialized:
            raise RuntimeError("Call initialize() first")
        return self._bridge.run_episode(**kwargs)

    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run evaluation across multiple episodes."""
        if not self._initialized:
            raise RuntimeError("Call initialize() first")
        return self._bridge.run_evaluation(**kwargs)

    def close(self):
        """Clean up and close the environment."""
        if self._bridge is not None:
            self._bridge.close()
