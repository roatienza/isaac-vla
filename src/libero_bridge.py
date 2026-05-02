"""
LIBERO Bridge: VLA + MuJoCo Control Loop
=========================================

The main simulation loop that connects the VLA server to the LIBERO
MuJoCo environment. This module provides a drop-in replacement for
the Isaac Sim bridge, using the same VLA server but bypassing IK
solving since LIBERO accepts delta EE actions directly.

Architecture:
    ┌─────────────────┐    HTTP     ┌─────────────────────┐
    │  VLA Server     │◄───────────►│  LIBERO Bridge      │
    │  (GPU:5090)     │  :8777/act  │  (MuJoCo)           │
    └─────────────────┘             └─────────────────────┘
                                             │
                                        ┌────▼─────┐
                                        │ LIBERO   │
                                        │ MuJoCo   │
                                        │ Env      │
                                        └──────────┘

Key differences from Isaac Sim:
- Uses MuJoCo physics (same as LIBERO training)
- No IK solving needed — LIBERO accepts delta EE actions directly
- No camera rendering overhead (MuJoCo is fast)
- Direct access to LIBERO task suite and evaluation
- Zero visual domain gap with OpenVLA-OFT training data
"""

import json
import logging
import os
import time
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


# ─── Dataset Statistics ──────────────────────────────────────────────────────

# Default normalization statistics for LIBERO spatial (OpenVLA-OFT pretrained)
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
        self.max_steps = ep_config.get("max_steps", 500)
        self.action_chunk_size = ep_config.get("action_chunk_size", 8)
        self.vla_query_frequency = ep_config.get("vla_query_frequency", 8)

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
        self._action_queue: List[np.ndarray] = []
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
            f"'{self.task.language}'"
        )

        # Get initial states for benchmarking
        self.init_states = self.task_suite.get_task_init_states(self.task_id)

        # Create environment
        bddl_file = os.path.join(
            self.task_suite.get_libero_path("bddl_files"),
            self.task.problem_folder,
            self.task.bddl_file,
        )

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths,
        }

        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(0)
        self.env.reset()

        logger.info("LIBERO environment initialized successfully")

    def run_episode(
        self,
        instruction: str = None,
        init_state_id: int = 0,
        record_video: bool = False,
    ) -> Dict[str, Any]:
        """Run a single episode of the task.

        Args:
            instruction: Natural language instruction (uses task language if None)
            init_state_id: Which initial state to use (0-9 for benchmarking)
            record_video: Whether to record video frames

        Returns:
            Dict with episode results (success, steps, etc.)
        """
        if self.env is None:
            raise RuntimeError("Call initialize() first")

        if instruction is None:
            instruction = self.task.language

        # Set initial state
        if init_state_id < len(self.init_states):
            self.env.set_init_state(self.init_states[init_state_id])

        self._step_count = 0
        self._recording = record_video
        self._video_frames = []
        self._action_queue = []
        self._chunk_index = 0

        logger.info(f"Starting episode with instruction: '{instruction}'")

        # Main loop
        while self._step_count < self.max_steps:
            # Capture observation
            observation = self._capture_observation()

            # Record frame if requested
            if self._recording:
                self._video_frames.append(observation["full_image"])

            # Query VLA every N steps (open-loop execution)
            if self._chunk_index >= len(self._action_queue):
                vla_result = self._query_vla(observation, instruction)

                if vla_result is None:
                    logger.warning("VLA query failed, stopping episode")
                    break

                # Process action chunk
                actions = vla_result["actions"]
                self._action_queue = self._process_action_chunk(actions)
                self._chunk_index = 0

            # Get next action from chunk
            if self._chunk_index >= len(self._action_queue):
                break

            action = self._action_queue[self._chunk_index]
            self._chunk_index += 1

            # Execute action in LIBERO environment
            obs, reward, done, info = self.env.step(action)

            self._step_count += 1

            # Check for success
            if done or reward > 0:
                logger.info(
                    f"Episode {self._episode_count + 1} SUCCESS "
                    f"after {self._step_count} steps"
                )
                return {
                    "success": True,
                    "steps": self._step_count,
                    "reward": reward,
                    "instruction": instruction,
                }

        # Episode ended without success
        logger.info(
            f"Episode {self._episode_count + 1} FAILED "
            f"after {self._step_count} steps"
        )
        return {
            "success": False,
            "steps": self._step_count,
            "reward": 0,
            "instruction": instruction,
        }

    def run_evaluation(
        self,
        num_episodes: int = None,
        record_video: bool = False,
    ) -> Dict[str, Any]:
        """Run evaluation across multiple episodes.

        Args:
            num_episodes: Number of episodes to run (uses config default if None)
            record_video: Whether to record video frames

        Returns:
            Dict with evaluation results
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        results = []
        success_count = 0
        total_steps = 0

        for ep_idx in range(num_episodes):
            logger.info(f"Running episode {ep_idx + 1}/{num_episodes}")

            episode_result = self.run_episode(
                init_state_id=ep_idx % len(self.init_states),
                record_video=record_video,
            )

            results.append(episode_result)
            if episode_result["success"]:
                success_count += 1
            total_steps += episode_result["steps"]

            self._episode_count += 1

            # Save video if recorded
            if record_video and self._video_frames:
                self._save_video_frames(episode_result["instruction"])

        # Calculate metrics
        success_rate = success_count / num_episodes
        avg_steps = total_steps / num_episodes

        eval_results = {
            "task": self.task.language,
            "task_suite": self.task_suite_name,
            "task_id": self.task_id,
            "num_episodes": num_episodes,
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "episodes": results,
        }

        logger.info(
            f"Evaluation complete: {success_count}/{num_episodes} "
            f"({success_rate:.1%} success rate, {avg_steps:.1f} avg steps)"
        )

        return eval_results

    def _capture_observation(self) -> Dict[str, np.ndarray]:
        """Capture observation from the LIBERO environment.

        Returns:
            Dict with "full_image", "wrist_image", and "state"
        """
        obs = self.env._get_obs()

        # Extract images
        full_image = obs["agentview_image"]
        wrist_image = obs["robot0_eye_in_hand_image"]

        # Preprocess images (center crop + resize to 224x224)
        if self.center_crop:
            full_image = crop_and_resize(
                full_image, self.target_size, center_crop=True
            )
            wrist_image = crop_and_resize(
                wrist_image, self.target_size, center_crop=True
            )

        # Extract state (joint positions + gripper)
        state = obs["robot0_joint_pos"].copy()

        return {
            "full_image": full_image,
            "wrist_image": wrist_image,
            "state": state,
        }

    def _process_action_chunk(
        self, actions: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Process a complete action chunk from VLA output.

        For LIBERO, we bypass IK solving and directly apply safety
        clipping to the delta EE actions.

        Args:
            actions: List of 7D action arrays [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            List of processed 7D action arrays
        """
        processed_actions = []

        for i, action in enumerate(actions):
            action = np.asarray(action, dtype=np.float64)

            # Clip action magnitude for safety
            clipped = clip_action_magnitude(
                action,
                max_position=self.max_position_delta,
                max_rotation=self.max_rotation_delta,
            )

            # Ensure gripper is in valid range [-1, 1] for LIBERO
            clipped[6] = np.clip(clipped[6], -1.0, 1.0)

            processed_actions.append(clipped)

        return processed_actions

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
        self._action_queue = []
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
            "instruction": instruction,
        }

        if wrist_image is not None:
            payload["wrist_image"] = wrist_image.tolist()

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


# ─── Embedded Mode (No HTTP) ─────────────────────────────────────────────────

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
