"""
Python API: Unified Interface for Isaac-VLA
==============================================

Provides a high-level Python API for controlling the Franka robot
in Isaac Sim via OpenVLA-OFT. Supports both embedded mode (direct
model loading) and remote mode (HTTP to VLA server + sim bridge).

Usage:
    # Embedded mode (single machine, no HTTP)
    client = IsaacVLAClient(mode="embedded")
    client.initialize()
    result = client.run_task("pick up the red block")

    # Remote mode (separate VLA server and sim bridge)
    client = IsaacVLAClient(mode="remote")
    client.initialize()
    result = client.run_task("pick up the red block")

    # Step-by-step control
    client.reset()
    for step in range(100):
        obs = client.get_observation()
        action = client.get_action(obs, "pick up the red block")
        client.apply_action(action)
        if client.check_success():
            break
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class IsaacVLAClient:
    """Unified Python API for Isaac-VLA.

    Args:
        mode: "embedded" (direct model import) or "remote" (HTTP)
        config_path: Path to configuration YAML file
        vla_server_url: URL of VLA server (remote mode only)
        sim_bridge_url: URL of sim bridge server (remote mode only)
        log_level: Logging level
    """

    def __init__(
        self,
        mode: str = "embedded",
        config_path: str = "config/default.yaml",
        vla_server_url: str = "http://localhost:8777",
        sim_bridge_url: str = "http://localhost:8889",
        log_level: str = "INFO",
    ):
        self.mode = mode
        self.config = load_config(config_path)
        self.vla_server_url = vla_server_url
        self.sim_bridge_url = sim_bridge_url

        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logging(
            name="isaac-vla",
            level=log_level or log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
        )

        # Internal state
        self._bridge = None
        self._initialized = False
        self._current_instruction = ""
        self._step_count = 0
        self._episode_count = 0

    def initialize(self):
        """Initialize the system (load model, start sim, etc.).

        In embedded mode, this loads the VLA model and starts Isaac Sim.
        In remote mode, this just verifies connectivity.
        """
        if self.mode == "embedded":
            from src.sim_bridge import EmbeddedVLABridge

            self._bridge = EmbeddedVLABridge()
            self._bridge.initialize()
        elif self.mode == "remote":
            from src.sim_bridge import SimBridge

            self._bridge = SimBridge(
                config_path=self.config.get("config_path", "config/default.yaml"),
                vla_server_url=self.vla_server_url,
            )
            self._bridge.initialize()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'embedded' or 'remote'.")

        self._initialized = True
        logger.info(f"IsaacVLAClient initialized in {self.mode} mode")

    def run_task(
        self,
        instruction: str,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run a complete task episode.

        Args:
            instruction: Natural language task description
            max_steps: Maximum steps per episode
            callback: Optional callback(step, obs, action)

        Returns:
            Episode result dict
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        self._current_instruction = instruction
        result = self._bridge.run_episode(
            instruction=instruction,
            max_steps=max_steps,
            callback=callback,
        )

        self._episode_count += 1
        return result

    def reset(self):
        """Reset the simulation scene."""
        if self._bridge is not None:
            self._bridge.reset()
        self._step_count = 0

    def get_observation(self) -> Dict[str, Any]:
        """Capture current observation.

        Returns:
            Dict with "full_image", "wrist_image", "state"
        """
        if self._bridge is not None:
            return self._bridge._capture_observation()
        return {}

    def get_action(
        self,
        observation: Dict[str, Any],
        instruction: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get next action from VLA.

        Args:
            observation: Observation dict
            instruction: Task instruction

        Returns:
            Action dict with joint_positions, gripper, etc.
        """
        if self._bridge is not None:
            return self._bridge.action_pipeline.get_next_action()
        return None

    def apply_action(self, action: Dict[str, np.ndarray]):
        """Apply an action to the simulation.

        Args:
            action: Action dict from get_action()
        """
        if self._bridge is not None:
            self._bridge._execute_action(action)

    def step(self, instruction: str) -> Dict[str, Any]:
        """Execute a single step.

        Args:
            instruction: Current task instruction

        Returns:
            Step result dict
        """
        if self._bridge is not None:
            return self._bridge.step_once(instruction)
        return {}

    def check_success(self, task_name: str) -> bool:
        """Check if the current task is completed successfully.

        Args:
            task_name: Name of the task to check

        Returns:
            True if task is completed
        """
        # This requires task-specific success detection
        # See evaluator.py for implementation
        return False

    def close(self):
        """Clean up and close."""
        if self._bridge is not None:
            self._bridge.close()
        self._initialized = False

    # ─── Context Manager ──────────────────────────────────────────────────

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.close()

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def step_count(self) -> int:
        return self._step_count


# ─── Remote API Client ───────────────────────────────────────────────────────

class RemoteVLAClient:
    """Client for the remote sim bridge HTTP API.

    Used when the sim bridge runs as a separate HTTP server.

    Args:
        base_url: URL of the sim bridge server
    """

    def __init__(self, base_url: str = "http://localhost:8889"):
        import requests
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def run_task(self, instruction: str, max_steps: int = 500) -> Dict:
        """Run a task via the remote API."""
        response = self.session.post(
            f"{self.base_url}/run_task",
            json={"instruction": instruction, "max_steps": max_steps},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def reset(self) -> Dict:
        """Reset the simulation via the remote API."""
        response = self.session.post(f"{self.base_url}/reset", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_observation(self) -> Dict:
        """Get current observation via the remote API."""
        response = self.session.get(f"{self.base_url}/observation", timeout=10)
        response.raise_for_status()
        return response.json()

    def step(self, instruction: str) -> Dict:
        """Execute a single step via the remote API."""
        response = self.session.post(
            f"{self.base_url}/step",
            json={"instruction": instruction},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Dict:
        """Get simulation status."""
        response = self.session.get(f"{self.base_url}/status", timeout=5)
        response.raise_for_status()
        return response.json()