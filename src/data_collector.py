"""
Data Collector: Demonstration Recording for Fine-Tuning
=========================================================

Records demonstration episodes in RLDS/OXE format compatible with
OpenVLA-OFT fine-tuning. Supports:

- Automatic recording during teleoperation
- Keyboard teleoperation interface
- Episode management (start, stop, save)
- Data augmentation pipeline
- Conversion to OpenVLA-OFT training format

Usage:
    collector = DataCollector(config_path="config/default.yaml")
    collector.initialize()
    collector.start_episode("pick up the red block")
    # ... teleoperate ...
    collector.stop_episode(save=True)
    collector.close()
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import crop_and_resize, load_config, setup_logging

logger = logging.getLogger(__name__)


class DataCollector:
    """Records demonstration data for VLA fine-tuning.

    Args:
        config_path: Path to configuration YAML file
        save_dir: Directory to save recorded data
        format: "rlds" or "hdf5"
    """

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        save_dir: Optional[str] = None,
        format: str = "rlds",
    ):
        self.config = load_config(config_path)
        dc_config = self.config.get("data_collection", {})

        self.save_dir = Path(save_dir or dc_config.get("save_dir", "./data/demonstrations"))
        self.format = format or dc_config.get("format", "rlds")
        self.save_images = dc_config.get("save_images", True)
        self.image_resolution = tuple(dc_config.get("image_resolution", [256, 256]))
        self.save_proprio = dc_config.get("save_proprio", True)
        self.save_actions = dc_config.get("save_actions", True)
        self.save_frequency = dc_config.get("save_frequency", 1)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Episode state
        self._current_episode: Optional[Episode] = None
        self._episode_count = 0

    def start_episode(self, task_description: str):
        """Start recording a new episode.

        Args:
            task_description: Natural language description of the task
        """
        self._current_episode = Episode(
            task_description=task_description,
            episode_id=self._episode_count,
        )
        self._episode_count += 1
        logger.info(f"Started episode {self._episode_count}: '{task_description}'")

    def record_step(
        self,
        third_person_image: np.ndarray,
        wrist_image: Optional[np.ndarray],
        proprioception: np.ndarray,
        action: np.ndarray,
        gripper: float,
    ):
        """Record a single step in the current episode.

        Args:
            third_person_image: (H, W, 3) uint8 image from third-person camera
            wrist_image: (H, W, 3) uint8 image from wrist camera (optional)
            proprioception: (8,) float32 proprioceptive state
            action: (7,) float32 action [dx, dy, dz, droll, dpitch, dyaw, gripper]
            gripper: float gripper width
        """
        if self._current_episode is None:
            logger.warning("No active episode. Call start_episode() first.")
            return

        # Preprocess images
        tp_processed = crop_and_resize(third_person_image, self.image_resolution)
        wr_processed = (
            crop_and_resize(wrist_image, (128, 128)) if wrist_image is not None
            else np.zeros((128, 128, 3), dtype=np.uint8)
        )

        step = StepData(
            third_person_image=tp_processed,
            wrist_image=wr_processed,
            proprioception=proprioception.copy(),
            action=action.copy(),
            gripper=gripper,
            timestamp=time.time(),
        )

        self._current_episode.steps.append(step)

    def stop_episode(self, save: bool = True, success: bool = False) -> Optional[str]:
        """Stop recording the current episode.

        Args:
            save: Whether to save the episode data
            success: Whether the episode was successful

        Returns:
            Path to saved episode directory, or None if not saved
        """
        if self._current_episode is None:
            logger.warning("No active episode to stop.")
            return None

        self._current_episode.success = success
        self._current_episode.end_time = time.time()

        save_path = None
        if save:
            save_path = self._save_episode(self._current_episode)

        logger.info(
            f"Episode {self._current_episode.episode_id} completed: "
            f"{len(self._current_episode.steps)} steps, "
            f"success={success}"
        )

        self._current_episode = None
        return save_path

    def _save_episode(self, episode: "Episode") -> str:
        """Save episode data to disk.

        Returns:
            Path to saved episode directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_dir = self.save_dir / f"episode_{episode.episode_id:04d}_{timestamp}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "episode_id": episode.episode_id,
            "task_description": episode.task_description,
            "num_steps": len(episode.steps),
            "success": episode.success,
            "start_time": episode.start_time,
            "end_time": episode.end_time,
            "format": self.format,
            "image_resolution": list(self.image_resolution),
        }
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save step data
        if self.format == "rlds":
            self._save_rlds(episode, episode_dir)
        elif self.format == "hdf5":
            self._save_hdf5(episode, episode_dir)
        else:
            raise ValueError(f"Unknown format: {self.format}")

        logger.info(f"Saved episode to {episode_dir}")
        return str(episode_dir)

    def _save_rlds(self, episode: "Episode", episode_dir: Path):
        """Save episode in RLDS format (compatible with OpenVLA-OFT fine-tuning).

        Creates:
            - actions.npy: (N, 7) float32 actions
            - observations/images/third_person/: PNG images
            - observations/images/wrist/: PNG images
            - observations/proprioception.npy: (N, 8) float32
            - language_instruction.txt: Task description
        """
        from PIL import Image

        n_steps = len(episode.steps)

        # Actions
        actions = np.stack([s.action for s in episode.steps])
        np.save(episode_dir / "actions.npy", actions)

        # Proprioception
        proprios = np.stack([s.proprioception for s in episode.steps])
        np.save(episode_dir / "proprioception.npy", proprios)

        # Images
        if self.save_images:
            tp_dir = episode_dir / "observations" / "images" / "third_person"
            wr_dir = episode_dir / "observations" / "images" / "wrist"
            tp_dir.mkdir(parents=True, exist_ok=True)
            wr_dir.mkdir(parents=True, exist_ok=True)

            for i, step in enumerate(episode.steps):
                tp_img = Image.fromarray(step.third_person_image)
                tp_img.save(tp_dir / f"frame_{i:06d}.png")

                wr_img = Image.fromarray(step.wrist_image)
                wr_img.save(wr_dir / f"frame_{i:06d}.png")

        # Language instruction
        with open(episode_dir / "language_instruction.txt", "w") as f:
            f.write(episode.task_description)

    def _save_hdf5(self, episode: "Episode", episode_dir: Path):
        """Save episode in HDF5 format."""
        import h5py

        n_steps = len(episode.steps)
        h5_path = episode_dir / "data.h5"

        with h5py.File(h5_path, "w") as f:
            # Actions
            actions = np.stack([s.action for s in episode.steps])
            f.create_dataset("actions", data=actions)

            # Proprioception
            proprios = np.stack([s.proprioception for s in episode.steps])
            f.create_dataset("proprioception", data=proprios)

            # Images
            if self.save_images:
                tp_images = np.stack([s.third_person_image for s in episode.steps])
                wr_images = np.stack([s.wrist_image for s in episode.steps])
                f.create_dataset("third_person_images", data=tp_images)
                f.create_dataset("wrist_images", data=wr_images)

            # Metadata
            f.attrs["task_description"] = episode.task_description
            f.attrs["success"] = episode.success
            f.attrs["num_steps"] = n_steps

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded episodes."""
        episodes = list(self.save_dir.glob("episode_*"))
        total_steps = 0
        successful = 0

        for ep_dir in episodes:
            meta_path = ep_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                total_steps += meta["num_steps"]
                if meta.get("success", False):
                    successful += 1

        return {
            "total_episodes": len(episodes),
            "total_steps": total_steps,
            "successful_episodes": successful,
            "save_dir": str(self.save_dir),
        }

    def close(self):
        """Finalize and close the data collector."""
        if self._current_episode is not None:
            logger.warning("Unclosed episode, saving...")
            self.stop_episode(save=True)


class StepData:
    """Data for a single step in an episode."""

    def __init__(
        self,
        third_person_image: np.ndarray,
        wrist_image: np.ndarray,
        proprioception: np.ndarray,
        action: np.ndarray,
        gripper: float,
        timestamp: float,
    ):
        self.third_person_image = third_person_image
        self.wrist_image = wrist_image
        self.proprioception = proprioception
        self.action = action
        self.gripper = gripper
        self.timestamp = timestamp


class Episode:
    """A recorded demonstration episode."""

    def __init__(self, task_description: str, episode_id: int):
        self.task_description = task_description
        self.episode_id = episode_id
        self.steps: List[StepData] = []
        self.success = False
        self.start_time = time.time()
        self.end_time: Optional[float] = None