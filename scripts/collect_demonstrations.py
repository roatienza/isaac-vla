#!/usr/bin/env python3
"""
Demonstration Data Collection Script
======================================

Collects demonstration data for fine-tuning OpenVLA-OFT on the
Franka kitchen environment. Supports:

- Keyboard teleoperation for data collection
- Automatic episode management
- RLDS format output for OpenVLA-OFT fine-tuning

Usage:
    <isaac_sim_install>/python.sh /abs/path/to/isaac-vla/scripts/collect_demonstrations.py \
        --task "pick up the red block" \
        --num-episodes 10 \
        --output-dir ./data/demonstrations

Controls:
    W/S: Move EE forward/backward (X)
    A/D: Move EE left/right (Y)
    Q/E: Move EE up/down (Z)
    R/F: Rotate EE (roll)
    T/G: Rotate EE (pitch)
    Y/H: Rotate EE (yaw)
    Space: Toggle gripper
    Enter: Save episode
    Escape: Discard episode
    P: Pause/resume

NOTE: When using Isaac Sim's python.sh, use an ABSOLUTE path to this script.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Resolve project root from this script's absolute location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to path (must be absolute since python.sh may change cwd)
sys.path.insert(0, str(_PROJECT_ROOT))

# Also set working directory to project root so config files resolve correctly
os.chdir(str(_PROJECT_ROOT))

from src.data_collector import DataCollector
from src.utils import load_config, setup_logging, FRANKA_HOME_JOINTS
import numpy as np

logger = logging.getLogger(__name__)


# Keyboard teleoperation mapping
TELEOP_KEYS = {
    "w": np.array([0.02, 0, 0, 0, 0, 0, 0]),   # Forward
    "s": np.array([-0.02, 0, 0, 0, 0, 0, 0]),  # Backward
    "a": np.array([0, 0.02, 0, 0, 0, 0, 0]),   # Left
    "d": np.array([0, -0.02, 0, 0, 0, 0, 0]),  # Right
    "q": np.array([0, 0, 0.02, 0, 0, 0, 0]),   # Up
    "e": np.array([0, 0, -0.02, 0, 0, 0, 0]),  # Down
    "r": np.array([0, 0, 0, 0.05, 0, 0, 0]),   # Roll+
    "f": np.array([0, 0, 0, -0.05, 0, 0, 0]),  # Roll-
    "t": np.array([0, 0, 0, 0, 0.05, 0, 0]),   # Pitch+
    "g": np.array([0, 0, 0, 0, -0.05, 0, 0]),  # Pitch-
    "y": np.array([0, 0, 0, 0, 0, 0.05, 0]),   # Yaw+
    "h": np.array([0, 0, 0, 0, 0, -0.05, 0]),  # Yaw-
    "space": "toggle_gripper",
}


def main():
    parser = argparse.ArgumentParser(description="Collect demonstration data")
    parser.add_argument("--task", type=str, required=True,
                        help="Task description for the demonstration")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to collect")
    parser.add_argument("--output-dir", type=str, default="./data/demonstrations",
                        help="Output directory for saved data")
    parser.add_argument("--format", type=str, default="rlds",
                        choices=["rlds", "hdf5"],
                        help="Data format")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file path")
    args = parser.parse_args()

    # Setup
    logger = setup_logging("data-collector", "INFO")
    config = load_config(args.config)

    # Initialize data collector
    collector = DataCollector(
        config_path=str(_PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else args.config,
        save_dir=args.output_dir,
        format=args.format,
    )

    logger.info(f"Collecting {args.num_episodes} episodes for task: '{args.task}'")
    logger.info("Controls:")
    logger.info("  W/S: Forward/Backward | A/D: Left/Right | Q/E: Up/Down")
    logger.info("  R/F: Roll | T/G: Pitch | Y/H: Yaw | Space: Gripper")
    logger.info("  Enter: Save episode | Escape: Discard | P: Pause")

    for ep in range(args.num_episodes):
        logger.info(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        logger.info("Press Enter to start, Escape to skip...")

        # In a real implementation, this would integrate with Isaac Sim's
        # keyboard input and simulation loop. Here we provide the structure.

        collector.start_episode(args.task)

        # ... simulation loop with keyboard teleoperation ...
        # Each step would call:
        #   collector.record_step(
        #       third_person_image=tp_image,
        #       wrist_image=wr_image,
        #       proprioception=proprio,
        #       action=teleop_action,
        #       gripper=gripper_width,
        #   )

        # For now, mark as incomplete
        logger.info("Episode recording (requires Isaac Sim integration)")

    # Print summary
    summary = collector.get_episode_summary()
    logger.info(f"\nCollection summary: {summary}")

    collector.close()


if __name__ == "__main__":
    main()