#!/usr/bin/env python3
"""
Sim Bridge Launcher
====================

Launch the Isaac Sim bridge server that connects the VLA model
to the Franka robot in the kitchen environment.

Usage:
    # With GUI visualization (default):
    <isaac_sim>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --config config/default.yaml

    # Headless (no GUI):
    <isaac_sim>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --headless

    # Run a specific task and watch the robot:
    <isaac_sim>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --task "pick up the red block"

    # With VLA server on a separate machine:
    <isaac_sim>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --vla-url http://192.168.1.100:8777

    # Save video of the episode:
    <isaac_sim>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --task "pick up the red block" --save-video

    # Interactive mode with command prompt:
    <isaac_sim>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --interactive

NOTE: This script must be run with Isaac Sim's Python interpreter.
      Use an ABSOLUTE path to this script, since Isaac Sim's python.sh
      changes the working directory:

    # CORRECT (absolute path):
    ~/isaacsim/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py

    # WRONG (relative path from a different directory):
    ~/isaacsim/python.sh scripts/run_sim_bridge.py
"""

import argparse
import logging
import os
import select
import sys
from pathlib import Path

# Resolve project root from this script's absolute location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to path (must be absolute since python.sh may change cwd)
sys.path.insert(0, str(_PROJECT_ROOT))

# Also set working directory to project root so config files resolve correctly
os.chdir(str(_PROJECT_ROOT))

from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Launch Isaac Sim Bridge")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--vla-url", type=str, default="http://localhost:8777",
                        help="VLA server URL")
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac Sim headless (no GUI)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task instruction to run (e.g., 'pick up the red block')")
    parser.add_argument("--task-name", type=str, default=None,
                        help="Task name from kitchen_tasks.yaml (e.g., 'pick_red_block')")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type instructions and watch the robot execute them")
    parser.add_argument("--save-video", action="store_true",
                        help="Save video of the episode to data/evaluation_videos/")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory to save videos (default: data/evaluation_videos/)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps per episode (overrides config)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("sim-bridge", args.log_level)

    # Load config (resolve relative to project root)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    config = load_config(str(config_path))

    # Override headless
    if args.headless:
        config.setdefault("sim_bridge", {}).setdefault("isaac_sim", {})["headless"] = True

    logger.info("Initializing Isaac Sim bridge...")

    # Import sim bridge (requires Isaac Sim)
    from src.sim_bridge import SimBridge

    # Create and initialize bridge
    bridge = SimBridge(
        config_path=str(config_path),
        vla_server_url=args.vla_url,
    )
    bridge.initialize()

    # Set up video recording if requested
    if args.save_video:
        video_dir = args.video_dir or str(_PROJECT_ROOT / "data" / "evaluation_videos")
        os.makedirs(video_dir, exist_ok=True)
        bridge.enable_video_recording(video_dir)
        logger.info(f"Video recording enabled: {video_dir}")

    logger.info("Sim bridge initialized. Ready for commands.")

    # Determine what to do
    instruction = None

    if args.task_name:
        # Load task from kitchen_tasks.yaml
        task_config_path = _PROJECT_ROOT / "config" / "kitchen_tasks.yaml"
        if task_config_path.exists():
            task_config = load_config(str(task_config_path))
            tasks = task_config.get("tasks", {})
            if args.task_name in tasks:
                instruction = tasks[args.task_name]["description"]
                logger.info(f"Loaded task '{args.task_name}': {instruction}")
            else:
                logger.error(f"Task '{args.task_name}' not found in kitchen_tasks.yaml")
                logger.info(f"Available tasks: {list(tasks.keys())}")
                bridge.close()
                return
        else:
            logger.error("kitchen_tasks.yaml not found")
            bridge.close()
            return

    elif args.task:
        instruction = args.task

    if instruction:
        # Run a specific task and visualize the robot
        logger.info(f"Running task: '{instruction}'")
        logger.info("Watch the Isaac Sim window to see the robot execute the task.")

        max_steps = args.max_steps
        result = bridge.run_episode(
            instruction=instruction,
            max_steps=max_steps,
        )
        logger.info(f"Task result: {result}")

        if args.save_video:
            bridge.stop_video_recording()
            logger.info("Video saved.")

    elif args.interactive:
        # Interactive mode: type instructions and watch
        logger.info("=" * 60)
        logger.info("INTERACTIVE MODE")
        logger.info("=" * 60)
        logger.info("Type a natural language instruction and press Enter to see")
        logger.info("the robot execute it in the Isaac Sim window.")
        logger.info("Type 'quit' or 'exit' to stop.")
        logger.info("Type 'reset' to reset the scene.")
        logger.info("Type 'status' to see robot state.")
        logger.info("")

        while True:
            # Non-blocking input to keep GUI responsive
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                try:
                    user_input = sys.stdin.readline().strip()
                except (EOFError, KeyboardInterrupt):
                    break
            else:
                # No input — step simulation to keep GUI responsive
                bridge.step_simulation()
                continue

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                logger.info("Exiting interactive mode.")
                break

            if user_input.lower() == "reset":
                bridge.reset()
                logger.info("Scene reset.")
                continue

            if user_input.lower() == "status":
                status = bridge.get_status()
                logger.info(f"Status: {status}")
                continue

            # Run the task
            logger.info(f"Running: '{user_input}'")
            logger.info("Watch the Isaac Sim window...")
            result = bridge.run_episode(
                instruction=user_input,
                max_steps=args.max_steps,
            )
            logger.info(f"Result: steps={result.get('steps', '?')}, "
                       f"success={result.get('success', 'unknown')}")

            if args.save_video:
                bridge.stop_video_recording()

    else:
        # Default: interactive mode with helpful message
        logger.info("=" * 60)
        logger.info("ISAAC SIM BRIDGE — INTERACTIVE MODE")
        logger.info("=" * 60)
        logger.info("")
        logger.info("The Isaac Sim window is now open and showing the kitchen scene.")
        logger.info("You can interact with the robot by typing instructions below.")
        logger.info("")
        logger.info("Examples:")
        logger.info("  pick up the red block")
        logger.info("  place the red block on the plate")
        logger.info("  move the yellow mug to the plate")
        logger.info("")
        logger.info("Commands:")
        logger.info("  reset   — Reset the scene to initial state")
        logger.info("  status  — Show robot state")
        logger.info("  quit    — Exit")
        logger.info("")

        while True:
            # Non-blocking input to keep GUI responsive
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                try:
                    user_input = sys.stdin.readline().strip()
                except (EOFError, KeyboardInterrupt):
                    break
            else:
                # No input — step simulation to keep GUI responsive
                bridge.step_simulation()
                continue

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower() == "reset":
                bridge.reset()
                logger.info("Scene reset.")
                continue

            if user_input.lower() == "status":
                status = bridge.get_status()
                for k, v in status.items():
                    logger.info(f"  {k}: {v}")
                continue

            # Run the task with visualization
            logger.info(f"Executing: '{user_input}' — watch the Isaac Sim window...")
            result = bridge.run_episode(
                instruction=user_input,
                max_steps=args.max_steps,
            )
            logger.info(f"Done: steps={result.get('steps', '?')}")

            if args.save_video:
                bridge.stop_video_recording()

    bridge.close()
    logger.info("Sim bridge closed.")


if __name__ == "__main__":
    main()