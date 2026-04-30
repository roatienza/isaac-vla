#!/usr/bin/env python3
"""
Quick Start Demo: OpenVLA-OFT on Franka in Isaac Sim
=======================================================

A minimal example showing how to:
1. Initialize the Isaac-VLA system
2. Set up a kitchen scene with Franka
3. Run a pick-and-place task using natural language

This is the simplest way to get started with isaac-vla.

Usage (embedded mode — single GPU):
    <isaac_sim>/python.sh /abs/path/to/isaac-vla/scripts/quick_start.py --instruction "pick up the red block"

Usage (remote mode — VLA server on separate machine):
    # Terminal 1: Start VLA server
    python scripts/run_vla_server.py --port 8777

    # Terminal 2: Start sim bridge (with Isaac Sim Python)
    <isaac_sim>/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py --vla-url http://localhost:8777

    # Terminal 3: Run quick start
    python scripts/quick_start.py --instruction "pick up the red block" --remote

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

from src.utils import setup_logging

logger = setup_logging("quick-start", "INFO")


def run_embedded(instruction: str, config_path: str, max_steps: int = 200):
    """Run in embedded mode (VLA model loaded directly).

    This is the simplest mode — everything runs in one process.
    Requires Isaac Sim and the VLA model on the same GPU.
    """
    logger.info("=" * 60)
    logger.info("ISAAC-VLA Quick Start (Embedded Mode)")
    logger.info("=" * 60)
    logger.info(f"Instruction: '{instruction}'")
    logger.info(f"Max steps: {max_steps}")
    logger.info("")

    # Step 1: Import and initialize
    logger.info("Step 1: Initializing system...")
    from src.api import IsaacVLAClient

    # Resolve config path relative to project root
    if not Path(config_path).is_absolute():
        config_path = str(_PROJECT_ROOT / config_path)

    client = IsaacVLAClient(mode="embedded", config_path=config_path)
    client.initialize()
    logger.info("✓ System initialized")

    # Step 2: Reset scene
    logger.info("Step 2: Resetting scene...")
    client.reset()
    logger.info("✓ Scene reset")

    # Step 3: Run task
    logger.info(f"Step 3: Running task '{instruction}'...")
    result = client.run_task(instruction, max_steps=max_steps)
    logger.info(f"✓ Task completed in {result.get('steps', '?')} steps")

    # Step 4: Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Instruction: {result.get('instruction', instruction)}")
    logger.info(f"  Steps taken: {result.get('steps', '?')}")
    logger.info(f"  Avg inference time: {result.get('avg_inference_time', 0):.3f}s")
    logger.info("")

    # Clean up
    client.close()
    logger.info("Done!")


def run_remote(instruction: str, vla_url: str, bridge_url: str, max_steps: int = 200):
    """Run in remote mode (VLA server and sim bridge as separate processes).

    This mode is for distributed setups where the VLA server runs on
    a GPU machine and the sim bridge runs on the Isaac Sim machine.
    """
    logger.info("=" * 60)
    logger.info("ISAAC-VLA Quick Start (Remote Mode)")
    logger.info("=" * 60)
    logger.info(f"Instruction: '{instruction}'")
    logger.info(f"VLA Server: {vla_url}")
    logger.info(f"Sim Bridge: {bridge_url}")
    logger.info("")

    from src.api import RemoteVLAClient

    client = RemoteVLAClient(base_url=bridge_url)

    # Check connectivity
    logger.info("Checking connectivity...")
    try:
        status = client.get_status()
        logger.info(f"✓ Sim bridge connected: {status}")
    except Exception as e:
        logger.error(f"✗ Cannot connect to sim bridge: {e}")
        logger.error("Make sure the sim bridge is running:")
        logger.error(f"  <isaac_sim>/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py --vla-url {vla_url}")
        return

    # Run task
    logger.info(f"Running task '{instruction}'...")
    result = client.run_task(instruction, max_steps=max_steps)
    logger.info(f"Task result: {result}")


def main():
    parser = argparse.ArgumentParser(description="Isaac-VLA Quick Start Demo")
    parser.add_argument("--instruction", type=str, default="pick up the red block",
                        help="Natural language instruction for the robot")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file path (relative to project root)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per episode")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote mode (VLA server + sim bridge)")
    parser.add_argument("--vla-url", type=str, default="http://localhost:8777",
                        help="VLA server URL (remote mode)")
    parser.add_argument("--bridge-url", type=str, default="http://localhost:8889",
                        help="Sim bridge URL (remote mode)")
    args = parser.parse_args()

    if args.remote:
        run_remote(args.instruction, args.vla_url, args.bridge_url, args.max_steps)
    else:
        run_embedded(args.instruction, args.config, args.max_steps)


if __name__ == "__main__":
    main()