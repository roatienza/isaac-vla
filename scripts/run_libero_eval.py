"""
LIBERO Evaluation Script
========================

Run OpenVLA-OFT evaluation on LIBERO MuJoCo environments.

This script implements the same evaluation pipeline as the reference
OpenVLA-OFT implementation:
https://github.com/roatienza/openvla-oft/experiments/robot/libero/run_libero_eval.py

Key features:
- Supports all LIBERO task suites (spatial, object, goal, 10, 90)
- Configurable number of episodes per task
- Video recording of rollouts
- Results saved as JSON with per-task and overall metrics
- HTTP mode (via VLA server) or embedded mode (direct model loading)

Usage:
    # Evaluate on a single task
    python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

    # Evaluate on all tasks in a suite
    python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks

    # Evaluate on all suites
    python scripts/run_libero_eval.py --all-suites

    # Run with VLA server
    python scripts/run_libero_eval.py --vla-server http://localhost:8777

    # Record videos
    python scripts/run_libero_eval.py --record-video

    # Embedded mode (no HTTP server needed)
    python scripts/run_libero_eval.py --embedded

Prerequisites:
    1. Install LIBERO: pip install libero
    2. Download datasets: python benchmark_scripts/download_libero_datasets.py
    3. Start VLA server (unless using --embedded):
       python scripts/run_vla_server.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Resolve project root from this script's absolute location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to path (must be absolute since python.sh may change cwd)
sys.path.insert(0, str(_PROJECT_ROOT))

# Also set working directory to project root so config files resolve correctly
os.chdir(str(_PROJECT_ROOT))

import numpy as np

from src.libero_bridge import LIBEROBridge, TASK_MAX_STEPS
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def run_single_task_evaluation(
    task_suite: str,
    task_id: int,
    num_episodes: int = 10,
    vla_server_url: str = "http://localhost:8777",
    record_video: bool = False,
    config_path: str = "config/default.yaml",
) -> Dict[str, Any]:
    """Run evaluation on a single LIBERO task.

    Args:
        task_suite: LIBERO task suite name (e.g., "libero_spatial")
        task_id: Task ID within the suite (0-9 for standard suites)
        num_episodes: Number of episodes to evaluate
        vla_server_url: URL of the VLA server
        record_video: Whether to record video frames
        config_path: Path to configuration YAML file

    Returns:
        Dict with evaluation results
    """
    # Load config and override with CLI args
    config = load_config(config_path)
    config["libero"] = {
        "task_suite": task_suite,
        "task_id": task_id,
        "num_episodes": num_episodes,
        "camera_heights": 256,
        "camera_widths": 256,
        "center_crop": True,
        "target_size": [224, 224],
        "num_steps_wait": 10,
    }

    # Save temporary config
    temp_config_path = "config/temp_libero.yaml"
    import yaml

    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    try:
        # Create bridge
        bridge = LIBEROBridge(
            config_path=temp_config_path,
            vla_server_url=vla_server_url,
        )
        bridge.initialize()

        # Run evaluation
        results = bridge.run_evaluation(
            num_episodes=num_episodes,
            record_video=record_video,
        )

        bridge.close()
        return results

    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def run_suite_evaluation(
    task_suite: str,
    num_tasks: int = 10,
    num_episodes: int = 10,
    vla_server_url: str = "http://localhost:8777",
    record_video: bool = False,
    config_path: str = "config/default.yaml",
) -> Dict[str, Any]:
    """Run evaluation on all tasks in a LIBERO suite.

    Args:
        task_suite: LIBERO task suite name
        num_tasks: Number of tasks to evaluate (default: 10)
        num_episodes: Number of episodes per task
        vla_server_url: URL of the VLA server
        record_video: Whether to record video frames
        config_path: Path to configuration YAML file

    Returns:
        Dict with suite-level evaluation results
    """
    suite_results = {
        "task_suite": task_suite,
        "tasks": [],
        "overall_success_rate": 0.0,
        "total_episodes": 0,
        "total_successes": 0,
    }

    total_episodes = 0
    total_successes = 0

    for task_id in range(num_tasks):
        logger.info("=" * 60)
        logger.info(f"Evaluating task {task_id}/{num_tasks} in {task_suite}")
        logger.info("=" * 60)

        try:
            task_result = run_single_task_evaluation(
                task_suite=task_suite,
                task_id=task_id,
                num_episodes=num_episodes,
                vla_server_url=vla_server_url,
                record_video=record_video,
                config_path=config_path,
            )

            suite_results["tasks"].append(task_result)
            total_episodes += num_episodes
            total_successes += task_result["success_count"]

        except Exception as e:
            logger.error(f"Failed to evaluate task {task_id}: {e}")
            suite_results["tasks"].append({
                "task_id": task_id,
                "error": str(e),
                "success_rate": 0.0,
            })

    suite_results["overall_success_rate"] = (
        total_successes / total_episodes if total_episodes > 0 else 0.0
    )
    suite_results["total_episodes"] = total_episodes
    suite_results["total_successes"] = total_successes

    return suite_results


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO Evaluation for OpenVLA-OFT"
    )

    # Task selection
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_spatial",
        help="LIBERO task suite name (libero_spatial, libero_object, libero_goal, libero_10, libero_90)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task ID within the suite (0-9 for standard suites)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all tasks in the specified suite",
    )
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="Evaluate all LIBERO suites",
    )

    # Evaluation settings
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes per task (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: auto from TASK_MAX_STEPS)",
    )

    # VLA server settings
    parser.add_argument(
        "--vla-server",
        type=str,
        default="http://localhost:8777",
        help="URL of the VLA server (default: http://localhost:8777)",
    )
    parser.add_argument(
        "--embedded",
        action="store_true",
        help="Use embedded mode (no HTTP server needed)",
    )

    # Output settings
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video frames during evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/libero_results",
        help="Directory to save results (default: data/libero_results)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration YAML file",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define suites to evaluate
    if args.all_suites:
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    else:
        suites = [args.task_suite]

    # Run evaluation
    start_time = time.time()
    all_results = {}

    for suite in suites:
        logger.info("=" * 60)
        logger.info(f"Evaluating suite: {suite}")
        logger.info("=" * 60)

        if args.all_tasks:
            suite_result = run_suite_evaluation(
                task_suite=suite,
                num_tasks=10,
                num_episodes=args.num_episodes,
                vla_server_url=args.vla_server,
                record_video=args.record_video,
                config_path=args.config,
            )
        else:
            suite_result = run_single_task_evaluation(
                task_suite=suite,
                task_id=args.task_id,
                num_episodes=args.num_episodes,
                vla_server_url=args.vla_server,
                record_video=args.record_video,
                config_path=args.config,
            )

        all_results[suite] = suite_result

    # Save results
    elapsed_time = time.time() - start_time
    results_path = output_dir / "evaluation_results.json"

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Total time: {elapsed_time:.1f}s")
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)

    # Print summary
    for suite, result in all_results.items():
        if "overall_success_rate" in result:
            logger.info(
                f"{suite}: {result['overall_success_rate']:.1%} success rate "
                f"({result['total_successes']}/{result['total_episodes']} episodes)"
            )
        else:
            logger.info(
                f"{suite} (task {result.get('task_id', 'N/A')}): "
                f"{result['success_rate']:.1%} success rate "
                f"({result['success_count']}/{result['num_episodes']} episodes)"
            )


if __name__ == "__main__":
    main()
