#!/usr/bin/env python3
"""
Task Evaluation Runner
========================

Runs automated evaluation of VLA performance on kitchen tasks.

Usage:
    python scripts/evaluate_tasks.py --tasks pick_red_block place_red_on_plate
    python scripts/evaluate_tasks.py --all --num-episodes 50
    python scripts/evaluate_tasks.py --task pick_red_block --num-episodes 10
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

from src.evaluator import TaskEvaluator
from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA on kitchen tasks")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all tasks")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes per task")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file path")
    parser.add_argument("--task-config", type=str, default="config/kitchen_tasks.yaml",
                        help="Task config file path")
    parser.add_argument("--output", type=str, default="./evaluation_report.json",
                        help="Output report path")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save evaluation videos")
    args = parser.parse_args()

    # Setup
    logger = setup_logging("evaluator", "INFO")

    # Resolve config paths relative to project root
    config_path = str(_PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else args.config
    task_config_path = str(_PROJECT_ROOT / args.task_config) if not Path(args.task_config).is_absolute() else args.task_config

    # Create evaluator
    evaluator = TaskEvaluator(
        config_path=config_path,
        task_config_path=task_config_path,
    )

    # Determine tasks to evaluate
    if args.all:
        task_filter = None  # Evaluate all tasks
    elif args.tasks:
        task_filter = args.tasks
    else:
        logger.error("Specify --tasks or --all")
        return

    # Run evaluation
    logger.info(f"Starting evaluation with {args.num_episodes} episodes per task")

    # Note: Actual evaluation requires Isaac Sim integration
    # This provides the evaluation framework and success criteria
    results = evaluator.evaluate_all(
        num_episodes=args.num_episodes,
        task_filter=task_filter,
    )

    # Print results
    evaluator.print_results(results)

    # Generate report
    report_path = evaluator.generate_report(results, args.output)
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()