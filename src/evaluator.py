"""
Task Evaluator: Automated Evaluation Framework
===============================================

Evaluates VLA performance on kitchen manipulation tasks with:
- Automatic success detection per task
- Multi-condition success criteria
- Metrics tracking (success rate, episode length, action variance)
- Video recording of episodes
- Evaluation report generation

Usage:
    evaluator = TaskEvaluator(config_path="config/default.yaml")
    evaluator.initialize()

    results = evaluator.evaluate_task("pick_red_block", num_episodes=10)
    print(results)

    # Evaluate all tasks
    all_results = evaluator.evaluate_all(num_episodes=50)
    evaluator.generate_report(all_results)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import load_config

logger = logging.getLogger(__name__)


class TaskEvaluator:
    """Evaluates VLA performance on kitchen manipulation tasks.

    Args:
        config_path: Path to configuration YAML file
        task_config_path: Path to kitchen tasks YAML file
    """

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        task_config_path: str = "config/kitchen_tasks.yaml",
    ):
        self.config = load_config(config_path)
        self.task_config = load_config(task_config_path)

        self.eval_config = self.config.get("evaluation", {})
        self.tasks = self.task_config.get("tasks", {})

        # Results storage
        self._results: Dict[str, List[Dict]] = {}

    def initialize(self):
        """Initialize the evaluator (load scene, etc.)."""
        logger.info(f"Initialized evaluator with {len(self.tasks)} tasks")

    def evaluate_task(
        self,
        task_name: str,
        num_episodes: int = 10,
        save_videos: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a single task across multiple episodes.

        Args:
            task_name: Name of the task (must match kitchen_tasks.yaml)
            num_episodes: Number of evaluation episodes
            save_videos: Whether to save episode videos

        Returns:
            Evaluation results dict
        """
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.tasks.keys())}")

        task_def = self.tasks[task_name]
        logger.info(f"Evaluating task '{task_name}': {task_def['description']}")

        results = []
        for ep in range(num_episodes):
            logger.info(f"  Episode {ep + 1}/{num_episodes}")

            # This would be called from the main sim loop
            # For now, we define the interface
            result = {
                "episode": ep,
                "task": task_name,
                "instruction": task_def["description"],
                "success": False,  # Will be set by _check_success
                "steps": 0,
                "actions": [],
            }
            results.append(result)

        # Compute summary statistics
        successes = sum(1 for r in results if r["success"])
        success_rate = successes / num_episodes if num_episodes > 0 else 0.0

        summary = {
            "task": task_name,
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "successful_episodes": successes,
            "results": results,
        }

        self._results[task_name] = results
        return summary

    def evaluate_all(
        self,
        num_episodes: int = 50,
        task_filter: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all (or filtered) tasks.

        Args:
            num_episodes: Episodes per task
            task_filter: List of task names to evaluate (None = all)

        Returns:
            Dict mapping task names to evaluation results
        """
        tasks_to_eval = task_filter or list(self.tasks.keys())
        all_results = {}

        for task_name in tasks_to_eval:
            logger.info(f"Evaluating task: {task_name}")
            result = self.evaluate_task(task_name, num_episodes)
            all_results[task_name] = result

        return all_results

    def check_success(
        self,
        task_name: str,
        scene,
        tolerance: float = 0.05,
    ) -> bool:
        """Check if a task has been completed successfully.

        Args:
            task_name: Name of the task
            scene: KitchenScene instance
            tolerance: Position tolerance in meters

        Returns:
            True if the task success condition is met
        """
        if task_name not in self.tasks:
            return False

        task_def = self.tasks[task_name]
        condition = task_def["success_condition"]
        obj_positions = scene.get_object_positions()

        return self._evaluate_condition(condition, obj_positions, tolerance)

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        obj_positions: Dict[str, np.ndarray],
        tolerance: float,
    ) -> bool:
        """Evaluate a success condition.

        Supports condition types:
            - object_at_location: Object is near a target position
            - object_on_object: Object is on top of another object
            - object_near_object: Object is near another object
            - multi_condition: Multiple conditions must all be true
            - all_objects_on_target: All listed objects are on a target object
        """
        cond_type = condition["type"]

        if cond_type == "object_at_location":
            obj_name = condition["object"]
            target_pos = np.array(condition["target_position"])
            if obj_name not in obj_positions:
                return False
            actual_pos = obj_positions[obj_name]
            distance = np.linalg.norm(actual_pos - target_pos)
            return distance < tolerance

        elif cond_type == "object_on_object":
            obj_name = condition["object"]
            target_name = condition["target_object"]
            if obj_name not in obj_positions or target_name not in obj_positions:
                return False
            obj_pos = obj_positions[obj_name]
            target_pos = obj_positions[target_name]
            # Check XY proximity and Z ordering (object on top)
            xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
            z_diff = obj_pos[2] - target_pos[2]
            return xy_dist < tolerance and z_diff > -0.01

        elif cond_type == "object_near_object":
            obj_name = condition["object"]
            target_name = condition["target_object"]
            if obj_name not in obj_positions or target_name not in obj_positions:
                return False
            obj_pos = obj_positions[obj_name]
            target_pos = obj_positions[target_name]
            distance = np.linalg.norm(obj_pos - target_pos)
            return distance < (tolerance + 0.05)  # Slightly larger tolerance for "near"

        elif cond_type == "multi_condition":
            conditions = condition["conditions"]
            return all(
                self._evaluate_condition(c, obj_positions, tolerance)
                for c in conditions
            )

        elif cond_type == "all_objects_on_target":
            objects = condition["objects"]
            target_name = condition["target_object"]
            return all(
                self._evaluate_condition(
                    {"type": "object_on_object", "object": obj, "target_object": target_name},
                    obj_positions,
                    tolerance,
                )
                for obj in objects
            )

        elif cond_type == "object_at_position":
            # Alias for object_at_location — used by rearrange_blocks task
            obj_name = condition["object"]
            target_pos = np.array(condition["target_position"])
            if obj_name not in obj_positions:
                return False
            actual_pos = obj_positions[obj_name]
            distance = np.linalg.norm(actual_pos - target_pos)
            return distance < tolerance

        else:
            logger.warning(f"Unknown condition type: {cond_type}")
            return False

    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: str = "./evaluation_report.json",
    ) -> str:
        """Generate an evaluation report.

        Args:
            results: Evaluation results from evaluate_all()
            output_path: Path to save the report JSON

        Returns:
            Path to the saved report
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_tasks": len(results),
            "tasks": {},
        }

        for task_name, task_result in results.items():
            report["tasks"][task_name] = {
                "success_rate": task_result["success_rate"],
                "num_episodes": task_result["num_episodes"],
                "successful_episodes": task_result["successful_episodes"],
            }

        # Compute overall statistics
        success_rates = [r["success_rate"] for r in results.values()]
        report["overall"] = {
            "mean_success_rate": np.mean(success_rates) if success_rates else 0.0,
            "std_success_rate": np.std(success_rates) if success_rates else 0.0,
            "min_success_rate": np.min(success_rates) if success_rates else 0.0,
            "max_success_rate": np.max(success_rates) if success_rates else 0.0,
        }

        # Save report
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output}")
        return str(output)

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 60)
        print("ISAAC-VLA EVALUATION RESULTS")
        print("=" * 60)

        for task_name, task_result in results.items():
            sr = task_result["success_rate"]
            n = task_result["num_episodes"]
            s = task_result["successful_episodes"]
            print(f"  {task_name:30s}  {sr:.1%} ({s}/{n})")

        if results:
            rates = [r["success_rate"] for r in results.values()]
            print("-" * 60)
            print(f"  {'OVERALL AVERAGE':30s}  {np.mean(rates):.1%}")
        print("=" * 60 + "\n")