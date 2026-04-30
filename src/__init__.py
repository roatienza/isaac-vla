"""
Isaac-VLA: OpenVLA-OFT on Franka Emika in Isaac Sim
====================================================

A complete system for deploying vision-language-action models on a
Franka Emika Panda robot in NVIDIA Isaac Sim with kitchen manipulation tasks.

Modules:
    - vla_server: OpenVLA-OFT inference server (FastAPI)
    - sim_bridge: Isaac Sim control bridge
    - action_pipeline: VLA actions → sim actions
    - ik_solver: Inverse kinematics for Franka
    - kitchen_scene: Kitchen environment builder
    - data_collector: Demonstration recording
    - evaluator: Task evaluation framework
    - api: Unified Python API
    - utils: Shared utilities
"""

__version__ = "0.1.0"

from .api import IsaacVLAClient

__all__ = ["IsaacVLAClient"]