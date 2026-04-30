#!/usr/bin/env python3
"""
Setup script for isaac-vla package.

Usage:
    pip install -e .  # Development install
    pip install .     # Regular install
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = this_dir / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path) as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="isaac-vla",
    version="0.1.0",
    description="OpenVLA-OFT on Franka Emika in Isaac Sim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="roatienza",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "tui": [
            "textual>=0.47.0",
            "rich>=13.6.0",
        ],
        "data": [
            "h5py>=3.10.0",
            "tensorflow-datasets>=4.9.0",
        ],
        "eval": [
            "matplotlib>=3.8.0",
            "pandas>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "isaac-vla-server=scripts.run_vla_server:main",
            "isaac-vla-bridge=scripts.run_sim_bridge:main",
            "isaac-vla-tui=scripts.run_tui_client:main",
            "isaac-vla-quickstart=scripts.quick_start:main",
            "isaac-vla-collect=scripts.collect_demonstrations:main",
            "isaac-vla-eval=scripts.evaluate_tasks:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Robotics",
    ],
)
