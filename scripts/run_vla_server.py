#!/usr/bin/env python3
"""
VLA Server Launcher
====================

Launch the OpenVLA-OFT inference server.

Usage:
    python scripts/run_vla_server.py --config config/default.yaml
    python scripts/run_vla_server.py --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial
    python scripts/run_vla_server.py --port 8777 --no-warmup
    python scripts/run_vla_server.py --openvla-oft-root /home/rowel/sandbox/openvla-oft
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

from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Launch OpenVLA-OFT Inference Server")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--host", type=str, default=None,
                        help="Server host (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (overrides config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (overrides config)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip model warmup on startup")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g., cuda:0, cpu)")
    parser.add_argument("--openvla-oft-root", type=str, default=None,
                        help="Path to openvla-oft repo root (if not pip-installed)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("vla-server", args.log_level)

    # Load config (resolve relative to project root)
    config_dict = {}
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    if config_path.exists():
        full_config = load_config(str(config_path))
        config_dict = full_config.get("vla_server", {})
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Override with CLI args
    if args.host:
        config_dict["host"] = args.host
    if args.port:
        config_dict["port"] = args.port
    if args.checkpoint:
        config_dict["pretrained_checkpoint"] = args.checkpoint
    if args.device:
        config_dict["device"] = args.device
    if args.openvla_oft_root:
        config_dict["openvla_oft_root"] = args.openvla_oft_root

    # Import and create server
    from src.vla_server import VLAServer, VLAConfig

    config = VLAConfig(**config_dict)
    server = VLAServer(config=config)

    logger.info("Loading model...")
    server.load_model()

    if not args.no_warmup:
        logger.info("Warming up model...")
        server.warmup()

    logger.info(f"Starting VLA server on {config.host}:{config.port}")
    server.run()


if __name__ == "__main__":
    main()