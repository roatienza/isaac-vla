#!/usr/bin/env python3
"""
Fine-tune OpenVLA-OFT on LIBERO dataset.

Usage:
    python scripts/run_libero_finetune.py --suite libero_spatial
    python scripts/run_libero_finetune.py --suite libero_object
    python scripts/run_libero_finetune.py --suite libero_goal
    python scripts/run_libero_finetune.py --suite libero_10
"""

import subprocess
import sys
import os
import argparse
import shutil
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/rowel/sandbox/isaac-vla")
OPENVLA_ROOT = Path("/home/rowel/sandbox/openvla-oft")
DATASET_ROOT = OPENVLA_ROOT / "datasets/rlds"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
FINETUNE_SCRIPT = OPENVLA_ROOT / "vla-scripts/finetune.py"

# Hyperparameters (from OpenVLA-OFT paper)
# RTX 5090 has 32GB VRAM - batch_size=1 needs ~25GB
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
MAX_STEPS = 150005
NUM_STEPS_BEFORE_DECAY = 100000
SAVE_FREQ = 10000
LORA_RANK = 32
NUM_IMAGES = 2
USE_PROPRIO = True
IMAGE_AUG = True


def ensure_modeling_files_synced():
    """Ensure the local modeling_prismatic.py is synced with the cached model."""
    local_modeling = OPENVLA_ROOT / "prismatic/extern/hf/modeling_prismatic.py"
    local_config = OPENVLA_ROOT / "prismatic/extern/hf/configuration_prismatic.py"
    local_processing = OPENVLA_ROOT / "prismatic/extern/hf/processing_prismatic.py"
    
    # Find the cached model directory
    cache_dir = Path.home() / ".cache/huggingface/hub/models--openvla--openvla-7b"
    snapshot_dir = cache_dir / "snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    
    if snapshot_dir.exists():
        # Copy local files to snapshot directory (overwriting symlinks)
        for src, dst_name in [(local_modeling, "modeling_prismatic.py"),
                              (local_config, "configuration_prismatic.py"),
                              (local_processing, "processing_prismatic.py")]:
            dst = snapshot_dir / dst_name
            if dst.is_symlink():
                dst.unlink()  # Remove symlink
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA-OFT on LIBERO dataset")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                        help="LIBERO task suite to fine-tune on")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max training steps")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (use cached model only)")
    parser.add_argument("--port", type=int, default=29501, help="Master port for distributed training (default: 29501)")
    args = parser.parse_args()

    # Create checkpoint directory
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    # Dataset name mapping
    dataset_name = f"{args.suite}_no_noops"

    # Build finetune.py arguments
    finetune_args = [
        "--vla_path", "openvla/openvla-7b",
        "--data_root_dir", str(DATASET_ROOT),
        "--dataset_name", dataset_name,
        "--run_root_dir", str(CHECKPOINT_ROOT),
        "--use_l1_regression", "True",
        "--use_diffusion", "False",
        "--use_film", "False",
        "--num_images_in_input", str(NUM_IMAGES),
        "--use_proprio", "True" if USE_PROPRIO else "False",
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.lr),
        "--num_steps_before_decay", str(NUM_STEPS_BEFORE_DECAY),
        "--max_steps", str(args.max_steps),
        "--save_freq", str(SAVE_FREQ),
        "--save_latest_checkpoint_only", "False",
        "--image_aug", "True" if IMAGE_AUG else "False",
        "--lora_rank", str(LORA_RANK),
        "--wandb_entity", "your-wandb-entity",
        "--wandb_project", "isaac-vla-libero",
        "--run_id_note", f"{args.suite}_ft_lora32_bs{args.batch_size}",
    ]

    # Run directly with python (no torchrun) - PartialState handles single-GPU case
    cmd = [sys.executable, str(FINETUNE_SCRIPT)] + finetune_args

    print("=" * 80)
    print(f"Fine-tuning OpenVLA-OFT on {args.suite.upper()}")
    print("=" * 80)
    print(f"Dataset: {DATASET_ROOT}/{dataset_name}")
    print(f"Checkpoints: {CHECKPOINT_ROOT}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Offline mode: {args.offline}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    # Environment variables
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    
    if args.offline:
        env["HF_HUB_OFFLINE"] = "true"
    
    # Speed up HuggingFace downloads
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_ETAG_TIMEOUT"] = "30"
    env["HF_HUB_CONNECTION_TIMEOUT"] = "30"

    # Pre-initialize distributed environment for PartialState
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = str(args.port)
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["LOCAL_WORLD_SIZE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "lo"

    # Ensure modeling files are synced before training
    ensure_modeling_files_synced()

    # Launch training directly with python (no torchrun)
    result = subprocess.run(cmd, cwd=str(OPENVLA_ROOT), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
