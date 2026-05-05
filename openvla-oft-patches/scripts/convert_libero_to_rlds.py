"""
Convert LIBERO HDF5 datasets to RLDS (TFDS ArrayRecord) format.

This script converts LIBERO HDF5 demonstration files into the RLDS format
expected by the OpenVLA-OFT fine-tuning pipeline (via dlimp/tfds).

Each HDF5 file contains multiple demonstrations (episodes). Each episode is
converted to a single trajectory example with time-batched observations and actions.

Usage:
    python scripts/convert_libero_to_rlds.py \
        --libero_data_dir /path/to/LIBERO/libero/datasets \
        --output_dir /path/to/openvla-oft/datasets/rlds \
        --suites libero_spatial libero_object libero_goal libero_10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def get_task_description_from_filename(filename):
    """Extract task description from HDF5 filename."""
    name = filename.replace("_demo.hdf5", "")
    desc = name.replace("_", " ")
    return desc


def load_hdf5_episode(hdf5_path, demo_idx):
    """Load a single episode from an HDF5 file.
    
    Returns a trajectory dict with time-batched data.
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_key = f"demo_{demo_idx}"
            if demo_key not in f["data"]:
                return None

            demo = f["data"][demo_key]
            dones = demo["dones"][()]
            if len(dones) == 0 or dones[-1] != 1:
                return None

            agentview_images = demo["obs"]["agentview_rgb"][()]
            wrist_images = demo["obs"]["eye_in_hand_rgb"][()]
            actions = demo["actions"][()]
            joint_states = demo["obs"]["joint_states"][()]
            gripper_states = demo["obs"]["gripper_states"][()]

            # Rotate images 180 degrees (matches training preprocessing)
            agentview_images = agentview_images[::-1, ::-1, :]
            wrist_images = wrist_images[::-1, ::-1, :]

            # Ensure images are 256x256
            if agentview_images.shape[1] != 256 or agentview_images.shape[2] != 256:
                from PIL import Image
                resized_agent = []
                resized_wrist = []
                for i in range(len(agentview_images)):
                    img = Image.fromarray(agentview_images[i])
                    img = img.resize((256, 256), Image.LANCZOS)
                    resized_agent.append(np.array(img))
                    img_w = Image.fromarray(wrist_images[i])
                    img_w = img_w.resize((256, 256), Image.LANCZOS)
                    resized_wrist.append(np.array(img_w))
                agentview_images = np.array(resized_agent)
                wrist_images = np.array(resized_wrist)

            # Handle gripper states - may be (T, 1) or (T, 2), take first column
            if gripper_states.shape[1] > 1:
                gripper_states = gripper_states[:, :1]

            # Build proprioception: joint states (7) + gripper (1) = 8D
            proprio = np.concatenate([joint_states, gripper_states], axis=1)

            # Ensure uint8 for images
            agentview_images = np.clip(agentview_images, 0, 255).astype(np.uint8)
            wrist_images = np.clip(wrist_images, 0, 255).astype(np.uint8)

            return {
                "observation": {
                    "image_primary": agentview_images,
                    "image_wrist": wrist_images,
                    "proprio": proprio.astype(np.float32),
                    "timestep": np.arange(len(actions), dtype=np.int64),
                },
                "action": actions.astype(np.float32),
            }
    except Exception as e:
        print(f"    ERROR loading {hdf5_path} demo_{demo_idx}: {e}")
        return None


def generate_trajectories(hdf5_dir, dataset_name):
    """Generate RLDS-format trajectories from HDF5 files.
    
    Each yielded example is a full trajectory (time-batched).
    """
    hdf5_files = sorted(Path(hdf5_dir).glob("*_demo.hdf5"))
    total_episodes = 0
    total_skipped = 0

    print(f"    Processing {len(hdf5_files)} HDF5 files...")

    for hdf5_path in hdf5_files:
        filename = hdf5_path.name
        task_desc = get_task_description_from_filename(filename)

        with h5py.File(hdf5_path, "r") as f:
            num_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

        for demo_idx in range(num_demos):
            episode = load_hdf5_episode(hdf5_path, demo_idx)
            if episode is None:
                total_skipped += 1
                continue

            total_episodes += 1

            # Yield full trajectory (time-batched)
            yield {
                "observation": {
                    "image_primary": episode["observation"]["image_primary"],  # (T, 256, 256, 3)
                    "image_wrist": episode["observation"]["image_wrist"],       # (T, 256, 256, 3)
                    "proprio": episode["observation"]["proprio"],               # (T, 8)
                    "timestep": episode["observation"]["timestep"],             # (T,)
                },
                "action": episode["action"],                                    # (T, 7)
                "task": {
                    "language_instruction": task_desc.encode("utf-8"),
                },
                "dataset_name": dataset_name.encode("utf-8"),
            }

    print(f"    Total episodes: {total_episodes}, Skipped: {total_skipped}")


def convert_suite(libero_data_dir, output_dir, suite_name):
    """Convert a single LIBERO suite to RLDS format."""
    hdf5_dir = Path(libero_data_dir) / suite_name
    rlds_dir = Path(output_dir) / f"{suite_name}_no_noops"

    if not hdf5_dir.exists():
        print(f"  [SKIP] HDF5 directory not found: {hdf5_dir}")
        return False

    if rlds_dir.exists():
        print(f"  [SKIP] RLDS directory already exists: {rlds_dir}")
        return False

    print(f"  [CONVERTING] {suite_name}...")
    print(f"    Source: {hdf5_dir}")
    print(f"    Output: {rlds_dir}")

    hdf5_files = list(hdf5_dir.glob("*_demo.hdf5"))
    print(f"    Found {len(hdf5_files)} HDF5 files")

    # Create output directory
    rlds_dir.mkdir(parents=True, exist_ok=True)

    # Create generator function
    def gen():
        return generate_trajectories(hdf5_dir, f"{suite_name}_no_noops")

    # Create output signature (trajectory-level, with time dimension)
    output_signature = {
        "observation": {
            "image_primary": tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
            "image_wrist": tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
            "proprio": tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
            "timestep": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        },
        "action": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
        "task": {
            "language_instruction": tf.TensorSpec(shape=(), dtype=tf.string),
        },
        "dataset_name": tf.TensorSpec(shape=(), dtype=tf.string),
    }

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )

    # Write array records
    print(f"    Writing train split...")
    tf.data.experimental.save(
        dataset,
        str(rlds_dir),
        compression="GZIP",
    )

    # Save dataset info
    dataset_info = {
        "description": f"LIBERO {suite_name} dataset in RLDS format",
        "citation": "",
        "features": {
            "observation": {
                "image_primary": {"type": "image", "shape": [None, 256, 256, 3], "dtype": "uint8"},
                "image_wrist": {"type": "image", "shape": [None, 256, 256, 3], "dtype": "uint8"},
                "proprio": {"type": "tensor", "shape": [None, 8], "dtype": "float32"},
                "timestep": {"type": "tensor", "shape": [None], "dtype": "int64"},
            },
            "action": {"type": "tensor", "shape": [None, 7], "dtype": "float32"},
            "task": {
                "language_instruction": {"type": "text"},
            },
            "dataset_name": {"type": "text"},
        },
        "supervised": False,
        "version": "1.0.0",
        "config": f"{suite_name}_no_noops",
        "splits": {
            "train": {
                "filename_to_num_examples": {},
                "num_examples": 0,
            }
        },
        "download_size": 0,
        "dataset_size": 0,
    }

    # Count examples and populate filename info
    import glob
    array_record_files = glob.glob(str(rlds_dir / "train-*.array_record*"))
    total_examples = 0
    for f in array_record_files:
        basename = os.path.basename(f)
        dataset_info["splits"]["train"]["filename_to_num_examples"][basename] = 0

    with open(rlds_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"    [DONE] {suite_name} - {len(array_record_files)} shards written")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to RLDS format")
    parser.add_argument(
        "--libero_data_dir",
        type=str,
        default="/home/rowel/sandbox/LIBERO/libero/datasets",
        help="Path to LIBERO datasets directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/rowel/sandbox/openvla-oft/datasets/rlds",
        help="Path to output RLDS directory"
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO suites to convert"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LIBERO HDF5 → RLDS Converter")
    print("=" * 80)
    print(f"Source: {args.libero_data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Suites: {args.suites}")
    print("=" * 80)

    for suite in args.suites:
        convert_suite(args.libero_data_dir, args.output_dir, suite)

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
