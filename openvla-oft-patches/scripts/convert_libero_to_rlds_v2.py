"""
Convert LIBERO HDF5 datasets to RLDS (TFRecord) format.

Uses TensorFlow operations for image processing to avoid memory leaks.
Writes trajectory-level TFRecords compatible with the RLDS pipeline.

Usage:
    python scripts/convert_libero_to_rlds_v2.py \
        --libero_data_dir /path/to/LIBERO/libero/datasets \
        --output_dir /path/to/openvla-oft/datasets/rlds \
        --suites libero_spatial
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only for conversion
tf.get_logger().setLevel("ERROR")


def get_task_description_from_filename(filename):
    """Extract task description from HDF5 filename."""
    name = filename.replace("_demo.hdf5", "")
    desc = name.replace("_", " ")
    return desc


def serialize_example(image_primary, image_wrist, proprio, timestep, action, language_instruction, dataset_name):
    """Serialize a single trajectory to TFExample."""
    feature = {
        "observation/image_primary": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_primary.tobytes()])
        ),
        "observation/image_wrist": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_wrist.tobytes()])
        ),
        "observation/proprio": tf.train.Feature(
            float_list=tf.train.FloatList(value=proprio.flatten())
        ),
        "observation/timestep": tf.train.Feature(
            int64_list=tf.train.Int64List(value=timestep)
        ),
        "action": tf.train.Feature(
            float_list=tf.train.FloatList(value=action.flatten())
        ),
        "task/language_instruction": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[language_instruction.encode("utf-8")])
        ),
        "dataset_name": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[dataset_name.encode("utf-8")])
        ),
        "num_steps": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[len(action)])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def load_and_process_episode(hdf5_path, demo_idx, target_size=256):
    """Load and process a single episode from HDF5.
    
    Returns serialized TFExample or None if episode should be skipped.
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

            # Load raw data
            agentview_images = demo["obs"]["agentview_rgb"][()]
            wrist_images = demo["obs"]["eye_in_hand_rgb"][()]
            actions = demo["actions"][()]
            eef_pos = demo["obs"]["ee_pos"][()]  # (T, 3) - end-effector position
            eef_ori = demo["obs"]["ee_ori"][()]  # (T, 3) - end-effector orientation (axis-angle)
            gripper_states = demo["obs"]["gripper_states"][()]  # (T, 2)

            # Build proprioception: eef_pos (3) + eef_ori axis-angle (3) + gripper (2) = 8D
            # This matches the reference evaluation code which uses:
            #   np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            # The HDF5 ee_ori is already in axis-angle format (same as quat2axisangle of robot0_eef_quat)
            proprio = np.concatenate([eef_pos, eef_ori, gripper_states], axis=1).astype(np.float32)

            # Rotate images 180 degrees and resize using TensorFlow
            num_steps = len(actions)
            
            # Process images using TensorFlow (faster, no memory leak)
            agentview_tf = tf.constant(agentview_images, dtype=tf.uint8)
            wrist_tf = tf.constant(wrist_images, dtype=tf.uint8)
            
            # Rotate 180 degrees
            agentview_tf = tf.image.flip_up_down(tf.image.flip_left_right(agentview_tf))
            wrist_tf = tf.image.flip_up_down(tf.image.flip_left_right(wrist_tf))
            
            # Resize to target size
            agentview_tf = tf.image.resize(agentview_tf, [target_size, target_size], method="lanczos3")
            wrist_tf = tf.image.resize(wrist_tf, [target_size, target_size], method="lanczos3")
            
            # Convert back to numpy
            agentview_images = tf.cast(agentview_tf, tf.uint8).numpy()
            wrist_images = tf.cast(wrist_tf, tf.uint8).numpy()

            # Ensure uint8
            agentview_images = np.clip(agentview_images, 0, 255).astype(np.uint8)
            wrist_images = np.clip(wrist_images, 0, 255).astype(np.uint8)

            actions = actions.astype(np.float32)
            timestep = np.arange(num_steps, dtype=np.int64)

            return {
                "image_primary": agentview_images,
                "image_wrist": wrist_images,
                "proprio": proprio,
                "timestep": timestep,
                "action": actions,
            }
    except Exception as e:
        return None


def convert_suite(libero_data_dir, output_dir, suite_name, target_size=256):
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

    hdf5_files = sorted(hdf5_dir.glob("*_demo.hdf5"))
    print(f"    Found {len(hdf5_files)} HDF5 files")

    # Create output directory
    rlds_dir.mkdir(parents=True, exist_ok=True)

    # Write TFRecords
    dataset_name = f"{suite_name}_no_noops"
    tfrecord_path = rlds_dir / "train.tfrecord"
    
    total_episodes = 0
    total_skipped = 0
    episode_count = 0

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for hdf5_path in hdf5_files:
            filename = hdf5_path.name
            task_desc = get_task_description_from_filename(filename)

            with h5py.File(hdf5_path, "r") as f:
                num_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

            for demo_idx in range(num_demos):
                episode_count += 1
                episode = load_and_process_episode(hdf5_path, demo_idx, target_size)
                
                if episode is None:
                    total_skipped += 1
                    continue

                total_episodes += 1

                # Serialize and write
                serialized = serialize_example(
                    episode["image_primary"],
                    episode["image_wrist"],
                    episode["proprio"],
                    episode["timestep"],
                    episode["action"],
                    task_desc,
                    dataset_name
                )
                writer.write(serialized)

                # Clear episode data to free memory
                del episode

                if episode_count % 50 == 0:
                    print(f"    Processed {episode_count} episodes ({total_episodes} valid, {total_skipped} skipped)")

    print(f"    Total: {total_episodes} episodes written, {total_skipped} skipped")

    # Save dataset info
    dataset_info = {
        "description": f"LIBERO {suite_name} dataset in RLDS format",
        "citation": "",
        "features": {
            "observation": {
                "image_primary": {"type": "image", "shape": [None, target_size, target_size, 3], "dtype": "uint8"},
                "image_wrist": {"type": "image", "shape": [None, target_size, target_size, 3], "dtype": "uint8"},
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
        "config": dataset_name,
        "splits": {
            "train": {
                "filename_to_num_examples": {"train.tfrecord": total_episodes},
                "num_examples": total_episodes,
            }
        },
        "download_size": 0,
        "dataset_size": os.path.getsize(tfrecord_path),
    }

    with open(rlds_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    file_size = os.path.getsize(tfrecord_path) / (1024 * 1024)
    print(f"    [DONE] {suite_name} - {file_size:.1f}MB written")
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
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target image size (default: 256)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LIBERO HDF5 → RLDS Converter (v2)")
    print("=" * 80)
    print(f"Source: {args.libero_data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Suites: {args.suites}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print("=" * 80)

    for suite in args.suites:
        convert_suite(args.libero_data_dir, args.output_dir, suite, args.target_size)

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
