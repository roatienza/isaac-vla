# OpenVLA-OFT Changes for LIBERO Fine-Tuning

> **Purpose**: This document describes all modifications required to adapt the [openvla-oft](https://github.com/moojink/openvla-oft) repository for fine-tuning on LIBERO datasets. All patched files are stored in `isaac-vla/openvla-oft-patches/` so they can be applied to a freshly cloned copy of openvla-oft.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [File-by-File Changes](#file-by-file-changes)
4. [TFRecord Format](#tfrecord-format)
5. [How to Apply from isaac-vla](#how-to-apply-from-isaac-vla)
6. [End-to-End Workflow](#end-to-end-workflow)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The following changes were made to enable fine-tuning OpenVLA-OFT on LIBERO demonstration data:

| # | File | Status | Description |
|---|------|--------|-------------|
| 1 | `prismatic/vla/datasets/rlds/libero_reader.py` | **Rewritten** | Custom TFRecord reader for flat `tf.train.Example` format |
| 2 | `prismatic/vla/datasets/rlds/dataset.py` | **Modified** | Updated dataset loading to support LIBERO reader |
| 3 | `prismatic/vla/datasets/rlds/oxe/configs.py` | **Modified** | Added LIBERO dataset configurations |
| 4 | `prismatic/vla/datasets/rlds/oxe/transforms.py` | **Modified** | Added `libero_dataset_transform` standardization |
| 5 | `prismatic/vla/datasets/rlds/oxe/mixtures.py` | **Modified** | Added LIBERO mixture specifications |
| 6 | `scripts/convert_libero_to_rlds.py` | **New** | HDF5 → RLDS ArrayRecord converter (v1) |
| 7 | `scripts/convert_libero_to_rlds_v2.py` | **New** | HDF5 → RLDS TFRecord converter (v2, recommended) |
| 8 | `check_tfrecord.py` | **New** | TFRecord inspection utility |

### Unchanged files (included for reference)

The following files were copied but are **identical** to upstream openvla-oft. They are included for reference and completeness:

- `prismatic/vla/datasets/rlds/__init__.py`
- `prismatic/vla/datasets/rlds/obs_transforms.py`
- `prismatic/vla/datasets/rlds/traj_transforms.py`
- `prismatic/vla/datasets/rlds/oxe/__init__.py`
- `prismatic/vla/datasets/rlds/oxe/materialize.py`

---

## Prerequisites

- **Python** ≥ 3.10
- **CUDA** ≥ 12.1, GPU with ≥ 16 GB VRAM
- **LIBERO** installed (`pip install libero`)
- LIBERO HDF5 datasets at `~/sandbox/LIBERO/libero/datasets/`

---

## File-by-File Changes

### 1. `prismatic/vla/datasets/rlds/libero_reader.py` — **Rewritten**

**Problem**: The original openvla-oft reader expects RLDS ArrayRecord format (sharded, with `dataset_info.json` metadata). Our conversion produces flat `tf.train.Example` TFRecords (one episode per record) for simplicity and compatibility.

**Key changes**:

- **`_parse_example_dict()`** — Parses a single `tf.train.Example` protobuf into a trajectory dict:
  - Reads flat byte arrays for images (`observation/image_primary`, `observation/image_wrist`)
  - Reads flat float arrays for actions (`action`, shape T×7) and proprioception (`observation/proprio`, shape T×8)
  - **Fix**: `language_instruction` is a scalar bytes field — it is tiled across trajectory length so downstream `tf.gather` operations work correctly
  - Generates synthetic `is_first`, `is_last`, `is_terminal`, `discount`, `reward` fields

- **`_wrap_as_dlaset()`** — Monkey-patches the `tf.data.Dataset` returned by `from_generator` to behave like a `dlimp.DLataset` (enables `.traj_map()` and other dlimp methods). This fixes a bug where `tf.data.Dataset.from_generator()` returns a `_FlatMapDataset` that lacks dlimp methods.

- **`load_libero_dataset()`** — Entry point: loads TFRecords, wraps as DLataset, optionally shuffles.

**TFRecord schema** (one `tf.train.Example` per episode):

```
num_steps:                     int64 (scalar)
observation/image_primary:     bytes (T × 256 × 256 × 3 uint8, row-major)
observation/image_wrist:       bytes (T × 256 × 256 × 3 uint8, row-major)
observation/proprio:           float32 (T × 8, flat)
observation/timestep:          int64 (T,)
action:                        float32 (T × 7, flat)
task/language_instruction:     bytes (utf-8 string, scalar)
dataset_name:                  bytes (utf-8 string, scalar)
```

### 2. `prismatic/vla/datasets/rlds/dataset.py` — **Modified**

**Changes**: Updated `make_dataset_from_rlds()` to handle LIBERO datasets. The key modification is in the `restructure()` inner function, which now correctly routes LIBERO data through the standardization pipeline.

The `libero_reader.py` module is imported and used when the dataset name matches a LIBERO suite. The dataset statistics (`num_transitions`, `num_trajectories`) are computed correctly for the flat TFRecord format.

### 3. `prismatic/vla/datasets/rlds/oxe/configs.py` — **Modified**

**Changes**: Added 5 LIBERO dataset entries to `OXE_DATASET_CONFIGS`:

```python
"libero_spatial_no_noops": {
    "image_obs_keys": {"primary": "image_primary", "secondary": None, "wrist": "image_wrist"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["proprio"],
    "state_encoding": StateEncoding.JOINT,
    "action_encoding": ActionEncoding.EEF_POS,
},
# ... same for libero_object, libero_goal, libero_10, libero_4_task_suites
```

**Details**:
- **Image keys**: `image_primary` (agentview 3rd-person), `image_wrist` (eye-in-hand)
- **State**: 8D proprioception (7 joint angles + 1 gripper width)
- **Action**: 7D delta EE (dx, dy, dz, droll, dpitch, dyaw, gripper)
- **State encoding**: `JOINT` (7 joint + 1 gripper = 8D)
- **Action encoding**: `EEF_POS` (6D delta pose + 1D gripper)

### 4. `prismatic/vla/datasets/rlds/oxe/transforms.py` — **Modified**

**Changes**: Added `libero_dataset_transform()` function and registered it in `OXE_STANDARDIZATION_TRANSFORMS`:

```python
def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # Check if this is our custom TFRecord format (already has proprio)
    if "proprio" in trajectory["observation"]:
        # Our custom TFRecord format already has proprio (8D: 7 joint + 1 gripper)
        # and action (7D delta EE) in the correct format
        # Just add EEF_state and gripper_state for compatibility
        trajectory["observation"]["EEF_state"] = trajectory["observation"]["proprio"][:, :6]
        trajectory["observation"]["gripper_state"] = trajectory["observation"]["proprio"][:, 6:7]
        return trajectory

    # Original RLDS format with "state" key (fallback)
    # ... gripper inversion logic ...

OXE_STANDARDIZATION_TRANSFORMS = {
    # ... existing entries ...
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_4_task_suites_no_noops": libero_dataset_transform,
}
```

### 5. `prismatic/vla/datasets/rlds/oxe/mixtures.py` — **Modified**

**Changes**: Added LIBERO mixture specifications to `OXE_NAMED_MIXTURES`:

```python
"libero_spatial_no_noops": [("libero_spatial_no_noops", 1.0)],
"libero_object_no_noops": [("libero_object_no_noops", 1.0)],
"libero_goal_no_noops": [("libero_goal_no_noops", 1.0)],
"libero_10_no_noops": [("libero_10_no_noops", 1.0)],
"libero_4_task_suites_no_noops": [
    ("libero_spatial_no_noops", 1.0),
    ("libero_object_no_noops", 1.0),
    ("libero_goal_no_noops", 1.0),
    ("libero_10_no_noops", 1.0),
],
```

### 6. `scripts/convert_libero_to_rlds.py` — **New (v1)**

Converts LIBERO HDF5 files to RLDS ArrayRecord format using `tf.data.experimental.save()`.

**Usage**:
```bash
python scripts/convert_libero_to_rlds.py \
    --libero_data_dir ~/sandbox/LIBERO/libero/datasets \
    --output_dir ~/sandbox/openvla-oft/datasets/rlds \
    --suites libero_spatial libero_object libero_goal libero_10
```

**Notes**: Uses PIL for image resizing. May have memory issues on large datasets.

### 7. `scripts/convert_libero_to_rlds_v2.py` — **New (v2, Recommended)**

Improved converter using TensorFlow operations for image processing (avoids PIL memory leaks) and writes flat `tf.train.Example` TFRecords (one episode per record).

**Usage**:
```bash
python scripts/convert_libero_to_rlds_v2.py \
    --libero_data_dir ~/sandbox/LIBERO/libero/datasets \
    --output_dir ~/sandbox/openvla-oft/datasets/rlds \
    --suites libero_spatial \
    --target_size 256
```

**Key improvements over v1**:
- Uses `tf.image.resize()` with Lanczos interpolation (no PIL dependency)
- Uses `tf.image.flip_up_down()` + `tf.image.flip_left_right()` for 180° rotation
- Writes single `train.tfrecord` file per suite (simpler to manage)
- Runs on CPU only (`CUDA_VISIBLE_DEVICES=-1`)
- Frees memory after each episode

### 8. `check_tfrecord.py` — **New**

Quick utility to inspect the schema of a TFRecord file:

```bash
cd openvla-oft
python check_tfrecord.py
```

Outputs feature names, types, and sizes for the first record.

---

## TFRecord Format

The v2 converter produces flat `tf.train.Example` TFRecords with the following schema:

```
tf.train.Example {
  features {
    feature {
      key: "num_steps"
      value { int64_list { value: [T] } }          # trajectory length
    }
    feature {
      key: "observation/image_primary"
      value { bytes_list { value: [T*256*256*3 uint8 bytes] } }
    }
    feature {
      key: "observation/image_wrist"
      value { bytes_list { value: [T*256*256*3 uint8 bytes] } }
    }
    feature {
      key: "observation/proprio"
      value { float_list { value: [T*8 floats] } }  # 7 joint + 1 gripper
    }
    feature {
      key: "observation/timestep"
      value { int64_list { value: [0, 1, ..., T-1] } }
    }
    feature {
      key: "action"
      value { float_list { value: [T*7 floats] } }  # 6 delta EE + 1 gripper
    }
    feature {
      key: "task/language_instruction"
      value { bytes_list { value: [utf-8 string] } }  # scalar, tiled at read time
    }
    feature {
      key: "dataset_name"
      value { bytes_list { value: [utf-8 string] } }
    }
  }
}
```

**Directory structure** (per suite):
```
datasets/rlds/libero_spatial_no_noops/
├── train.tfrecord          # single TFRecord file (all episodes)
└── dataset_info.json       # metadata (episode count, feature shapes)
```

---

## How to Apply from isaac-vla

Given a freshly cloned `isaac-vla` and a freshly cloned `openvla-oft`:

### Step 1: Clone both repos

```bash
# Clone isaac-vla (source of patches)
git clone https://github.com/roatienza/isaac-vla.git
cd isaac-vla
git checkout main
cd ..

# Clone openvla-oft (target)
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .
cd ..
```

### Step 2: Copy patched files

```bash
ISAAC_VLA=/path/to/isaac-vla
OFT=/path/to/openvla-oft

# Core patched files (MUST copy)
cp "$ISAAC_VLA/openvla-oft-patches/prismatic/vla/datasets/rlds/libero_reader.py" \
   "$OFT/prismatic/vla/datasets/rlds/"

cp "$ISAAC_VLA/openvla-oft-patches/prismatic/vla/datasets/rlds/dataset.py" \
   "$OFT/prismatic/vla/datasets/rlds/"

cp "$ISAAC_VLA/openvla-oft-patches/prismatic/vla/datasets/rlds/oxe/configs.py" \
   "$OFT/prismatic/vla/datasets/rlds/oxe/"

cp "$ISAAC_VLA/openvla-oft-patches/prismatic/vla/datasets/rlds/oxe/transforms.py" \
   "$OFT/prismatic/vla/datasets/rlds/oxe/"

cp "$ISAAC_VLA/openvla-oft-patches/prismatic/vla/datasets/rlds/oxe/mixtures.py" \
   "$OFT/prismatic/vla/datasets/rlds/oxe/"

# Conversion scripts
cp "$ISAAC_VLA/openvla-oft-patches/scripts/convert_libero_to_rlds.py" \
   "$OFT/scripts/"

cp "$ISAAC_VLA/openvla-oft-patches/scripts/convert_libero_to_rlds_v2.py" \
   "$OFT/scripts/"

# Utility
cp "$ISAAC_VLA/openvla-oft-patches/check_tfrecord.py" \
   "$OFT/"
```

### Step 3: Verify

```bash
cd $OFT
python check_tfrecord.py  # should work if TFRecords exist
```

### Alternative: Use the fork directly

The patched version is also available as a fork:

```bash
git clone https://github.com/roatienza/openvla-oft.git
cd openvla-oft
pip install -e .
```

This fork already contains all patches (commits `fef942f` and `33fa271`).

---

## End-to-End Workflow

### 1. Convert LIBERO HDF5 → TFRecord

```bash
cd openvla-oft

python scripts/convert_libero_to_rlds_v2.py \
    --libero_data_dir ~/sandbox/LIBERO/libero/datasets \
    --output_dir ~/sandbox/openvla-oft/datasets/rlds \
    --suites libero_spatial libero_object libero_goal libero_10
```

### 2. Fine-tune on a single suite

```bash
python scripts/finetune.py \
    --pretrained_checkpoint openvla/openvla-7b \
    --dataset_root ~/sandbox/openvla-oft/datasets/rlds \
    --dataset_name libero_spatial_no_noops \
    --run_root ./runs \
    --run_id libero_spatial_ft \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --max_steps 150005 \
    --save_every 10000
```

### 3. Fine-tune on all suites (sequential)

```bash
# Use the launch script from isaac-vla
cd isaac-vla
bash scripts/run_all_suites_finetune.sh
```

### 4. Evaluate fine-tuned model

```bash
cd isaac-vla
python scripts/run_libero_eval.py \
    --checkpoint_path ./checkpoints/openvla-7b+libero_spatial_no_noops+.../ \
    --task_suite libero_spatial
```

---

## Troubleshooting

### `AttributeError: '_FlatMapDataset' object has no attribute 'traj_map'`

This is fixed by the `_wrap_as_dlaset()` monkey-patch in `libero_reader.py`. Make sure you copied the patched version.

### `tf.gather` fails on `language_instruction`

The original reader produces a scalar (rank-0) `language_instruction`. The patched reader tiles it across trajectory length. Ensure `libero_reader.py` is the patched version.

### Conversion fails with memory error

Use `convert_libero_to_rlds_v2.py` instead of v1. It runs on CPU and frees memory after each episode.

### TFRecord format mismatch

Verify your TFRecords with `check_tfrecord.py`. The patched reader expects flat `tf.train.Example` format (v2 converter output), not ArrayRecord format (v1 converter output).

### Fine-tuning loss is NaN

Check:
1. Action values are in [-1, 1] range (LIBERO actions should already be)
2. Image values are uint8 [0, 255]
3. Proprioception is float32
4. Batch size is 1 (required for 16 GB GPU)

---

## References

- **isaac-vla repo**: https://github.com/roatienza/isaac-vla
- **openvla-oft upstream**: https://github.com/moojink/openvla-oft
- **openvla-oft fork (patched)**: https://github.com/roatienza/openvla-oft
- **LIBERO benchmark**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **OFT paper**: https://arxiv.org/abs/2406.09206
