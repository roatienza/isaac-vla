# Production Fine-Tuning Guide for LIBERO

## Root Cause Analysis: Why Initial Evaluation Was 0%

After initial fine-tuning and evaluation, the model achieved **0% success rate** across all LIBERO tasks. Three critical issues were identified:

### 1. Action Clipping Was Too Aggressive

The `config/default.yaml` had safety clipping bounds that were far too restrictive:

| Dimension | Dataset Max (from stats) | Dataset Std | Old Clip Limit | New Clip Limit |
|-----------|--------------------------|-------------|----------------|----------------|
| Position (x, y, z) | ±0.94 | 0.35–0.51 | **±0.05** ❌ | ±1.0 ✅ |
| Rotation (roll, pitch, yaw) | ±0.19–0.38 | 0.04–0.07 | **±0.1** ❌ | ±1.0 ✅ |

The model's predicted actions were being **clipped to 5% of their intended magnitude**. Every action was essentially a tiny nudge instead of meaningful movement. The robot could not reach any target.

**Fix applied:** Increased `max_position_delta` from `0.05` to `1.0` and `max_rotation_delta` from `0.1` to `1.0` in `config/default.yaml`.

### 2. Only 5,000 Training Steps

The OpenVLA-OFT paper reports ~97% success on LIBERO-Spatial after **150,000 training steps**. Our initial run used only **5,000 steps** (3.3% of recommended amount). At 5K steps, the model has barely completed one epoch over the dataset.

**Fix applied:** Retrain with `--max-steps 150000 --save-freq 10000`.

### 3. Proprioception State Dimension Mismatch

The LIBERO bridge was concatenating `robot0_joint_pos` (7D) + `robot0_gripper_qpos` (4D) = **11D state**, but OpenVLA-OFT expects exactly **8D state** (7 joint angles + 1 gripper width). This caused shape mismatch errors during inference.

**Fix applied:** Clip proprio state to `robot0_joint_pos[:7]` + `robot0_gripper_qpos[:1]` = 8D.

---

## LIBERO Dataset Conversion Pipeline

### Overview

LIBERO datasets come in HDF5 format. We convert them to RLDS TFRecord format compatible with OpenVLA-OFT fine-tuning:

```
LIBERO HDF5 → TFRecord conversion → Custom TFRecord reader → OpenVLA-OFT training
```

### Conversion Script

```bash
cd /home/rowel/sandbox/openvla-oft

# Convert all LIBERO suites to RLDS format
python scripts/convert_libero_to_rlds_v2.py \
    --libero_data_dir /home/rowel/sandbox/LIBERO/libero/datasets \
    --output_dir datasets/rlds \
    --suites libero_spatial libero_object libero_goal libero_10
```

### TFRecord Format

Each trajectory is stored as a single TFExample with:
- `observation/image_primary`: Raw bytes (T, 256, 256, 3) — third-person camera
- `observation/image_wrist`: Raw bytes (T, 256, 256, 3) — wrist camera
- `observation/proprio`: Float32 (T, 8) — 7 joint angles + 1 gripper width
- `observation/timestep`: Int64 (T)
- `action`: Float32 (T, 7) — delta EE pose (dx, dy, dz, droll, dpitch, dyaw, gripper)
- `task/language_instruction`: String — task description
- `dataset_name`: String
- `num_steps`: Int64

### Custom Reader

`prismatic/vla/datasets/rlds/libero_reader.py` provides a custom TFRecord reader that:
- Reads TFRecords directly using `tf.data.TFRecordDataset`
- Wraps the result as a `dlimp.DLataset` hybrid class
- Parses images, proprioception, actions, and language instructions
- Handles both versioned (`1.0.0/`) and flat directory structures

### Dataset Config Changes

`prismatic/vla/datasets/rlds/oxe/configs.py` was updated for all LIBERO datasets:
- `image_obs_keys`: `{"primary": "image_primary", "wrist": "image_wrist"}` (was `image`/`wrist_image`)
- `state_obs_keys`: `["proprio"]` (was `["EEF_state", "gripper_state"]`)
- `state_encoding`: `StateEncoding.JOINT` (was `StateEncoding.POS_EULER`)

### Transform Changes

`prismatic/vla/datasets/rlds/oxe/transforms.py` was updated to handle the custom TFRecord format:
- Detects `proprio` key in observation
- Adds `EEF_state` and `gripper_state` for backward compatibility
- Skips action inversion (actions are already in correct format)

---

## Production Fine-Tuning Setup

### Prerequisites

1. **LIBERO datasets downloaded:**
   ```bash
   cd /path/to/LIBERO
   python benchmark_scripts/download_libero_datasets.py
   ```

2. **OpenVLA-OFT checkpoint available:**
   ```bash
   # Verify base model exists
   ls /home/rowel/sandbox/openvla-oft/weights/openvla-7b/
   ```

3. **GPU with ≥24GB VRAM** (RTX 3090/4090/5090/A100 recommended)

### Training Commands

#### Quick Test Run (5,000 steps)

```bash
python scripts/run_libero_finetune.py \
    --suite libero_spatial \
    --max-steps 5000 \
    --save-freq 5000 \
    --batch-size 1 \
    --lr 0.0005 \
    --lora-rank 32
```

**Expected:** ~23 minutes, saves checkpoint at step 5,000.

#### Production Run (150,000 steps)

```bash
python scripts/run_libero_finetune.py \
    --suite libero_spatial \
    --max-steps 150000 \
    --save-freq 10000 \
    --batch-size 1 \
    --lr 0.0005 \
    --lora-rank 32
```

**Expected:** ~42 hours on single A100, saves checkpoints every 10,000 steps.

#### All Task Suites

```bash
# Train on all 4 LIBERO suites sequentially
for suite in libero_spatial libero_object libero_goal libero_10; do
    python scripts/run_libero_finetune.py \
        --suite $suite \
        --max-steps 150000 \
        --save-freq 10000 \
        --batch-size 1 \
        --lr 0.0005 \
        --lora-rank 32
done
```

### Training Configuration Reference

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--suite` | `libero_spatial` | Per task | LIBERO task suite |
| `--max-steps` | 150000 | 150000 | Total training steps |
| `--save-freq` | 10000 | 10000 | Checkpoint save interval |
| `--batch-size` | 1 | 1 | Batch size (VRAM-limited) |
| `--lr` | 0.0005 | 0.0005 | Learning rate |
| `--lora-rank` | 32 | 32 | LoRA adapter rank |
| `--lora-alpha` | 16 | 16 | LoRA scaling factor |
| `--lora-dropout` | 0.0 | 0.0 | LoRA dropout rate |
| `--image-aug` | on | on | Image augmentation |
| `--port` | 29501 | 29501 | Master port for distributed training |

### Architecture

```
OpenVLA-7B (frozen)
├── Vision Backbone (frozen)
├── LLM Backbone (frozen)
├── LoRA adapters (trainable): q_proj, v_proj, k_proj, o_proj
├── Proprioception projector (trainable): ~16.8M params
└── L1 Regression action head (trainable): ~151.1M params
```

**Total trainable parameters:** ~278.7M (1.45% of 7.65B total)

---

## Evaluation Pipeline

### Using Your Trained Checkpoint

1. **Locate your checkpoint:**
   ```bash
   ls checkpoints/openvla-7b+libero_spatial_no_noops+*/
   # Find the latest checkpoint directory (e.g., checkpoint-10000, checkpoint-150000)
   ```

2. **Start the VLA server with your checkpoint:**
   ```bash
   python scripts/run_vla_server.py \
       --checkpoint /path/to/checkpoint-150000/
   ```

3. **Run evaluation:**
   ```bash
   # Single task
   python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

   # All tasks in a suite
   python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks

   # All suites
   python scripts/run_libero_eval.py --all-suites --num-episodes 50
   ```

### Expected Results

| Training Steps | Expected Success Rate (LIBERO-Spatial) | Notes |
|----------------|----------------------------------------|-------|
| 0 (base model) | ~15% | Zero-shot generalization |
| 5,000 | ~10–20% | Barely one epoch |
| 50,000 | ~60–75% | Partial convergence |
| 100,000 | ~85–92% | Near convergence |
| 150,000 | ~90–97% | Full convergence (OFT paper) |

### Results Output

Evaluation results are saved to:
```
data/libero_results/evaluation_results.json
```

Format:
```json
{
  "libero_spatial": {
    "task_0": {"success_rate": 0.92, "episodes": 50, "successes": 46},
    "task_1": {"success_rate": 0.88, "episodes": 50, "successes": 44},
    ...
    "mean_success_rate": 0.90
  }
}
```

---

## Troubleshooting

### 0% Success Rate

1. **Check action clipping bounds** in `config/default.yaml`:
   ```yaml
   libero:
     max_position_delta: 1.0    # Must be ≥ 0.5
     max_rotation_delta: 1.0    # Must be ≥ 0.2
   ```

2. **Verify checkpoint is loaded** — check VLA server logs for:
   ```
   Loading checkpoint from: /path/to/checkpoint-XXXXX/
   ```

3. **Verify proprio state is 8D** — check logs for:
   ```
   State shape: (8,)
   ```

### Training Crashes

1. **Out of memory:** Reduce batch size to 1 or use gradient accumulation
2. **NaN losses:** Reduce learning rate to 1e-4
3. **Slow training:** Check GPU utilization with `nvidia-smi`
4. **TFDS builder not found:** This is expected — the custom TFRecord reader handles this automatically

### Evaluation Crashes

1. **VLA server not running:** Start with `python scripts/run_vla_server.py`
2. **LIBERO files missing:** Run `python benchmark_scripts/download_libero_datasets.py`
3. **BDDL file errors:** Ensure `task_suite.get_task_bddl_file_path()` is used (not manual path construction)

### TFRecord Conversion Issues

1. **HDF5 files missing:** Ensure LIBERO datasets are downloaded first
2. **Memory errors during conversion:** Convert one suite at a time
3. **Image shape mismatch:** Verify conversion script uses correct image keys (`agentview_rgb`, `eye_in_hand_rgb`)

---

## Monitoring Training

```bash
# Live progress (check the managed process logs)
# Training is running as a background process

# GPU utilization
nvidia-smi

# Checkpoint sizes
du -sh checkpoints/openvla-7b+libero_spatial_no_noops+*/checkpoint-*/

# Check training log
tail -f checkpoints/openvla-7b+libero_spatial_no_noops+*/train.log
```

---

## Current Training Status (2026-05-04)

- **Suite:** `libero_spatial_no_noops`
- **Steps:** 0/150,000 (in progress)
- **GPU:** CUDA:0 (NVIDIA A100-SXM4-40GB)
- **Speed:** ~1 step/second
- **ETA:** ~42 hours
- **Checkpoint interval:** Every 10,000 steps
- **Checkpoint dir:** `checkpoints/openvla-7b+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_spatial_ft_lora32/`
