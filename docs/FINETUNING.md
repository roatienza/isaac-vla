# Fine-Tuning Guide

## Overview

Fine-tuning OpenVLA-OFT on your Franka kitchen tasks involves:

1. **Collecting demonstrations** in Isaac Sim
2. **Converting to RLDS format** for OpenVLA-OFT training
3. **Fine-tuning** the model with LoRA or full fine-tuning
4. **Evaluating** the fine-tuned model

---

## Step 1: Collect Demonstrations

### 1.1 Keyboard Teleoperation

Use the built-in teleoperation interface to collect demonstrations:

```bash
# IMPORTANT: Use ABSOLUTE path to script, since python.sh changes the working directory
<isaac_sim>/python.sh /abs/path/to/isaac-vla/scripts/collect_demonstrations.py \
    --task "pick up the red block" \
    --num-episodes 50 \
    --output-dir ./data/demonstrations
```

### 1.2 Teleoperation Controls

| Key | Action |
|---|---|
| W/S | Move EE forward/backward (X axis) |
| A/D | Move EE left/right (Y axis) |
| Q/E | Move EE up/down (Z axis) |
| R/F | Rotate EE (roll) |
| T/G | Rotate EE (pitch) |
| Y/H | Rotate EE (yaw) |
| Space | Toggle gripper |
| Enter | Save episode |
| Escape | Discard episode |

### 1.3 Recommended Demonstration Counts

| Task Level | Tasks | Episodes/Task | Total Episodes |
|---|---|---|---|
| Level 1 (Simple pick) | 2 | 25 | 50 |
| Level 2 (Pick & place) | 2 | 50 | 100 |
| Level 3 (Multi-object) | 2 | 75 | 150 |
| Level 4 (Complex) | 2 | 100 | 200 |
| Level 5 (Long-horizon) | 2 | 100 | 200 |
| **Total** | **10** | | **~700** |

---

## Step 2: Convert to RLDS Format

### 2.1 Data Structure

Each demonstration episode should contain:
- RGB images (third-person + wrist)
- Proprioceptive state (8D)
- Actions (7D delta EE + gripper)
- Language instruction
- Episode metadata

### 2.2 Dataset Statistics

After collecting demonstrations, compute dataset statistics:

```python
import numpy as np
import json
from pathlib import Path

data_dir = Path("./data/demonstrations")
actions = []
proprios = []

for ep_dir in data_dir.glob("episode_*"):
    actions.append(np.load(ep_dir / "actions.npy"))
    proprios.append(np.load(ep_dir / "proprioception.npy"))

all_actions = np.concatenate(actions, axis=0)
all_proprios = np.concatenate(proprios, axis=0)

# Compute statistics for normalization
stats = {
    "franka_kitchen": {
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "q01": np.percentile(all_actions, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99, axis=0).tolist(),
        },
        "proprio": {
            "mean": all_proprios.mean(axis=0).tolist(),
            "std": all_proprios.std(axis=0).tolist(),
            "min": all_proprios.min(axis=0).tolist(),
            "max": all_proprios.max(axis=0).tolist(),
        }
    }
}

with open("dataset_statistics.json", "w") as f:
    json.dump(stats, f, indent=2)
```

### 2.3 Update Configuration

Update `config/default.yaml` with your dataset statistics:

```yaml
vla_server:
  model:
    unnorm_key: "franka_kitchen"  # Your custom key
```

And update `config/finetune.yaml`:

```yaml
dataset:
  name: "franka_kitchen"
  data_dir: "./data/demonstrations"
  dataset_statistics:
    franka_kitchen:
      action:
        min: [...]  # From computed statistics
        max: [...]  # From computed statistics
        mean: [...]  # From computed statistics
        std: [...]  # From computed statistics
```

---

## Step 3: Fine-Tune the Model

### 3.1 LoRA Fine-Tuning (Recommended)

LoRA fine-tuning is recommended for single-GPU setups. It updates only a small
percentage of parameters while achieving good performance.

```bash
cd openvla-oft

# LoRA fine-tuning
python -m prismatic.vla.train \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --dataset_dir /path/to/isaac-vla/data/demonstrations \
    --run_output_dir ./checkpoints/franka-kitchen-lora \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --proprio_dim 8 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    --save_steps 500 \
    --bf16 \
    --wandb_project isaac-vla \
    --wandb_name franka-kitchen-lora32
```

### 3.2 Full Fine-Tuning

Full fine-tuning updates all model parameters. Requires more GPU memory and
careful learning rate selection.

```bash
# Full fine-tuning (requires gradient accumulation)
python -m prismatic.vla.train \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --dataset_dir /path/to/isaac-vla/data/demonstrations \
    --run_output_dir ./checkpoints/franka-kitchen-full \
    --use_l1_regression True \
    --num_images_in_input 2 \
    --use_proprio True \
    --proprio_dim 8 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --save_steps 500 \
    --bf16 \
    --fsdp 1 \
    --wandb_project isaac-vla \
    --wandb_name franka-kitchen-full
```

### 3.3 Key Hyperparameters

| Parameter | LoRA | Full | Notes |
|---|---|---|---|
| Learning Rate | 5e-5 | 1e-5 | Lower for full fine-tuning |
| Batch Size | 2 | 2 | Per GPU |
| Grad Accumulation | 4 | 8 | Effective batch = 8/16 |
| LoRA Rank | 32 | N/A | Higher = more capacity |
| LoRA Alpha | 32 | N/A | Usually = rank |
| Epochs | 20 | 10 | More for LoRA |
| Warmup Steps | 500 | 500 | |
| LR Schedule | Cosine | Cosine | |

---

## Step 4: Evaluate the Fine-Tuned Model

### 4.1 Update Configuration

Point the VLA server to your fine-tuned checkpoint:

```yaml
# config/default.yaml
vla_server:
  model:
    pretrained_checkpoint: "./checkpoints/franka-kitchen-lora"
    unnorm_key: "franka_kitchen"
```

### 4.2 Run Evaluation

```bash
# Evaluate all tasks
python scripts/evaluate_tasks.py --all --num-episodes 50

# Evaluate specific tasks
python scripts/evaluate_tasks.py --tasks pick_red_block place_red_on_plate --num-episodes 50
```

### 4.3 Expected Performance

| Task | Pretrained (Zero-shot) | Fine-tuned (LoRA) | Fine-tuned (Full) |
|---|---|---|---|
| pick_red_block | ~5% | ~60% | ~75% |
| place_red_on_plate | ~2% | ~50% | ~65% |
| stack_red_on_blue | ~1% | ~30% | ~45% |
| move_mug_to_plate | ~1% | ~25% | ~40% |

*Note: These are estimated targets. Actual performance depends on data quality and quantity.*

---

## Tips for Better Fine-Tuning

1. **Data Quality > Data Quantity**: 50 high-quality demos beat 200 noisy ones
2. **Diverse Starting Positions**: Randomize object positions across episodes
3. **Consistent Language**: Use consistent phrasing for similar tasks
4. **Augmentation**: Use color jitter and random crop during training
5. **Evaluation-in-the-Loop**: Evaluate every 500 training steps
6. **Start from OFT Checkpoint**: Always start from `openvla-7b-oft-finetuned-libero-spatial`, not the base model
7. **Proprioception Helps**: Always include proprioceptive state (8D)
8. **Dual Camera**: Use both third-person and wrist cameras