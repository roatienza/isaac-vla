# Isaac-VLA: OpenVLA-OFT on Franka Emika

A complete system for deploying OpenVLA-OFT vision-language-action models on a Franka Emika Panda robot arm. Supports **LIBERO MuJoCo** (benchmark evaluation, zero domain gap) and **Isaac Sim** (realistic rendering, real-world deployment prep).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [LIBERO Benchmark Evaluation](#libero-benchmark-evaluation)
- [Fine-Tuning on LIBERO Data](#fine-tuning-on-libero-data)
- [Isaac Sim (Optional)](#isaac-sim-optional)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Quick Start

Get up and running with LIBERO benchmark evaluation in 5 minutes:

```bash
# 1. Clone and install
git clone https://github.com/roatienza/isaac-vla
cd isaac-vla
pip install -r requirements.txt
pip install libero

# 2. Download LIBERO assets
python -c "import libero; libero.utils.download_libero_assets()"

# 3. Start VLA server (Terminal 1)
python scripts/run_vla_server.py

# 4. Run evaluation (Terminal 2)
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Conda recommended |
| CUDA | 12.1+ | RTX 5090 tested |
| GPU | 16GB+ VRAM | OpenVLA-OFT 7B needs ~16GB |
| Isaac Sim | 4.2.0+ | Optional — for realistic rendering |

### Step 1: Clone Repository

```bash
git clone https://github.com/roatienza/isaac-vla
cd isaac-vla
```

### Step 2: Create Conda Environment

```bash
conda create -n isaac-vla python=3.10 -y
conda activate isaac-vla
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# LIBERO (MuJoCo benchmark environment)
pip install libero

# Download LIBERO benchmark datasets and assets
python -c "import libero; libero.utils.download_libero_assets()"
```

### Step 4: Verify Installation

```bash
# Test LIBERO import
python -c "import libero; print('LIBERO installed:', libero.__version__)"

# Test VLA server startup
python scripts/run_vla_server.py --help
```

---

## LIBERO Benchmark Evaluation

LIBERO provides **zero domain gap** with OpenVLA-OFT training data — same MuJoCo renderer, physics, and camera viewpoints.

### Task Suites

| Suite | Tasks | Description | Max Steps |
|-------|-------|-------------|-----------|
| `libero_spatial` | 10 | Spatial reasoning (pick & place to different locations) | 220 |
| `libero_object` | 10 | Object manipulation (different target objects) | 280 |
| `libero_goal` | 10 | Goal conditioning (achieve different end states) | 300 |
| `libero_10` | 10 | Combined spatial + object + goal tasks | 520 |
| `libero_90` | 90 | Full benchmark (all tasks combined) | 400 |

### Running Evaluation

```bash
# Terminal 1: Start VLA server
python scripts/run_vla_server.py

# Terminal 2: Evaluate single task
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

# Evaluate all tasks in a suite
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks

# Evaluate with multiple episodes per task
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks --num-episodes 10

# Evaluate all suites
python scripts/run_libero_eval.py --all-suites --num-episodes 5

# Record videos during evaluation
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0 --record-video

# Custom output directory
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0 --output-dir ./results
```

### Evaluation Output

Results are saved to `data/libero_results/evaluation_results.json`:

```json
{
  "libero_spatial": {
    "task_suite": "libero_spatial",
    "task_id": 0,
    "task_name": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "num_episodes": 10,
    "success_count": 2,
    "success_rate": 0.2,
    "avg_episode_length": 220.0
  }
}
```

### Configuration

Edit `config/default.yaml` to tune LIBERO evaluation:

```yaml
libero:
  task_suite: "libero_spatial"
  task_id: 0
  num_episodes: 10
  camera_heights: 256
  camera_widths: 256
  center_crop: true
  target_size: [224, 224]
  num_steps_wait: 10           # Physics stabilization steps at episode start
  max_position_delta: 0.05     # Safety clipping (meters)
  max_rotation_delta: 0.1      # Safety clipping (radians)
  chunk_size: 8                # Action chunking (predict 8 steps ahead)
```

### Understanding the Evaluation Pipeline

```
┌─────────────┐    HTTP     ┌─────────────────┐
│  VLA Server  │◄───────────►│  LIBERO Bridge  │
│             │             │                 │
│ OpenVLA-OFT │             │ MuJoCo Env      │
│ (7B params) │             │ Franka Robot    │
│             │             │ 3P + Wrist Cam  │
└─────────────┘             └─────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Observation    │
                           │  224×224 images │
                           │  8D proprio     │
                           │  Task language  │
                           └─────────────────┘
```

1. **Reset**: Environment resets with random initial state from task suite
2. **Stabilize**: First 10 steps use dummy action for physics settling
3. **Observe**: Capture third-person + wrist camera images, extract 8D proprioception
4. **Query VLA**: Send images + language instruction + state to VLA server
5. **Execute**: Apply 8 predicted delta EE actions open-loop
6. **Repeat**: Until task success or max steps reached

---

## Fine-Tuning on LIBERO Data

Fine-tune OpenVLA-OFT on LIBERO demonstrations for improved task success rates.

### Overview

```
LIBERO Datasets → Fine-Tune OpenVLA-OFT → Evaluate → Improved Success Rate
```

### Step 1: Prepare Dataset

LIBERO datasets are downloaded during installation. Verify they exist:

```bash
ls ~/.libero/datasets/
# libero_spatial/  libero_object/  libero_goal/  libero_10/  libero_90/
```

Each dataset contains demonstration trajectories in RLDS format:
- `agentview_image`: Third-person camera (256×256)
- `eye_in_hand_image`: Wrist camera (256×256)
- `robot0_joint_pos`: 7 joint positions
- `robot0_gripper_qpos`: Gripper state
- `actions`: 7D delta EE actions

### Step 2: Configure Fine-Tuning

Edit `config/finetune.yaml`:

```yaml
model:
  base_checkpoint: "moojink/openvla-7b-oft-finetuned-libero-spatial"
  use_l1_regression: true
  num_images_in_input: 2
  use_proprio: true
  proprio_dim: 8

dataset:
  name: "libero_spatial"
  data_dir: "~/.libero/datasets/libero_spatial"
  format: "rlds"

training:
  method: "lora"
  lora:
    rank: 32
    alpha: 32
    dropout: 0.0
  learning_rate: 5.0e-5
  batch_size: 2
  gradient_accumulation_steps: 4
  num_epochs: 20
  warmup_steps: 500
  lr_schedule: "cosine"
```

### Step 3: Run Fine-Tuning

```bash
cd /home/rowel/sandbox/openvla-oft

# LoRA fine-tuning (recommended for single GPU)
python -m prismatic.vla.train \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --dataset_dir ~/.libero/datasets/libero_spatial \
    --run_output_dir ./checkpoints/libero-spatial-lora \
    --use_l1_regression True \
    --num_images_in_input 2 \
    --use_proprio True \
    --proprio_dim 8 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    --bf16 \
    --wandb_project isaac-vla \
    --wandb_name libero-spatial-lora32
```

### Step 4: Evaluate Fine-Tuned Model

```bash
# Point VLA server to fine-tuned checkpoint
python scripts/run_vla_server.py --checkpoint ./checkpoints/libero-spatial-lora

# Run evaluation
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks --num-episodes 50
```

### Hyperparameter Guide

| Parameter | LoRA | Full FT | Notes |
|-----------|------|---------|-------|
| Learning Rate | 5e-5 | 1e-5 | Lower for full FT |
| Batch Size | 2 | 2 | Per GPU |
| Grad Accumulation | 4 | 8 | Effective batch = 8/16 |
| LoRA Rank | 32 | N/A | Higher = more capacity |
| Epochs | 20 | 10 | More for LoRA |
| Warmup Steps | 500 | 500 | Cosine LR schedule |

### Expected Results

| Model | libero_spatial | libero_object | libero_goal | libero_10 |
|-------|---------------|---------------|-------------|-----------|
| OpenVLA-OFT (base) | ~10-20% | ~5-15% | ~10-20% | ~5-15% |
| LoRA Fine-tuned | ~60-80% | ~50-70% | ~55-75% | ~40-60% |
| Full Fine-tuned | ~70-90% | ~60-80% | ~65-85% | ~50-70% |

---

## Isaac Sim (Optional)

For realistic rendering and real-world deployment preparation.

### When to Use Isaac Sim

| Use Case | Recommended Environment |
|----------|------------------------|
| Benchmark evaluation | ✅ LIBERO MuJoCo |
| Fine-tuning | ✅ LIBERO MuJoCo |
| Realistic rendering | ✅ Isaac Sim |
| Real-world deployment prep | ✅ Isaac Sim |
| PhysX physics | ✅ Isaac Sim |

### Isaac Sim Setup

```bash
# Install Isaac Sim 4.2.0+
# Download from: https://developer.nvidia.com/isaac-sim

# Launch sim bridge
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --headless --save-video
```

### LIBERO vs Isaac Sim Comparison

| Feature | LIBERO (MuJoCo) | Isaac Sim |
|---------|----------------|-----------|
| **Visual Domain Gap** | ✅ Zero (matches training) | ❌ Different renderer |
| **Physics** | ✅ MuJoCo (matches training) | ❌ PhysX (different) |
| **IK Failures** | ✅ None (direct EE control) | ⚠️ Possible |
| **Speed** | ✅ Fast (~10-30 FPS) | ⚠️ Slower (~5-15 FPS) |
| **Realism** | ❌ Synthetic | ✅ Realistic rendering |
| **Use Case** | Benchmark evaluation, fine-tuning | Real-world deployment prep |

---

## Project Structure

```
isaac-vla/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── config/
│   ├── default.yaml                 # Master configuration
│   ├── kitchen_tasks.yaml           # Task definitions
│   └── finetune.yaml                # Fine-tuning configuration
├── src/
│   ├── __init__.py                  # Package init
│   ├── api.py                       # Python API (embedded + remote)
│   ├── vla_server.py                # OpenVLA-OFT inference server
│   ├── sim_bridge.py                # Isaac Sim bridge
│   ├── libero_bridge.py             # LIBERO MuJoCo bridge
│   ├── action_pipeline.py           # VLA action → sim action pipeline
│   ├── ik_solver.py                 # Inverse kinematics solvers
│   ├── kitchen_scene.py             # Kitchen scene builder
│   ├── data_collector.py            # Demonstration data collection
│   ├── evaluator.py                 # Task evaluation framework
│   └── utils.py                     # Shared utilities
├── scripts/
│   ├── run_vla_server.py            # VLA server launcher
│   ├── run_sim_bridge.py            # Sim bridge launcher
│   ├── run_libero_eval.py           # LIBERO evaluation runner
│   ├── run_tui_client.py            # Rich TUI client
│   ├── collect_demonstrations.py    # Data collection script
│   ├── evaluate_tasks.py            # Evaluation runner
│   └── quick_start.py               # Quick start demo
└── docs/
    ├── STEP_BY_STEP.md              # Detailed setup guide
    ├── ARCHITECTURE.md              # Architecture deep-dive
    ├── FINETUNING.md                # Fine-tuning guide
    └── KITCHEN_TASKS.md             # Kitchen task definitions
```

---

## Documentation

- [Step-by-Step Setup Guide](docs/STEP_BY_STEP.md) — Complete installation and setup instructions
- [Architecture Deep-Dive](docs/ARCHITECTURE.md) — System architecture and data flow
- [Fine-Tuning Guide](docs/FINETUNING.md) — How to fine-tune on kitchen tasks
- [Kitchen Task Definitions](docs/KITCHEN_TASKS.md) — Task catalog and success criteria
- [Environment Comparison](ENVIRONMENT_COMPARISON.md) — LIBERO vs Isaac Sim detailed comparison

---

## Citation

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
