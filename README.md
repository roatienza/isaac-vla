# Isaac-VLA: OpenVLA-OFT on Franka Emika

A complete system for deploying [OpenVLA-OFT](https://github.com/moojink/openvla-oft) (Optimized Fine-Tuning of Vision-Language-Action models) on a **Franka Emika Panda** robot arm. Supports **LIBERO MuJoCo** (benchmark evaluation, zero domain gap) and **Isaac Sim** (realistic rendering, real-world deployment prep).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [LIBERO Benchmark Evaluation](#libero-benchmark-evaluation)
- [Fine-Tuning on LIBERO Data](#fine-tuning-on-libero-data)
- [Using Your Trained Checkpoint](#using-your-trained-checkpoint)
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
conda create -n vla-oft python=3.10 -y
conda activate vla-oft
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
conda create -n vla-oft python=3.10 -y
conda activate vla-oft
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# LIBERO (MuJoCo benchmark environment)
pip install libero

# Clone LIBERO repository (needed for BDDL task files)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO

# Download LIBERO benchmark datasets and assets
cd LIBERO && python benchmark_scripts/download_libero_datasets.py
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
  max_position_delta: 1.0      # Safety clipping (meters) — DO NOT reduce below 0.5
  max_rotation_delta: 1.0      # Safety clipping (radians) — DO NOT reduce below 0.5
  chunk_size: 8                # Action chunking (predict 8 steps ahead)
```

> **Important**: The `max_position_delta` and `max_rotation_delta` values must be large enough (≥0.5) to allow meaningful robot movements. Values that are too small (e.g., 0.05) will clip actions to near-zero, resulting in 0% success rate regardless of model quality.

### Understanding the Evaluation Pipeline

```
┌─────────────────┐    HTTP     ┌─────────────────────────┐
│  VLA Server      │◄───────────►│  LIBERO Bridge          │
│                  │             │                         │
│ OpenVLA-OFT 7B   │             │ MuJoCo Env              │
│ (L1 Regression)  │             │ Franka Robot            │
│                  │             │ 3P + Wrist Cam          │
└─────────────────┘             └─────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────────────┐
                           │  Observation            │
                           │  224×224 images         │
                           │  8D proprio             │
                           │  Task language           │
                           └─────────────────────────┘
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
- `robot0_gripper_qpos`: Gripper width
- `actions`: 7D delta EE actions (dx, dy, dz, droll, dpitch, dyaw, gripper)

### Step 2: Fine-Tune

```bash
# Fine-tune on LIBERO-Spatial (150K steps, LoRA rank 32)
python scripts/run_libero_finetune.py --suite libero_spatial --max-steps 150000 --lora-rank 32

# Fine-tune on LIBERO-Object
python scripts/run_libero_finetune.py --suite libero_object --max-steps 150000 --lora-rank 32

# Fine-tune on LIBERO-Goal
python scripts/run_libero_finetune.py --suite libero_goal --max-steps 150000 --lora-rank 32

# Fine-tune on LIBERO-10 (combined)
python scripts/run_libero_finetune.py --suite libero_10 --max-steps 150000 --lora-rank 32

# Quick test run (5K steps)
python scripts/run_libero_finetune.py --suite libero_spatial --max-steps 5000
```

### Fine-Tuning Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--suite` | `libero_spatial` | LIBERO task suite |
| `--max-steps` | 150000 | Total training steps |
| `--save-freq` | 10000 | Checkpoint save interval |
| `--batch-size` | 1 | Batch size (VRAM-limited) |
| `--lr` | 0.0005 | Learning rate |
| `--lora-rank` | 32 | LoRA adapter rank |
| `--lora-alpha` | 16 | LoRA scaling factor |
| `--lora-dropout` | 0.0 | LoRA dropout rate |
| `--image-aug` | off | Image augmentation |

### Expected Results

| Training Steps | Expected Success Rate (LIBERO-Spatial) | Notes |
|----------------|----------------------------------------|-------|
| 0 (base model) | ~15% | Zero-shot generalization |
| 5,000 | ~10–20% | Barely one epoch |
| 50,000 | ~60–75% | Partial convergence |
| 100,000 | ~85–92% | Near convergence |
| 150,000 | ~90–97% | Full convergence (OFT paper) |

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

## Using Your Trained Checkpoint

### 1. Locate Your Checkpoint

```bash
ls checkpoints/openvla-7b+libero_spatial_no_noops+*/
# Find the latest checkpoint directory (e.g., checkpoint-10000, checkpoint-150000)
```

### 2. Start the VLA Server with Your Checkpoint

```bash
python scripts/run_vla_server.py \
    --checkpoint /path/to/checkpoint-150000/
```

### 3. Run Evaluation

```bash
# Single task
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

# All tasks in a suite
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks

# All suites
python scripts/run_libero_eval.py --all-suites --num-episodes 50
```

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
│   ├── run_libero_finetune.py       # LIBERO fine-tuning runner
│   ├── run_tui_client.py            # Rich TUI client
│   ├── collect_demonstrations.py    # Data collection script
│   ├── evaluate_tasks.py            # Evaluation runner
│   └── quick_start.py               # Quick start demo
└── docs/
    ├── STEP_BY_STEP.md              # Detailed setup guide
    ├── ARCHITECTURE.md              # Architecture deep-dive
    ├── FINETUNING.md                # Fine-tuning guide
    ├── KITCHEN_TASKS.md             # Kitchen task definitions
    └── PRODUCTION_FINETUNING.md     # Production fine-tuning guide
```

---

## Documentation

- [Step-by-Step Setup Guide](docs/STEP_BY_STEP.md) — Complete installation and setup instructions
- [Architecture Deep-Dive](docs/ARCHITECTURE.md) — System architecture and data flow
- [Fine-Tuning Guide](docs/FINETUNING.md) — How to fine-tune on kitchen tasks
- [Kitchen Task Definitions](docs/KITCHEN_TASKS.md) — Task catalog and success criteria
- [Production Fine-Tuning Guide](docs/PRODUCTION_FINETUNING.md) — Root cause analysis, production configs, and troubleshooting
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
