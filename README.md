# Isaac-VLA: OpenVLA-OFT on Franka Emika

A complete system for deploying OpenVLA-OFT vision-language-action models on a Franka Emika Panda robot arm. Supports **LIBERO MuJoCo** (benchmark evaluation, zero domain gap) and **Isaac Sim** (realistic rendering, real-world deployment prep).

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
  max_position_delta: 1.0      # Safety clipping (meters) — DO NOT reduce below 0.5
  max_rotation_delta: 1.0      # Safety clipping (radians) — DO NOT reduce below 0.5
  chunk_size: 8                # Action chunking (predict 8 steps ahead)
```

> **Important**: The `max_position_delta` and `max_rotation_delta` values must be large enough (≥0.5) to allow meaningful robot movements. Values that are too small (e.g., 0.05) will clip actions to near-zero, resulting in 0% success rate regardless of model quality.

### Understanding the Evaluation Pipeline

```
┌──────────────────┐    HTTP     ┌──────────────────────┐
│  VLA Server      │◄───────────►│  LIBERO Bridge       │
│                  │             │                      │
│ OpenVLA-OFT 7B   │             │ MuJoCo Env           │
│ (L1 Regression)  │             │ Franka Robot         │
│                  │             │ 3P + Wrist Cam       │
└──────────────────┘             └──────────────────────┘
                                    │
                                    ▼
                           ┌──────────────────────┐
                           │  Observation         │
                           │  224×224 images      │
                           │  8D proprio          │
                           │  Task language       │
                           └──────────────────────┘
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

### Step 2: Quick Test Run (5,000 Steps)

For a quick validation that fine-tuning works:

```bash
python scripts/run_libero_finetune.py \
    --suite libero_spatial \
    --max-steps 5000 \
    --save-freq 5000 \
    --batch-size 1 \
    --lr 0.0005 \
    --lora-rank 32
```

This takes ~23 minutes on an RTX 5090 and saves a checkpoint at step 5,000.

### Step 3: Production Fine-Tuning (150,000 Steps)

For production-quality results matching the OFT paper (~97% success rate):

```bash
python scripts/run_libero_finetune.py \
    --suite libero_spatial \
    --max-steps 150000 \
    --save-freq 10000 \
    --batch-size 1 \
    --lr 0.0005 \
    --lora-rank 64
```

This takes ~11 hours on an RTX 5090 and saves checkpoints every 10,000 steps.

### Step 4: Fine-Tune All Task Suites

To train a generalist model across all LIBERO tasks:

```bash
python scripts/run_libero_finetune.py \
    --suite libero_90 \
    --max-steps 150000 \
    --save-freq 10000 \
    --batch-size 1 \
    --lr 0.0005 \
    --lora-rank 64
```

### Hyperparameter Guide

| Parameter | Quick Test | Production | Notes |
|-----------|-----------|------------|-------|
| `--max-steps` | 5,000 | 150,000 | 150K recommended for best results |
| `--save-freq` | 5,000 | 10,000 | Must be ≤ max-steps |
| `--batch-size` | 1 | 1 | Larger batches need more VRAM |
| `--lr` | 0.0005 | 0.0005 | Learning rate for LoRA |
| `--lora-rank` | 32 | 64 | Higher rank = more capacity |
| `--lora-alpha` | 32 | 64 | Typically equals rank |
| `--lora-dropout` | 0.0 | 0.0 | Dropout for LoRA layers |

### Expected Results

| Training Steps | libero_spatial | libero_object | libero_goal | libero_10 |
|---------------|---------------|---------------|-------------|-----------|
| 0 (base model) | ~10-20% | ~5-15% | ~10-20% | ~5-15% |
| 5,000 | ~20-40% | ~15-30% | ~15-35% | ~10-25% |
| 50,000 | ~70-85% | ~60-75% | ~65-80% | ~50-65% |
| 100,000 | ~90-95% | ~85-92% | ~85-93% | ~75-88% |
| 150,000 | ~95-98% | ~92-96% | ~90-95% | ~85-92% |

> **Note**: Results vary based on hyperparameters, dataset quality, and random seed. The OFT paper reports ~97% on spatial tasks after 150K steps.

---

## Using Your Trained Checkpoint

After fine-tuning, your checkpoint will be saved in the `checkpoints/` directory. Here's how to use it:

### 1. Locate Your Checkpoint

```bash
# List available checkpoints
ls -la checkpoints/

# Example checkpoint path pattern:
# checkpoints/openvla-7b+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_spatial_ft_lora32_bs1/checkpoint-5000/
```

### 2. Start VLA Server with Your Checkpoint

```bash
# Terminal 1: Start VLA server pointing to your checkpoint
python scripts/run_vla_server.py \
    --checkpoint /path/to/your/checkpoint-5000/
```

### 3. Evaluate on LIBERO Tasks

```bash
# Terminal 2: Evaluate single task
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

# Evaluate all tasks in a suite
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks --num-episodes 10

# Evaluate all suites with your fine-tuned model
python scripts/run_libero_eval.py --all-suites --num-episodes 5
```

### 4. Troubleshooting Poor Performance

If you're getting 0% success rate, check these common issues:

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Action clipping too aggressive** | Robot barely moves | Set `max_position_delta: 1.0` and `max_rotation_delta: 1.0` in `config/default.yaml` |
| **Insufficient training** | Random movements | Train for at least 50K steps (150K recommended) |
| **Wrong checkpoint** | Server uses base model | Verify `--checkpoint` path points to fine-tuned weights |
| **Proprio dimension mismatch** | Server 500 errors | Ensure state is 8D (7 joints + 1 gripper) |
| **Missing LIBERO files** | Env creation fails | Run `python -c "import libero; libero.utils.download_libero_assets()"` |

For detailed analysis and production fine-tuning guide, see [docs/PRODUCTION_FINETUNING.md](docs/PRODUCTION_FINETUNING.md).

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

## Development with OnIt Agent

The **OnIt agent** is an autonomous development assistant that can manage isaac-vla development tasks, run experiments, and automate workflows.

### Quick Start

```bash
# Launch OnIt agent with thinking mode and unrestricted access
onit --think --unrestricted --target-env vla-oft

# The agent will:
# - Activate the vla-oft conda environment
# - Clone/update the isaac-vla repository
# - Execute development tasks autonomously
# - Commit and push changes to the repository
```

### Agent Capabilities

| Capability | Description |
|------------|-------------|
| **Code Development** | Write, modify, and debug Python code |
| **Experiment Management** | Run fine-tuning, evaluation, and data collection |
| **Documentation** | Generate and update documentation |
| **Git Management** | Commit, push, and manage branches |
| **File System** | Read, write, and organize project files |
| **Web Research** | Search for documentation and solutions |
| **Process Management** | Start/stop servers and background processes |

### Example Agent Commands

```bash
# Ask the agent to implement a feature
onit --think --unrestricted --target-env vla-oft "implement LIBERO evaluation pipeline"

# Ask the agent to debug an issue
onit --think --unrestricted --target-env vla-oft "fix the 0% success rate in LIBERO evaluation"

# Ask the agent to update documentation
onit --think --unrestricted --target-env vla-oft "update README with fine-tuning guide"

# Ask the agent to run experiments
onit --think --unrestricted --target-env vla-oft "fine-tune OpenVLA-OFT on libero_spatial for 150K steps"
```

### Agent Workflow

```
User Request → OnIt Agent → Execute Tools → Modify Code → Test → Commit → Push
     ↑                                                                              ↓
     └──────────────────────────────────────────────────────────────────────────────┘
```

The agent operates in the `vla-oft` conda environment and has full access to:
- File system operations (read, write, edit)
- Shell command execution
- Git operations (commit, push, branch management)
- Web search and documentation lookup
- Process management (start/stop servers)

### Best Practices

1. **Be specific**: Provide clear, detailed instructions for complex tasks
2. **Review changes**: The agent commits and pushes automatically — review the git history
3. **Use `--think`**: Enables deeper reasoning for complex development tasks
4. **Use `--unrestricted`**: Allows full tool access for comprehensive development
5. **Specify `--target-env`**: Ensures the agent uses the correct conda environment

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
