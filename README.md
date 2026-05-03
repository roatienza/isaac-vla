# Isaac-VLA: OpenVLA-OFT on Franka Emika

A complete system for deploying OpenVLA-OFT vision-language-action models on a Franka Emika Panda robot arm. Supports **LIBERO MuJoCo** (benchmark evaluation, zero domain gap) and **Isaac Sim** (realistic rendering, real-world deployment prep).

---

## Quick Start: LIBERO Benchmark Evaluation (Recommended)

The fastest way to get started — zero visual domain gap with OpenVLA-OFT training data.

### 1. Install Dependencies

```bash
# Install LIBERO (includes MuJoCo and Robosuite)
pip install libero

# Download LIBERO benchmark datasets
cd /home/rowel/miniconda3/envs/vla-oft/lib/python3.10/site-packages/libero
python benchmark_scripts/download_libero_datasets.py
```

### 2. Start VLA Server

```bash
cd /home/rowel/sandbox/isaac-vla
python scripts/run_vla_server.py
```

### 3. Run Evaluation

```bash
# Single task
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

# All tasks in a suite
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks

# All suites
python scripts/run_libero_eval.py --all-suites --num-episodes 10
```

---

## Full Setup Guide

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Conda recommended |
| CUDA | 12.1+ | RTX 5090 tested |
| GPU | 16GB+ VRAM | OpenVLA-OFT 7B needs ~16GB |
| Isaac Sim | 4.2.0+ | Optional — for realistic rendering |

### Install from Source

```bash
# Clone the repository
git clone https://github.com/roatienza/isaac-vla
cd /home/rowel/sandbox/isaac-vla

# Create conda environment
conda create -n isaac-vla python=3.10 -y
conda activate isaac-vla

# Install dependencies
pip install -r requirements.txt

# Install LIBERO (for MuJoCo benchmark)
pip install libero
cd /home/rowel/miniconda3/envs/vla-oft/lib/python3.10/site-packages/libero
python benchmark_scripts/download_libero_datasets.py
```

---

## Running LIBERO Evaluation

LIBERO provides **zero domain gap** with OpenVLA-OFT training data — same MuJoCo renderer, physics, and camera viewpoints.

### Task Suites

| Suite | Tasks | Description | Max Steps |
|-------|-------|-------------|-----------|
| `libero_spatial` | 10 | Spatial reasoning (pick & place to different locations) | 220 |
| `libero_object` | 10 | Object manipulation (different target objects) | 280 |
| `libero_goal` | 10 | Goal conditioning (achieve different end states) | 300 |
| `libero_10` | 10 | Combined spatial + object + goal tasks | 520 |
| `libero_90` | 90 | Full benchmark (all tasks combined) | 400 |

### Usage

```bash
# Start VLA server (Terminal 1)
python scripts/run_vla_server.py --config config/default.yaml

# Evaluate single task (Terminal 2)
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0

# Evaluate all tasks in a suite
python scripts/run_libero_eval.py --task-suite libero_spatial --all-tasks --num-episodes 10

# Evaluate all suites
python scripts/run_libero_eval.py --all-suites --num-episodes 10

# Record videos during evaluation
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0 --record-video

# Custom output directory
python scripts/run_libero_eval.py --task-suite libero_spatial --task-id 0 --output-dir ./results
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
  num_steps_wait: 10           # Physics stabilization steps
  max_position_delta: 0.05     # Safety clipping (meters)
  max_rotation_delta: 0.1      # Safety clipping (radians)
```

### Output

Results are saved to `data/libero_results/evaluation_results.json`:

```json
{
  "libero_spatial": {
    "task_suite": "libero_spatial",
    "task_id": 0,
    "task_name": "pick up the black bowl...",
    "num_episodes": 10,
    "success_count": 2,
    "success_rate": 0.2,
    "avg_episode_length": 220.0
  }
}
```

---

## Fine-Tuning on LIBERO Data

Fine-tune OpenVLA-OFT on LIBERO demonstrations for improved performance.

### Step 1: Collect Demonstrations (Optional)

Use LIBERO's built-in demonstration datasets or collect your own:

```bash
# LIBERO datasets are pre-downloaded during setup
# Check available datasets:
ls ~/.libero/datasets/
```

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

---

## Running Isaac Sim (Optional)

For realistic rendering and real-world deployment preparation.

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

## Visualization Guide

### How to See the Robot

Isaac Sim provides a **3D viewport window** that shows the robot in the kitchen scene. When you run the sim bridge with `headless=False` (the default), this window opens automatically.

**What you'll see:**
- A kitchen counter with colored blocks, a plate, and a mug
- A Franka Emika Panda robot arm mounted at the edge of the counter
- Two camera views (third-person overhead and wrist-mounted)
- The robot moving in real-time as it executes your instructions

**How to interact:**
1. Run the sim bridge (see Launch options above)
2. Wait for the Isaac Sim window to appear (~10-30 seconds startup)
3. Type instructions at the terminal prompt — the robot will execute them
4. Use your mouse to orbit/zoom the 3D viewport:
   - **Left-click + drag**: Orbit the camera
   - **Right-click + drag**: Pan the camera
   - **Scroll wheel**: Zoom in/out

### Headless Mode (No GUI)

If you're running on a remote server without a display, use `--headless`:

```bash
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --headless --save-video
```

You can still capture video with `--save-video` to review the robot's behavior later.

### Livestreaming (Remote Visualization)

If Isaac Sim is running on a remote GPU server, you can livestream the viewport to your local machine using the Isaac Sim WebRTC Streaming Client:

```bash
# On the remote server, run Isaac Sim with streaming enabled:
cd ~/isaacsim
./isaac-sim.streaming.sh

# Or from Python:
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --headless
# (Add LIVESTREAM=1 environment variable to enable WebRTC)
```

Then connect using the **Isaac Sim WebRTC Streaming Client** on your local machine (download from NVIDIA).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        isaac-vla System                              │
│                                                                      │
│  ┌─────────────────┐    HTTP/REST    ┌──────────────────────────┐   │
│  │   VLA Server     │◄──────────────►│   Sim Bridge              │   │
│  │   (GPU:5090)     │   :8777/act    │   (Isaac Sim Process)     │   │
│  │                  │                 │                            │   │
│  │  ┌────────────┐ │                 │  ┌──────────────────────┐ │   │
│  │  │ OpenVLA-OFT│ │                 │  │ Kitchen Scene         │ │   │
│  │  │ 7B Model   │ │                 │  │ ┌────────┐ ┌───────┐ │ │   │
│  │  │ (L1 Regr)  │ │                 │  │ │ Franka │ │Objects│ │ │   │
│  │  └────────────┘ │                 │  │ │  Arm   │ │       │ │ │   │
│  │  ┌────────────┐ │                 │  │ └────────┘ └───────┘ │ │   │
│  │  │Action Head │ │                 │  │ ┌────────┐ ┌───────┐ │ │   │
│  │  │(MLP/FiLM)  │ │                 │  │ │ 3P Cam │ │Wrist  │ │ │   │
│  │  └────────────┘ │                 │  │ │256x256 │ │128x128│ │ │   │
│  │  ┌────────────┐ │                 │  │ └────────┘ └───────┘ │ │   │
│  │  │Proprio Proj│ │                 │  └──────────────────────┘ │   │
│  │  └────────────┘ │                 │                            │   │
│  └─────────────────┘                 │  ┌──────────────────────┐ │   │
│                                       │  │ Action Pipeline       │ │   │
│  ┌─────────────────┐    HTTP/REST    │  │ Denorm → Clip →       │ │   │
│  │   Rich TUI       │◄──────────────►│  │ Delta EE → IK →       │ │   │
│  │   Client         │   :8889         │  │ Joint Targets         │ │   │
│  │   (Textual)      │                 │  └──────────────────────┘ │   │
│  └─────────────────┘                 └──────────────────────────┘   │
│                                                                      │
│  ┌─────────────────┐                                                │
│  │   Python API     │◄──Direct Import──Sim Bridge                   │
│  │   IsaacVLAClient│                                                │
│  └─────────────────┘                                                │
└──────────────────────────────────────────────────────────────────────┘
```

## Features

- **Visual Robot Control**: Watch the Franka robot execute tasks in the Isaac Sim window in real-time
- **Interactive Mode**: Type natural language instructions and see the robot respond immediately
- **Video Recording**: Save MP4 videos of episodes for review and sharing
- **OpenVLA-OFT Model Serving**: FastAPI server for GPU inference with action chunking
- **Isaac Sim Integration**: Franka Emika + kitchen scene with cameras and objects
- **Rich Terminal UI**: Textual-based TUI for interactive control and monitoring
- **Python API**: Programmatic interface for scripting and agent integration
- **Action Pipeline**: VLA → denormalize → clip → IK → joint targets → sim step
- **Data Collection**: Built-in teleoperation and demonstration recording (RLDS format)
- **Evaluation Suite**: Automated task evaluation with success detection
- **IK Solvers**: LULA (recommended), damped least squares (fallback)

## Quick Start

### 1. Install OpenVLA-OFT (VLA Server Environment)

```bash
conda create -n vla-oft python=3.10 -y
conda activate vla-oft

git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .
pip install flash-attn --no-build-isolation
```

### 2. Install Isaac-VLA Dependencies

```bash
cd /home/rowel/sandbox/isaac-vla
pip install -r requirements.txt
```

### 3. Install Isaac Sim (on the desktop with RTX 5090)

```bash
# Option A: pip install (Isaac Sim 4.5+)
pip install isaacsim

# Option B: Download from NVIDIA Omniverse Launcher
# https://developer.nvidia.com/isaac-sim
```

### 4. Launch the System

#### Option A: Watch the Robot in Isaac Sim (Recommended)

The simplest way to **see the robot** is to run the sim bridge with `headless=False` (the default). This opens the Isaac Sim window showing the Franka robot in the kitchen scene. You can then type instructions and watch the robot execute them.

```bash
# Terminal 1: VLA Server (on GPU machine)
conda activate vla-oft
python scripts/run_vla_server.py --config config/default.yaml

# Terminal 2: Isaac Sim Bridge — opens the Isaac Sim window
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --config config/default.yaml
```

When the Isaac Sim window opens, you'll see the kitchen scene with the Franka robot. Type instructions at the `🤖 >` prompt and watch the robot respond:

```
🤖 > pick up the red block
Executing: 'pick up the red block' — watch the Isaac Sim window...
```

#### Option B: Run a Specific Task

Run a single task and watch the robot complete it:

```bash
# With a free-form instruction:
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --task "pick up the red block"

# With a named task from kitchen_tasks.yaml:
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --task-name pick_red_block
```

#### Option C: Save Video of the Episode

If you're running headless or want to save a video:

```bash
# Save video to data/evaluation_videos/
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block" --save-video

# Custom video directory:
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block" --save-video --video-dir /home/rowel/sandbox/isaac-vla/data/evaluation_videos

# Headless mode with video recording:
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py \
    --headless --task "pick up the red block" --save-video
```

#### Option D: Full Interactive Mode

```bash
# Interactive mode with explicit flag:
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --interactive

# Then type instructions, 'reset', 'status', or 'quit':
🤖 Enter instruction (or 'quit'): pick up the red block
🤖 Enter instruction (or 'quit'): reset
🤖 Enter instruction (or 'quit'): quit
```

#### Option E: Remote Mode with TUI Client

```bash
# Terminal 1: VLA Server
python scripts/run_vla_server.py --config config/default.yaml

# Terminal 2: Sim Bridge
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py --config config/default.yaml

# Terminal 3: Rich TUI Client (text-based control panel)
python scripts/run_tui_client.py
```

#### Option F: Embedded Mode (Single Process)

```bash
# Everything in one process (VLA model loaded directly, no HTTP)
<isaac_sim_install>/python.sh /home/rowel/sandbox/isaac-vla/scripts/quick_start.py --instruction "pick up the red block"
```

#### Option G: Python API

```python
from src.api import IsaacVLAClient
client = IsaacVLAClient(mode='embedded')
client.initialize()
result = client.run_task('pick up the red block')
print(result)
```

> **Important**: When using Isaac Sim's `python.sh`, always use **absolute paths** to scripts.
> Isaac Sim's `python.sh` changes the working directory, so relative paths like `scripts/run_sim_bridge.py`
> will not work unless you are in the project directory. For example:
>
> ```bash
> # CORRECT:
> ~/isaacsim/python.sh /home/rowel/sandbox/isaac-vla/scripts/run_sim_bridge.py
>
> # WRONG:
> ~/isaacsim/python.sh scripts/run_sim_bridge.py
> ```
>
> All scripts in this project automatically detect their location and set the working directory
> to the project root, so config file paths (e.g., `config/default.yaml`) will resolve correctly
> even when launched from outside the project directory.
```

## File Structure

```
isaac-vla/
├── README.md                        # This file
├── PLAN.md                          # Detailed execution plan
├── requirements.txt                 # Python dependencies
├── config/
│   ├── default.yaml                 # Master configuration
│   ├── kitchen_tasks.yaml           # Task definitions (10 tasks)
│   └── finetune.yaml                # Fine-tuning configuration
├── src/
│   ├── __init__.py                  # Package init
│   ├── api.py                       # Python API (embedded + remote)
│   ├── vla_server.py                # OpenVLA-OFT inference server
│   ├── sim_bridge.py                # Isaac Sim bridge (Franka + kitchen)
│   ├── libero_bridge.py             # LIBERO MuJoCo bridge (benchmark eval)
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
│   └── quick_start.py              # Quick start demo
├── assets/
│   └── kitchen_scenes/              # USD scene files (optional)
└── docs/
    ├── STEP_BY_STEP.md              # Detailed setup guide
    ├── ARCHITECTURE.md              # Architecture deep-dive
    ├── FINETUNING.md                # Fine-tuning guide
    └── KITCHEN_TASKS.md             # Kitchen task definitions
```

## Kitchen Tasks

| Level | Task | Instruction | Difficulty |
|---|---|---|---|
| 1 | pick_red_block | "pick up the red block" | ★ |
| 1 | pick_blue_block | "pick up the blue block" | ★ |
| 2 | place_red_on_plate | "place the red block on the plate" | ★★ |
| 2 | place_blue_on_plate | "place the blue block on the plate" | ★★ |
| 3 | stack_red_on_blue | "stack the red block on top of the blue block" | ★★★ |
| 3 | move_mug_to_plate | "move the yellow mug to the plate" | ★★★ |
| 4 | pick_and_place_mug_upright | "pick up the yellow mug and place it upright on the plate" | ★★★★ |
| 4 | rearrange_blocks | "put the red block on the left and the blue block on the right" | ★★★★ |
| 5 | clear_table | "put all blocks on the plate" | ★★★★★ |
| 5 | prepare_table | "place the mug on the plate and stack the red block on the blue block" | ★★★★★ |

## Action Pipeline

OpenVLA-OFT outputs **7D delta end-effector actions**:

```
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

The pipeline converts these to sim actions:
1. **Denormalize** using dataset statistics
2. **Clip** action magnitudes for safety
3. **Apply delta** to current EE pose → target EE pose
4. **Solve IK** → 7 joint position targets
5. **Clamp** to joint limits
6. **Interpret gripper** → open/close command

## RTX 5090 Notes

- **VRAM**: 32GB — sufficient for OpenVLA-OFT 7B inference (~16GB)
- **VLA Inference**: ~150-200ms per action chunk
- **Isaac Sim Rendering**: ~30 FPS at 256×256
- **Fine-tuning (LoRA rank 32)**: Fits in single GPU with batch_size=2
- **Full Fine-tuning**: Requires gradient accumulation (batch_size=2, grad_accum=8)

## Documentation

- [Step-by-Step Setup Guide](docs/STEP_BY_STEP.md) — Complete installation and setup instructions
- [Architecture Deep-Dive](docs/ARCHITECTURE.md) — System architecture and data flow
- [Fine-Tuning Guide](docs/FINETUNING.md) — How to fine-tune on kitchen tasks
- [Kitchen Task Definitions](docs/KITCHEN_TASKS.md) — Task catalog and success criteria
- [Execution Plan](PLAN.md) — Detailed implementation plan and status

## Citation

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```