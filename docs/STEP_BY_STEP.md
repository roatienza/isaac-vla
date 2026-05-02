# Step-by-Step Setup Guide

## Complete setup instructions for Isaac-VLA on an RTX 5090 desktop

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| NVIDIA GPU | RTX 5090 (32GB) | Minimum 16GB VRAM for inference |
| NVIDIA Driver | 550+ | Required for Isaac Sim 4.5 |
| Ubuntu | 22.04 LTS | Recommended OS |
| Python | 3.10 | Required by OpenVLA-OFT |
| CUDA | 12.x | Comes with Isaac Sim |
| Isaac Sim | 4.5+ | NVIDIA Omniverse Launcher |
| Conda/Mamba | Latest | Environment management |

---

## Step 1: Install Isaac Sim

### Option A: Via Omniverse Launcher (Recommended)
```bash
# 1. Download and install NVIDIA Omniverse Launcher
#    https://www.nvidia.com/en-us/omniverse/download/

# 2. Open Launcher → Exchange → Find "Isaac Sim" → Install

# 3. Verify installation
~/.local/share/ov/pkg/isaac-sim-*/python.sh -c "import isaacsim; print('Isaac Sim OK')"
```

### Option B: Via pip (Isaac Sim 4.5+)
```bash
pip install isaacsim
```

---

## Step 2: Create the VLA Environment

```bash
# Create conda environment for OpenVLA-OFT
conda create -n vla-oft python=3.10 -y
conda activate vla-oft

# Install PyTorch (match your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and install OpenVLA-OFT
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 (for training)
pip install packaging ninja
ninja --version  # Should return exit code 0
pip install "flash-attn==2.5.5" --no-build-isolation

# Verify installation
python -c "from experiments.robot.openvla_utils import get_vla; print('OpenVLA-OFT OK')"
```

---

## Step 3: Install Isaac-VLA Dependencies

```bash
# In the same vla-oft environment:
cd /path/to/isaac-vla
pip install -r requirements.txt

# Verify key packages
python -c "import fastapi; print('FastAPI OK')"
python -c "import textual; print('Textual OK')"
python -c "import yaml; print('PyYAML OK')"
```

---

## Step 4: Download the Pretrained Model

```bash
# The model will auto-download from HuggingFace on first use.
# Alternatively, pre-download:
python -c "
from experiments.robot.openvla_utils import get_vla, get_processor
from experiments.robot.libero.run_libero_eval import GenerateConfig

cfg = GenerateConfig(
    pretrained_checkpoint='moojink/openvla-7b-oft-finetuned-libero-spatial',
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
)
vla = get_vla(cfg)
processor = get_processor(cfg)
print('Model downloaded and loaded successfully!')
"
```

**Note:** The pretrained model (`openvla-7b-oft-finetuned-libero-spatial`) is trained on LIBERO tasks. For kitchen tasks, you'll need to fine-tune (see Step 8).

---

## Step 5: Verify Isaac Sim Works

```bash
# Test Isaac Sim with a simple Franka example
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac-sim-*

# NOTE: When using python.sh, use absolute paths for scripts
$ISAAC_SIM_PATH/python.sh -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
from isaacsim.core.api import World
world = World()
world.scene.add_default_ground_plane()
print('Isaac Sim works!')
app.close()
"
```

---

## Step 6: Test the VLA Server

```bash
# Terminal 1: Start the VLA server
conda activate vla-oft
python scripts/run_vla_server.py --port 8777

# Terminal 2: Test with a dummy request
python -c "
import requests
import numpy as np
import json

# Health check
resp = requests.get('http://localhost:8777/health')
print('Health:', resp.json())

# Model info
resp = requests.get('http://localhost:8777/model_info')
print('Model:', resp.json())

# Action prediction (with dummy image)
payload = {
    'image': np.zeros((256, 256, 3), dtype=np.uint8).tolist(),
    'instruction': 'pick up the red block',
    'state': np.zeros(8, dtype=np.float32).tolist(),
}
resp = requests.post('http://localhost:8777/act', json=payload)
print('Action:', resp.json())
"
```

---

## Step 7: Run the Full System

### How to Visualize and Watch the Robot

The key to seeing the robot is running Isaac Sim with `headless=False` (the default). This opens a 3D viewport window showing the Franka robot in the kitchen scene. You can type instructions and watch the robot execute them in real-time.

**What you'll see in the Isaac Sim window:**
- Kitchen counter with colored blocks (red, blue, green), a white plate, and a yellow mug
- Franka Emika Panda robot arm
- The robot moving as it executes your natural language instructions

**Viewport controls (mouse):**
- Left-click + drag: Orbit camera around the scene
- Right-click + drag: Pan the camera
- Scroll wheel: Zoom in/out

### Option A: Interactive Mode (Recommended for First-Time Users)

This is the easiest way to see the robot — type instructions and watch:

```bash
# Terminal 1: VLA Server
conda activate vla-oft
python scripts/run_vla_server.py --config config/default.yaml

# Terminal 2: Sim Bridge (opens Isaac Sim window)
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac-sim-*
# IMPORTANT: Use ABSOLUTE path to script, since python.sh changes the working directory
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --config config/default.yaml

# When the window opens, type at the prompt:
🤖 > pick up the red block
🤖 > place the red block on the plate
🤖 > reset
🤖 > quit
```

### Option B: Run a Single Task

```bash
# Run one task and watch the robot complete it:
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block"

# Or use a named task from kitchen_tasks.yaml:
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --task-name pick_red_block
```

### Option C: Save Video of the Episode

If you're running headless or want to save a recording:

```bash
# Save video (MP4) of the episode:
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block" --save-video

# Headless mode with video recording (for remote servers):
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --headless --task "pick up the red block" --save-video

# Custom video output directory:
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block" --save-video --video-dir /tmp/my_videos
```

### Option D: Embedded Mode (Single Machine, No HTTP)

```bash
# Everything in one process (VLA model loaded directly)
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/quick_start.py \
    --instruction "pick up the red block" \
    --config config/default.yaml
```

### Option E: Remote Mode (Separate Processes)

```bash
# Terminal 1: VLA Server (on GPU machine)
conda activate vla-oft
python scripts/run_vla_server.py --config config/default.yaml

# Terminal 2: Isaac Sim Bridge (on sim machine)
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac-sim-*
# IMPORTANT: Use ABSOLUTE path to script, since python.sh changes the working directory
$ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --vla-url http://localhost:8777 \
    --config config/default.yaml

# Terminal 3: TUI Client (any machine)
python scripts/run_tui_client.py --bridge-url http://localhost:8889
```

### Option F: Livestreaming (Remote Visualization)

If Isaac Sim runs on a remote server, you can livestream the viewport using WebRTC:

```bash
# On the remote server:
cd ~/isaacsim
./isaac-sim.streaming.sh

# Then connect from your local machine using the Isaac Sim WebRTC Streaming Client
# Download from: https://developer.nvidia.com/isaac-sim
```

> **Note**: When using Isaac Sim's `python.sh`, always use **absolute paths** to scripts.
> The `python.sh` script changes the working directory, so relative paths won't work
> unless you `cd` into the project directory first. All isaac-vla scripts automatically
> detect their location and set the working directory to the project root.

---

## Step 8: Fine-Tune on Kitchen Tasks

### 8.1 Collect Demonstrations

```bash
# IMPORTANT: Use ABSOLUTE path to script, since python.sh changes the working directory
ISAAC_SIM_PATH/python.sh /abs/path/to/isaac-vla/scripts/collect_demonstrations.py \
    --task "pick up the red block" \
    --num-episodes 50 \
    --output-dir ./data/demonstrations
```

### 8.2 Convert to RLDS Format

```bash
# Convert collected demonstrations to OpenVLA-OFT training format
python -c "
from src.data_collector import DataCollector
# ... conversion script ...
"
```

### 8.3 Fine-Tune the Model

```bash
# Fine-tune OpenVLA-OFT on kitchen data
cd openvla-oft

# LoRA fine-tuning (recommended, fits on single GPU)
python scripts/finetune.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --dataset_dir /path/to/isaac-vla/data/demonstrations \
    --run_output_dir ./checkpoints/franka-kitchen-lora \
    --use_l1_regression=True \
    --num_images_in_input=2 \
    --use_proprio=True \
    --lora_rank=32 \
    --batch_size=2 \
    --num_epochs=20
```

### 8.4 Evaluate Fine-Tuned Model

```bash
# Update config to use fine-tuned checkpoint
# In config/default.yaml, change:
#   pretrained_checkpoint: "./checkpoints/franka-kitchen-lora"

# Then evaluate
python scripts/evaluate_tasks.py --all --num-episodes 50
```

---

## Troubleshooting

### VLA Server Won't Start
```bash
# Check GPU memory
nvidia-smi

# If OOM, try 8-bit quantization
python scripts/run_vla_server.py --config config/default.yaml
# (Set load_in_8bit: true in config)

# If get_vla_action() fails with TypeError, check parameter names:
# The OpenVLA-OFT API uses 'obs' and 'task_label', not 'observation' and 'instruction'
```

### Isaac Sim Import Errors
```bash
# Make sure you're using Isaac Sim's Python
which python  # Should be isaac-sim's python

# Check Isaac Sim version
$ISAAC_SIM_PATH/python.sh -c "import isaacsim; print(isaacsim.__version__)"

# IMPORTANT: When running scripts with python.sh, use ABSOLUTE paths
# WRONG: ~/isaacsim/python.sh scripts/run_sim_bridge.py
# CORRECT: ~/isaacsim/python.sh /home/user/isaac-vla/scripts/run_sim_bridge.py
```

### Camera Issues
```bash
# Verify camera rendering works
$ISAAC_SIM_PATH/python.sh -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
world = World()
cam = Camera(prim_path='/World/Camera', resolution=(256, 256))
world.scene.add(cam)
world.reset()
for i in range(10):
    world.step(render=True)
rgba = cam.get_rgba()
print(f'Camera image shape: {rgba.shape}')
app.close()
"
```

### LULA IK Solver Issues
```bash
# The LULA IK solver requires Isaac Sim 4.5+ with the correct namespace
# Isaac Sim 4.5+ uses: isaacsim.robot_motion.motion_generation
# Older versions use: omni.isaac.motion_generation

# Test LULA IK solver (requires Isaac Sim)
$ISAAC_SIM_PATH/python.sh -c "
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
print('LULA IK solver available (Isaac Sim 4.5+ namespace)')
"
```

### `python.sh` Can't Find Script

If you get an error like `can't open file 'scripts/run_sim_bridge.py': [Errno 2] No such file or directory`,
it's because Isaac Sim's `python.sh` changes the working directory. Use **absolute paths**:

```bash
# WRONG:
~/isaacsim/python.sh scripts/run_sim_bridge.py

# CORRECT:
~/isaacsim/python.sh /path/to/isaac-vla/scripts/run_sim_bridge.py
```

All isaac-vla scripts automatically detect their location and set the working directory
to the project root, so config file paths like `config/default.yaml` will resolve correctly.

### `add_reference_to_stage()` TypeError: got an unexpected keyword argument 'path'

Isaac Sim 4.5+ has **two** `add_reference_to_stage` functions with different parameter names:

| Import path | Second parameter name |
|---|---|
| `isaacsim.core.utils.stage.add_reference_to_stage` | `prim_path` |
| `isaacsim.core.experimental.utils.stage.add_reference_to_stage` | `path` |

The **stable API** (`isaacsim.core.utils.stage`) uses `prim_path`:
```python
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path="...", prim_path="/World/MyAsset")  # ✅ Correct
add_reference_to_stage(usd_path="...", path="/World/MyAsset")      # ❌ TypeError!
```

The **experimental API** (`isaacsim.core.experimental.utils.stage`) uses `path`:
```python
import isaacsim.core.experimental.utils.stage as stage_utils
stage_utils.add_reference_to_stage(usd_path="...", path="/World/MyAsset")  # ✅ Correct
```

### `get_assets_root_path()` Import Changes

In Isaac Sim 4.5+, `get_assets_root_path()` was moved from `isaacsim.core.utils.nucleus`
to `isaacsim.storage.native`. The old import still works but emits a deprecation warning.
The isaac-vla code tries both import paths automatically:

```python
# Preferred (Isaac Sim 4.5+)
from isaacsim.storage.native import get_assets_root_path

# Deprecated (still works, emits warning)
from isaacsim.core.utils.nucleus import get_assets_root_path
```

If neither import works and you see a warning about missing assets, ensure your
Nucleus server is running or set the asset root path manually in Isaac Sim settings.

### Action Pipeline Issues
```bash
# Test IK solver
python -c "
from src.ik_solver import create_ik_solver
solver = create_ik_solver('damped_least_squares')
print('IK solver created (note: requires Isaac Sim for LULA)')
"
```

### I Can't See the Robot / Isaac Sim Window

If you don't see the Isaac Sim window when running the sim bridge:

1. **Make sure `headless` is not set** — The default config has `headless: false`, but check:
   ```yaml
   # In config/default.yaml:
   sim_bridge:
     isaac_sim:
       headless: false  # Must be false to see the window
   ```
   Or don't pass `--headless` on the command line.

2. **Check your display** — If you're running on a remote server via SSH, you need either:
   - X11 forwarding: `ssh -X user@server`
   - WebRTC livestreaming (see "Livestreaming" section above)
   - Headless mode with video recording: `--headless --save-video`

3. **Wait for startup** — Isaac Sim takes 10-30 seconds to start up. You'll see:
   ```
   [app ready]
   [Simulation App Startup Complete]
   ```
   Then the window should appear.

4. **Check GPU drivers** — Isaac Sim requires an NVIDIA GPU with proper drivers:
   ```bash
   nvidia-smi  # Should show your GPU and driver version
   ```

5. **Use the interactive prompt** — After the window opens, type instructions at the terminal:
   ```
   🤖 > pick up the red block
   ```
   You'll see the robot move in the Isaac Sim window.

### How to Save Videos

To record an MP4 video of the robot executing a task:

```bash
# With GUI (you can also watch live):
~/isaacsim/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block" --save-video

# Headless (no GUI, for remote servers):
~/isaacsim/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --headless --task "pick up the red block" --save-video

# Custom output directory:
~/isaacsim/python.sh /abs/path/to/isaac-vla/scripts/run_sim_bridge.py \
    --task "pick up the red block" --save-video --video-dir /tmp/videos
```

Videos are saved as `episode_YYYYMMDD_HHMMSS.mp4` in the output directory.

---

## Performance Benchmarks (RTX 5090)

| Component | Expected Performance |
|---|---|
| VLA Inference (7B) | ~150-200ms per action chunk |
| Isaac Sim Rendering | ~30 FPS at 256×256 |
| IK Solve (LULA) | ~1ms per solve |
| Full Pipeline Latency | ~200-300ms per step |
| VRAM Usage (Inference) | ~16GB |
| VRAM Usage (Fine-tuning LoRA) | ~24GB |