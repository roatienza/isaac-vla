# Architecture Deep-Dive

## System Architecture

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
│  ┌─────────────────┐    HTTP/REST    │  │ ┌──────────────────┐ │ │   │
│  │   Rich TUI       │◄──────────────►│  │ │ Denorm → Clip →  │ │ │   │
│  │   Client         │   :8889         │  │ │ Delta EE → IK →  │ │ │   │
│  │   (Textual)      │                 │  │ │ Joint Targets    │ │ │   │
│  └─────────────────┘                 │  │ └──────────────────┘ │ │   │
│                                       │  └──────────────────────┘ │   │
│  ┌─────────────────┐                 │                            │   │
│  │   Python API     │◄──Direct───────│  ┌──────────────────────┐ │   │
│  │   IsaacVLAClient│   Import        │  │ Data Collector        │ │   │
│  └─────────────────┘                 │  │ (RLDS/HDF5 format)   │ │   │
│                                       │  └──────────────────────┘ │   │
│                                       └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Camera      │────►│  Observation │────►│  VLA Server   │────►│  Action Chunk │
│  Capture     │     │  Preprocess  │     │  (OpenVLA-OFT)│     │  (8 actions)  │
│  (3P+Wrist)  │     │  (crop,resize│     │  (GPU infer)  │     │  (7D each)    │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                         │
                                                                         ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Isaac Sim   │◄────│  Joint       │◄────│  IK Solver   │◄────│  Action       │
│  Physics     │     │  Position    │     │  (LULA/DLS)  │     │  Pipeline     │
│  Step        │     │  Targets     │     │              │     │  (denorm,clip)│
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Action Representation

OpenVLA-OFT outputs **7D delta end-effector actions**:

| Dimension | Meaning | Range | Notes |
|---|---|---|---|
| 0 | Δx (forward/back) | [-1, 1] | Denormalized from dataset stats |
| 1 | Δy (left/right) | [-1, 1] | |
| 2 | Δz (up/down) | [-1, 1] | |
| 3 | Δroll | [-π, π] | |
| 4 | Δpitch | [-π, π] | |
| 5 | Δyaw | [-π, π] | |
| 6 | gripper | [0, 1] | >0.5 = close, ≤0.5 = open |

The action pipeline converts these to:
1. **Denormalize** using dataset statistics (unnorm_key)
2. **Clip** action magnitudes for safety
3. **Apply delta** to current EE pose → target EE pose
4. **Solve IK** → 7 joint position targets
5. **Interpret gripper** → open/close command

## Action Chunking

OpenVLA-OFT predicts **8 actions at once** (action chunking). The system
executes these in open-loop fashion:

```
Step 0: Query VLA → Get 8 actions
Step 1-8: Execute actions 1-8 (open loop)
Step 9: Query VLA again → Get next 8 actions
...
```

This reduces VLA inference overhead and enables smoother motion.

## Camera Configuration

| Camera | Resolution | Mount | Purpose |
|---|---|---|---|
| Third-person | 256×256 | Overhead/shoulder | Primary observation |
| Wrist | 128×128 | End-effector | Close-up manipulation |

Both images are center-cropped and resized before being sent to the VLA model.

## Proprioception Format

The 8D proprioceptive state sent to VLA:

| Index | Value | Notes |
|---|---|---|
| 0-6 | Joint angles (rad) | Franka 7-DOF |
| 7 | Gripper width (m) | 0.0 = closed, 0.04 = open |

## IK Solver Comparison

| Solver | Accuracy | Speed | Stability | Notes |
|---|---|---|---|---|
| LULA | High | Fast (~1ms) | High | Recommended for Isaac Sim |
| Damped Least Squares | Medium | Medium | Medium | Pure numpy fallback |
| Differential IK | High | Fast | High | Isaac Sim built-in |

## Memory Requirements (RTX 5090 - 32GB)

| Component | VRAM | Notes |
|---|---|---|
| OpenVLA-OFT 7B (bf16) | ~14GB | Inference only |
| OpenVLA-OFT 7B (8-bit) | ~8GB | Quantized inference |
| Isaac Sim Rendering | ~2-4GB | Headless mode less |
| Action Head + Proprio | ~0.5GB | |
| **Total (bf16)** | **~18-20GB** | Fits on RTX 5090 |
| **Total (8-bit)** | **~12-14GB** | More headroom |

## Fine-Tuning Memory (LoRA rank 32)

| Component | VRAM | Notes |
|---|---|---|
| Model (bf16) | ~14GB | |
| LoRA adapters | ~0.5GB | |
| Gradients | ~2GB | |
| Optimizer states | ~4GB | AdamW |
| Activations | ~4GB | |
| **Total** | **~24-26GB** | Fits on RTX 5090 with batch_size=2 |