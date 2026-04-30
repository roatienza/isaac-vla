# Isaac-VLA: Execution Plan

## Overview

This document details the step-by-step execution plan for implementing OpenVLA-OFT on a Franka Emika Panda robot arm in NVIDIA Isaac Sim with a kitchen manipulation environment, running on an RTX 5090 desktop.

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Environment Setup
- [x] Install Isaac Sim 4.5+ on RTX 5090 desktop
- [x] Create `vla-oft` conda environment with OpenVLA-OFT
- [x] Create `isaac-vla` project structure
- [x] Verify GPU detection and CUDA compatibility
- [ ] Test Isaac Sim standalone Franka example
- [ ] Test VLA server with dummy observation

### 1.2 VLA Server Implementation
- [x] Implement `src/vla_server.py` — FastAPI server wrapping OpenVLA-OFT
- [x] Implement action chunk caching and warm-up
- [x] Add health-check and model-info endpoints
- [ ] Test with sample LIBERO observation pickle
- [ ] Benchmark inference latency on RTX 5090

### 1.3 Isaac Sim Scene Construction
- [x] Implement `src/kitchen_scene.py` — Kitchen scene builder
- [x] Load Franka Emika Panda USD asset
- [x] Add kitchen counter, table legs, backsplash
- [x] Place manipulation objects (cubes, cylinders)
- [x] Add third-person camera (256×256 overhead)
- [x] Add wrist-mounted camera (128×128)
- [x] Implement camera-to-numpy capture pipeline
- [ ] Test scene rendering in Isaac Sim
- [ ] Verify camera images are correct resolution and format

---

## Phase 2: Core Pipeline (Week 3-4)

### 2.1 Action Pipeline
- [x] Implement `src/action_pipeline.py` — VLA output → sim action
- [x] Implement `src/ik_solver.py` — EE delta pose → joint positions
- [x] Implement action denormalization (dataset statistics)
- [x] Implement action chunking (open-loop execution of N steps)
- [x] Implement gripper open/close threshold logic
- [x] Add safety clamping and collision checking
- [ ] Test action pipeline with recorded VLA outputs
- [ ] Validate IK solutions in simulation

### 2.2 Sim Bridge
- [x] Implement `src/sim_bridge.py` — Isaac Sim control loop
- [x] Implement observation capture (images + proprioception)
- [x] Implement action execution (joint targets → sim step)
- [x] Implement episode management (reset, task loading)
- [x] Add HTTP API for external control
- [ ] Test end-to-end: VLA server → sim bridge → Franka action
- [ ] Verify action chunk timing and open-loop execution

### 2.3 Python API
- [x] Implement `src/api.py` — Unified client interface
- [x] `IsaacVLAClient` class with `run_task()`, `step()`, `reset()`
- [x] Embedded mode (direct import, no HTTP)
- [x] Remote mode (HTTP to bridge server)
- [ ] Test embedded mode in Isaac Sim
- [ ] Test remote mode with separate processes

---

## Phase 3: User Interface (Week 5)

### 3.1 Rich TUI Client
- [x] Implement `scripts/run_tui_client.py` — Textual-based TUI
- [x] Dashboard layout: camera feed, joint state, task status
- [x] Task input: type natural language instruction
- [x] Action log: real-time display of VLA actions
- [x] Manual control: keyboard teleoperation fallback
- [x] Episode recording controls
- [ ] Test TUI with live simulation
- [ ] Add camera image rendering (ASCII or sixel)

### 3.2 Configuration System
- [x] Implement `config/default.yaml` — Master config
- [x] Implement `config/kitchen_tasks.yaml` — Task definitions
- [x] Implement `config/finetune.yaml` — Fine-tuning config
- [x] Add environment variable overrides
- [ ] Add config validation schema

---

## Phase 4: Data & Evaluation (Week 6-7)

### 4.1 Data Collection
- [x] Implement `src/data_collector.py` — RLDS format recording
- [x] Implement teleoperation interface (keyboard → joint targets)
- [x] Auto-save episodes with task description labels
- [x] Convert to OpenVLA-OFT fine-tuning format
- [ ] Test data collection in simulation
- [ ] Validate RLDS output format

### 4.2 Evaluation Suite
- [x] Implement `src/evaluator.py` — Task evaluation framework
- [x] Define success criteria for each kitchen task
- [x] Implement automatic success/failure detection
- [x] Track metrics: success rate, episode length, action variance
- [x] Generate evaluation reports
- [ ] Test evaluation on pretrained model (zero-shot)
- [ ] Validate success detection accuracy

### 4.3 Fine-Tuning Pipeline
- [x] Create fine-tuning dataset from collected demos
- [x] Configure LoRA/Full fine-tuning for Franka action space
- [x] Document fine-tuning workflow
- [ ] Collect 50+ demonstrations per task
- [ ] Run LoRA fine-tuning on RTX 5090
- [ ] Evaluate fine-tuned model vs pretrained

---

## Phase 5: Integration & Polish (Week 8)

### 5.1 End-to-End Testing
- [ ] Test all 10 kitchen tasks with pretrained OpenVLA-OFT
- [ ] Test all 10 kitchen tasks with fine-tuned model
- [ ] Benchmark latency and throughput
- [ ] Stress test: long episodes, error recovery

### 5.2 Documentation
- [x] Complete STEP_BY_STEP.md setup guide
- [x] Complete ARCHITECTURE.md deep-dive
- [x] Complete FINETUNING.md guide
- [x] Complete KITCHEN_TASKS.md task catalog
- [ ] Add inline code documentation (docstrings)
- [ ] Add troubleshooting section

### 5.3 Polish
- [x] Add logging and error handling throughout
- [ ] Add unit tests for action pipeline
- [ ] Add integration tests for sim bridge
- [ ] Performance optimization (camera capture, action latency)

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| VLA Model | OpenVLA-OFT 7B (L1 regression) | Best performance on manipulation, fast inference |
| Action Space | 7-DOF delta EE pose + gripper | OpenVLA-OFT native output format |
| Action Conversion | Delta EE → IK → joint positions | Avoids IK ambiguity, uses LULA solver |
| Camera | Third-person (256×256) + wrist (128×128) | Matches OpenVLA-OFT dual-camera input |
| Proprioception | 8D (7 joint angles + gripper width) | Matches Franka DOF |
| Sim Frequency | 120 Hz physics, 30 Hz control, ~5 Hz VLA | Balance fidelity and compute |
| Communication | HTTP REST (VLA↔Bridge) | Simple, debuggable, language-agnostic |
| TUI Framework | Textual (Python) | Rich terminal UI, async support |
| Data Format | RLDS/OXE format | Compatible with OpenVLA-OFT fine-tuning |
| IK Solver | LULA (NVIDIA) | Fast, stable, integrated with Isaac Sim |

---

## Risk Mitigation

| Risk | Mitigation | Status |
|---|---|---|
| `python.sh` can't find script | Use absolute paths when running with python.sh | Fixed |
| VLA `get_vla_action()` TypeError | Fixed parameter names: `obs`/`task_label` instead of `observation`/`instruction` | Fixed |
| LULA IK import path | Updated to try `isaacsim.robot_motion.motion_generation` first, fallback to `omni.isaac.motion_generation` | Fixed |
| VLA zero-shot performance poor on kitchen | Start with simple pick-place tasks; fine-tune incrementally | Planned |
| IK solver instability | Use LULA IK with redundancy resolution; add joint limit clamping | Implemented |
| Isaac Sim camera latency | Use GPU-accelerated rendering pipeline; cache frames | Needs testing |
| Action chunk drift | Monitor EE pose error; re-query VLA when error exceeds threshold | Implemented |
| VRAM contention (VLA + Sim) | Run VLA server in separate process; use 8-bit quantization if needed | Config ready |
| Wrist camera mount issues | Use relative transform from EE link; test in sim first | Needs testing |
| OpenVLA-OFT action format mismatch | Verify unnorm_key and action dimensions match Franka | Needs testing |

---

## Success Criteria

1. **MVP**: Franka picks up a red block and places it on a plate, controlled by natural language via VLA
2. **Target**: 5+ kitchen tasks with >50% success rate using fine-tuned OpenVLA-OFT
3. **Stretch**: Full kitchen task suite (10 tasks) with agent integration from research plan

---

## Implementation Status

| Module | File | Status |
|---|---|---|
| VLA Server | `src/vla_server.py` | ✅ Implemented + Fixed (`obs`/`task_label` params) |
| Sim Bridge | `src/sim_bridge.py` | ✅ Implemented |
| Action Pipeline | `src/action_pipeline.py` | ✅ Implemented |
| IK Solver | `src/ik_solver.py` | ✅ Implemented + Fixed (Isaac Sim 4.5+ namespace) |
| Kitchen Scene | `src/kitchen_scene.py` | ✅ Implemented |
| Data Collector | `src/data_collector.py` | ✅ Implemented |
| Evaluator | `src/evaluator.py` | ✅ Implemented |
| Python API | `src/api.py` | ✅ Implemented |
| Utilities | `src/utils.py` | ✅ Implemented |
| VLA Server Script | `scripts/run_vla_server.py` | ✅ Implemented + Fixed (absolute path handling) |
| Sim Bridge Script | `scripts/run_sim_bridge.py` | ✅ Implemented + Fixed (absolute path + cwd handling) |
| TUI Client | `scripts/run_tui_client.py` | ✅ Implemented + Fixed (absolute path handling) |
| Data Collection Script | `scripts/collect_demonstrations.py` | ✅ Implemented + Fixed (absolute path handling) |
| Evaluation Script | `scripts/evaluate_tasks.py` | ✅ Implemented + Fixed (absolute path handling) |
| Quick Start | `scripts/quick_start.py` | ✅ Implemented + Fixed (absolute path handling) |
| Config (default) | `config/default.yaml` | ✅ Implemented |
| Config (tasks) | `config/kitchen_tasks.yaml` | ✅ Implemented |
| Config (finetune) | `config/finetune.yaml` | ✅ Implemented |
| Setup Guide | `docs/STEP_BY_STEP.md` | ✅ Implemented + Updated (python.sh path notes) |
| Architecture | `docs/ARCHITECTURE.md` | ✅ Implemented |
| Fine-tuning Guide | `docs/FINETUNING.md` | ✅ Implemented + Updated (python.sh path notes) |
| Kitchen Tasks | `docs/KITCHEN_TASKS.md` | ✅ Implemented |