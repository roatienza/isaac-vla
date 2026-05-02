# Isaac Sim vs LIBERO/OpenVLA-OFT Training Environment Comparison

## Executive Summary

The Isaac Sim environment is **moderately close** to the LIBERO training environment but has several **significant mismatches** that will degrade OpenVLA-OFT performance. The most critical issues are:

1. **Robot mounting geometry** — Franka base sits ON the table instead of behind it
2. **Third-person camera angle** — elevated/side view vs LIBERO's front-facing agentview
3. **Wrist camera orientation** — looks down vs LIBERO's forward-looking wrist cam
4. **Rendering engine** — Isaac Sim (ray-traced) vs MuJoCo (OpenGL rasterized)

---

## Detailed Comparison

### 1. Robot Setup

| Parameter | LIBERO (MuJoCo/Robosuite) | Isaac Sim (Current) | Match? |
|-----------|---------------------------|---------------------|--------|
| Robot | Franka Emika Panda | Franka Emika Panda | ✅ |
| Robot base position | Behind table, on floor | ON table surface at z=0.8 | ❌ **Critical** |
| Table height | 0.8m (standard) | 0.8m | ✅ |
| Table dimensions | ~1.5m × 1.5m | 1.5m × 1.5m | ✅ |
| Robot reach to table | Arm extends forward over table edge | Arm starts at table center, limited forward reach | ❌ **Critical** |
| Gripper | Panda gripper (2 fingers) | Panda gripper (2 fingers) | ✅ |
| Control mode | Delta EE (position + orientation) | Delta EE → IK → joint position | ✅ (equivalent) |

**Issue**: In LIBERO, the Franka base is positioned **behind the table on the floor**, with the arm extending forward over the table. In Isaac Sim, the base is placed **on top of the table** at z=0.8, which fundamentally changes the reachable workspace and the visual appearance of the arm.

### 2. Camera Setup

#### Third-Person Camera (Agentview)

| Parameter | LIBERO (Agentview) | Isaac Sim (Current) | Match? |
|-----------|--------------------|--------------------|--------|
| Resolution | 256×256 | 256×256 | ✅ |
| Preprocessing | Center crop → 224×224 | Center crop → 224×224 | ✅ |
| Position | Front-facing, ~1.5m from robot, elevated ~1m | Side-elevated: [0.8386, 0.0, 2.2904] | ❌ **Significant** |
| FOV | ~45-50° (MuJoCo default) | 75° | ❌ **Significant** |
| View angle | Front-facing, looking down at table | Side-elevated, looking at table center | ❌ **Significant** |
| Camera type | MuJoCo offscreen renderer | Isaac Sim ray-traced camera | ❌ **Moderate** |

**Issue**: LIBERO's "agentview" camera is positioned **in front of the robot**, providing a view similar to a human operator standing in front of the robot. The Isaac Sim camera is positioned **to the side and much higher**, giving a very different perspective. The FOV is also wider (75° vs ~45°).

#### Wrist Camera

| Parameter | LIBERO (Wrist) | Isaac Sim (Current) | Match? |
|-----------|----------------|--------------------|--------|
| Resolution | 256×256 | 256×256 | ✅ |
| Preprocessing | Center crop → 224×224 | Center crop → 224×224 | ✅ |
| Mount point | panda_hand (TCP) | panda_hand (TCP) | ✅ |
| Position offset | Small forward offset | [0.05, 0, 0] (5cm forward) | ✅ |
| Orientation | Looks forward along arm/fingers | Looks DOWN (-Z in hand frame) | ❌ **Significant** |
| FOV | ~30-45° (MuJoCo default) | ~37° (focal_length=0.8m) | ✅ |
| Camera type | MuJoCo offscreen renderer | Isaac Sim ray-traced camera | ❌ **Moderate** |

**Issue**: LIBERO's wrist camera looks **forward along the arm** (in the direction the gripper points), showing what the gripper is approaching. The Isaac Sim wrist camera looks **downward** (-Z in hand frame), which is a different viewpoint.

### 3. Action Space

| Parameter | LIBERO/OpenVLA-OFT | Isaac Sim (Current) | Match? |
|-----------|--------------------|--------------------|--------|
| Action type | 7D delta EE [dx, dy, dz, droll, dpitch, dyaw, gripper] | 7D delta EE → IK → joint positions | ✅ |
| Action normalization | min/max normalization | Same (libero_spatial_no_noops) | ✅ |
| Action chunking | 8 steps (NUM_ACTIONS_CHUNK) | 8 steps | ✅ |
| Proprioception | 8D (7 joints + gripper) | 8D (7 joints + gripper) | ✅ |
| Gripper encoding | Continuous [0, 1] | Continuous [0, 1] → threshold | ✅ |
| Control frequency | 10 Hz (LIBERO default) | 30 Hz | ❌ **Moderate** |
| IK method | MuJoCo built-in IK | LULA IK solver | ❌ **Moderate** |

### 4. Rendering & Visual Appearance

| Parameter | LIBERO (MuJoCo) | Isaac Sim | Match? |
|-----------|-----------------|-----------|--------|
| Rendering engine | MuJoCo (OpenGL rasterization) | Isaac Sim (ray-traced/RTX) | ❌ **Significant** |
| Lighting | Simple directional + ambient | Environment lighting + shadows | ❌ **Moderate** |
| Object textures | Simple colored materials | Can be more realistic | ❌ **Moderate** |
| Shadows | Basic | Realistic (RTX) | ❌ **Moderate** |
| Background | Solid color / simple | Ground plane + environment | ❌ **Moderate** |
| Image quality | Clean, synthetic | More realistic, with noise | ❌ **Moderate** |

**Issue**: MuJoCo produces clean, synthetic-looking images with simple lighting. Isaac Sim produces more realistic images with shadows, reflections, and potentially noise. This creates a **domain gap** that the model wasn't trained on.

### 5. Physics & Dynamics

| Parameter | LIBERO (MuJoCo) | Isaac Sim | Match? |
|-----------|-----------------|-----------|--------|
| Physics engine | MuJoCo | PhysX | ❌ **Moderate** |
| Contact model | Pyramidal + soft | PhysX contact | ❌ **Moderate** |
| Simulation frequency | 10 Hz (control) | 120 Hz physics, 30 Hz render | ❌ **Moderate** |
| Object dynamics | Rigid body | Rigid body | ✅ |

---

## Impact Assessment

### Critical Issues (Will cause failures)

1. **Robot mounting geometry**: The Franka base position fundamentally changes the reachable workspace and arm appearance. The model expects to see the arm extending from behind the table, not starting from the table surface.

2. **Third-person camera viewpoint**: The model was trained on front-facing agentview images. The side-elevated view will be completely unfamiliar, causing the model to misinterpret spatial relationships.

3. **Wrist camera orientation**: The model expects to see forward-looking wrist images. The downward-looking view shows different visual content.

### Significant Issues (Will degrade performance)

4. **Rendering domain gap**: MuJoCo vs Isaac Sim produces visually different images. The model may not generalize well to the more realistic Isaac Sim rendering.

5. **FOV mismatch**: Third-person camera FOV is 75° vs ~45° in LIBERO, changing the field of view.

### Minor Issues (Manageable)

6. **Control frequency**: 30 Hz vs 10 Hz — the model predicts actions at a different rate than it was trained on, but the action chunking should handle this.

7. **IK solver differences**: MuJoCo vs LULA IK may produce slightly different joint configurations for the same EE pose.

---

## Recommendations

### Short-term (Quick fixes)

1. **Fix robot mounting**: Move Franka base to floor position behind the table (e.g., `[0.0, 0.0, 0.0]` or similar), matching LIBERO's setup.

2. **Fix third-person camera**: Reposition to front-facing agentview:
   - Position: ~`[1.5, 0.0, 1.5]` (in front of robot, elevated)
   - Target: table center `[0.75, 0.0, 0.8]`
   - FOV: ~45° (focal_length ≈ 1.5m for 256×256)

3. **Fix wrist camera**: Rotate to look forward along fingers (+X direction):
   - Orientation: identity or small adjustment to look forward

### Medium-term (Better performance)

4. **Match rendering**: Use simpler lighting, disable shadows, use solid background to match MuJoCo appearance.

5. **Match object appearance**: Use simple colored materials (no textures) to match LIBERO objects.

### Long-term (Best performance)

6. **Fine-tune on Isaac Sim data**: Collect demonstrations in Isaac Sim and fine-tune the OpenVLA-OFT model. This is the **only way to fully bridge the domain gap**.

7. **Use domain randomization**: Randomize lighting, camera angles, and object appearances during fine-tuning to improve generalization.

---

## Conclusion

**The Isaac Sim environment is NOT close enough to the LIBERO training environment for reliable OpenVLA-OFT performance out-of-the-box.** The critical mismatches in robot mounting, camera viewpoints, and rendering will cause:

- **IK failures**: The model predicts actions based on LIBERO's workspace geometry, which doesn't match Isaac Sim's setup
- **Poor action predictions**: The model sees unfamiliar camera views and may output random or incorrect actions
- **Low task success rates**: Even if IK succeeds, the actions may not be appropriate for the task

**Minimum viable fix**: Correct the robot mounting and camera viewpoints to match LIBERO. This should reduce IK failures and improve action quality, but some domain gap will remain due to rendering differences.

**Recommended approach**: Fix the geometry/viewpoints first, then collect Isaac Sim demonstrations and fine-tune the model for best results.
