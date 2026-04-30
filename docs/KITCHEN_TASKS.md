# Kitchen Task Definitions

## Overview

The kitchen environment contains a Franka Emika Panda robot arm on a kitchen
table with various manipulation objects. Tasks are organized by difficulty level
from simple pick-and-place to long-horizon multi-step sequences.

---

## Environment Layout

```
                    ┌─────────────────────────┐
                    │      Backsplash          │
                    │                          │
    ┌───────────────┴─────────────────────────┴───────────────┐
    │                                                          │
    │   🟢(green)    🔵(blue)    🔴(red)     🟡(mug)         │
    │                                                          │
    │              🍽️(plate)                                   │
    │                                                          │
    │                      🤖 (Franka Base)                    │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    X: Forward (toward robot = 0.3, away = 0.8)
    Y: Left (-0.4), Right (0.4)
    Z: Up (table surface = 0.8)
```

## Object Specifications

| Object | Type | Size | Color | Default Position |
|---|---|---|---|---|
| red_block | Cube | 0.04m | Red [1,0,0] | [0.5, 0.0, 0.82] |
| blue_block | Cube | 0.04m | Blue [0,0,1] | [0.5, 0.15, 0.82] |
| green_block | Cube | 0.04m | Green [0,1,0] | [0.5, -0.15, 0.82] |
| white_plate | Cylinder | r=0.06m, h=0.01m | White [1,1,1] | [0.7, 0.0, 0.805] |
| yellow_mug | Cylinder | r=0.03m, h=0.08m | Yellow [1,1,0] | [0.3, 0.1, 0.84] |

---

## Task Catalog

### Level 1: Simple Pick-Up (Difficulty ★)

| Task | Instruction | Success Condition | Max Steps |
|---|---|---|---|
| pick_red_block | "pick up the red block" | Red block lifted >5cm from table | 200 |
| pick_blue_block | "pick up the blue block" | Blue block lifted >5cm from table | 200 |

**Success Detection**: Object Z position > table_height + 0.05m

### Level 2: Pick and Place (Difficulty ★★)

| Task | Instruction | Success Condition | Max Steps |
|---|---|---|---|
| place_red_on_plate | "place the red block on the plate" | Red block on white plate (XY < 0.05m, Z > plate) | 300 |
| place_blue_on_plate | "place the blue block on the plate" | Blue block on white plate | 300 |

**Success Detection**: Object on target (XY distance < tolerance, Z > target Z)

### Level 3: Multi-Object (Difficulty ★★★)

| Task | Instruction | Success Condition | Max Steps |
|---|---|---|---|
| stack_red_on_blue | "stack the red block on top of the blue block" | Red block on blue block (XY < 0.04m) | 400 |
| move_mug_to_plate | "move the yellow mug to the plate" | Yellow mug near white plate (dist < 0.06m) | 400 |

**Success Detection**: Object-on-object with tight tolerance

### Level 4: Complex Manipulation (Difficulty ★★★★)

| Task | Instruction | Success Condition | Max Steps |
|---|---|---|---|
| pick_and_place_mug_upright | "pick up the yellow mug and place it upright on the plate" | Mug on plate + upright orientation | 500 |
| rearrange_blocks | "put the red block on the left side and the blue block on the right side" | Red at left, blue at right | 600 |

**Success Detection**: Multi-condition (position + orientation)

### Level 5: Long-Horizon (Difficulty ★★★★★)

| Task | Instruction | Success Condition | Max Steps |
|---|---|---|---|
| clear_table | "put all blocks on the plate" | All 3 blocks on plate | 800 |
| prepare_table | "place the mug on the plate and stack the red block on the blue block" | Mug on plate + red on blue | 1000 |

**Success Detection**: All conditions must be met simultaneously

---

## Success Detection Implementation

Each task has a `success_condition` in `config/kitchen_tasks.yaml`. The evaluator
checks these conditions using object positions from the simulation:

```python
# Example: "object_on_object" condition
def check_object_on_object(obj_pos, target_pos, tolerance=0.05):
    xy_dist = np.linalg.norm(obj_pos[:2] - target_pos[:2])
    z_diff = obj_pos[2] - target_pos[2]
    return xy_dist < tolerance and z_diff > -0.01
```

### Condition Types

| Type | Parameters | Description |
|---|---|---|
| object_at_location | object, target_position, tolerance | Object near a fixed position |
| object_on_object | object, target_object, tolerance | Object on top of another |
| object_near_object | object, target_object, tolerance | Object near another |
| multi_condition | conditions[] | All conditions must be true |
| all_objects_on_target | objects[], target_object | All objects on target |

---

## Task Difficulty Progression

```
Level 1 ─── Level 2 ─── Level 3 ─── Level 4 ─── Level 5
  Pick      Pick&Place   Multi-Obj   Complex     Long-Horizon
  (2 tasks)  (2 tasks)   (2 tasks)   (2 tasks)   (2 tasks)
```

### Recommended Training Order

1. Start with Level 1 tasks (50 episodes each)
2. Progress to Level 2 (100 episodes each)
3. Add Level 3 (150 episodes each)
4. Attempt Level 4 (200 episodes each)
5. Challenge Level 5 (200 episodes each)

### Expected Success Rates

| Level | Zero-shot | LoRA Fine-tuned | Full Fine-tuned |
|---|---|---|---|
| 1 | 5-10% | 60-70% | 75-85% |
| 2 | 2-5% | 50-60% | 65-75% |
| 3 | 1-3% | 30-40% | 45-55% |
| 4 | 0-2% | 20-30% | 35-45% |
| 5 | 0-1% | 10-20% | 25-35% |