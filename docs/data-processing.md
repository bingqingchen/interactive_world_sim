# Data Processing: R1 Lite MCAP → IWS HDF5 Harmonization

## 1. Source Data Summary (R1 Lite ROS 2 MCAP Bags)

### 1.1 Archive Structure

```
data/
└── session_YYYYMMDD_HHMMSS/
    ├── session.json              # session-level metadata (host, instruction, topics list)
    ├── trajectories.jsonl        # per-trajectory metadata (one JSON object per line)
    └── trajectories/
        └── traj_NNNN/
            ├── meta.json         # per-trajectory metadata (labels, duration, instruction)
            ├── rosbag_record.log # rosbag2 recording log
            └── bag/
                ├── metadata.yaml # MCAP bag metadata (topics, message counts, QoS)
                └── bag_0.mcap    # serialized ROS 2 messages (CDR format)
```

### 1.2 Robot Platform

**R1 Lite** — a mobile bimanual humanoid with:
- **Chassis:** 3× steer + 3× wheel joints (6 joints)
- **Torso:** 3 joints (`torso_joint1..3`)
- **Left arm:** 6 joints (`left_arm_joint1..6`) + 2 gripper finger joints
- **Right arm:** 6 joints (`right_arm_joint1..6`) + 2 gripper finger joints
- **Total joints in `/joint_states`:** 25

This differs from the existing ALOHA setup which uses 2× Interbotix arms (7 joints per arm = 14 total, or 16 including finger positions).

### 1.3 ROS 2 Topics (Relevant for Conversion)

#### Proprioception & Control

| Topic | Type | Rate | Content |
|-------|------|------|---------|
| `/joint_states` | `JointState` | 500 Hz | All 25 joint positions, velocities, efforts |
| `/hdas/feedback_arm_left` | `JointState` | 200 Hz | Left arm 6 joint pos/vel/effort |
| `/hdas/feedback_arm_right` | `JointState` | 200 Hz | Right arm 6 joint pos/vel/effort |
| `/hdas/feedback_gripper_left` | `JointState` | 200 Hz | Left gripper position (scalar, range ~0–100) |
| `/hdas/feedback_gripper_right` | `JointState` | 200 Hz | Right gripper position (scalar, range ~0–100) |
| `/hdas/feedback_torso` | `JointState` | 500 Hz | Torso 3 joint positions |
| `/hdas/feedback_chassis` | `JointState` | 200 Hz | Chassis joint positions |
| `/motion_control/pose_ee_arm_left` | `PoseStamped` | 50 Hz | Left EE pose (position + quaternion) |
| `/motion_control/pose_ee_arm_right` | `PoseStamped` | 50 Hz | Right EE pose (position + quaternion) |
| `/motion_control/twist_ee_arm_left` | `TwistStamped` | 50 Hz | Left EE velocity |
| `/motion_control/twist_ee_arm_right` | `TwistStamped` | 50 Hz | Right EE velocity |
| `/motion_control/pose_floating_base` | `PoseStamped` | 50 Hz | Floating base pose |
| `/motion_target/target_joint_state_arm_left` | `JointState` | 200 Hz | Commanded left arm positions (6 joints) |
| `/motion_target/target_joint_state_arm_right` | `JointState` | 200 Hz | Commanded right arm positions (6 joints) |
| `/motion_target/target_position_gripper_left` | `JointState` | 20 Hz | Commanded left gripper position (scalar) |
| `/motion_target/target_position_gripper_right` | `JointState` | 20 Hz | Commanded right gripper position (scalar) |
| `/tabletop/hdas/feedback_arm_left` | `JointState` | 190 Hz | Tabletop left arm feedback (6 joints) |
| `/tabletop/hdas/feedback_arm_right` | `JointState` | 190 Hz | Tabletop right arm feedback (6 joints) |

#### Cameras

| Topic | Type | Rate | Resolution | Encoding |
|-------|------|------|------------|----------|
| `/hdas/camera_wrist_left/color/image_raw` | `Image` | 15 Hz | 640×360 | bgr8 |
| `/hdas/camera_wrist_right/color/image_raw` | `Image` | 15 Hz | 640×360 | bgr8 |
| `/hdas/camera_head/left_raw/image_raw_color/compressed` | `CompressedImage` | 15 Hz | — | jpeg |
| `/hdas/camera_head/right_raw/image_raw_color/compressed` | `CompressedImage` | 15 Hz | — | jpeg |
| `/hdas/camera_wrist_left/depth/image_raw` | `Image` | 15 Hz | 640×360 | 16UC1 |
| `/hdas/camera_wrist_right/depth/image_raw` | `Image` | 15 Hz | 640×360 | 16UC1 |
| `/hdas/camera_wrist_left/aligned_depth_to_color/image_raw` | `Image` | 15 Hz | 640×360 | 16UC1 |
| `/hdas/camera_wrist_right/aligned_depth_to_color/image_raw` | `Image` | 15 Hz | 640×360 | 16UC1 |
| `/hdas/camera_wrist_left/color/camera_info` | `CameraInfo` | 15 Hz | 640×360 | — |
| `/hdas/camera_wrist_right/color/camera_info` | `CameraInfo` | 15 Hz | 640×360 | — |

**Camera intrinsics (left wrist, representative):**
```
K = [326.14, 0, 320.64,
     0, 325.70, 178.89,
     0, 0, 1]
distortion_model = plumb_bob
D = [-0.054, 0.061, -0.0004, 0.0004, -0.020]
```

#### Other Sensors

| Topic | Type | Rate | Content |
|-------|------|------|---------|
| `/hdas/imu_chassis` | `Imu` | 200 Hz | Chassis IMU |
| `/hdas/imu_torso` | `Imu` | 100 Hz | Torso IMU |
| `/joy_left`, `/joy_right` | `Joy` | 10 Hz | Joystick inputs |
| `/hdas/feedback_left_arm_wrench` | `WrenchStamped` | — | Force/torque (empty in sample) |

### 1.4 Trajectory Metadata

From `trajectories.jsonl` and `meta.json`:
```json
{
  "trajectory": "traj_0001",
  "label": "good",
  "is_good": true,
  "instruction": "pick up green soda can and place it in the middle of the workspace",
  "duration_sec": 55.805,
  "recorded_topics_count": 72,
  "storage_id": "mcap"
}
```

Key metadata fields:
- **`instruction`**: Natural language task description (useful for language-conditioned models)
- **`label` / `is_good`**: Human quality label (allows filtering bad demonstrations)
- **`duration_sec`**: Episode wall-clock duration

### 1.5 Recording Duration & Data Volume

Sample trajectory `traj_0001`:
- Duration: **37.3 seconds** of actual data (55.8s wall clock)
- Total messages: **160,719**
- Image frames per camera: **~559** (at 15 Hz)
- Joint state messages: **18,665** (at 500 Hz)
- Arm feedback messages: **~7,470** per arm (at 200 Hz)
- MCAP file size: ~824 MB (compressed)

---

## 2. Target Data Format (IWS Per-Episode HDF5)

### 2.1 Sim ALOHA HDF5 Structure (Current)

```
episode_{ID}.hdf5
├── action             (T, 4)            float32   task-space action (bimanual XY)
├── env_state          (T, 7)            float32   environment state (sim only)
├── robot_bases        (T, 2, 4, 4)      float32   robot base poses in world
├── obs/
│   ├── joint_pos      (T, 14)           float32   puppet joint positions (7/arm)
│   ├── ee_pos         (T, 2, 4, 4)      float32   end-effector 4×4 SE(3) poses
│   └── images/
│       └── top_pov    (T, 128, 128, 3)  uint8     top-down camera view
```

- **T = 300** steps per episode (at ~10 Hz recording)
- **Action** is extracted from `ee_pos`: `concat(ee_pos[:, 0, :2, 3], ee_pos[:, 1, :2, 3])` → XY of both EEs

### 2.2 Real ALOHA HDF5 Structure (Current)

```
episode_{ID}.hdf5
├── timestamp             (T,)            float64
├── action                (T, D)          float32   raw action (varies by ctrl_mode)
├── obs/
│   ├── joint_pos         (T, 14)         float32   14 joints (7/arm)
│   ├── full_joint_pos    (T, 16)         float32   with finger positions
│   ├── ee_pos            (T, 2, 4, 4)    float32   end-effector SE(3) poses
│   ├── world_t_robot_base (T, 2, 4, 4)  float32   base transforms
│   └── images/
│       ├── camera_0_color     (T, H, W, 3)  uint8
│       ├── camera_0_depth     (T, H, W)     uint16   [optional]
│       ├── camera_0_intrinsics (T, 9)       float32
│       ├── camera_0_extrinsics (T, 16)      float32
│       └── camera_1_*         ...
```

### 2.3 Zarr Cache (Model Input)

The HDF5 episodes are converted once to a Zarr cache by `_convert_real_to_dp_replay()`:

```
cache.zarr.zip/
├── data/
│   ├── action             (N_total, D)           float32
│   ├── camera_0_color     (N_total, 128, 128, 3) uint8   [JPEG2K compressed]
│   └── camera_1_color     (N_total, 128, 128, 3) uint8   [JPEG2K compressed]
└── meta/
    └── episode_ends       (N_episodes,)          int64
```

### 2.4 DataLoader Batch Format

```python
{
    "obs": {
        "camera_0_color": (T, 3, 128, 128),   # float32 [0, 1]
        "camera_1_color": (T, 3, 128, 128),   # float32 [0, 1]
    },
    "goal": {
        "camera_0_color": (3, 128, 128),
        "camera_1_color": (3, 128, 128),
    },
    "action": (T, D),                          # float32 [-1, 1] (range-normalized)
}
```

---

## 3. Gap Analysis: R1 Lite MCAP vs. IWS HDF5

### 3.1 Structural Gaps

| Aspect | IWS Expected | R1 Lite Source | Gap |
|--------|-------------|----------------|-----|
| **File format** | Per-episode HDF5 | Per-trajectory MCAP bag | Full format conversion required |
| **Episode segmentation** | One HDF5 per episode | One MCAP per trajectory (already segmented) | Direct 1:1 mapping |
| **Train/val split** | `train/` and `val/` subdirs | Flat `trajectories/` dir | Need to split |
| **Naming** | `episode_{N}.hdf5` | `traj_NNNN/bag/bag_0.mcap` | Rename |

### 3.2 Proprioception Gaps

| Field | IWS Expected | R1 Lite Source | Mapping |
|-------|-------------|----------------|---------|
| `obs/joint_pos` | `(T, 14)` — 7 joints/arm | 6 joints/arm (no wrist-roll or gripper in arm joints) | **Joint count mismatch**: 6 vs 7 per arm. Need to decide: pad with gripper, or change model dim. |
| `obs/full_joint_pos` | `(T, 16)` — arm joints + finger joints | Arm (6) + gripper (1 scalar, range 0–100) per arm. No per-finger resolution. | Gripper representation differs (scalar % vs. joint angle). Need normalization. |
| `obs/ee_pos` | `(T, 2, 4, 4)` — SE(3) poses | Available from `/motion_control/pose_ee_arm_{left,right}` as position + quaternion | Convert `PoseStamped` (pos + quat) → 4×4 SE(3) matrix |
| `obs/world_t_robot_base` | `(T, 2, 4, 4)` — base poses | Available from `/motion_control/pose_floating_base` | Single mobile base vs. 2 fixed ALOHA bases. Need redesign. |
| `action` | `(T, D)` — task-space action | Available from `/motion_target/target_joint_state_arm_{left,right}` (commanded joints) or EE poses | Compute from EE pose or use joint commands directly |
| `timestamp` | `(T,)` float64 | Available from ROS message timestamps | Extract from `log_time_ns` |

### 3.3 Camera / Vision Gaps

| Field | IWS Expected | R1 Lite Source | Mapping |
|-------|-------------|----------------|---------|
| Camera names | `camera_0_color`, `camera_1_color`, or `top_pov` | `camera_wrist_left`, `camera_wrist_right`, `camera_head_left`, `camera_head_right` | Rename to `camera_0_color` etc. or use semantic names |
| Camera type | External workspace cameras (ALOHA) or sim top-down | Wrist-mounted + head-mounted (egocentric) | Different viewpoint semantics — wrist cameras move with EE |
| Resolution | 128×128 (in cache) from 1280×720 (raw) | 640×360 (raw) | Center-crop + resize to 128×128 |
| Color encoding | RGB uint8 | BGR8 (OpenCV convention) | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Depth | Optional `camera_N_depth` uint16 | Available: aligned_depth_to_color 16UC1 | Direct mapping |
| Intrinsics | `(T, 9)` float32 (flattened 3×3 K) | Available from `CameraInfo.k` | Direct mapping |
| Extrinsics | `(T, 16)` float32 (flattened 4×4) | Not directly available (would need TF tree) | Must compute or omit |
| Number of views | 1 (sim) or 2 (real) | 4 available (2 wrist + 2 head) | Select subset or use all |

### 3.4 Rate / Temporal Gaps

| Aspect | IWS Expected | R1 Lite Source | Gap |
|--------|-------------|----------------|-----|
| Recording rate | 10 Hz (fixed) | Mixed: joints 200–500 Hz, images 15 Hz, EE 50 Hz | Need temporal alignment & resampling to common rate |
| Episode length | ~300 steps (~30s at 10 Hz) | ~559 image frames (~37s at 15 Hz) | Variable — acceptable, sampler handles this |
| Time synchronization | Implicit (all arrays same length) | Topic-level ROS timestamps | Must time-align across topics |

### 3.5 Robot Kinematics Gaps

| Aspect | IWS (ALOHA) | R1 Lite |
|--------|-------------|---------|
| Robot model | Trossen VX300S | Custom R1 Lite arms |
| DOF per arm | 6 + 1 gripper | 6 + 1 gripper |
| FK/IK | `KinHelper("trossen_vx300s")` | Not available — need R1 Lite URDF or use EE poses directly |
| Gripper range | Joint angle (rad) | Percentage 0–100 |
| Base configuration | 2 fixed bases (1.06m apart) | Single mobile base with torso |

---

## 4. Harmonization Plan

### Phase 1: MCAP → HDF5 Converter Script

Create `scripts/data_collection/convert_r1lite_mcap_to_hdf5.py`:

#### 4.1 Message Extraction & Temporal Alignment

1. **Read all messages** from the MCAP bag, grouped by topic
2. **Choose reference clock**: use camera image timestamps as the master timeline (15 Hz native, downsample to 10 Hz for IWS compatibility)
3. **Interpolate proprioception** to camera timestamps using nearest-neighbor or linear interpolation:
   - `/hdas/feedback_arm_{left,right}` (200 Hz → 10 Hz)
   - `/hdas/feedback_gripper_{left,right}` (200 Hz → 10 Hz)
   - `/motion_control/pose_ee_arm_{left,right}` (50 Hz → 10 Hz)
   - `/motion_target/target_joint_state_arm_{left,right}` (200 Hz → 10 Hz)
   - `/motion_target/target_position_gripper_{left,right}` (20 Hz → 10 Hz)

#### 4.2 Field Construction

| Target HDF5 Field | Source | Transformation |
|-------------------|--------|----------------|
| `timestamp` | Camera timestamps | `log_time_ns / 1e9` → float64 |
| `obs/joint_pos` | `/hdas/feedback_arm_{left,right}` + `/hdas/feedback_gripper_{left,right}` | Concatenate: `[left_arm(6), left_gripper_normalized(1), right_arm(6), right_gripper_normalized(1)]` → `(T, 14)`. Gripper: `gripper_pct / 100.0 * GRIPPER_JOINT_RANGE + GRIPPER_JOINT_MIN` |
| `obs/ee_pos` | `/motion_control/pose_ee_arm_{left,right}` | Convert `(pos, quat)` → 4×4 SE(3) matrix, stack to `(T, 2, 4, 4)` |
| `obs/world_t_robot_base` | `/motion_control/pose_floating_base` | Convert to 4×4, replicate for both arms: `(T, 2, 4, 4)`. Or: store single base `(T, 1, 4, 4)` and update dataset code. |
| `obs/images/camera_0_color` | `/hdas/camera_wrist_left/color/image_raw` | BGR→RGB, center-crop, resize to 128×128 |
| `obs/images/camera_1_color` | `/hdas/camera_wrist_right/color/image_raw` | BGR→RGB, center-crop, resize to 128×128 |
| `obs/images/camera_2_color` | `/hdas/camera_head/left_raw/image_raw_color/compressed` | JPEG decode, resize to 128×128 |
| `obs/images/camera_3_color` | `/hdas/camera_head/right_raw/image_raw_color/compressed` | JPEG decode, resize to 128×128 |
| `obs/images/camera_0_depth` | `/hdas/camera_wrist_left/aligned_depth_to_color/image_raw` | Direct uint16 copy, resize |
| `obs/images/camera_0_intrinsics` | `/hdas/camera_wrist_left/color/camera_info` | Flatten K matrix → `(T, 9)` float32 |
| `action` | `/motion_target/target_joint_state_arm_{left,right}` + `/motion_target/target_position_gripper_{left,right}` | Concatenate commanded positions: `[left_arm(6), left_grip_norm(1), right_arm(6), right_grip_norm(1)]` → `(T, 14)` for joint mode. Or compute task-space action from EE poses. |
| `robot_bases` | `/motion_control/pose_floating_base` | Convert to `(T, 2, 4, 4)` — both entries identical (single mobile base) |

#### 4.3 Gripper Normalization

R1 Lite grippers report in percentage (0–100). To map to joint-angle representation:

```python
# Option A: Normalize to [0, 1]
gripper_normalized = gripper_pct / 100.0

# Option B: Map to ALOHA-equivalent joint range (if model requires it)
# ALOHA gripper joint range: ~[-0.6213, 1.4910] rad
# This requires knowing the R1 Lite gripper physical range
```

**Recommendation**: Use Option A (`[0, 1]`) with a new `ctrl_mode` for R1 Lite, avoiding the ALOHA-specific gripper normalization functions entirely.

#### 4.4 Action Space Design for R1 Lite

**Option 1: Joint-space actions (simplest)**
```python
action_dim = 14  # 6 arm + 1 gripper per arm × 2 arms
ctrl_mode = "r1lite_joint"
```

**Option 2: Task-space actions (compatible with existing model)**
```python
# From EE poses, extract XY (bimanual push equivalent)
action = concat([ee_left[:2, 3], ee_right[:2, 3]])  # (T, 4)
ctrl_mode = "bimanual_push"
```

**Option 3: Full task-space with gripper**
```python
# XYZ + gripper per arm
action = concat([ee_left[:3, 3], grip_left, ee_right[:3, 3], grip_right])  # (T, 8)
ctrl_mode = "r1lite_bimanual_grasp"
```

**Recommendation**: Start with **Option 2** for maximum compatibility with the existing model architecture (action_dim=4). Extend to Option 3 when gripper control is needed.

### Phase 2: Dataset Class Extension

#### 4.5 New Dataset Class: `R1LiteDataset`

Create `interactive_world_sim/datasets/latent_dynamics/r1lite_dataset.py`:

- Inherit structure from `SimAlohaDataset` / `RealAlohaDataset`
- Override `_convert_real_to_dp_replay()` to handle R1 Lite HDF5 format
- Skip ALOHA-specific gripper normalization (`PUPPET_GRIPPER_JOINT_NORMALIZE_FN`)
- Skip ALOHA-specific FK (`KinHelper("trossen_vx300s")`) — use stored EE poses directly
- Support new camera names / counts

#### 4.6 New Config: `r1lite_dataset.yaml`

```yaml
defaults:
  - base_dataset

dataset_dir: data/r1lite/session_YYYYMMDD
horizon: 16
val_horizon: 16
aug_mode: none
skip_frame: 1
pad_after: 7
pad_before: 1
seed: 42
skip_idx: 4
use_cache: true
resolution: 128
obs_keys: [camera_0_color, camera_1_color]  # wrist cameras
low_dim_keys: []
goal_sample: intermediate
shape_meta:
  action:
    shape: [4]    # bimanual_push: [left_x, left_y, right_x, right_y]
  obs:
    camera_0_color:
      shape: [3, 128, 128]
      type: rgb
    camera_1_color:
      shape: [3, 128, 128]
      type: rgb
```

### Phase 3: Conversion Pipeline

#### 4.7 End-to-End Pipeline

```bash
# 1. Extract archive
tar -xzf data/test.tar.gz -C data/

# 2. Convert MCAP → HDF5
python scripts/data_collection/convert_r1lite_mcap_to_hdf5.py \
    --input_dir data/data/session_20260428_052521 \
    --output_dir /fs/scratch/.../iws/data/r1lite/session_20260428 \
    --target_hz 10 \
    --resolution 128 \
    --action_mode bimanual_push \
    --cameras camera_wrist_left,camera_wrist_right

# 3. Validate output
python scripts/validate_hdf5.py \
    --dataset_dir /fs/scratch/.../iws/data/r1lite/session_20260428/train

# 4. Train
python main.py +name=r1lite_stage_1 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=r1lite_dataset \
    dataset.dataset_dir=/fs/scratch/.../iws/data/r1lite/session_20260428 \
    algorithm.training_stage=1
```

### Phase 4: Validation & Testing

#### 4.8 Validation Checks

1. **Shape consistency**: All HDF5 fields match expected shapes from `shape_meta`
2. **Range validation**: Joint positions in plausible range, images in [0, 255], actions in expected bounds
3. **Temporal alignment**: Verify timestamp monotonicity and consistent dt
4. **Visual sanity**: Render image sequences as video, overlay EE trajectories
5. **Zarr cache**: Verify `cache.zarr.zip` generation succeeds and loads correctly
6. **DataLoader**: Verify batches come out with correct shapes and dtypes

---

## 5. Key Design Decisions

### Decision 1: Target Recording Rate

**Recommendation: 10 Hz**
- Matches existing IWS pipeline (sim records at ~10 Hz equivalent)
- 15 Hz camera native rate → subsample every 1.5 frames or nearest-neighbor resample to 10 Hz
- All proprioception topics are ≥50 Hz so downsampling is safe

### Decision 2: Camera Selection

| Priority | Camera | Topic | Rationale |
|----------|--------|-------|-----------|
| 1 | Wrist left | `/hdas/camera_wrist_left/color/image_raw` | Best manipulation view |
| 2 | Wrist right | `/hdas/camera_wrist_right/color/image_raw` | Stereo wrist pair |
| 3 | Head left | `/hdas/camera_head/left_raw/image_raw_color/compressed` | Workspace overview |
| 4 | Head right | `/hdas/camera_head/right_raw/image_raw_color/compressed` | Stereo head pair |

Start with 2 cameras (both wrist) for compatibility with the existing 2-camera model input. Head cameras can be added later.

### Decision 3: Action Representation

For initial compatibility: use **EE-position-based actions** (same as `SimAlohaDataset`):
```python
# From EE poses:
action = [left_ee_x, left_ee_y, right_ee_x, right_ee_y]  # (T, 4)
```

This lets us reuse `SimAlohaDataset` / the existing model with `action_dim=4` and test training immediately. Joint-space or richer action spaces can be added as new `ctrl_mode` entries.

### Decision 4: Robot Base Representation

Since R1 Lite has a **single mobile base** (vs. ALOHA's 2 fixed bases):
- Store `robot_bases` as `(T, 2, 4, 4)` where both entries are the floating base pose
- The model doesn't directly consume `robot_bases` — it's only used in `_convert_real_to_dp_replay()` for FK, which we bypass by using EE poses directly

### Decision 5: Gripper Data

- Store as 7th joint (normalized to `[0, 1]`) in `obs/joint_pos` to maintain the `(T, 14)` shape
- For actions: initially omit gripper from action (use `bimanual_push` mode with action_dim=4)
- For grasping tasks: add a new `r1lite_bimanual_grasp` ctrl_mode with action_dim=8

### Decision 6: Missing `env_state`

The sim dataset has `env_state` (T, 7) for the environment state — this is **simulation-only** (object poses from the physics engine). R1 Lite real data does not have this. The field is not consumed by the model or dataset class — it can be omitted.

---

## 6. Implementation Checklist

- [ ] **Converter script** (`scripts/data_collection/convert_r1lite_mcap_to_hdf5.py`)
  - [ ] MCAP reading with `mcap_ros2`
  - [ ] Temporal alignment to 10 Hz reference clock
  - [ ] BGR→RGB color conversion
  - [ ] Image cropping and resizing to 128×128
  - [ ] PoseStamped → 4×4 SE(3) matrix conversion
  - [ ] Joint + gripper concatenation to `(T, 14)` format
  - [ ] Action computation from EE poses
  - [ ] Train/val split (e.g., 90/10)
  - [ ] HDF5 writing in IWS-compatible layout
  - [ ] Quality filtering via `is_good` flag from metadata
- [ ] **Dataset class** (`r1lite_dataset.py`)
  - [ ] Inherit from `BaseImageDataset`
  - [ ] `_convert_real_to_dp_replay()` without ALOHA-specific FK
  - [ ] Camera key mapping
  - [ ] Gripper normalization
- [ ] **Config** (`configurations/dataset/r1lite_dataset.yaml`)
- [ ] **Validation script** (`scripts/validate_hdf5.py`)
- [ ] **Update data pipeline docs**

---

## 7. Dependencies

### Already Available (in `iws` conda env)
- `h5py` (3.7.0) — HDF5 read/write
- `cv2` (OpenCV) — image processing
- `numpy`, `torch` — array manipulation

### Newly Installed
- `mcap` (1.3.1) — MCAP file reading
- `mcap-ros2-support` (0.5.7) — ROS 2 message deserialization

### May Need
- `scipy` — for interpolation (`scipy.interpolate.interp1d` for temporal alignment)
- R1 Lite URDF — only if FK/IK is needed (not needed for EE-pose-based actions)

---

## 8. Open Questions

1. **External cameras**: The session requested `/external_realsense/camera_external_1` and `camera_external_2` topics but they were `topic_not_found`. Are external workspace cameras available in other sessions? Should the pipeline support them?

2. **Tabletop arms**: Topics `/tabletop/hdas/feedback_arm_{left,right}` contain arm feedback at 190 Hz. Are these the same arms as `/hdas/feedback_arm_{left,right}` or separate tabletop-mounted actuators?

3. **Multi-session handling**: How should multiple sessions be combined? Sequential episode numbering across sessions?

4. **Hand feedback**: Topics `/hdas/feedback_hand_{left,right}` exist but had no messages in this sample. When active, should hand data be included?

5. **Torso & chassis**: The R1 Lite torso (3 joints) and chassis (6 joints) are not represented in the ALOHA-style `joint_pos`. Should they be stored as additional low-dim observations for the model to condition on?
