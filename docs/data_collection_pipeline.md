# Data Collection & Training Pipeline

## 1. Teleoperation & Data Collection

### 1.1 Hardware Setup

The system uses an **ALOHA bimanual** configuration with two Interbotix robot arms in a **leader-follower (master-puppet)** teleoperation setup.

| Role | Robot Model | Frequency | Function |
|------|------------|-----------|----------|
| **Master** (leader) | Interbotix WX250S × 2 | 200 Hz | Human operator moves these; gripper torque disabled |
| **Puppet** (follower) | Interbotix VX300S × 2 | 50 Hz | Mirrors master commands with velocity-limited interpolation |

Communication between master and puppet uses `SharedMemoryRingBuffer` (state readout) and `SharedMemoryQueue` (timestamped commands), both running in separate processes to avoid GIL blocking.

### 1.2 Robot Base Extrinsics

The two puppet arms are mounted facing each other on a shared table. Their base poses in the world frame are stored in [`interactive_world_sim/real_world/aloha_extrinsics/`](../interactive_world_sim/real_world/aloha_extrinsics/):

```
Right arm base pose (world frame):          Left arm base pose (world frame):
┌                          ┐                ┌                           ┐
│  0   1   0   0.11        │                │  0  -1   0   0.11         │
│ -1   0   0   0.42        │                │  1   0   0  -0.64         │
│  0   0   1   0.02        │                │  0   0   1   0.02         │
│  0   0   0   1.00        │                │  0   0   0   1.00         │
└                          ┘                └                           ┘
Position: (0.11, 0.42, 0.02)m              Position: (0.11, -0.64, 0.02)m
Robot X-axis → world -Y (inward)           Robot X-axis → world +Y (inward)
```

**Key observations:**
- Both arms are at the same X (0.11m) and Z (0.02m) — they sit side-by-side along the Y-axis.
- They are **1.06m apart** along Y, facing each other (robot X-axes point inward toward workspace center).
- Z-axes both point up — the arms are upright on a flat table.
- The workspace center is approximately at (0.5, -0.1, 0.02) in world coordinates.

![Workspace Layout](aloha_workspace_layout.png)

To **calibrate your own setup**, run:
```bash
python interactive_world_sim/real_world/calibrate_robot.py
```
This launches an Open3D visualizer showing robot point clouds overlaid with camera point clouds. Adjust the `robot_base_in_world` matrix until alignment is correct, then the script saves `.npy` files to `aloha_extrinsics/`.

### 1.3 Camera Setup

**All cameras are external (workspace-mounted) — no wrist cameras are used.**

| Setup | Cameras | Names | Raw Resolution | Notes |
|-------|---------|-------|----------------|-------|
| **Real robot** | 2× Intel RealSense D435 | `camera_0_color`, `camera_1_color` | 1280×720 @ 30fps | RGB + optional depth |
| **Simulation** | 1× MuJoCo camera | `top_pov` | 128×128 | Top-down view |

Camera extrinsics are calibrated via a ChArUco board:
```bash
python interactive_world_sim/real_world/calibrate_realsenses.py \
    --rows 5 --cols 6 --checker_width 0.04 --marker_width 0.03
```

Camera naming (`camera_0`, `camera_1`) is device enumeration order, **not** a fixed semantic role. Camera placement is flexible — the extrinsics calibration handles arbitrary positions. The only viewpoint hint in the codebase: `deploy/extract_init_imgs.py` maps one task to a folder named `real_pusht_topdown` using `camera_1_color`.

### 1.4 Data Collection Scripts

#### Real Robot Teleoperation

```bash
python scripts/data_collection/collect_real_aloha.py \
    -o <output_dir> \
    -r [left|right] \     # which arm(s) to record
    -f 10 \               # recording frequency in Hz
    -cm joint \            # control mode (see §2.3)
    -ts 200                # auto-save after N steps
```

**Interactive controls during collection:**
| Key | Action |
|-----|--------|
| `C` | Start recording episode |
| `S` | Stop / save episode |
| `Backspace` | Delete last episode |
| `Q` | Quit |

The operator closes the master gripper (past midpoint) to signal recording start. Episodes auto-save after `--total_steps` steps.

#### Simulated Data Collection

```bash
python scripts/data_collection/sim_aloha_dataset_collection_scripted.py
```

Uses scripted policies with a mix of motion primitives:
- 30% linear pushes (coordinated multi-arm)
- 30% rotations (gripper corners on T-shape)
- 30% random contact (collision-aware)
- 10% random exploration (non-contact)

Auto-records every ~3 seconds.

### 1.5 Per-Episode HDF5 Structure

Each episode is saved as an individual HDF5 file:

```
episode_{ID}.hdf5
├── timestamp              (T,)           float64    observation timestamps
├── obs/
│   ├── joint_pos          (T, 14)        float32    puppet joint positions (7/arm)
│   ├── full_joint_pos     (T, 16)        float32    with finger positions
│   ├── ee_pos             (T, 2, 4, 4)   float32    end-effector poses (4×4 SE(3))
│   ├── world_t_robot_base (T, 2, 4, 4)   float32    robot base poses in world
│   └── images/
│       ├── camera_0_color      (T, H, W, 3)  uint8
│       ├── camera_0_depth      (T, H, W)     uint16   [optional]
│       ├── camera_0_intrinsics (T, 9)         float32
│       ├── camera_0_extrinsics (T, 16)        float32
│       └── camera_1_*          ...
├── joint_action           (T, 14)        float32    commanded joint positions
├── action                 (T, D)         float32    task-space action (D varies by ctrl_mode)
└── videos/
    └── {episode_id}/
        ├── 0.mp4                                     camera 0 video
        ├── 1.mp4                                     camera 1 video
        └── multi_cam.mp4                             grid visualization
```

---

## 2. Data Pipeline: Disk → DataLoader → Model

### 2.1 HDF5 → Zarr Cache (one-time conversion)

On first dataset load, `_convert_real_to_dp_replay()` converts all episode HDF5 files into a single Zarr cache for efficient random access.

```
Per-episode HDF5 files
    ↓  iterate episodes, extract & transform
Zarr store (cache.zarr.zip)
```

**Image preprocessing during conversion:**
```
Raw: (T, 720, 1280, 3) uint8
  → center_crop
  → cv2.resize(INTER_AREA)
  → (T, 128, 128, 3) uint8
```

**Action computation during conversion:**
- Joint positions → forward kinematics → end-effector pose → task-space action
- Action dimension depends on `ctrl_mode` / `action_mode` (see §2.3)

**Zarr layout:**
```
cache.zarr.zip/
├── data/
│   ├── camera_0_color     (N_total, 128, 128, 3) uint8   [JPEG2K compressed]
│   ├── camera_1_color     (N_total, 128, 128, 3) uint8   [JPEG2K compressed]
│   ├── action             (N_total, D)           float32  [Blosc LZ4]
│   └── ...
└── meta/
    └── episode_ends       (N_episodes,)          int64    [cumulative indices]
```

Episodes are concatenated into flat temporal arrays. `episode_ends` tracks boundaries (e.g., `[100, 250, 380]` → episode 0 = steps 0–99, episode 1 = 100–249, …).

### 2.2 Dataset Classes

```
BaseLowdimDataset (abstract)
BaseImageDataset (abstract)
├── RealAlohaDataset    interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py
└── SimAlohaDataset     interactive_world_sim/datasets/latent_dynamics/sim_aloha_dataset.py
```

**Sequence sampling** is handled by `SequenceSampler` (`interactive_world_sim/utils/sampler.py`):
- Pre-computes valid sequence start indices respecting episode boundaries
- Supports `pad_before` / `pad_after` for context padding
- `skip_frame`: temporal subsampling (e.g., `skip_frame=2` samples every other frame)
- Goal sampling strategies: `"final"` (last frame), `"intermediate"` (random from sequence end to episode end), `"aggressive"` (20% early-stop chance)

**`__getitem__` output** (per sample):
```python
{
    "obs": {
        "camera_0_color": (T, 3, 128, 128),   # float32, [0, 1]
        "camera_1_color": (T, 3, 128, 128),   # float32, [0, 1]
    },
    "goal": {
        "camera_0_color": (3, 128, 128),       # single frame
        "camera_1_color": (3, 128, 128),
    },
    "action": (T, D),                          # float32
}
```

Image normalization: `uint8 / 255.0 → [0, 1]` with channels moved to CHW format. Optional augmentation (20% probability): affine transforms, Gaussian noise, hue/saturation.

### 2.3 Action Space

The action representation is set by `ctrl_mode` and determines dimension `D`:

| Control Mode | Dim | Representation |
|---|---|---|
| `joint` | 14 | Raw joint angles (bimanual) |
| `single_push` | 2 | End-effector XY |
| `single_sweep` | 4 | X, Y, height, gripper |
| `single_grasp` | 4 | X, Y, Z, gripper |
| `single_rope` | 5 | X, Y, Z, θ_wrist, gripper |
| `bimanual_push` | 4 | Right XY + Left XY |
| `bimanual_sweep` | 4 | Right XY + Left XY |
| `bimanual_rope` | 8 | 2× (X, Y, Z, gripper) |
| `bimanual_pack` | 6 | 2× (X, Y, Z) |

All task-space actions are computed from joint positions via forward kinematics in `action_utils.py`. The EE workspace is clipped to `[0.25, 1.0]m × [-0.25, 0.25]m` in the robot base frame.

Actions are **range-normalized** to [-1, 1] using per-dimension min/max statistics from the training data.

### 2.4 Multi-View Video → Model

The world model uses **Consistency Trajectory Matching (CTM)** for action-conditioned video generation. Here is the full data flow from DataLoader batch to model:

```
DataLoader batch
    batch["obs"]["camera_0_color"]  →  (B, T, 3, 128, 128)
    batch["obs"]["camera_1_color"]  →  (B, T, 3, 128, 128)
    batch["action"]                 →  (B, T, D)
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  1. MULTI-VIEW CHANNEL CONCATENATION        │
    │     torch.cat(views, dim=2)                 │
    │     → (B, T, 3×N_views, 128, 128)          │
    │     e.g. 2 cameras → (B, T, 6, 128, 128)   │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │  2. ENCODER (lightweight CNN)               │
    │     Conv2d(6→8, k=3) + 2× [SiLU, Conv,     │
    │       SiLU, Conv(stride=2)]                 │
    │     → (B×T, 8, 32, 32)                     │
    │                                             │
    │     Per-view L2 normalization:              │
    │     channels [0:4] = view0 / ‖view0‖        │
    │     channels [4:8] = view1 / ‖view1‖        │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │  3. DYNAMICS MODEL (CMLatentDynamics)       │
    │     3D U-Net with temporal attention         │
    │     Input: (T, B, 8, 32, 32) latent seq     │
    │                                             │
    │     Action conditioning via FiLM:           │
    │     action (D) → MLP → emb (512)            │
    │     h = norm(h) · (1 + scale) + shift       │
    │     Applied at every ResNet block            │
    │                                             │
    │     Output: (T, B, 8, 32, 32) predicted     │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │  4. DECODER (CMDecoder + ControlNet)        │
    │     2D U-Net guided by ControlNet            │
    │     Latent → ControlNet conditions           │
    │     Noise (B×T, 6, 128, 128) → denoise      │
    │                                             │
    │     Output: (B×T, 6, 128, 128)              │
    │     → split: [0:3]=cam0, [3:6]=cam1          │
    └─────────────────────────────────────────────┘
```

**Resolution summary:**

| Stage | Spatial | Channels | Dtype |
|-------|---------|----------|-------|
| Raw capture | 1280×720 | 3 per view | uint8 |
| Zarr cache | 128×128 | 3 per view | uint8 (JPEG2K) |
| Model input | 128×128 | 3 × N_views | float32 [0,1] |
| Encoder output | 32×32 | 4 × N_views | float32 (L2-normed) |
| Decoder output | 128×128 | 3 × N_views | float32 |

### 2.5 Training Stages

| Stage | Trains | Frozen | Steps | Loss |
|-------|--------|--------|-------|------|
| **1** | Encoder + Decoder | — | ~50k | Consistency loss (denoise at two noise levels) |
| **2** | Dynamics network | Encoder | 100k–200k | Consistency loss on latent trajectories, action-conditioned |
| **3** | Decoder (0.1× LR) | Encoder, Dynamics | Variable | Reconstruction loss for visual quality refinement |

Key hyperparameters: AdamW (β=0.9/0.999), LR=8e-5, warmup=10k steps, gradient clip=1.0, noise schedule=sigmoid with 1000 training timesteps.
