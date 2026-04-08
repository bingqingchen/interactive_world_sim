# RealSense Camera Setup & Calibration Guide

## 1. Hardware Requirements

- **2× Intel RealSense D435** cameras (D400-series)
- **ChArUco calibration board** (see §3 for specs)
- USB 3.0 cables (one per camera)
- Camera mounts / clamps for workspace mounting

> **No wrist cameras are used.** Both cameras are external, workspace-mounted.

---

## 2. Physical Camera Placement

Mount the two D435 cameras looking at the robot workspace. The exact placement is flexible — the extrinsic calibration (§4) handles arbitrary positions and orientations.

Some general guidance:
- Both cameras should have a clear, overlapping view of the workspace area
- Avoid placing them where the robot arms will occlude critical workspace regions
- Typical setups use one top-down and one angled view, but this is not required
- Camera naming (`camera_0`, `camera_1`) is determined by **device serial number sort order**, not physical position

### Verify Connected Cameras

The system auto-discovers plugged-in D400-series cameras:

```python
from interactive_world_sim.real_world.single_realsense import SingleRealsense
serials = SingleRealsense.get_connected_devices_serial()
print(serials)  # e.g. ['123456789', '987654321']
```

If no serial numbers are explicitly passed to `MultiRealsense` or `RealAlohaEnv`, all connected D400 cameras are used automatically.

---

## 3. ChArUco Calibration Board

### Board Specification

The **`calibrate_realsenses.py` CLI** uses these defaults (which override the method-level defaults in `single_realsense.py`):

| Parameter | CLI default (`calibrate_realsenses.py`) | Method default (`calibrate_extrinsics()`) |
|---|---|---|
| Board size (cols × rows) | **6 × 5** | 6 × 9 |
| Square side length | **0.04 m** (40 mm) | 0.03 m (30 mm) |
| Marker side length | **0.03 m** (30 mm) | 0.022 m (22 mm) |
| ArUco dictionary | `DICT_4X4_250` | `DICT_4X4_250` |

> **Important:** The CLI defaults take precedence when you run `calibrate_realsenses.py` without arguments. The method defaults in `single_realsense.py:calibrate_extrinsics()` are only used if you call the method directly without overriding. **Make sure your printed board matches whichever parameters you actually use.**

### Printing the Board

1. Generate the board image using OpenCV:
   ```python
   import cv2
   dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
   board = cv2.aruco.CharucoBoard(
       size=(6, 5),            # (cols, rows) — matches CLI defaults
       squareLength=0.04,       # 40 mm
       markerLength=0.03,       # 30 mm
       dictionary=dictionary,
   )
   img = board.generateImage((600, 500))  # pixel size
   cv2.imwrite("charuco_board.png", img)
   ```
2. Print at **exact scale** — verify with a ruler that the squares measure 40 mm
3. Mount on a flat, rigid surface (clipboard, foam board, etc.)

---

## 4. Camera Extrinsic Calibration

This determines each camera's pose in the world frame (`T_cam_in_world`).

### 4.1 Board Placement Convention

The calibration code applies a hardcoded `tf_world2board` rotation (in `single_realsense.py`):

```python
tf_world2board[:3, :3] = np.array([[0, -1, 0],
                                    [-1,  0, 0],
                                    [ 0,  0, -1]])
```

This encodes the relationship between the workspace world frame and the ChArUco board frame. Derived from the rotation matrix, the required board placement is:

| Board axis (OpenCV convention) | Points toward in world frame |
|-------------------------------|------------------------------|
| **X** — rightward along columns | **−Y_world** (toward right arm) |
| **Y** — upward along rows | **−X_world** (toward operator) |
| **Z** — out of printed face | **−Z_world** (downward into table) |

The **world origin** is defined as the board's **bottom-left corner** (column 0, row 0 in OpenCV ChArUco numbering).

---

#### Top-down view (looking down from above the table)

```
                    +X_world  (away from operator → into workspace)
                        ▲
                        │
      Left arm          │                         Right arm
      (Y=+0.42m)        │                         (Y=−0.64m)
          ●             │                              ●
          │             │                              │
          │  ┌──────────┼──────────────────────┐       │
          │  │          │  Robot Workspace      │       │
          │  │          │                       │       │
          │  └──────────┼──────────────────────┘       │
          │             │                              │
  ────────┼─────────────┼──────────────────────────────┼──────  Y_world
          │          (0,0,0)                            │
          │         World    ← board col 0, row 0       │
          │          Origin                             │
          │                                             │
          │  Board placed FACE UP, flat on table:       │
          │                                             │
          │  col→  0    1    2    3    4    5            │
          │                                             │
          │  row4  ·────·────·────·────·────·           │  ← furthest from workspace
          │        │    │    │    │    │    │           │
          │  row3  ·────·────·────·────·────·           │
          │        │    │    │    │    │    │           │
          │  row2  ·────·────·────·────·────·           │
          │        │    │    │    │    │    │           │
          │  row1  ·────·────·────·────·────·           │
          │        │    │    │    │    │    │           │
          │  row0  ★────·────·────·────·────·           │  ← world origin (★)
          │      (0,0)                    (5,0)         │
          │        │                        │           │
          │        └────────────────────────┘           │
          │         columns run in −Y direction         │
          │         (toward right arm) ──────────────►  │
          │                                             │
          │             OPERATOR                        │
          │                                             │
          ┼─── −X_world ────────────────────────────────┼
```

**Key points:**
- Place the board **flat on the table**, printed face **facing upward** (toward the cameras)
- The **★ corner** (bottom-left of the printed board, where ArUco ID 0 is placed) goes at the **world origin**
- **Columns** extend from the ★ corner in the **−Y direction** (toward the right arm)
- **Rows** extend from the ★ corner in the **−X direction** (toward the operator, away from the workspace)

---

#### Side view — camera geometry

```
     Camera               Camera
       ↓ ↘             ↙ ↓
       │   ╲         ╱   │
       │    ╲       ╱    │
       │     ╲     ╱     │
       │      ╲   ╱      │
  ─────┼───────╲─╱───────┼─────  table surface (Z=0)
       │        ▼        │
       │   [ ChArUco ]   │   ← board face UP, cameras look DOWN at it
       │   ★ = origin    │
```

The cameras must have a clear, unobstructed view of the full board surface. All ArUco markers must be visible — partial occlusion will reduce corner count and increase reprojection error.

---

#### Math: how the extrinsic is computed

```
tf_world2board:  maps a point in world frame   → board frame
T_board2cam:     maps a point in board frame   → camera frame  (from estimatePoseCharucoBoard)

Saved extrinsic = T_board2cam @ T_world2board  =  T_world2cam
                  (transforms world-frame points into camera-frame coordinates)
```

### 4.2 Run Calibration

```bash
python interactive_world_sim/real_world/calibrate_realsenses.py \
    --rows 5 --cols 6 --checker_width 0.04 --marker_width 0.03
```

**What happens:**
1. Opens all connected RealSense cameras at 5 fps (lower fps for stable calibration frames)
2. Sets exposure=60, gain=64, white_balance=3800
3. For each camera:
   - Detects ArUco markers → interpolates ChArUco corners
   - Estimates board pose (`R_board2cam`, `t_board2cam`) via `cv2.aruco.estimatePoseCharucoBoard`
   - Computes `T_world2cam = T_board2cam @ T_world2board`
   - Prints reprojection error (aim for **< 1 px**)
   - Saves result to `interactive_world_sim/real_world/cam_extrinsics/{serial_number}.npy`
4. Loops continuously — reposition the board and re-run until satisfied with the error

**Console output to watch:**
```
Number of corners: 20        # More corners = more stable estimate
Reprojection Error: 0.34     # < 1.0 is good, < 0.5 is excellent
R_board2cam: [...]
t_board2cam: [...]
```

Press `Ctrl+C` to stop once you have a good calibration.

### 4.3 Output Files

Extrinsics are saved per camera serial number:

```
interactive_world_sim/real_world/cam_extrinsics/
├── {serial_number_0}.npy    # 4×4 float64 T_world2cam
└── {serial_number_1}.npy    # 4×4 float64 T_world2cam
```

This directory is auto-created on first run. If a `.npy` file doesn't exist for a serial number, the system falls back to an **identity matrix** and prints a warning.

---

## 5. Visual Verification (Robot + Camera Overlay)

After camera calibration, verify alignment by overlaying camera point clouds with robot point clouds:

```bash
python interactive_world_sim/real_world/calibrate_robot.py
```

**What happens:**
1. Loads robot arm base poses from `aloha_extrinsics/{side}_base_pose_in_world.npy`
2. Starts both puppet and master robots
3. Opens all RealSense cameras with depth enabled
4. Launches an Open3D visualizer showing:
   - **Camera point clouds** (using the camera extrinsics from §4)
   - **Robot mesh point clouds** (using forward kinematics + robot base poses)
5. You can teleoperate the robot and visually verify that the robot mesh aligns with the camera-observed point cloud

If misaligned, adjust the `robot_base_in_world` matrix in `calibrate_robot.py` and re-run.

### Robot Base Extrinsics

These are **not** camera poses — they define where each robot arm base sits in the world frame:

```
aloha_extrinsics/
├── right_base_pose_in_world.npy    # 4×4 SE(3) matrix
└── left_base_pose_in_world.npy     # 4×4 SE(3) matrix
```

See `docs/aloha_workspace.ipynb` for visualizations of the workspace layout.

---

## 6. Runtime Camera Settings

During data collection, `RealAlohaEnv` (in `real_aloha_env.py`) applies these settings:

| Setting | Value | Unit |
|---------|-------|------|
| Color exposure | 60 | ×100 µs (= 6 ms) |
| Color gain | 64 | — |
| White balance | 3800 | Kelvin |
| Depth preset | High Density | — |
| Depth exposure | 7000 | ×100 µs |
| Depth gain | 16 | — |
| Capture resolution | 1280 × 720 | pixels |
| Capture FPS | 30 | Hz |

These are set programmatically — no manual camera adjustment needed.

---

## 7. How Extrinsics Are Loaded at Runtime

In `SingleRealsense.__init__()`, each camera loads its extrinsics from:

```python
extrinsics_dir = os.path.join(os.path.dirname(__file__), "cam_extrinsics")
# Loads: cam_extrinsics/{serial_number}.npy
```

Every frame pushed to shared memory includes both `intrinsics` (3×3 `K` matrix) and `extrinsics` (4×4 world-to-camera transform). These are stored per-timestep in the HDF5 episodes as `obs/images/camera_{i}_intrinsics` and `obs/images/camera_{i}_extrinsics`.

---

## 8. Quick Reference Checklist

| # | Step | Command / Action |
|---|------|-----------------|
| 1 | Mount cameras | Position 2× D435 looking at workspace, connect via USB 3.0 |
| 2 | Verify detection | Run `SingleRealsense.get_connected_devices_serial()` |
| 3 | Print ChArUco board | 6×5 grid, 40mm squares, 30mm markers, `DICT_4X4_250`. **Verify printed dimensions with a ruler.** |
| 4 | Place board | Flat at world origin, following the board placement convention (§4.1) |
| 5 | Calibrate cameras | `python interactive_world_sim/real_world/calibrate_realsenses.py` |
| 6 | Check error | Reprojection error < 1 px in console output |
| 7 | Verify alignment | `python interactive_world_sim/real_world/calibrate_robot.py` |
| 8 | Collect data | `python scripts/data_collection/collect_real_aloha.py -o <output_dir>` |

---

## 9. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| "No markers detected" | Board too far, bad lighting, wrong board spec | Move board closer, increase exposure, verify board dimensions match CLI args |
| High reprojection error (> 2 px) | Board not flat, motion blur, partial occlusion | Use rigid board, lower capture FPS, ensure full board is visible |
| Identity matrix warning | Missing `.npy` in `cam_extrinsics/` | Run `calibrate_realsenses.py` first |
| Point clouds misaligned in Open3D | Robot base pose wrong, or camera extrinsics bad | Re-calibrate cameras, then adjust `robot_base_in_world` in `calibrate_robot.py` |
| Camera not detected | USB 2.0 port, kernel driver issue | Use USB 3.0, run `rs-enumerate-devices` to check |
