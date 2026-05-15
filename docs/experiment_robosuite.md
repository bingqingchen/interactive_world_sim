# Robosuite Lift World Model Training

## TODO:
To re-enable it for Stage 2/3: you'll need to either pre-download i3d_torchscript.pt from a machine with unrestricted internet access and place it at interactive_world_sim/algorithms/common/metrics/i3d_torchscript.pt

## Dataset

**Source:** [Hugging Face](https://huggingface.co/datasets/jacob3333/RobosuiteLiftConverted)

```bash
python scripts/download_data_hf.py \
    --repo jacob3333/RobosuiteLiftConverted \
    --local_dir data/robosuite/lift
```

### Metadata

| Property | Value |
|---|---|
| Task | Lift (single-arm in Robosuite) |
| Train episodes | 520 |
| Val episodes | 200 |
| Steps per episode | 43–150 (variable) |
| Image resolution | 128 × 128 × 3 (uint8) |
| Action dim | 4 (end-effector XYZ + gripper) |
| Action mode | `single_grasp` |
| Collection | Unknown (converted from Robosuite rollouts) |

### HDF5 Structure (per episode)

```
episode_N.hdf5
├── action                  (T, 7)      float32   # stored actions (NOT used by loader)
├── timestamp               (T,)        float64   # step timestamps
└── obs/
    ├── ee_pos              (T, 1, 4, 4) float32  # end-effector 4×4 transform (1 arm)
    ├── joint_pos           (T, 7)       float32  # 6 joint angles + 1 gripper
    ├── world_t_robot_base  (T, 1, 4, 4) float32  # robot base transform
    └── images/
        ├── camera_0_color  (T, 128, 128, 3) uint8  # RGB camera 0
        └── camera_1_color  (T, 128, 128, 3) uint8  # RGB camera 1
```

### Directory layout

```
data/robosuite/lift/
├── train/          # 520 episodes (non-contiguous indices)
│   └── cache.zarr.zip   # auto-generated on first load
└── val/            # 200 episodes
    └── cache.zarr.zip
```

### Note on actions

The `SimAlohaDataset` loader does **not** use the stored `action` field directly. With `action_mode: single_grasp`, it derives 4D actions from `obs/ee_pos` and `obs/joint_pos`:

```python
ee_xyz = ee_pos[:, 0, :3, 3]      # (T, 3) — XYZ from 4×4 transform
gripper = joint_pos[:, -1:]        # (T, 1) — gripper state
action = concat(ee_xyz, gripper)   # (T, 4)
```

The stored `action` field (T, 7) is only used to determine episode length. No normalization or clipping is applied to the derived action values; the range normalizer handles scaling at training time.

### Differences from MuJoCo PushT

| Property | MuJoCo PushT | Robosuite Lift |
|---|---|---|
| Robot | Bimanual ALOHA (2 arms) | Single arm |
| Task | 2D planar push | 3D reach + grasp + lift |
| `ee_pos` shape | `(T, 2, 4, 4)` | `(T, 1, 4, 4)` |
| `joint_pos` shape | `(T, 14)` | `(T, 7)` |
| Action mode | `bimanual_push` | `single_grasp` |
| Action derivation | `[left_xy, right_xy]` from ee_pos | `[ee_xyz, gripper]` from ee_pos + joint_pos |
| Action dim | 4 | 4 |
| Image keys | `top_pov` | `camera_0_color`, `camera_1_color` |
| Episodes (train/val) | 10,000 / 100 | 520 / 200 |
| Steps per episode | 300 (fixed) | 43–150 (variable) |

---

## Configuration

### Dataset config: `sim_robosuite_dataset`

Key fields:

| Field | Value | Notes |
|---|---|---|
| `obs_keys` | `[camera_0_color, camera_1_color]` | Must match HDF5 keys under `obs/images/` |
| `action_mode` | `single_grasp` | Controls action extraction in `SimAlohaDataset` |
| `shape_meta.action.shape` | `[4]` | 4D: XYZ + gripper |
| `shape_meta.obs.camera_0_color` | `[3, 128, 128]`, type `rgb` | CHW format |
| `shape_meta.obs.camera_1_color` | `[3, 128, 128]`, type `rgb` | CHW format |
| `resolution` | `128` | Native resolution, no resize needed |
| `skip_frame` | `1` | Use every frame |
| `use_cache` | `true` | First run converts HDF5 → zarr cache |

### Algorithm config

These are passed as command-line overrides and must match the dataset:

| Field | Value | Notes |
|---|---|---|
| `algorithm.action_dim` | `4` | Must equal `shape_meta.action.shape[0]` |
| `algorithm.latent_dim` | `512` | Latent space dimension |
| `algorithm.training_stage` | `1` / `2` / `3` | See training stages below |
| `algorithm.obs_keys` | Inherited from `${dataset.obs_keys}` | Auto-resolved by Hydra |

### Fields that do NOT apply

| Field | Reason |
|---|---|
| `delta_action` | Defined in `real_aloha_dataset` config but never read by any dataset class. |

---

## Training Stages

All three stages use the same dataset and algorithm configs; they differ in `training_stage`, `horizon`, and checkpoint loading.

### Stage 1: Autoencoder

Train encoder + diffusion decoder to compress RGB → latent space.

```bash
python main.py +name=robosuite_lift_stage_1 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_robosuite_dataset \
  dataset.dataset_dir=data/robosuite/lift \
  dataset.horizon=1 dataset.val_horizon=1 \
  experiment.training.batch_size=16 \
  experiment.training.precision=bf16-mixed \
  experiment.training.data.num_workers=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=32 \
  experiment.validation.val_every_n_step=6000 \
  experiment.validation.precision=bf16-mixed \
  experiment.validation.data.num_workers=16 \
  "algorithm.metrics=[]" \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1
```

Reconstruction should be **near-perfect** before proceeding to stage 2.

### Stage 2: Dynamics

Train latent dynamics model. Requires a stage 1 checkpoint.

```bash
python main.py +name=robosuite_lift_stage_2 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_robosuite_dataset \
  dataset.dataset_dir=data/robosuite/lift \
  dataset.horizon=10 dataset.val_horizon=150 \
  experiment.training.batch_size=4 \
  experiment.training.precision=bf16-mixed \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.validation.precision=bf16-mixed \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=16 \
  experiment.validation.data.num_workers=16 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="path_to_stage_1.ckpt" \
  algorithm.training_stage=2
```

### Stage 3: Decoder Finetuning

Finetune decoder for robustness to latent noise. Requires stage 2 checkpoint.

```bash
python main.py +name=robosuite_lift_stage_3 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_robosuite_dataset \
  dataset.dataset_dir=data/robosuite/lift \
  dataset.horizon=1 dataset.val_horizon=150 \
  experiment.training.batch_size=16 \
  experiment.training.precision=bf16-mixed \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.validation.precision=bf16-mixed \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=16 \
  experiment.validation.data.num_workers=16 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="path_to_stage_2.ckpt" \
  algorithm.training_stage=3
```

---

## LSF Job Script

The job script is at [jobs/train_stage1.bsub](../jobs/train_stage1.bsub). Submit from the project root:

```bash
bsub < jobs/train_stage1.bsub
```

Logs go to `jobs/logs/iws_robosuite.<JOB_ID>.{out,err}`.

---

## Cache Invalidation

The zarr cache (`cache.zarr.zip`) does not encode which `action_mode` was used. If you change `action_mode` or the action extraction logic, **delete the existing cache files** before rerunning:

```bash
rm data/robosuite/lift/train/cache.zarr.zip
rm data/robosuite/lift/val/cache.zarr.zip
```
