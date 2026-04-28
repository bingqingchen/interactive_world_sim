# MuJoCo PushT World Model Training

## TODO: 
To re-enable it for Stage 2/3: you'll need to either pre-download i3d_torchscript.pt from a machine with unrestricted internet access and place it at interactive_world_sim/algorithms/common/metrics/i3d_torchscript.pt

## Dataset

**Source:** [Hugging Face](https://huggingface.co/datasets/yixuan1999/interactive-world-sim-mujoco-data)

```bash
python scripts/download_data_hf.py \
    --repo yixuan1999/interactive-world-sim-mujoco-data \
    --local_dir data/mujoco
```

### Metadata

| Property | Value |
|---|---|
| Task | PushT (bimanual ALOHA in MuJoCo) |
| Train episodes | 10,000 |
| Val episodes | 100 |
| Steps per episode | 300 |
| Image resolution | 128 × 128 × 3 (uint8) |
| Action dim | 4 (bimanual end-effector XY) |
| Action range | [-0.219, 0.219] |
| Collection | Scripted policy (linear, rotating, random_contact, random_no_contact) |

### HDF5 Structure (per episode)

```
episode_N.hdf5
├── action                (300, 4)    float32   # commanded EE actions
├── env_state             (300, 7)    float32   # ground-truth environment state
├── robot_bases           (300, 2, 4, 4) float32 # robot base transforms
└── obs/
    ├── ee_pos            (300, 2, 4, 4) float32 # end-effector 4×4 transforms (2 arms)
    ├── joint_pos         (300, 14)   float32    # joint positions (7 per arm)
    └── images/
        └── top_pov       (300, 128, 128, 3) uint8  # top-down RGB camera
```

### Directory layout

```
data/mujoco/
├── train/          # 10,000 episodes (episode_0.hdf5 … episode_9999.hdf5)
│   └── cache.zarr.zip   # auto-generated on first load
└── val/            # 100 episodes
    └── cache.zarr.zip
```

### Note on actions

The `SimAlohaDataset` loader does **not** use the stored `action` field directly. It derives 4D actions from `obs/ee_pos`:

```python
action = concat(ee_pos[:, 0, :2, 3], ee_pos[:, 1, :2, 3])  # (T, 4)
```

This extracts the XY translation of each arm's end-effector from the 4×4 transform matrices. There is a small discrepancy (~0.009 max) between this derived action and the stored `action` field.

---

## Configuration

### Dataset config: `sim_aloha_dataset`


Key fields:

| Field | Value | Notes |
|---|---|---|
| `obs_keys` | `[top_pov]` | Must match the HDF5 key under `obs/images/` |
| `shape_meta.action.shape` | `[4]` | 4D bimanual XY |
| `shape_meta.obs.top_pov` | `[3, 128, 128]`, type `rgb` | CHW format |
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
| `action_mode` | Only used by `RealAlohaDataset` (real robot joint → action conversion). `SimAlohaDataset` ignores it. Do not set it. |
| `delta_action` | Defined in `real_aloha_dataset` config but never read by any dataset class. |

---

## Training Stages

All three stages use the same dataset and algorithm configs; they differ in `training_stage`, `horizon`, and checkpoint loading.

### Stage 1: Autoencoder

Train encoder + diffusion decoder to compress RGB → latent space.

```bash
python main.py +name=pusht_stage_1 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_mujoco_dataset \
  dataset.dataset_dir=data/mujoco \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[top_pov] \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=10 \
  experiment.validation.val_every_n_step=6000 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1
```

Reconstruction should be **near-perfect** before proceeding to stage 2.

### Stage 2: Dynamics

Train latent dynamics model. Requires a stage 1 checkpoint.

```bash
python main.py +name=pusht_stage_2 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_mujoco_dataset \
  dataset.dataset_dir=data/mujoco \
  dataset.horizon=10 dataset.val_horizon=200 \
  dataset.obs_keys=[top_pov] \
  experiment.training.batch_size=4 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.noise_scheduler.loss_weighting=uniform \
  algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="path_to_stage_1.ckpt" \
  algorithm.training_stage=2
```

### Stage 3: Decoder Finetuning

Finetune decoder for robustness to latent noise. Requires stage 2 checkpoint.

```bash
python main.py +name=pusht_stage_3 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=sim_mujoco_dataset \
  dataset.dataset_dir=data/mujoco \
  dataset.horizon=1 dataset.val_horizon=200 \
  dataset.obs_keys=[top_pov] \
  experiment.training.batch_size=16 \
  experiment.training.max_steps=1000005 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.limit_batch=1.0 \
  experiment.validation.batch_size=2 \
  experiment.validation.val_every_n_step=30000 \
  experiment.training.checkpointing.every_n_train_steps=10000 \
  experiment.training.data.num_workers=4 \
  experiment.validation.data.num_workers=4 \
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

Logs go to `jobs/logs/iws_mujoco.<JOB_ID>.{out,err}`.
