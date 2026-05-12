# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive World Simulator (IWS) — a latent world model for robot manipulation that predicts future image observations conditioned on robot actions. Supports both simulated (MuJoCo/ALOHA) and real-world (ALOHA bimanual, R1 Lite) robot data. Based on [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing).

## Environment Setup

```bash
mamba env create -f conda_env.yaml
conda activate iws
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/
pip install -e .
# For MuJoCo sim environments:
git submodule update --init --recursive
uv pip install -e external/gym-aloha/
```

Set W&B entity before training:
```yaml
# configurations/config.yaml
wandb:
  entity: YOUR_WANDB_ENTITY
```

## Commands

**Lint / format:**
```bash
pre-commit run --all-files   # runs ruff, black, mypy, check-yaml, etc.
ruff check . --fix           # lint only
black .                      # format only
```

**Training (3-stage pipeline):**
```bash
# Stage 1: Autoencoder
python main.py +name=<run_name> algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1

# Stage 2: Dynamics (requires stage 1 checkpoint)
python main.py +name=<run_name> algorithm=latent_world_model \
  ... algorithm.training_stage=2 \
  algorithm.load_ae="path/to/stage1.ckpt"

# Stage 3: Decoder finetuning (requires stage 2 checkpoint)
python main.py +name=<run_name> algorithm=latent_world_model \
  ... algorithm.training_stage=3 \
  algorithm.load_ae="path/to/stage2.ckpt"
```

**Inference (keyboard, no robot):**
```bash
python scripts/inference/teleoperate_keyboard.py \
  +output_dir='data/wm_demo' +use_joystick=false +use_dataset=false \
  +act_horizon=1 +scene=real \
  "+ckpt_paths=['outputs/pusht_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht/val \
  "dataset.obs_keys=['camera_1_color']"
```

**Local demo server:**
```bash
bash deploy/start_demo.sh
```

**Data collection (real robot):**
```bash
python scripts/data_collection/collect_real_aloha.py \
  --output_dir data/<task> --robot_sides right left \
  --frequency 10 --ctrl_mode bimanual_push --total_steps 200
python -m interactive_world_sim.real_world.robot_sleep --left --right
```

**Cluster (LSF/bsub):**
```bash
bsub < jobs/train_stage1.bsub
bsub < jobs/train_stage2.bsub
```

## Architecture

### Configuration System (Hydra)

All config is composed from YAML files under `configurations/`:
- `config.yaml` — root, selects experiment/dataset/algorithm/cluster
- `configurations/experiment/` — training loop settings (batch size, steps, val frequency)
- `configurations/algorithm/` — model hyperparameters (latent_dim, action_dim, diffusion params)
- `configurations/dataset/` — dataset path, horizon, obs_keys, action_mode, shape_meta

Override any config at the command line with Hydra syntax (`key=value`, `+key=value` for new keys, `"key=[a,b]"` for lists).

### Experiment / Training Loop

`main.py` → `BaseLightningExperiment` → PyTorch Lightning `Trainer`

`experiments/exp_base.py:BaseLightningExperiment` builds the dataset, dataloaders, algorithm, and runs `trainer.fit()`. `experiments/exp_latent_dyn.py:LatentDynExperiment` wires the compatible algorithm (`LatentWorldModel`) and datasets (`RealAlohaDataset`, `SimAlohaDataset`). Adding a new experiment means subclassing `BaseLightningExperiment`, registering in `compatible_algorithms`/`compatible_datasets`, and adding a YAML under `configurations/experiment/`.

### Model (`LatentWorldModel`)

`algorithms/latent_dynamics/latent_world_model.py` — the Lightning module for all three stages.

| Component | Location | Role |
|-----------|----------|------|
| Encoder | `_build_model()` — `nn.Sequential` of Conv2d + SiLU | RGB obs → compact latent |
| Decoder | `algorithms/models/cm_decoder.py:CMDecoder` | Latent → RGB via diffusion UNet (CMControlledUnetModel) |
| Dynamics | `algorithms/latent_dynamics/models/cm_latent_dynamics.py:CMLatentDynamics` | Predicts next latent from current latent + action; causal temporal attention |

Stage loading: `algorithm.load_ae` in the YAML points to a `.ckpt` file; the `LatentWorldModel` constructor loads encoder+decoder weights from it before training the next stage.

Checkpoints are saved alongside their Hydra config at `outputs/<date>/<time>/checkpoints/` with a `.hydra/config.yaml` sibling directory — the inference scripts rely on this layout to reload the model config.

### Data Pipeline

Raw data (HDF5 episodes) → Zarr cache → `ReplayBuffer` → `SequenceSampler` → DataLoader

`datasets/latent_dynamics/real_aloha_dataset.py:_convert_real_to_dp_replay()` converts per-episode HDF5 files into a single Zarr archive (`cache.zarr.zip`) on the first load. Subsequent runs use the cache.

**HDF5 episode layout:**
- Sim: `action (T,4)`, `obs/joint_pos (T,14)`, `obs/ee_pos (T,2,4,4)`, `obs/images/top_pov (T,128,128,3)`
- Real ALOHA: `action (T,D)`, `obs/joint_pos (T,14)`, `obs/ee_pos (T,2,4,4)`, `obs/images/camera_{0,1}_color (T,H,W,3)`

**DataLoader batch:**
```python
{
    "obs":  {"camera_0_color": (T, 3, 128, 128)},   # float32 [0,1]
    "goal": {"camera_0_color": (3, 128, 128)},
    "action": (T, D),                                # float32 [-1,1] normalized
}
```

`dataset.action_mode` and `dataset.action_dim` must be consistent — `bimanual_push` = 4-dim XY for both arms.

### Active Work: R1 Lite Integration

`docs/data-processing.md` documents the in-progress effort to harmonize R1 Lite MCAP data (mobile bimanual humanoid, 6 joints/arm, 4 cameras, mixed-rate ROS 2 topics) to IWS HDF5 format. Key deliverables tracked there:
- `scripts/data_collection/convert_r1lite_mcap_to_hdf5.py` (converter)
- `interactive_world_sim/datasets/latent_dynamics/r1lite_dataset.py` (new dataset class)
- `configurations/dataset/r1lite_dataset.yaml` (new config)

Start with `bimanual_push` action mode (EE XY, action_dim=4) for maximum model compatibility. Avoid ALOHA-specific gripper normalization functions (`PUPPET_GRIPPER_JOINT_NORMALIZE_FN`) and `KinHelper("trossen_vx300s")` in R1 Lite code.

### Real-World Interface

`real_world/` contains hardware drivers: `multi_realsense.py` / `single_realsense.py` (Intel RealSense), `aloha_bimanual_master.py` / `aloha_bimanual_puppet.py` (Interbotix ALOHA arms), `real_aloha_env.py` (environment wrapper). Camera calibration extrinsics are stored in `real_world/aloha_extrinsics/`.

### Deploy (Browser Demo)

`deploy/server.py` is a FastAPI + WebSocket server that runs inference from a pre-loaded `LatentWorldModel` and streams rendered frames to the browser. Start with `bash deploy/start_demo.sh`, then connect from the project page or `localhost`.
