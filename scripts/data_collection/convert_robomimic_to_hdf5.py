"""Convert one or more robomimic image HDF5 files to the IWS episode-per-file format.

Pass multiple -i flags to merge several source datasets into one output directory.
Episodes are numbered sequentially across all source files.

Usage (single file):
    python scripts/data_collection/convert_robomimic_to_hdf5.py \\
        -i datasets/lift/ph/image_v15.hdf5 \\
        -o data/lift_ph

Usage (merge):
    python scripts/data_collection/convert_robomimic_to_hdf5.py \\
        -i datasets/lift/ph/image_v15.hdf5 \\
        -i datasets/lift/mh/image_v15_128.hdf5 \\
        -o data/lift_merged

Output layout::

    <output_dir>/
    ├── train/
    │   ├── episode_0.hdf5
    │   └── ...
    └── val/
        ├── episode_0.hdf5
        └── ...

Each episode HDF5 contains::

    action                    (T, A)         float32
    timestamp                 (T,)           float64
    obs/
        joint_pos             (T, 7)         float32
        ee_pos                (T, 1, 4, 4)   float32   SE(3) end-effector pose
        world_t_robot_base    (T, 1, 4, 4)   float32   identity for fixed-base Panda
        images/
            camera_0_color    (T, H, W, 3)   uint8     agentview
            camera_1_color    (T, H, W, 3)   uint8     robot0_eye_in_hand
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import click
import cv2
import h5py
import numpy as np
from tqdm import tqdm

try:
    from yixuan_utilities.hdf5_utils import save_dict_to_hdf5
except ImportError:
    from interactive_world_sim.algorithms.common.hdf5_utils import save_dict_to_hdf5

from interactive_world_sim.utils.pose_utils import pos_quat_to_mat

ROBOSUITE_CONTROL_FREQ = 20  # Hz


def _get_split(
    f: h5py.File,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Return (train_demo_keys, val_demo_keys)."""
    if "mask" in f and "train" in f["mask"] and "valid" in f["mask"]:
        train_keys = [k.decode("utf-8") for k in np.array(f["mask"]["train"][:])]
        val_keys = [k.decode("utf-8") for k in np.array(f["mask"]["valid"][:])]
        print(
            f"Using existing mask split: {len(train_keys)} train / {len(val_keys)} val"
        )
        return train_keys, val_keys

    all_keys = sorted(f["data"].keys())
    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(all_keys))
    n_val = max(1, int(len(shuffled) * val_ratio))
    train_keys = shuffled[: len(shuffled) - n_val]
    val_keys = shuffled[len(shuffled) - n_val :]
    print(
        f"Random split (seed={seed}, val_ratio={val_ratio}): "
        f"{len(train_keys)} train / {len(val_keys)} val"
    )
    return train_keys, val_keys


def _resize_images(imgs: np.ndarray, resolution: int) -> np.ndarray:
    """INTER_AREA resize each frame in (T, H, W, 3) to (T, resolution, resolution, 3)."""
    assert False, "We should not need to resizez images"
    return np.stack(
        [
            cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
            for frame in imgs
        ],
        axis=0,
    )


def build_episode(demo_group: h5py.Group, resolution: int) -> Dict:
    """Read one robomimic demo group and return an IWS-format episode dict."""
    T = int(demo_group.attrs["num_samples"])

    actions = demo_group["actions"][()].astype(np.float32)  # (T, A)
    joint_pos = demo_group["obs"]["robot0_joint_pos"][()].astype(np.float32)  # (T, 7)

    eef_pos = demo_group["obs"]["robot0_eef_pos"][()]  # (T, 3) float64
    eef_quat_xyzw = demo_group["obs"]["robot0_eef_quat"][()]  # (T, 4) float64 xyzw
    # pos_quat_to_mat expects wxyz quaternion order
    eef_quat_wxyz = eef_quat_xyzw[:, [3, 0, 1, 2]]
    pose_in_pos_quat = np.concatenate([eef_pos, eef_quat_wxyz], axis=1)  # (T, 7)
    ee_pos_mat = pos_quat_to_mat(pose_in_pos_quat).astype(np.float32)  # (T, 4, 4)
    ee_pos = ee_pos_mat[:, None, :, :]  # (T, 1, 4, 4)

    # Panda is a fixed-base robot; base pose in world is an identity transform
    world_t_robot_base = np.broadcast_to(
        np.eye(4, dtype=np.float32)[None, None], (T, 1, 4, 4)
    ).copy()

    cam0 = demo_group["obs"]["agentview_image"][()]  # (T, H, W, 3) uint8
    cam1 = demo_group["obs"]["robot0_eye_in_hand_image"][()]

    if cam0.shape[1] != resolution or cam0.shape[2] != resolution:
        cam0 = _resize_images(cam0, resolution)
        cam1 = _resize_images(cam1, resolution)

    timestamp = (np.arange(T) / ROBOSUITE_CONTROL_FREQ).astype(np.float64)

    return {
        "action": actions,
        "timestamp": timestamp,
        "obs": {
            "joint_pos": joint_pos,
            "ee_pos": ee_pos,
            "world_t_robot_base": world_t_robot_base,
            "images": {
                "camera_0_color": cam0,
                "camera_1_color": cam1,
            },
        },
    }


def _verify_episode(episode: Dict, resolution: int) -> None:
    """Sanity-check shapes and dtypes before writing."""
    T = episode["action"].shape[0]
    assert episode["action"].dtype == np.float32
    assert episode["timestamp"].shape == (T,)
    assert episode["obs"]["joint_pos"].shape == (T, 7)
    assert episode["obs"]["ee_pos"].shape == (T, 1, 4, 4)
    assert episode["obs"]["world_t_robot_base"].shape == (T, 1, 4, 4)
    for cam_key in ("camera_0_color", "camera_1_color"):
        img = episode["obs"]["images"][cam_key]
        assert img.shape == (T, resolution, resolution, 3), (
            f"{cam_key} shape {img.shape} != expected ({T}, {resolution}, {resolution}, 3)"
        )
        assert img.dtype == np.uint8


def _save_episode(
    episode: Dict,
    split_dir: str,
    episode_id: int,
    attr_dict: Dict,
    resolution: int,
) -> None:
    """Write one episode dict to <split_dir>/episode_<episode_id>.hdf5."""
    H = W = resolution
    config_dict: Dict = {
        "timestamp": {"dtype": "float64"},
        "obs": {
            "images": {
                "camera_0_color": {"chunks": (1, H, W, 3), "dtype": "uint8"},
                "camera_1_color": {"chunks": (1, H, W, 3), "dtype": "uint8"},
            }
        }
    }
    episode_path = os.path.join(split_dir, f"episode_{episode_id}.hdf5")
    save_dict_to_hdf5(episode, config_dict, episode_path, attr_dict=attr_dict)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a robomimic image HDF5 file. Repeat to merge multiple files.",
)
@click.option(
    "--output_dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Root output directory. train/ and val/ sub-directories are created here.",
)
@click.option(
    "--val_ratio",
    default=0.1,
    show_default=True,
    type=float,
    help="Fraction of demos to hold out for val (ignored when mask/train+valid exist).",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Random seed for the fallback val split. Incremented per file when merging.",
)
@click.option(
    "--resolution",
    default=128,
    show_default=True,
    type=int,
    help="Output image resolution (H=W). Images are resized only when needed.",
)
def main(
    input_paths: Tuple[str, ...],
    output_dir: str,
    val_ratio: float,
    seed: int,
    resolution: int,
) -> None:
    """Convert one or more robomimic HDF5 datasets to IWS episode-per-file format."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    train_ep_id = 0
    val_ep_id = 0
    total_train = 0
    total_val = 0

    # First pass: collect splits from all files to know the total demo count.
    splits: List[Tuple[str, Dict, List[str], List[str]]] = []
    for file_idx, input_path in enumerate(input_paths):
        with h5py.File(input_path, "r", swmr=True) as f:
            env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
            attr_dict = {
                "sim": True,
                "source": "robomimic",
                "source_file": os.path.basename(input_path),
                "env_name": env_args.get("env_name", "unknown"),
                "env_version": env_args.get("env_version", "unknown"),
            }
            train_keys, val_keys = _get_split(f, val_ratio, seed + file_idx)
            splits.append((input_path, attr_dict, train_keys, val_keys))

    total_demos = sum(len(tk) + len(vk) for _, _, tk, vk in splits)

    with tqdm(total=total_demos, unit="demo") as pbar:
        for file_idx, (input_path, attr_dict, train_keys, val_keys) in enumerate(splits):
            pbar.set_description(f"[{file_idx + 1}/{len(splits)}] {os.path.basename(input_path)}")
            with h5py.File(input_path, "r", swmr=True) as f:
                for demo_key in train_keys:
                    episode = build_episode(f["data"][demo_key], resolution)
                    _verify_episode(episode, resolution)
                    _save_episode(episode, train_dir, train_ep_id, attr_dict, resolution)
                    pbar.set_postfix(split="train", ep=train_ep_id)
                    train_ep_id += 1
                    pbar.update(1)

                for demo_key in val_keys:
                    episode = build_episode(f["data"][demo_key], resolution)
                    _verify_episode(episode, resolution)
                    _save_episode(episode, val_dir, val_ep_id, attr_dict, resolution)
                    pbar.set_postfix(split="val", ep=val_ep_id)
                    val_ep_id += 1
                    pbar.update(1)

            total_train += len(train_keys)
            total_val += len(val_keys)

    print(
        f"\nDone. {total_train + total_val} demos → {total_train} train / {total_val} val"
        f" in {output_dir}"
    )


if __name__ == "__main__":
    main()
