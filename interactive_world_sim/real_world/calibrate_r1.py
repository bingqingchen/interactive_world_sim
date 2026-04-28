"""
Calibrate the rigid transform between R1 Lite's world frame and the
camera-calibrated world frame.

The R1 Lite firmware publishes end-effector poses in its own world frame via
ROS 2 topics. The RealSense cameras have been calibrated to a separate world
frame (defined by the ChArUco board). This script computes the rigid transform
T_r1_to_cam that maps R1's world frame into the camera world frame.

Usage:
  1. Ensure ROS 2 is running and R1 Lite EE pose topics are publishing.
  2. Ensure your RealSense cameras are connected and calibrated
     (cam_extrinsics/ populated).
  3. Run:
       python interactive_world_sim/real_world/calibrate_r1.py

Interactive controls:
  - Press 'c' in the terminal to collect a correspondence point:
    the current R1 EE position (from ROS 2) and the corresponding centroid
    from the camera point cloud (the point closest to where the EE appears
    in the depth cloud) are recorded.
  - Press 's' to solve for the rigid transform after collecting >= 3 points.
  - Press 'v' to toggle visual verification mode (overlay transformed EE
    markers onto the camera point cloud).
  - Press 'q' to quit.

Output:
  interactive_world_sim/real_world/r1_extrinsics/T_r1_to_cam_world.npy
"""

import argparse
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from yixuan_utilities.draw_utils import (
        aggr_point_cloud_from_data,
        np2o3d,
        o3dVisualizer,
    )
except ImportError:
    print("ERROR: yixuan_utilities is required. Install from requirements.txt.")
    sys.exit(1)

from interactive_world_sim.real_world.multi_realsense import MultiRealsense

# ---------------------------------------------------------------------------
# ROS 2 subscriber for R1 Lite EE poses
# ---------------------------------------------------------------------------

_HAS_ROS2 = False
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    _HAS_ROS2 = True
except ImportError:
    pass


def pose_stamped_to_mat(msg) -> np.ndarray:
    """Convert a geometry_msgs/PoseStamped to a 4x4 SE(3) matrix."""
    p = msg.pose.position
    q = msg.pose.orientation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    mat[:3, 3] = [p.x, p.y, p.z]
    return mat


class R1EEPoseListener:
    """Subscribes to R1 Lite EE pose topics and caches the latest poses."""

    LEFT_TOPIC = "/motion_control/pose_ee_arm_left"
    RIGHT_TOPIC = "/motion_control/pose_ee_arm_right"

    def __init__(self) -> None:
        if not _HAS_ROS2:
            raise RuntimeError(
                "rclpy is not available. Source your ROS 2 workspace first."
            )
        rclpy.init()
        self._node = rclpy.create_node("r1_calibration_listener")

        self._left_pose: Optional[np.ndarray] = None
        self._right_pose: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self._node.create_subscription(
            PoseStamped, self.LEFT_TOPIC, self._left_cb, 10
        )
        self._node.create_subscription(
            PoseStamped, self.RIGHT_TOPIC, self._right_cb, 10
        )

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()

    def _left_cb(self, msg: "PoseStamped") -> None:
        with self._lock:
            self._left_pose = pose_stamped_to_mat(msg)

    def _right_cb(self, msg: "PoseStamped") -> None:
        with self._lock:
            self._right_pose = pose_stamped_to_mat(msg)

    def get_ee_positions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (left_xyz, right_xyz) in R1's world frame, or None if not yet received."""
        with self._lock:
            left = self._left_pose[:3, 3].copy() if self._left_pose is not None else None
            right = self._right_pose[:3, 3].copy() if self._right_pose is not None else None
        return left, right

    def get_ee_poses(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (left_4x4, right_4x4) in R1's world frame."""
        with self._lock:
            left = self._left_pose.copy() if self._left_pose is not None else None
            right = self._right_pose.copy() if self._right_pose is not None else None
        return left, right

    def shutdown(self) -> None:
        self._node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Rigid transform solver (Kabsch / Procrustes)
# ---------------------------------------------------------------------------


def solve_rigid_transform(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """
    Compute the rigid transform T (4x4) such that dst ≈ T @ src (homogeneous).

    Uses the Kabsch algorithm (SVD-based least-squares).

    Parameters
    ----------
    src : (N, 3) — source points (R1 EE positions in R1 world frame)
    dst : (N, 3) — target points (corresponding positions in camera world frame)

    Returns
    -------
    T : (4, 4) rigid transform matrix
    """
    assert src.shape == dst.shape and src.shape[0] >= 3, (
        f"Need >= 3 point pairs, got {src.shape[0]}"
    )

    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    src_c = src - centroid_src
    dst_c = dst - centroid_dst

    H = src_c.T @ dst_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    t = centroid_dst - R @ centroid_src

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Camera point cloud helpers
# ---------------------------------------------------------------------------


def get_camera_point_cloud(
    realsense: MultiRealsense,
    boundaries: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Capture one frame from all cameras and return the aggregated point cloud
    as (pcd_o3d, points_np) where points_np is (N, 3) in camera world frame.
    """
    out = realsense.get()
    colors = np.stack([v["color"] for v in out.values()])[..., ::-1]  # BGR→RGB
    depths = np.stack([v["depth"] for v in out.values()]) / 1000.0  # mm→m
    intrinsics = np.stack([v["intrinsics"] for v in out.values()])
    extrinsics = np.stack([v["extrinsics"] for v in out.values()])

    pcd_o3d = aggr_point_cloud_from_data(
        colors=colors,
        depths=depths,
        Ks=intrinsics,
        poses=extrinsics,
        downsample=False,
        boundaries=boundaries,
    )
    points = np.asarray(pcd_o3d.points)
    return pcd_o3d, points


def find_nearest_point(cloud: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Find the nearest point in *cloud* (N,3) to *query* (3,)."""
    tree = cKDTree(cloud)
    _, idx = tree.query(query)
    return cloud[idx]


# ---------------------------------------------------------------------------
# Marker helpers for visualisation
# ---------------------------------------------------------------------------


def make_sphere_pcd(center: np.ndarray, radius: float = 0.015, n: int = 200) -> np.ndarray:
    """Create a small sphere point cloud around *center* for visualisation."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius
    return pts + center


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------


def run_calibration(args: argparse.Namespace) -> None:
    print("\n=== R1 Lite ↔ Camera World Frame Calibration ===\n")

    # --- ROS 2 listener ---
    print("Starting ROS 2 EE pose listener …")
    ee_listener = R1EEPoseListener()
    time.sleep(1.0)

    # --- Cameras ---
    print("Starting RealSense cameras …")
    boundaries = {
        "x_lower": -2.0, "x_upper": 2.0,
        "y_lower": -2.0, "y_upper": 2.0,
        "z_lower": -2.0, "z_upper": 2.0,
    }

    realsense = MultiRealsense(
        resolution=(640, 480),
        put_downsample=False,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        verbose=False,
    )
    realsense.start()
    realsense.set_exposure(exposure=60, gain=64)
    realsense.set_white_balance(white_balance=3800)
    realsense.set_depth_preset("High Density")
    realsense.set_depth_exposure(3000, 16)
    time.sleep(3.0)

    # --- Open3D visualizer ---
    vis = o3dVisualizer()
    vis.start()
    vis.add_triangle_mesh("origin", "origin", size=0.1)
    vis.render()

    # --- State ---
    src_points: List[np.ndarray] = []  # R1 EE positions (R1 world frame)
    dst_points: List[np.ndarray] = []  # corresponding camera-world positions
    T_r1_to_cam: Optional[np.ndarray] = None
    verify_mode = False

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_extrinsics")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "T_r1_to_cam_world.npy")

    arm_side = args.arm  # "left" or "right"

    print(
        "\nInstructions:\n"
        "  Move one R1 EE to a clearly visible point in the camera view.\n"
        "  Press 'c' to collect a correspondence (R1 EE pos ↔ nearest camera point).\n"
        "  Collect at least 3 (ideally 5+) correspondences spread across the workspace.\n"
        "  Press 's' to solve for the rigid transform.\n"
        "  Press 'v' to toggle verification overlay.\n"
        "  Press 'q' to quit.\n"
        f"\n  Using {arm_side} arm for EE positions.\n"
    )

    try:
        while True:
            # Update camera point cloud
            pcd_o3d, cloud_pts = get_camera_point_cloud(realsense, boundaries)
            vis.update_pcd(pcd_o3d, "camera_pcd")

            # If in verify mode, overlay transformed EE markers
            if verify_mode and T_r1_to_cam is not None:
                left_pos, right_pos = ee_listener.get_ee_positions()
                marker_pts = []
                for pos in (left_pos, right_pos):
                    if pos is not None:
                        transformed = (T_r1_to_cam[:3, :3] @ pos + T_r1_to_cam[:3, 3])
                        marker_pts.append(make_sphere_pcd(transformed))
                if marker_pts:
                    all_markers = np.concatenate(marker_pts, axis=0)
                    vis.update_pcd(np2o3d(all_markers), "ee_markers")

            vis.render()

            # Non-blocking keyboard input
            if sys.stdin is not None and sys.stdin.readable():
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.readline().strip().lower()
                else:
                    continue
            else:
                time.sleep(0.05)
                continue

            if key == "c":
                # Collect correspondence
                left_pos, right_pos = ee_listener.get_ee_positions()
                ee_pos = left_pos if arm_side == "left" else right_pos
                if ee_pos is None:
                    print(f"  [!] No EE pose received for {arm_side} arm yet. Wait for topic data.")
                    continue

                if len(cloud_pts) == 0:
                    print("  [!] No camera point cloud available.")
                    continue

                # If we already have a partial transform, use it to seed the search;
                # otherwise use the raw R1 position as the query.
                if T_r1_to_cam is not None:
                    query = T_r1_to_cam[:3, :3] @ ee_pos + T_r1_to_cam[:3, 3]
                else:
                    query = ee_pos

                nearest = find_nearest_point(cloud_pts, query)
                src_points.append(ee_pos.copy())
                dst_points.append(nearest.copy())
                n = len(src_points)
                print(
                    f"  [+] Correspondence #{n} collected:\n"
                    f"      R1  EE pos: {ee_pos}\n"
                    f"      Cam point:  {nearest}\n"
                )

            elif key == "s":
                if len(src_points) < 3:
                    print(f"  [!] Need >= 3 correspondences, have {len(src_points)}.")
                    continue

                src = np.array(src_points)
                dst = np.array(dst_points)
                T_r1_to_cam = solve_rigid_transform(src, dst)

                # Compute residual
                transformed = (T_r1_to_cam[:3, :3] @ src.T).T + T_r1_to_cam[:3, 3]
                residuals = np.linalg.norm(transformed - dst, axis=1)
                rmse = np.sqrt(np.mean(residuals ** 2))

                print(
                    f"\n  === Rigid Transform Solved ({len(src_points)} correspondences) ===\n"
                    f"  RMSE: {rmse * 1000:.2f} mm\n"
                    f"  Per-point residuals (mm): {np.round(residuals * 1000, 2)}\n"
                    f"  T_r1_to_cam_world:\n{T_r1_to_cam}\n"
                )

                np.save(save_path, T_r1_to_cam)
                print(f"  Saved to: {save_path}\n")
                print("  Press 'v' to toggle verification overlay, 'c' to add more points, 'q' to quit.\n")

            elif key == "v":
                if T_r1_to_cam is None:
                    print("  [!] Solve first (press 's').")
                else:
                    verify_mode = not verify_mode
                    status = "ON" if verify_mode else "OFF"
                    print(f"  Verification overlay: {status}")

            elif key == "q":
                break

    finally:
        print("\nShutting down …")
        realsense.stop()
        ee_listener.shutdown()
        print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate R1 Lite world frame to camera-calibrated world frame."
    )
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="right",
        help="Which arm's EE to use for collecting correspondences (default: right).",
    )
    args = parser.parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
