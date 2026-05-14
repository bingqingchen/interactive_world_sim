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
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from yixuan_utilities.open3d_utils import (
        aggr_point_cloud_from_data,
        np2o3d,
        o3dVisualizer,
    )
except ImportError:
    print("ERROR: yixuan_utilities is required. Install from requirements.txt.")
    sys.exit(1)

from interactive_world_sim.real_world.multi_realsense import MultiRealsense

# ---------------------------------------------------------------------------
# ROS 2 — prefer subprocess-based listener (avoids Python version mismatch
# between ROS 2 Humble / Python 3.10 and the conda env / Python 3.11).
# Fall back to rclpy if it happens to be importable in this interpreter.
# ---------------------------------------------------------------------------

_ROS2_CLI = shutil.which("ros2")  # None if ros2 not on PATH

_HAS_ROS2_RCLPY = False
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    _HAS_ROS2_RCLPY = True
except ImportError:
    pass

_HAS_ROS2 = bool(_ROS2_CLI) or _HAS_ROS2_RCLPY


def pose_stamped_yaml_to_mat(data: dict) -> np.ndarray:
    """Convert a parsed YAML dict from `ros2 topic echo` to a 4x4 SE(3) matrix."""
    pos = data["pose"]["position"]
    ori = data["pose"]["orientation"]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat(
        [ori["x"], ori["y"], ori["z"], ori["w"]]
    ).as_matrix()
    mat[:3, 3] = [pos["x"], pos["y"], pos["z"]]
    return mat


def pose_stamped_to_mat(msg) -> np.ndarray:
    """Convert a geometry_msgs/PoseStamped to a 4x4 SE(3) matrix."""
    p = msg.pose.position
    q = msg.pose.orientation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    mat[:3, 3] = [p.x, p.y, p.z]
    return mat


class ManualEEPoseProvider:
    """Fallback provider that stores EE positions entered manually on stdin."""

    def __init__(self) -> None:
        self._left_pos: Optional[np.ndarray] = None
        self._right_pos: Optional[np.ndarray] = None

    def set_position(self, arm: str, xyz: np.ndarray) -> None:
        if arm == "left":
            self._left_pos = xyz.copy()
        else:
            self._right_pos = xyz.copy()

    def get_ee_positions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._left_pos, self._right_pos

    def get_ee_poses(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        def _to_mat(p: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if p is None:
                return None
            m = np.eye(4, dtype=np.float64)
            m[:3, 3] = p
            return m
        return _to_mat(self._left_pos), _to_mat(self._right_pos)

    def shutdown(self) -> None:
        pass


class ROS2SubprocessEEListener:
    """
    EE pose listener that reads poses by calling `ros2 topic echo --once`.

    This avoids the Python-version mismatch between ROS 2 Humble (Python 3.10)
    and the conda runtime (Python 3.11): the ros2 CLI is a separate process so
    the C-extension ABI difference does not matter.
    """

    LEFT_TOPIC = "/motion_control/pose_ee_arm_left"
    RIGHT_TOPIC = "/motion_control/pose_ee_arm_right"

    _POLL_INTERVAL = 0.1  # seconds between background refreshes

    def __init__(self) -> None:
        self._left_pose: Optional[np.ndarray] = None
        self._right_pose: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _fetch_once(self, topic: str) -> Optional[np.ndarray]:
        try:
            result = subprocess.run(
                [_ROS2_CLI, "topic", "echo", "--once", topic],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            # The output contains the YAML block between '---' separators.
            text = result.stdout
            # Strip the trailing '---' separator and any warning lines
            blocks = [b.strip() for b in text.split("---") if b.strip()]
            if not blocks:
                return None
            # Take the last complete block (most recent message)
            data = yaml.safe_load(blocks[-1])
            if data and "pose" in data:
                return pose_stamped_yaml_to_mat(data)
        except (subprocess.TimeoutExpired, yaml.YAMLError, KeyError, TypeError):
            pass
        return None

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            left = self._fetch_once(self.LEFT_TOPIC)
            right = self._fetch_once(self.RIGHT_TOPIC)
            with self._lock:
                if left is not None:
                    self._left_pose = left
                if right is not None:
                    self._right_pose = right
            self._stop.wait(self._POLL_INTERVAL)

    def get_ee_positions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            left = self._left_pose[:3, 3].copy() if self._left_pose is not None else None
            right = self._right_pose[:3, 3].copy() if self._right_pose is not None else None
        return left, right

    def get_ee_poses(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            left = self._left_pose.copy() if self._left_pose is not None else None
            right = self._right_pose.copy() if self._right_pose is not None else None
        return left, right

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


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

import open3d as o3d  # noqa: E402 (placed here to keep top imports clean)
import cv2 as _cv2  # noqa: E402


# ---------------------------------------------------------------------------
# Headless web-stream visualizer (MJPEG over HTTP, no X11 required)
# ---------------------------------------------------------------------------

class WebStreamVisualizer:
    """
    Drop-in replacement for o3dVisualizer that renders the scene offscreen
    and serves each frame as an MJPEG stream over HTTP.

    No X11 display is required.  Open in a browser at:
      http://localhost:<port>/
    If you are accessing remotely, forward the port first:
      ssh -L <port>:localhost:<port> <user>@<host>
    """

    _HTML = (
        b"<html><head><title>Calibration Viewer</title>"
        b"<style>body{margin:0;background:#111;display:flex;"
        b"justify-content:center;align-items:center;height:100vh;}"
        b"img{max-width:100%;max-height:100vh;}</style></head>"
        b"<body><img src='/stream' alt='point cloud stream'></body></html>"
    )

    def __init__(self, port: int = 8888, width: int = 1280, height: int = 720) -> None:
        self.port = port
        self.width = width
        self.height = height
        self._renderer: Optional["o3d.visualization.rendering.OffscreenRenderer"] = None
        self._geometries: dict = {}
        self._frame_lock = threading.Lock()
        self._jpeg_frame: bytes = b""

    def start(self) -> None:
        # EGL surfaceless mode: no X display required, falls back to Mesa software
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self._renderer.scene.set_background([0.15, 0.15, 0.15, 1.0])
        # Default camera: above-front view of a ~2 m robot workspace
        self._renderer.setup_camera(
            60.0,
            np.array([0.0, 0.0, 0.5]),   # look-at center
            np.array([0.0, -2.0, 1.5]),  # eye position
            np.array([0.0, 0.0, 1.0]),   # up vector
        )
        self._start_http_server()
        print(
            f"\n  [WebVis] Browser stream ready — open in your browser:\n"
            f"    http://localhost:{self.port}/\n"
            f"  [WebVis] If viewing remotely, set up the SSH tunnel first:\n"
            f"    ssh -L {self.port}:localhost:{self.port} <user>@<host>\n"
        )

    def _start_http_server(self) -> None:
        import http.server
        import socketserver

        vis = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:  # suppress access logs
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(WebStreamVisualizer._HTML)))
                    self.end_headers()
                    self.wfile.write(WebStreamVisualizer._HTML)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while True:
                            with vis._frame_lock:
                                frame = vis._jpeg_frame
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                    + frame
                                    + b"\r\n"
                                )
                                self.wfile.flush()
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads = True

        server = _Server(("0.0.0.0", self.port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def add_triangle_mesh(
        self,
        type: str,
        mesh_name: str,
        color: Optional[np.ndarray] = None,
        radius: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
        depth: float = 0.1,
        size: float = 0.1,
    ) -> None:
        if type == "origin":
            geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        elif type == "sphere":
            geom = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        elif type == "box":
            geom = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        else:
            raise NotImplementedError(f"Unsupported mesh type: {type!r}")
        geom.compute_vertex_normals()
        if color is not None:
            geom.paint_uniform_color(color)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        if mesh_name in self._geometries:
            self._renderer.scene.remove_geometry(mesh_name)
        self._geometries[mesh_name] = geom
        self._renderer.scene.add_geometry(mesh_name, geom, mat)

    def update_pcd(self, pcd: "o3d.geometry.PointCloud", mesh_name: str) -> None:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        if mesh_name in self._geometries:
            self._renderer.scene.remove_geometry(mesh_name)
        self._geometries[mesh_name] = pcd
        self._renderer.scene.add_geometry(mesh_name, pcd, mat)

    def render(self, **_kwargs: object) -> None:
        img = self._renderer.render_to_image()
        img_np = np.asarray(img)  # H x W x 3, RGB uint8
        _, jpeg_buf = _cv2.imencode(".jpg", img_np[..., ::-1], [_cv2.IMWRITE_JPEG_QUALITY, 75])
        with self._frame_lock:
            self._jpeg_frame = jpeg_buf.tobytes()


def _is_valid_extrinsic(ext: np.ndarray) -> bool:
    """Return True if *ext* looks like an invertible 4x4 pose matrix."""
    if ext.shape != (4, 4):
        return False
    det = float(np.linalg.det(ext[:3, :3]))
    return abs(det) > 0.5  # proper rotation matrix has det ≈ ±1


def get_camera_point_cloud(
    realsense: MultiRealsense,
    boundaries: dict,
) -> Tuple["o3d.geometry.PointCloud", np.ndarray]:
    """
    Capture one frame from all cameras and return the aggregated point cloud
    as (pcd_o3d, points_np) where points_np is (N, 3) in camera world frame.

    Cameras whose extrinsics matrix is all-zero (ring buffer not yet populated
    or camera failed to start) are silently skipped.
    """
    out = realsense.get()

    # Filter to cameras with a valid (invertible) extrinsics matrix.
    valid = {k: v for k, v in out.items() if _is_valid_extrinsic(v["extrinsics"])}
    if not valid:
        return o3d.geometry.PointCloud(), np.zeros((0, 3))

    colors = np.stack([v["color"] for v in valid.values()])[..., ::-1]  # BGR→RGB
    depths = np.stack([v["depth"] for v in valid.values()]) / 1000.0  # mm→m
    intrinsics = np.stack([v["intrinsics"] for v in valid.values()])
    extrinsics = np.stack([v["extrinsics"] for v in valid.values()])

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
    print("\n=== R1 Lite \u2194 Camera World Frame Calibration ===\n")

    # --- EE pose provider ---
    if args.manual:
        print("Manual mode: ROS 2 not used. Enter EE positions when prompted.")
        ee_listener: "ManualEEPoseProvider | ROS2SubprocessEEListener | R1EEPoseListener" = ManualEEPoseProvider()
    elif _ROS2_CLI:
        print("Starting ROS 2 EE pose listener (subprocess mode) \u2026")
        ee_listener = ROS2SubprocessEEListener()
        time.sleep(1.0)
    elif _HAS_ROS2_RCLPY:
        print("Starting ROS 2 EE pose listener (rclpy mode) \u2026")
        ee_listener = R1EEPoseListener()
        time.sleep(1.0)
    else:
        print(
            "ERROR: ros2 CLI not found and rclpy is not importable.\n"
            "  Install ROS 2 or re-run with --manual to enter EE positions by hand."
        )
        sys.exit(1)

    # --- Cameras ---
    print("Starting RealSense cameras \u2026")
    boundaries = {
        "x_lower": -2.0, "x_upper": 2.0,
        "y_lower": -2.0, "y_upper": 2.0,
        "z_lower": -2.0, "z_upper": 2.0,
    }

    realsense = MultiRealsense(
        resolution=(640, 480),
        put_downsample=True,
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

    # --- Open3D visualizer (web stream when DISPLAY is not available) ---
    _use_web = not os.environ.get("DISPLAY", "").strip()
    if _use_web:
        vis: "o3dVisualizer | WebStreamVisualizer" = WebStreamVisualizer(port=args.web_port)
    else:
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

    collect_hint = (
        "  Press 'c' to collect a correspondence \u2014 you will be prompted to type\n"
        "  the EE position as 'x,y,z' (in metres)."
        if args.manual else
        "  Press 'c' to collect a correspondence (R1 EE pos \u2194 nearest camera point)."
    )
    print(
        "\nInstructions:\n"
        "  Move one R1 EE to a clearly visible point in the camera view.\n"
        f"{collect_hint}\n"
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
                if isinstance(ee_listener, ManualEEPoseProvider):
                    raw = input(f"  Enter {arm_side} EE position as 'x,y,z' (metres): ").strip()
                    try:
                        xyz = np.array([float(v) for v in raw.split(",")])
                        if xyz.shape != (3,):
                            raise ValueError
                        ee_listener.set_position(arm_side, xyz)
                    except (ValueError, IndexError):
                        print("  [!] Invalid input. Enter three comma-separated floats, e.g. 0.1,0.2,0.3")
                        continue

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
        print("\nShutting down \u2026")
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
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual mode: enter EE positions as x,y,z on stdin instead of reading from ROS 2 topics.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8888,
        metavar="PORT",
        help="HTTP port for the browser-based visualizer (used when DISPLAY is not set, default: 8888).",
    )
    args = parser.parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
