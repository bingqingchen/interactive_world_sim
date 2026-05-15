import argparse
import time

import pyrealsense2 as rs

from interactive_world_sim.real_world.multi_realsense import MultiRealsense


def get_d435i_serials() -> list:
    """Return serial numbers of connected D435I cameras only."""
    ctx = rs.context()
    serials = []
    for d in ctx.query_devices():
        name = d.get_info(rs.camera_info.name)
        if "D435I" in name.upper():
            serials.append(d.get_info(rs.camera_info.serial_number))
    return serials


def calibrate_all(
    rows: int, cols: int, checker_width: float, marker_width: float
) -> None:
    serials = get_d435i_serials()
    if not serials:
        print("No D435I cameras found.")
        return
    print(f"Calibrating D435I cameras: {serials}")

    with MultiRealsense(
        serial_numbers=serials,
        put_downsample=False,
        # resolution=(640, 480),
        capture_fps=15,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        verbose=False,
    ) as realsense:
        realsense.set_exposure(exposure=60, gain=64)
        realsense.set_white_balance(white_balance=3800)
        time.sleep(1.0)
        realsense.calibrate_extrinsics(
            visualize=True,
            board_size=(cols, rows),
            squareLength=checker_width,
            markerLength=marker_width,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--checker_width", type=float, default=0.04)
    parser.add_argument("--marker_width", type=float, default=0.03)

    args = parser.parse_args()
    calibrate_all(args.rows, args.cols, args.checker_width, args.marker_width)
