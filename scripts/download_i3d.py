#!/usr/bin/env python3
"""Download i3d_torchscript.pt for FVD metric computation."""

import urllib.request
from pathlib import Path

URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
DEST = Path(__file__).parent.parent / "interactive_world_sim/algorithms/common/metrics/i3d_torchscript.pt"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading I3D model to {DEST} ...")
    urllib.request.urlretrieve(URL, DEST)
    print("Done.")


if __name__ == "__main__":
    main()
