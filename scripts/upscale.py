"""Enlarge your own images with the trained network.

Every image in inputs/ (or every file given) is enlarged by the scale. <name>_bicubic_x<scale>.png and
<name>_srcnn_x<scale>.png are written to outputs/. The network enlarges the luminance; the colour
channels are enlarged by bicubic interpolation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srcnn.config import load_config, pick_device  # noqa: E402
from srcnn.data import IMAGE_SUFFIXES  # noqa: E402
from srcnn.inference import upscale_image  # noqa: E402
from srcnn.model import load_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="default: every image in inputs/")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--scale", type=int, help="override the scale in the config")
    parser.add_argument("--weights", type=Path, help="default: the path in the config")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config, args.scale)
    scale = cfg["scale"]
    device = pick_device(args.device)
    model, _ = load_weights(args.weights or ROOT / cfg["weights"], device)
    images = args.images or sorted(p for p in (ROOT / "inputs").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit("no images: put some in inputs/ or give their paths")
    args.out.mkdir(parents=True, exist_ok=True)
    for path in images:
        with Image.open(path) as im:
            img = np.array(im if im.mode == "L" else im.convert("RGB"))
        bicubic, srcnn = upscale_image(model, img, scale, device)
        for label, result in (("bicubic", bicubic), ("srcnn", srcnn)):
            target = args.out / f"{path.stem}_{label}_x{scale}.png"
            Image.fromarray(result).save(target)
        print(f"{path.name}: {img.shape[1]} x {img.shape[0]} -> {srcnn.shape[1]} x {srcnn.shape[0]}")


if __name__ == "__main__":
    main()
