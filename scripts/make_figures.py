"""Figures for the README: the training curve and a visual comparison.

Reads results/x<scale>/training_log.csv, results/x<scale>/summary.json (written by evaluate.py) and the
trained weights; writes training_curve.png and comparison.png next to them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srcnn.config import load_config, shown  # noqa: E402
from srcnn.data import load_pair, modcrop  # noqa: E402
from srcnn.evaluate import score  # noqa: E402
from srcnn.inference import super_resolve, upscale_image  # noqa: E402
from srcnn.metrics import to_uint8  # noqa: E402
from srcnn.model import load_weights  # noqa: E402
from srcnn.resize import imresize  # noqa: E402

SURFACE, TEXT, TEXT_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"
# test image -> (top, left, size) of the region shown, in ground-truth pixels
CROPS = {"Set5/butterfly_GT.bmp": (100, 140, 100), "Set14/zebra.bmp": (120, 410, 100)}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def training_curve(log_path: Path, summary: dict, monitor: str, out: Path) -> None:
    with open(log_path) as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["iteration"]) > 0]
    it = np.array([int(r["iteration"]) for r in rows]) / 1000
    psnr = np.array([float(r[f"{monitor}_psnr"]) for r in rows])
    entry = summary["sets"][monitor]
    refs = [("authors' weights (= paper)", entry.get("authors_weights_psnr")),
            ("bicubic", entry["bicubic_psnr"])]
    low = min(entry["bicubic_psnr"], psnr.min()) - 0.3
    high = max(psnr.max(), refs[0][1] or 0) + 0.25

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    style(ax)
    for label, y in refs:  # labels mid-way along the line, where the curve has long left the bicubic level
        if y is not None:
            ax.axhline(y, color=MUTED, linewidth=1)
            ax.text(it[-1] * 0.45, y + 0.04, f"{label}: {y:.2f} dB", color=TEXT_2, fontsize=8.5, va="bottom")
    ax.plot(it, psnr, color=SERIES, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(it[-1], psnr[-1], "o", markersize=8, color=SERIES, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{psnr[-1]:.2f} dB", (it[-1], psnr[-1]), xytext=(0, -16), textcoords="offset points",
                ha="right", va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, it[-1] * 1.02)
    ax.set_ylim(low, high)
    ax.set_xlabel("training iterations (thousands, batches of 128)", color=TEXT_2, fontsize=9)
    ax.set_ylabel(f"{monitor} PSNR (dB)", color=TEXT_2, fontsize=9)
    ax.set_title(f"SRCNN 9-1-5, x{summary['scale']}: {monitor} PSNR during training", color=TEXT,
                 fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def comparison(model, scale: int, out: Path) -> None:
    fig, axes = plt.subplots(len(CROPS), 3, figsize=(8.4, 3.15 * len(CROPS)), facecolor=SURFACE, squeeze=False)
    for row, (rel, (top, left, size)) in enumerate(CROPS.items()):
        path = ROOT / "data" / rel
        with Image.open(path) as im:
            hr = modcrop(np.array(im.convert("RGB")), scale)
        lr = to_uint8(imresize(hr / 255.0, 1.0 / scale))  # an 8-bit low-resolution image
        bicubic, srcnn = upscale_image(model, lr, scale, "cpu")
        gt_y, bicubic_y = load_pair(path, scale)  # the scores are those of the evaluation (Y channel)
        b_psnr, _ = score(gt_y, bicubic_y, scale)
        s_psnr, _ = score(gt_y, super_resolve(model, bicubic_y, "cpu"), scale)
        panels = [(hr, "original"), (bicubic, f"bicubic, {b_psnr:.2f} dB"), (srcnn, f"SRCNN, {s_psnr:.2f} dB")]
        for ax, (img, title) in zip(axes[row], panels):
            ax.imshow(img[top:top + size, left:left + size], interpolation="nearest")
            ax.set_title(title, color=TEXT, fontsize=9)
            ax.axis("off")
        axes[row][0].text(-0.04, 0.5, path.stem.replace("_GT", ""), transform=axes[row][0].transAxes,
                          rotation=90, ha="right", va="center", color=TEXT_2, fontsize=9)
    fig.suptitle(f"x{scale}, enlarged crops (PSNR of the whole image, Y channel)", color=TEXT_2, fontsize=9, y=0.995)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--scale", type=int, help="override the scale in the config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.scale)
    results = ROOT / cfg["results"]
    summary = json.loads((results / "summary.json").read_text())
    training_curve(results / "training_log.csv", summary, cfg["train"]["monitor"], results / "training_curve.png")
    model, _ = load_weights(ROOT / cfg["weights"])
    comparison(model, cfg["scale"], results / "comparison.png")
    print(f"wrote {shown(results, ROOT)}/training_curve.png and comparison.png")


if __name__ == "__main__":
    main()
