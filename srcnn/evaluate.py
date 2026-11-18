"""Scores on a test set with the paper's protocol.

For every test image: Y channel cropped to a multiple of the scale (ground truth), bicubic down- and
up-sampling with MATLAB's kernel (input), the network with replicate padding (output). Both the
bicubic and the network output are converted to 8-bit, ``scale`` pixels are removed from each border,
and PSNR and SSIM are computed on what remains.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import list_images, load_pair
from .inference import super_resolve
from .metrics import psnr, shave, ssim, to_uint8
from .model import SRCNN

# Published results for x3 on the Y channel: (PSNR in dB, SSIM or None).
# "srcnn" is the 9-1-5 network trained on the 91 images (ECCV 2014 version, Tables 1 and 2).
# Bicubic, with SSIM: TPAMI 2016 version, Tables 2-4.
PAPER_X3 = {
    "Set5": {"bicubic": (30.39, 0.8682), "srcnn": (32.39, None)},
    "Set14": {"bicubic": (27.54, 0.7736), "srcnn": (29.00, None)},
    "BSD200": {"bicubic": (25.94, 0.7469), "srcnn": None},
}


def prepare(folder: str | Path, scale: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, ground truth uint8, bicubic input) for every image of a test set."""
    return [(path.stem, *load_pair(path, scale)) for path in list_images(folder)]


def score(gt: np.ndarray, estimate: np.ndarray, scale: int) -> tuple[float, float]:
    """(PSNR, SSIM) of a float estimate in [0, 1] against the uint8 ground truth."""
    a, b = shave(gt, scale), shave(to_uint8(estimate), scale)
    return psnr(a, b), ssim(a, b)


def evaluate(model: SRCNN, prepared, scale: int, device) -> list[dict]:
    rows = []
    for name, gt, bicubic in prepared:
        b_psnr, b_ssim = score(gt, bicubic, scale)
        s_psnr, s_ssim = score(gt, super_resolve(model, bicubic, device), scale)
        rows.append({"image": name, "bicubic_psnr": b_psnr, "bicubic_ssim": b_ssim,
                     "srcnn_psnr": s_psnr, "srcnn_ssim": s_ssim})
    return rows


def mean_scores(rows: list[dict]) -> dict:
    keys = ("bicubic_psnr", "bicubic_ssim", "srcnn_psnr", "srcnn_ssim")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
