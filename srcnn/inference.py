"""Applying a trained network to an image."""
from __future__ import annotations

import numpy as np
import torch

from .color import rgb_to_ycbcr, ycbcr_to_rgb
from .metrics import to_uint8
from .model import SRCNN
from .resize import imresize


def super_resolve(model: SRCNN, y: np.ndarray, device) -> np.ndarray:
    """Network output for a bicubic-upscaled Y channel (float, [0, 1]); same size as the input."""
    x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        out = model(x, same_size=True)
    return out[0, 0].double().cpu().numpy()


def upscale_image(model: SRCNN, img: np.ndarray, scale: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Enlarge a uint8 image, (H, W) grey or (H, W, 3) RGB, by ``scale``.

    Returns (bicubic result, SRCNN result) as uint8 images. The network enlarges the luminance; the
    colour channels Cb and Cr are enlarged by bicubic interpolation only, as in the paper.
    """
    if img.ndim == 2:
        bicubic = imresize(img.astype(np.float64) / 255.0, scale)
        return to_uint8(bicubic), to_uint8(super_resolve(model, bicubic, device))
    ycbcr = imresize(rgb_to_ycbcr(img[..., :3]) / 255.0, scale)
    bicubic_rgb = ycbcr_to_rgb(ycbcr * 255.0) / 255.0
    ycbcr[..., 0] = super_resolve(model, ycbcr[..., 0], device)
    srcnn_rgb = ycbcr_to_rgb(ycbcr * 255.0) / 255.0
    return to_uint8(bicubic_rgb), to_uint8(srcnn_rgb)
