"""RGB <-> YCbCr conversion as in MATLAB's rgb2ycbcr / ycbcr2rgb (ITU-R BT.601, Y in [16, 235]).

The network works on the Y (luminance) channel only. Cb and Cr are upscaled by bicubic interpolation.
"""
from __future__ import annotations

import numpy as np

# rows: Y, Cb, Cr; columns: R, G, B (inputs and outputs in [0, 255])
_M = np.array([
    [65.481, 128.553, 24.966],
    [-37.797, -74.203, 112.0],
    [112.0, -93.786, -18.214],
]) / 255.0
_OFFSET = np.array([16.0, 128.0, 128.0])
_M_INV = np.linalg.inv(_M)


def round_half_away(x: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, halves away from zero (MATLAB's round)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) RGB in [0, 255] -> YCbCr in [0, 255], not rounded."""
    return np.asarray(rgb, dtype=np.float64) @ _M.T + _OFFSET


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_ycbcr`; not rounded or clipped."""
    return (np.asarray(ycbcr, dtype=np.float64) - _OFFSET) @ _M_INV.T


def luma_uint8(rgb: np.ndarray) -> np.ndarray:
    """Y channel of a uint8 RGB image, rounded to uint8 as MATLAB's rgb2ycbcr does for uint8 input."""
    y = rgb_to_ycbcr(rgb)[..., 0]
    return np.clip(round_half_away(y), 0, 255).astype(np.uint8)
