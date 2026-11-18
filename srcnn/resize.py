"""Bicubic resizing that reproduces MATLAB's ``imresize(img, scale, 'bicubic')``.

The paper creates its low-resolution inputs, for training and for testing, with MATLAB's imresize: a
cubic convolution kernel (a = -0.5) that is stretched by 1/scale when the image is shrunk
(antialiasing), with symmetric padding at the borders. Other libraries (PIL, OpenCV, torch) handle
the kernel and the borders differently. That changes the bicubic baseline and every number measured
against it.
"""
from __future__ import annotations

import math

import numpy as np


def cubic(x: np.ndarray) -> np.ndarray:
    """Keys' cubic convolution kernel with a = -0.5."""
    x = np.abs(x)
    x2, x3 = x * x, x * x * x
    near = (1.5 * x3 - 2.5 * x2 + 1.0) * (x <= 1)
    far = (-0.5 * x3 + 2.5 * x2 - 4.0 * x + 2.0) * ((x > 1) & (x <= 2))
    return near + far


def contributions(in_len: int, out_len: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights and 0-based source indices of every output sample along one axis.

    A direct translation of the contributions() routine inside MATLAB's imresize.
    """
    width = 4.0
    if scale < 1:  # shrinking: widen the kernel so that it also low-pass filters
        def kernel(t):
            return scale * cubic(scale * t)
        width /= scale
    else:
        kernel = cubic
    # output sample x (1-based) sits at u in input coordinates (1-based)
    u = np.arange(1, out_len + 1, dtype=np.float64) / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - width / 2.0)
    taps = math.ceil(width) + 2
    idx = left[:, None] + np.arange(taps)[None, :]
    weights = kernel(u[:, None] - idx)
    weights /= weights.sum(axis=1, keepdims=True)
    # symmetric padding: indices outside the image are reflected back into it
    mirror = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    idx = mirror[np.mod(idx - 1, 2 * in_len).astype(np.int64)]
    used = np.any(weights != 0, axis=0)
    return weights[:, used], idx[:, used]


def _resize_axis(img: np.ndarray, axis: int, out_len: int, scale: float) -> np.ndarray:
    weights, idx = contributions(img.shape[axis], out_len, scale)
    src = np.moveaxis(img, axis, 0)
    out = np.zeros((out_len,) + src.shape[1:], dtype=np.float64)
    extra = (1,) * (src.ndim - 1)
    for k in range(idx.shape[1]):
        out += weights[:, k].reshape((out_len,) + extra) * src[idx[:, k]]
    return np.moveaxis(out, 0, axis)


def imresize(img: np.ndarray, scale: float | None = None, output_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Resize an (H, W) or (H, W, C) image like MATLAB's bicubic imresize.

    Give either ``scale`` (output size = ceil(scale * input size), as in MATLAB) or ``output_shape``.
    The image is processed in float64 and nothing is rounded or clipped.
    """
    img = np.asarray(img, dtype=np.float64)
    in_shape = img.shape[:2]
    if output_shape is None:
        if scale is None:
            raise ValueError("give scale or output_shape")
        scales = (float(scale), float(scale))
        # the small tolerance keeps e.g. 510 * (1/3) from rounding up to 171
        output_shape = tuple(math.ceil(n * s - 1e-9) for n, s in zip(in_shape, scales))
    else:
        output_shape = tuple(int(n) for n in output_shape)
        scales = tuple(o / n for o, n in zip(output_shape, in_shape))
    out = img
    # MATLAB resizes the dimension with the smaller scale factor first (rows first on a tie)
    for axis in sorted(range(2), key=lambda a: scales[a]):
        out = _resize_axis(out, axis, output_shape[axis], scales[axis])
    return out
