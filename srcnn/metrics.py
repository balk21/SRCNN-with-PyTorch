"""PSNR and SSIM on the Y channel, as in the paper's evaluation.

Both outputs are converted to 8-bit (``uint8(x * 255)`` in MATLAB) and ``scale`` pixels are removed
from every border before the metrics are computed.
"""
from __future__ import annotations

import numpy as np


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Float image in [0, 1] -> uint8, rounded halves up and clipped, like MATLAB's uint8(x * 255)."""
    return np.clip(np.floor(np.asarray(x, dtype=np.float64) * 255.0 + 0.5), 0, 255).astype(np.uint8)


def shave(img: np.ndarray, border: int) -> np.ndarray:
    return img[border:-border, border:-border] if border > 0 else img


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mse = float(np.mean(diff * diff))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)


def _gaussian(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    r = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-r * r / (2.0 * sigma * sigma))
    return g / g.sum()


def _filter_valid(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Filter with the separable window outer(g, g), keeping only fully covered positions."""
    k = len(g)
    rows = sum(g[i] * img[i:img.shape[0] - k + 1 + i] for i in range(k))
    return sum(g[i] * rows[:, i:rows.shape[1] - k + 1 + i] for i in range(k))


def ssim(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """Mean SSIM with the 11 x 11 Gaussian window (sigma 1.5) of Wang et al. (2004)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = _gaussian()
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mu_a, mu_b = _filter_valid(a, g), _filter_valid(b, g)
    var_a = _filter_valid(a * a, g) - mu_a * mu_a
    var_b = _filter_valid(b * b, g) - mu_b * mu_b
    cov = _filter_valid(a * b, g) - mu_a * mu_b
    index = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))
    return float(index.mean())
