"""Reading images, the bicubic degradation and the training sub-images, as in the paper's code.

Ground truth: the Y channel of an image, cropped so that its size is a multiple of the scale.
Network input: the ground truth shrunk by the scale and enlarged back, both with bicubic imresize.
Training samples: 33 x 33 input sub-images taken with a stride of 14, each paired with the centre of
the ground truth that the unpadded network can predict (21 x 21 for the 9-1-5 network).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .color import luma_uint8
from .resize import imresize

IMAGE_SUFFIXES = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; see 'Data' in the README")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_luma(path: str | Path) -> np.ndarray:
    """Y channel (uint8) of an image file. A single-channel image is taken as Y directly."""
    with Image.open(path) as im:
        if im.mode == "L":
            return np.array(im)
        return luma_uint8(np.array(im.convert("RGB")))


def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    """Crop the bottom and right edges so that height and width are multiples of ``scale``."""
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]


def degrade(hr: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Bicubic down-sampling and the bicubic up-sampling of the result.

    ``hr`` is a float image whose size is a multiple of ``scale``. Returns (low-resolution image,
    bicubic estimate at the original size).
    """
    lr = imresize(hr, 1.0 / scale)
    return lr, imresize(lr, output_shape=hr.shape[:2])


def load_pair(path: str | Path, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """(ground-truth Y as uint8, bicubic estimate of Y as float in [0, 1]) for one test image."""
    gt = modcrop(read_luma(path), scale)
    _, bicubic = degrade(gt.astype(np.float64) / 255.0, scale)
    return gt, bicubic


def training_pairs(folder: str | Path, scale: int, size: int = 33, shrink: int = 12,
                   stride: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """All training sub-images of the images in ``folder``.

    Returns inputs of shape (N, size, size) and labels of shape (N, size - shrink, size - shrink),
    float32 in [0, 1]. ``shrink`` is the number of pixels the unpadded network loses (12 for 9-1-5).
    """
    border = shrink // 2
    label = size - shrink
    inputs, labels = [], []
    for path in list_images(folder):
        hr = modcrop(read_luma(path), scale).astype(np.float64) / 255.0
        _, bicubic = degrade(hr, scale)
        h, w = hr.shape
        for r in range(0, h - size + 1, stride):
            for c in range(0, w - size + 1, stride):
                inputs.append(bicubic[r:r + size, c:c + size])
                labels.append(hr[r + border:r + border + label, c + border:c + border + label])
    if not inputs:
        raise ValueError(f"no {size} x {size} sub-images in {folder}")
    return np.stack(inputs).astype(np.float32), np.stack(labels).astype(np.float32)
