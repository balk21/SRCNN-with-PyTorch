import numpy as np
from PIL import Image

from srcnn.color import luma_uint8
from srcnn.data import load_pair, modcrop, read_luma, training_pairs


def test_modcrop():
    assert modcrop(np.zeros((512, 344)), 3).shape == (510, 342)


def test_read_luma(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (10, 12, 3), dtype=np.uint8)
    grey = rng.integers(0, 256, (10, 12), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "colour.png")
    Image.fromarray(grey).save(tmp_path / "grey.png")
    assert np.array_equal(read_luma(tmp_path / "colour.png"), luma_uint8(rgb))
    assert np.array_equal(read_luma(tmp_path / "grey.png"), grey)  # used as Y directly


def test_training_pairs_grid_and_alignment(tmp_path):
    grey = np.random.default_rng(1).integers(0, 256, (62, 77), dtype=np.uint8)  # cropped to 60 x 75
    Image.fromarray(grey).save(tmp_path / "a.png")
    inputs, labels = training_pairs(tmp_path, scale=3, size=33, shrink=12, stride=14)
    # sub-images start at rows 0, 14 (28 + 33 > 60) and columns 0, 14, 28, 42
    assert inputs.shape == (8, 33, 33)
    assert labels.shape == (8, 21, 21)
    hr = grey[:60, :75] / 255.0
    _, bicubic = load_pair(tmp_path / "a.png", 3)
    assert np.allclose(inputs[5], bicubic[14:47, 14:47])  # row 14, column 14
    assert np.allclose(labels[5], hr[20:41, 20:41])  # its centre
