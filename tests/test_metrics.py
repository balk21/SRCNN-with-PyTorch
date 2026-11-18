import numpy as np

from srcnn.metrics import psnr, shave, ssim, to_uint8


def test_psnr():
    a, b = np.zeros((8, 8)), np.ones((8, 8))  # MSE 1
    assert abs(psnr(a, b) - 20 * np.log10(255)) < 1e-12
    assert psnr(a, a) == float("inf")


def test_to_uint8_rounds_and_clips():
    x = np.array([-0.1, 0.6, 1.4, 254.6, 306.0]) / 255
    assert to_uint8(x).tolist() == [0, 1, 1, 255, 255]


def test_shave():
    assert shave(np.zeros((10, 12)), 3).shape == (4, 6)
    assert shave(np.zeros((10, 12)), 0).shape == (10, 12)


def test_ssim_is_one_for_identical_images_and_symmetric():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (40, 40)).astype(np.float64)
    b = np.clip(a + rng.normal(0, 10, a.shape), 0, 255)
    assert abs(ssim(a, a) - 1) < 1e-12
    assert 0 < ssim(a, b) < 1
    assert abs(ssim(a, b) - ssim(b, a)) < 1e-12


def test_ssim_of_a_single_window():
    # an 11 x 11 image has exactly one window: SSIM is the formula with Gaussian-weighted moments
    rng = np.random.default_rng(1)
    a, b = rng.random((11, 11)) * 255, rng.random((11, 11)) * 255
    r = np.arange(11) - 5
    g = np.exp(-r * r / (2 * 1.5 ** 2))
    w = np.outer(g, g) / np.outer(g, g).sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    va, vb = (w * a * a).sum() - ma ** 2, (w * b * b).sum() - mb ** 2
    cab = (w * a * b).sum() - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    expected = (2 * ma * mb + c1) * (2 * cab + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    assert abs(ssim(a, b) - expected) < 1e-12
