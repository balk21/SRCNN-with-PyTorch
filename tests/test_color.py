import numpy as np

from srcnn.color import luma_uint8, rgb_to_ycbcr, round_half_away, ycbcr_to_rgb


def test_luma_of_black_white_and_primaries():
    rgb = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    # MATLAB: rgb2ycbcr gives Y = 16, 235, 81, 145, 41
    assert luma_uint8(rgb).tolist() == [[16, 235, 81, 145, 41]]


def test_round_trip():
    rgb = np.random.default_rng(0).uniform(0, 255, size=(4, 5, 3))
    assert np.allclose(ycbcr_to_rgb(rgb_to_ycbcr(rgb)), rgb)


def test_halves_are_rounded_away_from_zero():
    assert round_half_away(np.array([0.5, 1.5, 2.5, -0.5])).tolist() == [1, 2, 3, -1]
