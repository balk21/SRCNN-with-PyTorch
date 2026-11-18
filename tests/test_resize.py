import numpy as np

from srcnn.resize import cubic, imresize


def test_cubic_kernel_is_interpolating():
    assert cubic(np.array([0.0]))[0] == 1.0
    assert np.allclose(cubic(np.array([-2.0, -1.0, 1.0, 2.0, 2.5])), 0.0)


def test_upscaling_keeps_the_original_samples():
    img = np.random.default_rng(0).random((17, 23))
    up = imresize(img, 3)
    assert up.shape == (51, 69)
    # output pixel 3i + 1 lies exactly on input pixel i, where the kernel is 1 and 0 at the neighbours
    assert np.allclose(up[1::3, 1::3], img)


def test_constant_image_stays_constant():
    img = np.full((30, 21), 0.37)
    assert np.allclose(imresize(img, 1 / 3), 0.37)
    assert np.allclose(imresize(img, 3), 0.37)


def test_downscaling_keeps_a_ramp_away_from_the_borders():
    ramp = np.tile(np.arange(60, dtype=np.float64), (9, 1))  # value = column index
    down = imresize(ramp, 1 / 3)
    assert down.shape == (3, 20)
    # output column i is centred on input column 3i + 1, and the widened kernel is symmetric there
    i = np.arange(2, 18)
    assert np.allclose(down[1, i], 3 * i + 1)


def test_output_size():
    lr = imresize(np.zeros((510, 342)), 1 / 3)
    assert lr.shape == (170, 114)
    assert imresize(lr, output_shape=(510, 342)).shape == (510, 342)


def test_colour_channels_are_resized_independently():
    img = np.random.default_rng(1).random((12, 15, 3))
    out = imresize(img, 1 / 3)
    for c in range(3):
        assert np.allclose(out[..., c], imresize(img[..., c], 1 / 3))
