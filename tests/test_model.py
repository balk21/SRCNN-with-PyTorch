import numpy as np
import pytest
import torch

from srcnn.model import SRCNN, load_matlab_weights, load_weights, save_weights


def random_model(**arch) -> SRCNN:
    torch.manual_seed(0)
    model = SRCNN(**arch)
    for p in model.parameters():  # larger than the N(0, 0.001) start, so the output is not ~0
        torch.nn.init.normal_(p, std=0.1)
    return model.eval()


def test_parameter_count_of_the_basic_network():
    model = SRCNN()
    assert sum(p.numel() for p in model.parameters()) == 8129  # 8 032 weights + 97 biases
    assert model.shrink == 12


def test_valid_and_same_size_outputs_agree_inside():
    model = random_model()
    x = torch.rand(2, 1, 40, 37)
    with torch.no_grad():
        valid, same = model(x), model(x, same_size=True)
    assert valid.shape == (2, 1, 28, 25)
    assert same.shape == x.shape
    # 6 pixels from the border the padding is never reached
    assert torch.allclose(same[..., 6:-6, 6:-6], valid, atol=1e-6)


def test_save_and_load(tmp_path):
    model = random_model(f2=3, n2=16)
    save_weights(model, tmp_path / "w.pth", scale=3)
    loaded, info = load_weights(tmp_path / "w.pth")
    assert info == {"scale": 3}
    assert loaded.arch == model.arch
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_matlab_weights_follow_the_authors_test_code(tmp_path):
    """Compare with a direct translation of SRCNN.m (imfilter 'same', 'replicate')."""
    scipy_io = pytest.importorskip("scipy.io")
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    f1, f2, f3, n1, n2 = 9, 1, 5, 4, 3
    m = {
        "weights_conv1": rng.normal(0, 0.1, (f1 * f1, n1)), "biases_conv1": rng.normal(0, 0.1, (n1, 1)),
        "weights_conv2": rng.normal(0, 0.1, (n1, f2 * f2, n2)), "biases_conv2": rng.normal(0, 0.1, (n2, 1)),
        "weights_conv3": rng.normal(0, 0.1, (n2, f3 * f3)), "biases_conv3": rng.normal(0, 0.1, (1, 1)),
    }
    scipy_io.savemat(tmp_path / "x3.mat", m)
    image = rng.random((20, 23))

    def imfilter(x, w):
        return ndimage.correlate(x, w, mode="nearest")

    conv1 = [np.maximum(imfilter(image, m["weights_conv1"][:, i].reshape(f1, f1, order="F")) + m["biases_conv1"][i, 0], 0)
             for i in range(n1)]
    conv2 = [np.maximum(sum(imfilter(conv1[j], m["weights_conv2"][j, :, i].reshape(f2, f2, order="F")) for j in range(n1))
                        + m["biases_conv2"][i, 0], 0) for i in range(n2)]
    expected = sum(imfilter(conv2[i], m["weights_conv3"][i].reshape(f3, f3, order="F")) for i in range(n2)) + m["biases_conv3"][0, 0]

    model = load_matlab_weights(tmp_path / "x3.mat")
    with torch.no_grad():
        out = model(torch.from_numpy(image).float()[None, None], same_size=True)[0, 0].double().numpy()
    assert np.allclose(out, expected, atol=1e-5)
