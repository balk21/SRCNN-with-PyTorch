"""The three-layer super-resolution network."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SRCNN(nn.Module):
    """Patch extraction (f1 x f1, n1 filters), non-linear mapping (f2 x f2, n2 filters) and
    reconstruction (f3 x f3), with ReLU after the first two layers.

    The default is the paper's basic 9-1-5 network with n1 = 64 and n2 = 32 (8 129 parameters).
    The convolutions have no padding, as in the paper's training, so the output is ``shrink`` =
    f1 + f2 + f3 - 3 pixels smaller than the input. With ``same_size=True`` every layer first pads its
    input by repeating the border pixels, as the paper's test code does
    (``imfilter(..., 'same', 'replicate')``), and the output has the size of the input.
    """

    def __init__(self, f1: int = 9, f2: int = 1, f3: int = 5, n1: int = 64, n2: int = 32,
                 channels: int = 1):
        super().__init__()
        if min(f1, f2, f3) < 1 or not (f1 % 2 and f2 % 2 and f3 % 2):
            raise ValueError("filter sizes must be odd")
        self.extract = nn.Conv2d(channels, n1, f1)
        self.map = nn.Conv2d(n1, n2, f2)
        self.reconstruct = nn.Conv2d(n2, channels, f3)
        self.shrink = f1 + f2 + f3 - 3
        self.arch = dict(f1=f1, f2=f2, f3=f3, n1=n1, n2=n2, channels=channels)
        # initialisation of the paper: N(0, 0.001) weights, zero biases
        for conv in self.layers():
            nn.init.normal_(conv.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(conv.bias)

    def layers(self) -> tuple[nn.Conv2d, nn.Conv2d, nn.Conv2d]:
        return self.extract, self.map, self.reconstruct

    @staticmethod
    def _conv(conv: nn.Conv2d, x: torch.Tensor, same_size: bool) -> torch.Tensor:
        pad = conv.kernel_size[0] // 2
        if same_size and pad:
            x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return conv(x)

    def forward(self, x: torch.Tensor, same_size: bool = False) -> torch.Tensor:
        x = F.relu(self._conv(self.extract, x, same_size))
        x = F.relu(self._conv(self.map, x, same_size))
        return self._conv(self.reconstruct, x, same_size)


def save_weights(model: SRCNN, path, **info) -> None:
    torch.save({"arch": model.arch, "state_dict": model.state_dict(), **info}, path)


def load_matlab_weights(path, device="cpu") -> SRCNN:
    """The authors' released MATLAB weights (e.g. model/9-1-5(91 images)/x3.mat of SRCNN_v1.zip).

    The .mat file stores each filter as a column of length f*f in MATLAB's column-major order;
    the test code applies it with imfilter, a correlation, as torch's conv2d does.
    """
    from scipy.io import loadmat

    m = loadmat(path)
    w1, w3 = m["weights_conv1"], m["weights_conv3"]
    w2 = m["weights_conv2"].reshape(m["weights_conv2"].shape[0], -1, m["weights_conv2"].shape[-1])
    f1, n1 = int(round(w1.shape[0] ** 0.5)), w1.shape[1]
    f2, n2 = int(round(w2.shape[1] ** 0.5)), w2.shape[2]
    f3 = int(round(w3.shape[1] ** 0.5))

    def filters(columns, f):  # (f*f, n) -> (n, f, f)
        return torch.from_numpy(columns.T.reshape(-1, f, f).transpose(0, 2, 1).copy())

    model = SRCNN(f1, f2, f3, n1, n2)
    with torch.no_grad():
        model.extract.weight.copy_(filters(w1, f1)[:, None])
        model.map.weight.copy_(torch.stack([filters(w2[:, :, i].T, f2) for i in range(n2)]))
        model.reconstruct.weight.copy_(filters(w3.T, f3)[None])
        for conv, bias in zip(model.layers(), ("biases_conv1", "biases_conv2", "biases_conv3")):
            conv.bias.copy_(torch.from_numpy(m[bias].ravel()))
    return model.to(device).eval()


def load_weights(path, device="cpu") -> tuple[SRCNN, dict]:
    """Model and the extra information saved with it (scale, iterations, ...)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SRCNN(**checkpoint["arch"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    info = {k: v for k, v in checkpoint.items() if k not in ("arch", "state_dict")}
    return model, info
