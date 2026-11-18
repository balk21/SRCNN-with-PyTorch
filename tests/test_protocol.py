"""The evaluation code against the published numbers. Runs only after scripts/download_data.py."""
from pathlib import Path

import pytest

from srcnn.evaluate import evaluate, mean_scores, prepare
from srcnn.model import load_matlab_weights

DATA = Path(__file__).resolve().parents[1] / "data"
WEIGHTS = DATA / "paper_weights" / "x3.mat"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="data not downloaded (scripts/download_data.py)")
@pytest.mark.parametrize("name, bicubic, bicubic_ssim, srcnn", [
    ("Set5", 30.39, 0.8682, 32.39),
    ("Set14", 27.54, 0.7736, 29.00),
])
def test_published_numbers_are_reproduced(name, bicubic, bicubic_ssim, srcnn):
    """Bicubic and the authors' 9-1-5 weights, x3 (ECCV 2014 Tables 1-2, TPAMI 2016 Tables 2-3)."""
    model = load_matlab_weights(WEIGHTS)
    scores = mean_scores(evaluate(model, prepare(DATA / name, 3), 3, "cpu"))
    assert abs(scores["bicubic_psnr"] - bicubic) < 0.006
    assert abs(scores["bicubic_ssim"] - bicubic_ssim) < 0.00006
    assert abs(scores["srcnn_psnr"] - srcnn) < 0.006
