"""Train SRCNN on the 91-image set.

The sub-images, initialisation, learning rates (10^-4, and 10^-5 for the last layer) and batch size are
the paper's. The optimiser is Adam instead of SGD with momentum, and training is much shorter than the
paper's 8 x 10^8 backpropagations. The weights after the last iteration are saved. PSNR on the monitor
set (Set5) is only logged to follow the training; it is not used to choose a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srcnn.config import load_config, pick_device, shown  # noqa: E402
from srcnn.data import training_pairs  # noqa: E402
from srcnn.evaluate import evaluate, mean_scores, prepare  # noqa: E402
from srcnn.model import SRCNN, save_weights  # noqa: E402


def sub_images(cfg: dict, shrink: int) -> tuple[np.ndarray, np.ndarray]:
    """Training inputs and labels, cached in data/cache after the first run."""
    t = cfg["train"]
    name = f"{Path(cfg['data']['train']).name}_x{cfg['scale']}_{t['sub_image']}px_stride{t['stride']}_shrink{shrink}.npz"
    cache = ROOT / cfg["data"]["cache"] / name
    if cache.exists():
        with np.load(cache) as f:
            return f["inputs"], f["labels"]
    print("building the training sub-images ...")
    inputs, labels = training_pairs(ROOT / cfg["data"]["train"], cfg["scale"], t["sub_image"], shrink, t["stride"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, inputs=inputs, labels=labels)
    return inputs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--scale", type=int, help="override the scale in the config")
    parser.add_argument("--iterations", type=int, help="override train.iterations")
    parser.add_argument("--device", help="override train.device (cuda or cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.scale)
    t, scale = cfg["train"], cfg["scale"]
    iterations = args.iterations or int(t["iterations"])
    device = pick_device(args.device or t["device"])
    torch.manual_seed(int(t["seed"]))

    model = SRCNN(**cfg["model"]).to(device)
    inputs, labels = sub_images(cfg, model.shrink)
    x_all = torch.from_numpy(inputs)[:, None].to(device)
    y_all = torch.from_numpy(labels)[:, None].to(device)
    n, batch = len(x_all), int(t["batch_size"])
    batches_per_epoch = n // batch  # the last incomplete batch of every epoch is dropped
    print(f"{n} sub-images, {sum(p.numel() for p in model.parameters())} parameters, "
          f"{iterations} iterations of {batch} on {device}")

    optimizer = torch.optim.Adam([
        {"params": [*model.extract.parameters(), *model.map.parameters()], "lr": float(t["lr"])},
        {"params": model.reconstruct.parameters(), "lr": float(t["lr_last"])},
    ])
    monitor_name = t["monitor"]
    monitor = prepare(ROOT / cfg["data"]["test"][monitor_name], scale)
    results = ROOT / cfg["results"]
    results.mkdir(parents=True, exist_ok=True)
    weights = ROOT / cfg["weights"]
    weights.parent.mkdir(parents=True, exist_ok=True)

    log_every = int(t["log_every"])
    start = time.time()
    loss_sum, loss_count = torch.zeros((), device=device), 0
    with open(results / "training_log.csv", "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["iteration", "train_mse", f"{monitor_name}_psnr", "seconds"])

        def checkpoint(it: int) -> None:
            nonlocal loss_sum, loss_count
            mse = loss_sum.item() / loss_count if loss_count else float("nan")
            psnr = mean_scores(evaluate(model, monitor, scale, device))["srcnn_psnr"]
            seconds = time.time() - start
            log.writerow([it, f"{mse:.7f}", f"{psnr:.4f}", f"{seconds:.0f}"])
            fh.flush()
            print(f"iteration {it:>8}  train MSE {mse:.6f}  {monitor_name} PSNR {psnr:.2f} dB  ({seconds / 60:.1f} min)")
            loss_sum, loss_count = torch.zeros((), device=device), 0

        checkpoint(0)
        model.train()
        order = None
        for it in range(1, iterations + 1):
            k = (it - 1) % batches_per_epoch
            if k == 0:
                order = torch.randperm(n, device=device)
            idx = order[k * batch:(k + 1) * batch]
            loss = F.mse_loss(model(x_all[idx]), y_all[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            loss_count += 1
            if it % log_every == 0 or it == iterations:
                checkpoint(it)

    save_weights(model, weights, scale=scale, iterations=iterations, sub_images=n,
                 optimizer="Adam", seconds=round(time.time() - start))
    print(f"saved {shown(weights, ROOT)}")


if __name__ == "__main__":
    main()
