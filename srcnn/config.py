"""Reading config.yaml, which every script uses."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, scale: int | None = None) -> dict:
    """The configuration, with ``scale`` overridden if given and {scale} filled in the output paths."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["scale"] = int(scale if scale is not None else cfg["scale"])
    for key in ("weights", "results"):
        cfg[key] = cfg[key].format(scale=cfg["scale"])
    return cfg


def shown(path: str | Path, root: str | Path) -> str:
    """``path`` relative to ``root`` if it lies inside it, for messages."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def pick_device(name: str):
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, using the CPU")
        name = "cpu"
    return torch.device(name)
