"""Evaluate trained weights on the test sets with the paper's protocol.

Each image's Y channel is cropped to a multiple of the scale, shrunk and enlarged again with MATLAB's
bicubic kernel, and passed through the network with replicate padding. PSNR and SSIM are computed on
8-bit values after removing `scale` pixels at every border. The authors' released 9-1-5 weights
(fetched by download_data.py) are evaluated alongside, to show that this code reproduces the
published numbers.

Writes results/x<scale>/metrics.md, per_image.csv and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srcnn.config import load_config, pick_device, shown  # noqa: E402
from srcnn.evaluate import PAPER_X3, evaluate, mean_scores, prepare  # noqa: E402
from srcnn.model import load_matlab_weights, load_weights  # noqa: E402


def cell(psnr: float, ssim: float | None = None) -> str:
    return f"{psnr:.2f}" if ssim is None else f"{psnr:.2f} / {ssim:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--scale", type=int, help="override the scale in the config")
    parser.add_argument("--weights", type=Path, help="default: the path in the config")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config, args.scale)
    scale = cfg["scale"]
    device = pick_device(args.device)
    model, info = load_weights(args.weights or ROOT / cfg["weights"], device)
    paper_file = ROOT / "data" / "paper_weights" / f"x{scale}.mat"
    paper_model = load_matlab_weights(paper_file, device) if paper_file.exists() else None
    published = PAPER_X3 if scale == 3 else {}

    out = ROOT / cfg["results"]
    out.mkdir(parents=True, exist_ok=True)
    summary = {"scale": scale, "weights": info, "sets": {}}
    per_image = []
    for name, folder in cfg["data"]["test"].items():
        if not (ROOT / folder).is_dir():
            print(f"{name}: {folder} not found, skipped")
            continue
        prepared = prepare(ROOT / folder, scale)
        rows = evaluate(model, prepared, scale, device)
        entry = {"images": len(rows), **mean_scores(rows), "published": published.get(name)}
        if paper_model is not None:
            paper_rows = evaluate(paper_model, prepared, scale, device)
            entry["authors_weights_psnr"] = mean_scores(paper_rows)["srcnn_psnr"]
            entry["authors_weights_ssim"] = mean_scores(paper_rows)["srcnn_ssim"]
            for row, paper_row in zip(rows, paper_rows):
                row["authors_weights_psnr"] = paper_row["srcnn_psnr"]
                row["authors_weights_ssim"] = paper_row["srcnn_ssim"]
        summary["sets"][name] = entry
        per_image += [{"set": name, **row} for row in rows]
        print(f"{name}: bicubic {cell(entry['bicubic_psnr'], entry['bicubic_ssim'])}"
              + (f"   authors' weights {cell(entry['authors_weights_psnr'], entry['authors_weights_ssim'])}"
                 if paper_model is not None else "")
              + f"   this repository {cell(entry['srcnn_psnr'], entry['srcnn_ssim'])}")

    with open(out / "per_image.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_image[0]))
        writer.writeheader()
        for row in per_image:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"PSNR (dB) / SSIM on the Y channel, x{scale}, all measured with this code.",
        "",
        "| test set | images | bicubic | authors' 9-1-5 weights | this repository | published: bicubic, 9-1-5 |",
        "|---|---|---|---|---|---|",
    ]
    for name, e in summary["sets"].items():
        authors = cell(e["authors_weights_psnr"], e["authors_weights_ssim"]) if "authors_weights_psnr" in e else "-"
        pub = e["published"]
        pub_text = "-" if not pub else ", ".join(cell(*pub[k]) if pub.get(k) else "-" for k in ("bicubic", "srcnn"))
        lines.append(f"| {name} | {e['images']} | {cell(e['bicubic_psnr'], e['bicubic_ssim'])} | {authors} "
                     f"| {cell(e['srcnn_psnr'], e['srcnn_ssim'])} | {pub_text} |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {shown(out / 'metrics.md', ROOT)}")


if __name__ == "__main__":
    main()
