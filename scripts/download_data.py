"""Download the training and test images into data/.

- T91 (training), Set5 and Set14: the image files of the authors' training package (SRCNN_train.zip
  on the paper's project page).
- BSD200: the 200 training images of the Berkeley Segmentation Dataset (BSDS300), the paper's third
  test set.
- The authors' 9-1-5 network trained on the 91 images (SRCNN_v1.zip), which evaluate.py --paper-weights
  uses to check the evaluation code against the published numbers.

The archives are kept in data/downloads/, so a second run only extracts.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "https://mmlab.ie.cuhk.edu.hk/projects/SRCNN"
TRAIN_ZIP = f"{PROJECT}/SRCNN_train.zip"
TEST_ZIP = f"{PROJECT}/SRCNN_v1.zip"
BSDS300 = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"


def fetch(url: str, folder: Path) -> Path:
    target = folder / url.rsplit("/", 1)[1]
    if not target.exists():
        print(f"downloading {url}")
        folder.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
    return target


def copy_member(src, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        shutil.copyfileobj(src, out)


def extract_zip(archive: Path, folders: dict[str, Path]) -> None:
    """Copy the files directly inside each archive folder (key) to a local folder (value)."""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parent, _, name = info.filename.rpartition("/")
            if parent + "/" in folders and name and not info.is_dir():
                with z.open(info) as src:  # only the base name is used, so nothing lands outside
                    copy_member(src, folders[parent + "/"] / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data", help="target folder (default: data/)")
    args = parser.parse_args()
    out = args.out
    downloads = out / "downloads"

    extract_zip(fetch(TRAIN_ZIP, downloads), {
        "SRCNN/Train/": out / "T91",
        "SRCNN/Test/Set5/": out / "Set5",
        "SRCNN/Test/Set14/": out / "Set14",
    })
    extract_zip(fetch(TEST_ZIP, downloads), {"SRCNN/model/9-1-5(91 images)/": out / "paper_weights"})
    with tarfile.open(fetch(BSDS300, downloads)) as tar:
        for member in tar.getmembers():
            parent, _, name = member.name.rpartition("/")
            if parent == "BSDS300/images/train" and member.isfile() and name.endswith(".jpg"):
                copy_member(tar.extractfile(member), out / "BSD200" / name)

    for name in ("T91", "Set5", "Set14", "BSD200", "paper_weights"):
        print(f"{name}: {len(list((out / name).glob('*')))} files")


if __name__ == "__main__":
    main()
