# SRCNN PyTorch Implementation

This repository is a PyTorch implementation of the Super-Resolution Convolutional Neural Network (SRCNN) based on the paper:
> **Image Super-Resolution Using Deep Convolutional Networks**
> Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. (2015)

## Features
- **9-1-5 Architecture:** Faithful replication of the original paper structure.
- **Valid Padding:** No padding in convolution layers to avoid border artifacts.
- **Y-Channel Training:** Training performed on the Y channel (YCbCr) as per the paper.
- **Dynamic Crop:** Automated handling of input/output size mismatch.

## Requirements
- Python 3.9
- PyTorch 2.5.1 (CUDA 12.1)
- See `environment.yaml` for full list.

## Usage
1. **Install Environment:** `conda env create -f environment.yaml`
2. **Prepare Data:** `python src/prepare.py`
3. **Train:** `python src/train.py`
4. **Test:** `python src/test.py`
5. **Inference:** Place images in `inputs/` and run `python src/inference.py`

## License
MIT License