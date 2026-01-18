# SRCNN PyTorch Implementation

This repository is a PyTorch implementation of the Super-Resolution Convolutional Neural Network (SRCNN) based on the paper:
> **Image Super-Resolution Using Deep Convolutional Networks**
> Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. (2015)
> 

## Features
- **9-1-5 Architecture:** Faithful replication of the original paper structure.
- **Valid Padding:** No padding in convolution layers to avoid border artifacts.
- **Y-Channel Training:** Training performed on the Y channel (YCbCr) as per the paper.
- **Dynamic Crop:** Automated handling of input/output size mismatch.
- **Optimizer:** Adam Optimizer is used instead of SDG.

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

## Folder Structure & Data Preparation

To keep the repository light, large files (`data/` and `experiments/`) are not included in the git history. You can download the pre-processed datasets and trained models from the link below:

🔗 **[Download Datasets & Pre-trained Models (Google Drive Link)](https://drive.google.com/drive/folders/1IkD847CYN7umXRaDYyQFw--5A3iAfq37?usp=sharing)**

After downloading, extract the contents to match the following structure:

```text
SRCNN-PyTorch/
│
├── data/
│   ├── raw/                # Original datasets for train
│   ├── test/               # Test datasets for evaluation
│   └── processed/          # Generated .h5 files (train.h5, test.h5)
│
├── experiments/
│   └── checkpoints/        # Saved model weights (best_model.pth)
│
├── inputs/                 # Put your images here for inference
├── outputs/                # Results will be saved here
├── src/                    # Source code (model, train, test, utils...)
.
.
.
```

## Quick Inference

You can test the model immediately using the pre-trained weights without training from scratch.

1.  **Download Model:** Get the `best_model.pth` file from the [Google Drive Link](https://drive.google.com/file/d/1iiqt87zxvgIN03IGY8TvFXO2xyjhGAlF/view?usp=drive_link) and place it into `experiments/checkpoints/` directory.
2.  **Add Images:** Place your test images (any format) into the `inputs/` folder.
3.  **Run Script:**
    ```bash
    python src/inference.py
    ```
4.  **View Results:** Check the `outputs/` folder. You will see both the `_bicubic` (baseline) and `_srcnn` (enhanced) versions of your images.

> **Note regarding Upscaling Factors:**
> The pre-trained model provided in the Drive link is trained for **Scale x3**.
>
> This implementation supports any scale factor. If you want to use **x2** or **x4**:
> 1. Open `config.yaml` and change `scale: 3` to your desired value.
> 2. Run `python src/prepare.py` to generate new dataset patches.
> 3. Run `python src/train.py` to train a new model for that scale.