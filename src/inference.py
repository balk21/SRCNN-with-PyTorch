import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model import SRCNN
from src.utils import convert_rgb_to_y, convert_rgb_to_ycbcr

MODEL_PATH = "experiments/checkpoints/best_model.pth"
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
SCALE = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference():
    input_path = ROOT / INPUT_DIR
    output_path = ROOT / OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Couldn't find directory: '{INPUT_DIR}'")
        input_path.mkdir()
        return

    print(f"=== SRCNN Inference (Scale: x{SCALE}) ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    model = SRCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(ROOT / MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        try:
             model.load_state_dict(torch.load(ROOT / MODEL_PATH, map_location=DEVICE))
        except FileNotFoundError:
             print("Couldn't find pretrained model.")
             return
    model.eval()

    extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))

    if not image_files:
        print("Couldn't find image to process.")
        return

    for img_file in image_files:
        filename = img_file.stem
        print(f"Processing: {filename}")

        img = Image.open(img_file).convert('RGB')

        # Bicubic Upscale
        target_w = img.width * SCALE
        target_h = img.height * SCALE
        
        img_bicubic = img.resize((target_w, target_h), resample=Image.BICUBIC)

        img_bicubic.save(output_path / f"{filename}_bicubic.png")

        # Enhance the channel Y.       
        img_ycbcr = img_bicubic.convert('YCbCr')
        y, cb, cr = img_ycbcr.split()
        
        y_np = np.array(y).astype(np.float32) / 255.0
        y_tensor = torch.from_numpy(y_np).to(DEVICE).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            pred_y_tensor = model(y_tensor)
        
        pred_y_np = pred_y_tensor.cpu().squeeze().numpy()
        pred_y_np = np.clip(pred_y_np, 0.0, 1.0) * 255.0
        pred_y_pil = Image.fromarray(pred_y_np.astype(np.uint8), mode='L')

        # Fix the scale by cropping channels Cb & Cr.        
        out_w, out_h = pred_y_pil.size
        
        diff_w = img_bicubic.width - out_w
        diff_h = img_bicubic.height - out_h
        
        left = diff_w // 2
        top = diff_h // 2
        right = left + out_w
        bottom = top + out_h
        
        cb_cropped = cb.crop((left, top, right, bottom))
        cr_cropped = cr.crop((left, top, right, bottom))
        
        img_srcnn = Image.merge('YCbCr', (pred_y_pil, cb_cropped, cr_cropped)).convert('RGB')
        img_srcnn.save(output_path / f"{filename}_srcnn.png")

    print("\nDone! Check out the folder '/outputs'.")

if __name__ == '__main__':
    inference()