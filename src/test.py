import torch
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model import SRCNN
from src.utils import convert_rgb_to_y, calc_psnr

import warnings
warnings.filterwarnings("ignore")

# --- AYARLAR ---
MODEL_PATH = "experiments/checkpoints/best_model.pth"
DATASETS = {
    'Set5': 'data/test/Set5',
    'Set14': 'data/test/Set14',
    'BSD200': 'data/test/BSD200'
}
SCALE = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def test():
    print(f"Test (Scale: x{SCALE})")
    print(f"Device: {DEVICE}")
    
    model = SRCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        try:
             model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except FileNotFoundError:
             print("Couldn't find pretrained model.")
             return
    model.eval()

    for dataset_name, dataset_path in DATASETS.items():
        data_dir = Path(dataset_path)
        if not data_dir.exists():
            continue
            
        image_files = sorted(list(data_dir.glob('*.png')) + list(data_dir.glob('*.bmp')) + list(data_dir.glob('*.jpg')))
        if not image_files:
            continue

        avg_bicubic_psnr = 0.0
        avg_srcnn_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_srcnn_ssim = 0.0
        count = 0
        
        print(f"\n====================== {dataset_name} ({len(image_files)} images) ======================")
        print(f"{'Image':<20} | {'Bicubic PSNR':<12} | {'SRCNN PSNR':<12} | {'Difference (dB)':<10}")
        print("-" * 65)

        for img_path in image_files:
            img_gt_pil = Image.open(img_path).convert('RGB')
            w, h = img_gt_pil.size
            
            w_new = (w // SCALE) * SCALE
            h_new = (h // SCALE) * SCALE
            img_gt_pil = img_gt_pil.resize((w_new, h_new), resample=Image.BICUBIC)
            
            img_lr_small = img_gt_pil.resize((w_new // SCALE, h_new // SCALE), resample=Image.BICUBIC)
            
            # Bicubic Upscale
            img_bicubic_pil = img_lr_small.resize((w_new, h_new), resample=Image.BICUBIC)
            
            # Convert to channel Y
            img_gt_np = np.array(img_gt_pil).astype(np.float32)
            img_bicubic_np = np.array(img_bicubic_pil).astype(np.float32)
            
            img_gt_y = convert_rgb_to_y(img_gt_np)
            img_bicubic_y = convert_rgb_to_y(img_bicubic_np)
            
            # Normalization
            img_gt_y /= 255.0
            img_bicubic_y /= 255.0
            
            # SRCNN Prediction
            img_lr_tensor = torch.from_numpy(img_bicubic_y).to(DEVICE).unsqueeze(0).unsqueeze(0).float()
            
            with torch.no_grad():
                pred_tensor = model(img_lr_tensor)
            
            pred_srcnn_y = pred_tensor.cpu().squeeze().numpy()
            pred_srcnn_y = np.clip(pred_srcnn_y, 0.0, 1.0)
            
            # Cropping for adjusting shapes 
            out_h, out_w = pred_srcnn_y.shape
            gt_h, gt_w = img_gt_y.shape
            
            crop_h = (gt_h - out_h) // 2
            crop_w = (gt_w - out_w) // 2
            
            gt_cropped = img_gt_y[crop_h:crop_h+out_h, crop_w:crop_w+out_w]
            bicubic_cropped = img_bicubic_y[crop_h:crop_h+out_h, crop_w:crop_w+out_w]
    
            # Bicubic Scores
            psnr_b = calc_psnr(torch.tensor(gt_cropped), torch.tensor(bicubic_cropped)).item()
            ssim_b = ssim_func(gt_cropped, bicubic_cropped, data_range=1.0)
            
            # SRCNN Scores
            psnr_s = calc_psnr(torch.tensor(gt_cropped), torch.tensor(pred_srcnn_y)).item()
            ssim_s = ssim_func(gt_cropped, pred_srcnn_y, data_range=1.0)
            
            avg_bicubic_psnr += psnr_b
            avg_srcnn_psnr += psnr_s
            avg_bicubic_ssim += ssim_b
            avg_srcnn_ssim += ssim_s
            count += 1
            
            diff = psnr_s - psnr_b
            print(f"{img_path.name[:20]:<20} | {psnr_b:<12.2f} | {psnr_s:<12.2f} | {diff:+.2f} dB")

        final_bicubic_psnr = avg_bicubic_psnr / count
        final_srcnn_psnr = avg_srcnn_psnr / count
        final_bicubic_ssim = avg_bicubic_ssim / count
        final_srcnn_ssim = avg_srcnn_ssim / count
        
        print("-" * 65)
        print(f"Average [{dataset_name}]:")
        print(f"Bicubic -> PSNR: {final_bicubic_psnr:.2f} dB | SSIM: {final_bicubic_ssim:.4f}")
        print(f"SRCNN   -> PSNR: {final_srcnn_psnr:.2f} dB | SSIM: {final_srcnn_ssim:.4f}")
        print(f"DIFFERENCE  -> PSNR: +{final_srcnn_psnr - final_bicubic_psnr:.2f} dB")
        print("==================================================================\n")

if __name__ == "__main__":
    test()