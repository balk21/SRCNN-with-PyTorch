import h5py
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils import convert_rgb_to_y

# CONFIGURATION
SCALE = 3
PATCH_SIZE = 33
STRIDE = 14 # indicated in paper.

TASKS = [
    {
        'name': 'Training Data (T91)',
        'mode': 'train',
        'input_dir': ROOT / 'data/raw/T91',
        'output_file': ROOT / 'data/processed/train.h5'
    },
    {
        'name': 'Test Data (Set5)',
        'mode': 'eval',
        'input_dir': ROOT / 'data/raw/Set5',
        'output_file': ROOT / 'data/processed/test.h5'
    }
]

def prepare():
    print(f"Training data is preparing. (Scale: x{SCALE}) ===\n")

    for task in TASKS:
        input_dir = task['input_dir']
        output_path = task['output_file']
        mode = task['mode']
        
        if not input_dir.exists():
            print(f"Couldn't find directory: '{input_dir}'")
            continue
    
        output_path.parent.mkdir(parents=True, exist_ok=True)

        extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp'}
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(ext))
            
        if not image_files:
            print(f"Warning: There is no image in '{input_dir}'.")
            continue

        print(f">> {task['name']} is getting ready... ({len(image_files)} images)")
        
        lr_patches = []
        hr_patches = []
        
        if mode == 'eval':
            h5_file = h5py.File(output_path, 'w')
            lr_group = h5_file.create_group('lr')
            hr_group = h5_file.create_group('hr')

        for idx, img_path in enumerate(tqdm(image_files)):
            img = Image.open(img_path).convert('RGB')
    
            w, h = img.size
            w_new = (w // SCALE) * SCALE
            h_new = (h // SCALE) * SCALE
            img = img.resize((w_new, h_new), resample=Image.BICUBIC)
            
            # LR & HR 
            img_lr_small = img.resize((w_new // SCALE, h_new // SCALE), resample=Image.BICUBIC)
            img_lr = img_lr_small.resize((w_new, h_new), resample=Image.BICUBIC)
            
            # Channel Y
            img_hr_np = np.array(img).astype(np.float32)
            img_lr_np = np.array(img_lr).astype(np.float32)
            img_hr_y = convert_rgb_to_y(img_hr_np)
            img_lr_y = convert_rgb_to_y(img_lr_np)
            
            if mode == 'eval':
                # Save directly for test mode.
                lr_group.create_dataset(str(idx), data=img_lr_y)
                hr_group.create_dataset(str(idx), data=img_hr_y)
            else:
                # Patch for training mode.
                h, w = img_lr_y.shape
                for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                    for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                        lr_patches.append(img_lr_y[y : y+PATCH_SIZE, x : x+PATCH_SIZE])
                        hr_patches.append(img_hr_y[y : y+PATCH_SIZE, x : x+PATCH_SIZE])

        if mode == 'train':
            lr_patches = np.array(lr_patches)
            hr_patches = np.array(hr_patches)
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('lr', data=lr_patches)
                f.create_dataset('hr', data=hr_patches)
            print(f"Total of {len(lr_patches)} patches created.")
        
        elif mode == 'eval':
            h5_file.close()

        print(f"Saved: {output_path}\n")

    print("Done.")

if __name__ == '__main__':
    prepare()