import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model import SRCNN
from src.dataset import TrainDataset, EvalDataset
from src.utils import calc_psnr, AverageMeter

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    save_dir = Path(config['model']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # (9-1-5)
    model = SRCNN(num_channels=config['model']['in_channels']).to(device)
    
    train_dataset = TrainDataset(config['train']['train_data_path'])
    test_dataset = EvalDataset(config['train']['eval_data_path'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )
    
    """
        Adam optimizer is prefered instead of SGD, which is used in original paper.
        Learning Rate is 1e-4 for first and second layer, 1e-5 for last layer, as in the paper.
    """
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': config['train']['learning_rate'] * 0.1}
    ], lr=config['train']['learning_rate'])
    
    criterion = nn.MSELoss()
    
    best_psnr = 0.0
    
    epochs = config['train']['epochs']
    print(f"Starting training: {epochs} Epoch, Data: {len(train_dataset)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = AverageMeter()
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward Pass
                preds = model(inputs)
                
                # 33x33 -> 21x21
                diff = targets.shape[2] - preds.shape[2] # 33 - 21 = 12
                crop = diff // 2 # 6 pixel
                
                # Take the center of the target image
                if crop > 0:
                    targets_cropped = targets[:, :, crop:-crop, crop:-crop]
                else:
                    targets_cropped = targets
                
                loss = criterion(preds, targets_cropped)
                
                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss.update(loss.item(), inputs.size(0))
                pbar.set_postfix({'loss': f"{epoch_loss.avg:.6f}"})
                pbar.update(1)
        
        # Validation
        model.eval()
        avg_psnr = validate(model, test_dataset, device)
        print(f"Epoch: {epoch+1}, Train Loss: {epoch_loss.avg:.5f} | Test PSNR: {avg_psnr:.2f} dB")
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"Best model saved. ({best_psnr:.2f} dB)")

def validate(model, dataset, device):
    total_psnr = 0.0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in DataLoader(dataset, batch_size=1):
            inputs, targets = inputs.to(device), targets.to(device)
            
            preds = model(inputs)
            
            diff = targets.shape[2] - preds.shape[2]
            crop = diff // 2
            if crop > 0:
                targets = targets[:, :, crop:-crop, crop:-crop]
                
            preds = torch.clamp(preds, 0.0, 1.0)
            
            psnr = calc_psnr(preds, targets)
            total_psnr += psnr.item()
            count += 1
            
    return total_psnr / count if count > 0 else 0

if __name__ == "__main__":
    train()