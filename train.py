import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataloader import get_dataloaders
from model import LightweightUNet

# Configuration
DATA_DIR = '/content/drive/MyDrive/DrivableSpaceDataset'
EPOCHS = 80
BATCH_SIZE = 8 # Safe starting point for consumer GPUs to prevent CUDA out of memory
BASE_FILTERS = 32
LR = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_MODEL_PATH = 'best_unet_model.pth'

# --- Custom Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Convert logits to probabilities (0 to 1)
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2): # Alpha > 0.5 critically boosts the minority road class
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce_with_logits(logits, targets)
        pt = torch.exp(-bce_loss)  # pt is the probability of the correct class
        
        # Dynamically apply alpha to positive class and (1-alpha) to negative class
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        
    def forward(self, logits, targets):
        # Masks shape: [B, H, W], but model outputs [B, 1, H, W]
        # We add an axis to targets so they align numerically
        targets = targets.unsqueeze(1) 
        
        # 0.3 Focal / 0.7 Dice blend to prevent Focal instability on imbalanced datasets
        return (0.3 * self.focal(logits, targets)) + (0.7 * self.dice(logits, targets))

# --- Metric: mIoU ---
def calculate_iou(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.unsqueeze(1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return 1e-6
    return (intersection / union).item()

# --- Main Training Loop ---
def main():
    print(f"Initializing Training on {DEVICE}...")
    
    train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    model = LightweightUNet(n_channels=3, n_classes=1, base_filters=BASE_FILTERS).to(DEVICE)
    
    criterion = ComboLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # AdamW with explicit weight decay
    scaler = GradScaler() # For Mixed Precision (Speed!)
    
    # State Persistence Logic for Colab fault-tolerance
    RESUME_CHECKPOINT = '/content/drive/MyDrive/checkpoint_epoch_X.pth'  # change X to resume specific epoch
    if os.path.exists(RESUME_CHECKPOINT):
        checkpoint = torch.load(RESUME_CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        print(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        start_epoch = 1
        best_iou = 0.0
    
    # 5-Epoch Linear Warmup into Cosine Annealing to stabilize raw uninitialized gradients
    warmup_epochs = 5
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 15)
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc='Training')
        for images, masks in train_pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast enables hardware acceleration (Tensor Cores) if running on modern GPUs
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_pred_drivable = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for images, masks in val_pbar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, masks)
                    
                val_loss += loss.item()
                val_iou += calculate_iou(logits, masks)
                
                # Sanity Check: What % of pixels is the model actually guessing is a road?
                pred_binary = (torch.sigmoid(logits) > 0.5).float()
                val_pred_drivable += pred_binary.mean().item()
                
        # Epoch Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_pred_perc = val_pred_drivable / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_iou:.4f} | Pred Road: {avg_pred_perc*100:.2f}%")
        
        # Save Best Checkpoint unconditionally based purely on Validation mIoU performance
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            print(f"✨ New Best Model! Saving checkpoint to {BEST_MODEL_PATH}...")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            
        # Save snapshot every 10 Epochs safely containing entirely network & optimizer tensors to prevent data loss
        if epoch % 10 == 0:
            print(f"Saving checkpoint state for Epoch {epoch}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, f'/content/drive/MyDrive/checkpoint_epoch_{epoch}.pth')
            
    print("\nTraining Complete! Ready for Inference testing.")

if __name__ == '__main__':
    # Validate dataset exists before starting
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Cannot find {DATA_DIR}. Please finish Phase 1 visually generating masks first.")
    else:
        main()
