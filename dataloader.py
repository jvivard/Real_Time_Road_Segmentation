import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DrivableDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=(256, 512)):
        """
        Custom PyTorch Dataset for Drivable Space Segmentation.
        Args:
            data_dir (str): Root directory of the generated dataset.
            split (str): 'train' or 'val'.
            img_size (tuple): (height, width) of the input images.
        """
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        self.img_names = sorted(os.listdir(self.img_dir))
        
        # Hardcoded size to explicitly follow the finalized (512x256 width x height) plan
        # Note: albumentations and standard ML convention uses (height, width) so its 256x512
        self.height = img_size[0] 
        self.width = img_size[1]
        
        # Define Albumentations Pipelines based on split
        if split == 'train':
            # Aggressive augmentation to offset small ~700 frame dataset footprint
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.RandomBrightnessContrast(p=0.4),
                # Standardize input exactly to 256H x 512W
                A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        base_name = img_name.replace('.jpg', '')
        mask_name = f"{base_name}.png"
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # OpenCV loads in BGR, convert to RGB for neural networks
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Masks are single channel (0 or 1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Ensure mask is float tensor [H, W] and mathematically restrict to binary [0.0, 1.0] for PyTorch Loss functions
        mask = mask.float() / 255.0
        
        return image, mask

def get_dataloaders(data_dir, batch_size=8, num_workers=4):
    """
    Returns (train_loader, val_loader) configured for fast loading.
    """
    # Create dataset instances
    train_dataset = DrivableDataset(data_dir=data_dir, split='train')
    val_dataset = DrivableDataset(data_dir=data_dir, split='val')
    
    # Pack into DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, # Critical for faster GPU transfer
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2, # Inference doesn't need to backtrack, so double the batch size
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Very fast sanity check of the newly created dataloader
    print("Testing PyTorch DataLoader & Augmentations...")
    # Will point this directly to our dataset folder generated in Phase 1
    root = 'c:/Users/ggaka/Downloads/DrivableSpaceDataset'
    
    # Provide dummy fallback so this doesn't strictly crash if Phase 1 is still exporting
    if os.path.exists(root):
        train_loader, _ = get_dataloaders(root, batch_size=4, num_workers=0)
        try:
            images, masks = next(iter(train_loader))
            print("Successfully loaded batch!")
            print(f"Image tensor shape: {images.shape} (Expected: [4, 3, 256, 512])")
            print(f"Mask tensor shape: {masks.shape} (Expected: [4, 256, 512])")
            print(f"Mask unique values: {torch.unique(masks)}")
        except StopIteration:
            print("Dataset generation is still running. Folders exist but might be empty.")
    else:
        print("Waiting for Phase 1 to finish... the DrivableSpaceDataset folder does not exist yet.")
