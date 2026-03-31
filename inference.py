import torch
import cv2
import numpy as np
import time
from model import LightweightUNet

# Config
MODEL_PATH = '/content/drive/MyDrive/nuscenes_project/best_unet_model.pth'
IMAGE_PATH = '/content/drive/MyDrive/DrivableSpaceDataset/val/images'
OUTPUT_PATH = '/content/drive/MyDrive/inference_results'
TARGET_SIZE = (512, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_model():
    model = LightweightUNet(n_classes=1).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle both old and new checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, TARGET_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    # Normalize same as training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(DEVICE), img_resized

def predict(model, tensor):
    with torch.no_grad():
        start = time.time()
        logits = model(tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
    
    prob = torch.sigmoid(logits)
    mask = (prob > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    fps = 1.0 / elapsed
    return mask, fps

def create_overlay(image, mask, gt_mask_path=None):
    overlay = image.copy()
    
    # Green overlay for predicted drivable
    green_mask = np.zeros_like(image)
    green_mask[mask == 1] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)
    
    # If ground truth available, show red for missed areas
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt > 127).astype(np.uint8)
        
        # Red = ground truth but not predicted (missed road)
        missed = np.zeros_like(image)
        missed[(gt_binary == 1) & (mask == 0)] = [0, 0, 255]
        overlay = cv2.addWeighted(overlay, 0.8, missed, 0.2, 0)
    
    return overlay

def run_inference(num_samples=20):
    model = load_model()
    print(f"Model loaded on {DEVICE}")
    
    # Warmup
    dummy = torch.randn(1, 3, 256, 512).to(DEVICE)
    for _ in range(10):
        _ = model(dummy)
    
    # FPS benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Benchmark FPS: {100/elapsed:.1f}")
    
    # Run on val images
    val_images = os.listdir(IMAGE_PATH)[:num_samples]
    fps_list = []
    
    for fname in val_images:
        img_path = os.path.join(IMAGE_PATH, fname)
        mask_path = os.path.join(
            IMAGE_PATH.replace('images', 'masks'),
            fname.replace('.jpg', '.png')
        )
        
        tensor, img_resized = preprocess(img_path)
        mask, fps = predict(model, tensor)
        fps_list.append(fps)
        
        overlay = create_overlay(img_resized, mask, mask_path)
        
        # Side by side: original | overlay
        side_by_side = np.hstack([img_resized, overlay])
        out_path = os.path.join(OUTPUT_PATH, f'result_{fname}')
        cv2.imwrite(out_path, side_by_side)
    
    print(f"Average inference FPS: {np.mean(fps_list):.1f}")
    print(f"Results saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    run_inference(num_samples=20)
