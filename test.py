import cv2
import numpy as np

mask = cv2.imread(r'C:\Users\ggaka\Downloads\DrivableSpaceDataset\train\masks\scene-0916_e6e877f31dd447199b56cae07f86daad.png', cv2.IMREAD_GRAYSCALE)
print("Unique values:", np.unique(mask))
print("Max value:", mask.max())