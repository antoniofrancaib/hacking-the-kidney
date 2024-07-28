from visualize import *
import torch
import cv2

# tile = "8_6"
# tile = "15_7"
tile = "11_8"
# img, shape = read_image(f"scratch/hubmap/processed/cropped_train/0486052bb/tile_{tile}.tiff", scale=1)
# cv2.imwrite(f"tile_{tile}.jpg", img)
mask = torch.load(f"scratch/hubmap/mask/0486052bb/tile_{tile}.pt")
breakpoint()
