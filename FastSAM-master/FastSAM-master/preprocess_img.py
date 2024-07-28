"""
Author: Lingao Xiao
Date: 10/8/2023
AI Tools declaration: GPT-4
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import label
import json
import re

import random

from visualize import *

# For REPRODUCIBILITY https://pytorch.org/docs/stable/notes/randomness.html
def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_and_resize_polygons(image_np, mask_np, file_path=None, padding=20):
    """
    For training data, we need to crop the image and mask into smaller images of size 224x224.
    """
    image = torch.tensor(image_np).permute(2, 0, 1)  # HxWxC to CxHxW
    mask = torch.tensor(mask_np.squeeze())  # Remove the singleton dimension

    # find contours in the mask
    contours, _ = cv2.findContours(mask_np.squeeze().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_resized_images = []

    for i, cnt in enumerate(tqdm(contours, desc=f"crop_n_resize")):
        # get bounding rectangle or rotated rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        x, y, w, h = cv2.boundingRect(cnt)

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding

        # Determine the maximum dimension to make the bounding box square
        max_dim = max(w, h)

        # Adjust x, y to keep the bounding box centered
        x_center = x + w // 2
        y_center = y + h // 2
        x = x_center - max_dim // 2
        y = y_center - max_dim // 2

        # Check if bounding box coordinates are within image boundaries
        height, width = image.shape[1:3]
        x = max(0, min(x, width - max_dim))
        y = max(0, min(y, height - max_dim))

        # Use max_dim for both width and height to get a square bounding box
        cropped_img = image[:, y:y+max_dim, x:x+max_dim]

        cropped_img_pil = Image.fromarray(cropped_img.permute(1, 2, 0).byte().numpy())

        transform = transforms.Compose([transforms.Resize((224, 224))])
        resized_img_pil = transform(cropped_img_pil)
        
        if file_path:   # save file to path
            os.makedirs(file_path, exist_ok=True)
            resized_img_pil.save(f'{file_path}/{i:03}.png')

        resized_img = transforms.ToTensor()(resized_img_pil)

        cropped_resized_images.append(resized_img)

    return cropped_resized_images

def is_mostly_black_or_white(img_array, threshold=100, accept_ratio=0.98):
    # If the image has more than one channel (e.g., RGB), convert it to grayscale
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=-1).astype(np.uint8)

    total_pixels = img_array.size

    # Count black and white pixels
    black_pixels = np.sum(img_array < threshold)
    white_pixels = np.sum(img_array > 255 - threshold)

    # Check if the image is mostly black or white
    if black_pixels / total_pixels > accept_ratio:
        return True
    elif white_pixels / total_pixels > accept_ratio:
        return True
    elif (black_pixels+white_pixels) / total_pixels > accept_ratio:
        return True
    else:
        return False

def identify_valid_tiles(base_path, scale = 0.25):
    images = glob.glob(f"{base_path}/*.tiff")
    valid_tiles = []  # List to store paths of valid tiles
    for image_path in images:
        image = read_image(f"{image_path}", scale = scale)[0]
        if not is_mostly_black_or_white(image): # register valid tiles
            valid_tiles.append(image_path)
            
    print(f"Finished identifying valid tiles. In total {len(valid_tiles)} valid tiles out of {len(images)} tiles.")
    # write to json file
    json.dump(valid_tiles, open(f"{base_path}/valid_tiles.json", "w"))

def crop_large_image(image_path, output_dir, tile_size=(1024, 1024), shift=512):
    """
    For test data, we need to crop the image into smaller images of size 1024x1024 to inference
    """
    image = tifffile.imread(image_path)

    if len(image.shape) == 5:
        image = image.squeeze().squeeze().transpose(1, 2, 0)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    height, width = image.shape[:2]
    print("Image shape:", image.shape)

    # some tiles may be missing on the right side or bottom side
    # num_tiles_y = -(-height // tile_size[0])
    # num_tiles_x = -(-width // tile_size[1])
    # fix the bug when some tiles are missing by using ceil
    num_tiles_y = math.ceil(height / tile_size[0])+1
    num_tiles_x = math.ceil(width / tile_size[1])+1
    print("Number of tiles:", num_tiles_y, num_tiles_x)
    
    valid_tiles = []  # List to store paths of valid tiles

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            start_y = i * tile_size[0]
            end_y = min(start_y + tile_size[0], height)
            start_x = j * tile_size[1]
            end_x = min(start_x + tile_size[1], width)

            cropped_image = image[start_y:end_y, start_x:end_x]
            output_path = f"{output_dir}/tile_{i}_{j}.tiff"
            os.makedirs(output_dir, exist_ok=True)

            if not is_mostly_black_or_white(cropped_image): # register valid tiles
                valid_tiles.append(output_path)

            # cv2.imwrite(output_path, cropped_image)
            tifffile.imwrite(output_path, cropped_image)

    print(f"Finished cropping {image_path} into {tile_size[0]}x{tile_size[1]} tiles.")
    print(f"Number of valid tiles: {len(valid_tiles)}")
    json.dump(valid_tiles, open(f"{output_dir}/valid_tiles.json", "w"))

def extract_info(filename):
    """
    Luojie's Part
    """
    # Regular expression to match the pattern
    prefix = './scratch/hubmap/processed/cropped_train/'
    pattern = prefix + r'(\w+)/tile_(\d+)_(\d+).tiff'
    match = re.match(pattern, filename)

    if match:
        identifier, row, col = match.groups()
        return identifier, int(row), int(col)
    else:
        raise ValueError(f"Filename {filename} doesn't match the expected format")

def visualize_check_image(data_directory, json_file):
    """
    Luojie's Part
    """
    valid = json.load(open(json_file, "r"))
    file_pattern = "*.tiff"  # You can use "*" to match all files or specify a specific pattern

    # Use glob.glob() to get a list of matching files
    images = glob.glob(f"{data_directory}/{file_pattern}")
    images = [s.replace('\\','/') for s in images]

    # [1:] to discard to folder name
    valid_row_col = [extract_info(valid_image)[1:] for valid_image in valid]
    # print(valid_row_col)

    # max row and col
    no_row = max([row for row, _ in valid_row_col]) + 1
    no_col = max([col for _, col in valid_row_col]) + 1

    scale = 0.25

    full_image = []
    for row in range(no_row - 1):
        row_image = []
        for col in range(no_col - 1):
            image = read_image(f"{data_directory}/tile_{row}_{col}.tiff", scale = scale)[0]
            
            # border
            image[0,:,:] = 0
            image[-1,:,:] = 0
            image[:,0,:] = 0
            image[:,-1,:] = 0
            
            if (row, col) in valid_row_col:
                # print("valid", end = ',')
                image[:,:,0] = image[:,:,0]/200*150 #blue channel
                image[:,:,1] = image[:,:,1]/140*150 #green channel
                image[:,:,2] = image[:,:,2]/200*150 #red channel
                
            row_image.append(image)
        row_image = np.hstack(row_image)
        print(row_image.shape)
        full_image.append(row_image)
    full_image = np.vstack(full_image)
    print(full_image.shape)
            
    # result_image = np.vstack(row_image)
    cv2.imwrite(f"{base_path}/combined_image_{scale}.jpg", full_image)

def merge_mask(image_folder_path, no_row, no_col):
    images = glob.glob(f"{image_folder_path}/*.jpg")
    masks = glob.glob(f"{image_folder_path}/*.pt")

    full_image = []
    full_mask = []
    for row in trange(no_row, desc=f"{image_folder_path.split('/')[-1]}"):
        row_image = []
        row_mask = []
        for col in range(no_col):
            image_path = os.path.join(image_folder_path, f'tile_{row}_{col}.jpg')
            if image_path in images:
                image = cv2.imread(image_path)
            else:
                image = np.full((1024, 1024, 3), 255, dtype = 'uint8')
            
            # border for visualization only
            image[0,:,:] = 0
            image[-1,:,:] = 0
            image[:,0,:] = 0
            image[:,-1,:] = 0
                
            row_image.append(image)

            # has mask
            temp_mask = torch.zeros((1024, 1024), dtype=torch.bool)
            if os.path.join(image_folder_path, f'tile_{row}_{col}.pt') in masks:
                mask = torch.load(os.path.join(image_folder_path, f'tile_{row}_{col}.pt'))
                # create tensor of all False of shape (1024, 1024)
                for m in mask:
                    if (np.sum(m==True) / (1024*1024)) < 0.05:
                        continue
                    elif (np.sum(m==True) / (1024*1024)) > 0.1:
                        continue
                    else:
                        temp_mask = temp_mask | m   # merge mask

            row_mask.append(temp_mask)

        # process image
        row_image = np.hstack(row_image)
        full_image.append(row_image)

        # process masks
        row_mask = np.hstack(row_mask)
        full_mask.append(row_mask)

    full_image = np.vstack(full_image)
    full_mask = np.vstack(full_mask)
    torch.save(full_mask, f"{image_folder_path}/combined_mask.pt")
    
    cv2.imwrite(f"{image_folder_path}/combined_image.jpg", full_image, [cv2.IMWRITE_JPEG_QUALITY,50])
    cv2.imwrite(f"{image_folder_path}/combined_mask.jpg", full_mask*255, [cv2.IMWRITE_JPEG_QUALITY,10])
    

# base_path=f"./scratch/hubmap/"
# train_files = sorted(glob.glob(os.path.join(base_path, 'train/*.tiff')))

#     # Crop and resize
#     results = crop_and_resize_polygons(image, mask, f'visualization/cropped_resized/{image_id}')

# for i in range(0, 5):
#     image_id = get_image_id(test_files[i])
#     crop_large_image(os.path.join(base_path, f'test/{image_id}.tiff'), f'scratch/hubmap/processed/cropped_test/{image_id}')

# # ======================= CREATE MASK =======================
# for i in range(0, 15):
#     list_ = ["54f2eec69", "e79de561c"]
#     image_id = get_image_id(train_files[i])
#     if image_id not in list_:
#         continue
#     crop_large_image(f"{train_files[i]}", f'./scratch/hubmap/processed/cropped_train/{image_id}')

#     # base_path=f"./scratch/hubmap/processed/cropped_test/{image_id}"
#     # identify_valid_tiles(base_path, scale = 0.25)
#     visualize_check_image(base_path, f"{base_path}/valid_tiles.json")

# ========================= MERGE MASK =========================
idx = 1
tiles_dict = {
    0: (25, 34),
    1: (38, 39),
}
image_id = get_image_id(train_files[idx])
# merge_mask(f"scratch/hubmap/mask/{image_id}", tiles_dict[1][0], tiles_dict[1][1])

# ======================= CROP AND RESIZE ACCORDING TO MERGED MASK =======================
base_path=f"./scratch/hubmap/"
image, shape = read_image(os.path.join(base_path, f'train/{image_id}.tiff'), scale=1.0)
mask = torch.load(f"scratch/hubmap/mask/{image_id}/combined_mask.pt")
results = crop_and_resize_polygons(image, mask, f'scratch/hubmap/cropped_for_training/{image_id}')