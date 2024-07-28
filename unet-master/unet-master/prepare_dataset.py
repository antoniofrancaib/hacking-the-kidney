import cv2
import datetime
import gc
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skimage.morphology
import sys
import tifffile
import json
import pickle

from tqdm import tqdm, trange

plot_full_image = True

# Number of glomeruli to display for each image
num_glom_display = 5

# Number of glomberuli to save as tiff files.
num_glom_save = 5

glob_scale = 0.25
base_path = 'scratch/hubmap'
#Directory Contents
print("Directory Contents")
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' )
print('\n'.join(os.listdir(base_path)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' )
train_files = sorted(glob.glob(os.path.join(base_path, 'train/*.tiff')))
print(f'Number of training images: {len(train_files)}')
print('\n'.join(train_files))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' )

#Test Images
test_files = sorted(glob.glob(os.path.join(base_path, 'test/*.tiff')))
print(f'Number of test images: {len(test_files)}')
print('\n'.join(test_files))

df_train = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_info = pd.read_csv(os.path.join(base_path,'HuBMAP-20-dataset_information.csv'))

#credit https://www.kaggle.com/harshsharma511/one-stop-understanding-eda-efficientunet
def rle_to_image(rle_mask, image_shape):
    """
    Converts an rle string to an image represented as a numpy array.
    Reference: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

    :param rle_mask: string with rle mask.
    :param image_shape: (width, height) of array to return
    :return: Image as a numpy array. 1 = mask, 0 = background.
    """

    # Processing
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    image = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        image[lo:hi] = 1

    return image.reshape(image_shape).T


def overlay_image_mask(image, mask, mask_color=(0,255,0), alpha=1.0):
    im_f= image.astype(np.float32)
#     if mask.ndim == 2:
#         mask = np.expand_dims(mask,-1)        
    mask_col = np.expand_dims(np.array(mask_color)/255.0, axis=(0,1))
    return (im_f + alpha * mask * (np.mean(0.8 * im_f + 0.2 * 255, axis=2, keepdims=True) * mask_col - im_f)).astype(np.uint8)


def overlay_image_mask_original(image, mask, mask_color=(0,255,0), alpha=1.0):
    return  np.concatenate((image, overlay_image_mask(image, mask)), axis=1)

def get_image_id(image_file):
    return os.path.splitext(os.path.split(image_file)[1])[0]


def read_image(image_file, scale=1.0):
    image = tifffile.imread(image_file).squeeze()
    if image.shape[0] == 3:
        image = np.transpose(image, (1,2,0))
    
    orig_shape = image.shape
    if scale != 1.0:
        image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    return image, orig_shape


def read_mask(image_file, image_shape, scale=1.0):
    image_id = get_image_id(image_file)
    train_info = df_train.loc[df_train['id'] == image_id]
    rle = train_info['encoding'].values[0] if len(train_info) > 0 else None
    if rle is not None:
        mask = rle_to_image(rle, (image_shape[1], image_shape[0]))
        if scale != 1.0:
            mask = cv2.resize(mask, (0,0), fx=scale, fy=scale)
        return np.expand_dims(mask,-1)
    else:
        return None        

    
def read_image_mask(image_file, scale=1.0):
    image, image_shape = read_image(image_file, scale)
    mask = read_mask(image_file, image_shape, scale)
    return image, mask


def get_tile(image, mask, x, y, tile_size, scale=1.0):
    x = round(x * scale)
    y = round(y * scale)
    size = int(round(tile_size / 2 * scale))
    image_s = image[y-size:y+size, x-size:x+size, :] 
    mask_s = mask[y-size:y+size, x-size:x+size, :]
    return image_s, mask_s


def get_particles(mask, scale=1.0):
    """
    In summary: The function get_particles takes a binary image and an optional scale factor.
    It identifies and labels connected components (particles) in the image,
    extracts statistics about each particle, organizes this data into a pandas dataframe,
    sorts the particles based on their coordinates, assigns a unique number to each particle,
    and finally returns the dataframe.
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    df_particles = pd.DataFrame(dict(zip(['x','y','left','top','width','height','area'],
                               [(centroids[1:,0]) / scale,
                                (centroids[1:,1]) / scale,
                                (stats[1:,cv2.CC_STAT_LEFT]) / scale,
                                (stats[1:,cv2.CC_STAT_TOP]) / scale,
                                (stats[1:,cv2.CC_STAT_WIDTH]) / scale,
                                (stats[1:,cv2.CC_STAT_HEIGHT]) / scale,
                                (stats[1:,cv2.CC_STAT_AREA]) / (scale * scale)])))
    df_particles.sort_values(['x','y'], inplace=True, ignore_index=True)
    df_particles['no'] = range(len(df_particles))
    return df_particles


def analyze_image(image_file, overlay=True):
    image_id = get_image_id(image_file)
    image, image_shape = read_image(image_file, glob_scale)
    mask = read_mask(image_file, image_shape, glob_scale)
    mask_full = read_mask(image_file, image_shape, scale=1.0)
    df_glom = get_particles(mask_full, scale=1.0)
    df_glom['id'] = image_id
    del mask_full
    gc.collect()
    
    info = df_info[df_info['image_file'] == f'{image_id}.tiff']
    print(f'Image ID:        {image_id:}')
    print(f'Image Size:      {info["width_pixels"].values[0]} x {info["height_pixels"].values[0]}')
    print(f'Patient No:      {info["patient_number"].values[0]}')
    print(f'Sex:             {info["sex"].values[0]}')
    print(f'Age:             {info["age"].values[0]}')
    print(f'Race:            {info["race"].values[0]}')
    print(f'Height:          {info["height_centimeters"].values[0]} cm')
    print(f'Weight:          {info["weight_kilograms"].values[0]} kg')
    print(f'BMI:             {info["bmi_kg/m^2"].values[0]} kg/m^2')
    print(f'Laterality:      {info["laterality"].values[0]}')
    print(f'Percent Cortex:  {info["percent_cortex"].values[0]} %')
    print(f'Percent Medulla: {info["percent_medulla"].values[0]} %')
    
    save_path = 'visualization/'
    save_path += 'overlay/' if overlay else 'no_overlay/'
    save_path += f'{image_id}'
    os.makedirs(save_path, exist_ok=True)
    # Plot full image
    if plot_full_image:
        scale = 0.1
        image_small = cv2.resize(image, (0,0), fx=scale, fy=scale)
        mask_small = cv2.resize(mask, (0,0), fx=scale, fy=scale)
        mask_small = np.expand_dims(mask_small,-1) 
    
        plt.figure(figsize=(16, 16))
        img = overlay_image_mask(image_small, mask_small)
        plt.axis('off')
        # save image
        plt.imsave(f'{save_path}/full_image.png', img)

    return
    # plot glomeruli images, save all
    for i in range(len(df_glom)):
        image_s, mask_s = get_tile(image,mask, df_glom['x'][i], df_glom['y'][i], 1000, scale=glob_scale)
        if overlay:
            ovl = overlay_image_mask(image_s, mask_s)
            cv2.imwrite(f'{save_path}/{image_id}_{i:03}.png', cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))    
        else:
            # check if image_s satisfies the condition: !_src.empty()
            if image_s.size != 0:
                cv2.imwrite(f'{save_path}/{image_id}_{i:03}.png', cv2.cvtColor(image_s, cv2.COLOR_RGB2BGR))    
    
    del image, mask
    gc.collect()
    return df_glom

def plot_glom(df, image_id, glom_no, train_or_test='train'):
    image, mask = read_image_mask(os.path.join(base_path, f'{train_or_test}/{image_id}.tiff'), scale=glob_scale)
    glom = df.loc[(df['id'] == image_id) & (df['no'] == glom_no)]
    im, ma = get_tile(image, mask, glom['x'].iloc[0], glom['y'].iloc[0], 1000, scale=glob_scale)
    del image, mask
    gc.collect()
    plt.figure(figsize=(16,8))
    plt.imshow(overlay_image_mask_original(im, ma))
    plt.title(f'Image: {image_id}, Glomeruli No: {glom_no}, Area: {glom["area"].iloc[0]}')
    
# ================================ LINGAO's NEW CODE =================================
def crop_large_image(image_file, output_dir, tile_size=(512, 512)):
    image_id = get_image_id(image_file)
    image, image_shape = read_image(image_file, scale=1.0)
    mask_full = read_mask(image_file, image_shape, scale=1.0)

    if len(image.shape) == 5:
        image = image.squeeze().squeeze().transpose(1, 2, 0)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    assert image.shape[:2] == mask_full.shape[:2], "Image and mask must have the same shape."

    height, width = image.shape[:2]
    print("Image shape:", image.shape)

    num_tiles_y = math.ceil(height / tile_size[0]) + 1
    num_tiles_x = math.ceil(width / tile_size[1]) + 1
    print("Number of tiles:", num_tiles_y, num_tiles_x)
    
    image_mask_dict = {}
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            start_y = i * tile_size[0]
            end_y = min(start_y + tile_size[0], height)
            start_x = j * tile_size[1]
            end_x = min(start_x + tile_size[1], width)

            cropped_image = image[start_y:end_y, start_x:end_x]
            cropped_mask = mask_full[start_y:end_y, start_x:end_x]

            image_mask_dict.update({f"tile_{i}_{j}": {"image": cropped_image, "mask": cropped_mask}})

    print(f"Finished cropping {image_file} into {tile_size[0]}x{tile_size[1]} tiles.")
    os.makedirs(f"{output_dir}", exist_ok=True)
    with open(f"{output_dir}/{image_id}.pkl", 'wb') as f:
        pickle.dump(image_mask_dict, f)
    
    del image, mask_full, image_mask_dict
    gc.collect()

if __name__ == '__main__':
    for i in trange(len(train_files)):
        # df_glom = pd.DataFrame()
        # df_glom = pd.concat([df_glom, analyze_image(train_files[i], overlay=True)], ignore_index=True)
        crop_large_image(train_files[i], "scratch/kidney_dataset/train")