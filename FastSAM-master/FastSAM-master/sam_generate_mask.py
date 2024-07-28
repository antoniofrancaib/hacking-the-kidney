import os
import glob
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import label
import tifffile
import json

import random
import matplotlib.pyplot as plt
import time

from visualize import *

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()        
        print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        return result    
    return timed

# obtained from
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    # only show the top 5 masks
    # for ann in sorted_anns:
    for ann in sorted_anns[:5]:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

glob_scale = 1.0
base_path = 'scratch/hubmap/processed/cropped_train'
test_files = sorted(glob.glob(os.path.join(base_path, '*.tiff')))
for i in trange(0, 15, desc=f"Images"):
    # image_id = get_image_id(train_files[i])
    image_id = get_image_id(train_files[i])
    tiff_images = glob.glob(f"{base_path}/{image_id}/*.tiff")
    skipped_tiles = []
    # find valid tiles
    valid_tiles = json.load(open(f"{base_path}/{image_id}/valid_tiles.json"))
    valid_tiles = [tile_name.split("/")[-1] for tile_name in valid_tiles]
    for tiff_image in tqdm(tiff_images, desc=f"Tiles"):
        if tiff_image.split("/")[-1] not in valid_tiles:   # skip invalid tiles
            continue
        image, shape = read_image(tiff_image, scale=glob_scale)
        if shape != (1024, 1024, 3):
            skipped_tiles.append(tiff_image)
            continue

        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        sam.to("cuda")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
        )
        generated_mask = mask_generator.generate(image)
        # save the mask to json file
        json.dump(generated_mask, open(f"{tiff_image.replace('tiff', 'json')}", "w"), cls=NumpyEncoder)
        print(f"saved {tiff_image.replace('tiff', 'json')}")

        # # plot the image and mask
        # plt.figure(figsize=(16, 8))
        # plt.imshow(image)
        # plt.tight_layout()
        # show_anns(generated_mask)
    json.dump(skipped_tiles, open(f"{base_path}/{image_id}/skipped_tiles.json", "w"), cls=NumpyEncoder)