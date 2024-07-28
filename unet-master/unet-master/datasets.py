import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

import torch
from tqdm import tqdm, trange
import glob
import pickle

import tifffile
import cv2

VALIDATION_SPLIT = 0.01

class KidneyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.files = glob.glob(f"{path}/*.pkl")
        self.images = []
        self.masks = []
        
        for file_path in tqdm(self.files, desc='Training images'):
            image_mask_dict = pickle.load(open(f'{file_path}', 'rb'))
            for key, value in image_mask_dict.items():
                image = value['image']
                mask = value['mask']

                if np.sum(mask) > 0: # only take patches which are a 1
                    self.images.append(image)
                    self.masks.append(mask)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        
        # Apply transformations using albumentations to ensure consistency for both image and mask
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask

def read_image(image_file, scale=1.0):
    image = tifffile.imread(image_file).squeeze()
    if image.shape[0] == 3:
        image = np.transpose(image, (1,2,0))
    
    orig_shape = image.shape
    if scale != 1.0:
        image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    return image, orig_shape

class KidneyDatasetTest(Dataset):
    def __init__(self, path, mask_path=None):
        """
        path should be the path to the tiff file
        """
        self.large_image, self.image_shape = read_image(path)
        self.images = self.crop_image()
        self.masks = []
        # self.mean, self.std = self.get_dataset_normalization()
        self.image_id = os.path.basename(path).split('.')[0]

        if mask_path is not None:
            predicted_masks = pickle.load(open(mask_path, 'rb'))
            self.masks = self.merge_crops(predicted_masks, self.image_shape)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6090, 0.4145, 0.6645], std=[0.1277, 0.1694, 0.1009]),  # training set mean and std
            # transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.images)
    
    def get_dataset_normalization(self):
        """
        Returns the mean and std of the dataset
        """
        mean = np.mean(self.large_image, axis=(0, 1))
        std = np.std(self.large_image, axis=(0, 1))
        return mean, std
    
    def crop_image(self, crop_size=(512, 512), overlap=0):
        """
        Crops the large image tensor into smaller patches of a specified size with overlapping.
        """
        crop_height, crop_width = crop_size
        overlap_height, overlap_width = int(crop_height * overlap), int(crop_width * overlap)
        step_height, step_width = crop_height - overlap_height, crop_width - overlap_width

        # Calculate padding
        pad_height = (step_height - self.large_image.shape[0] % step_height) % step_height
        pad_width = (step_width - self.large_image.shape[1] % step_width) % step_width

        # Pad the image
        padded_image = np.pad(self.large_image, 
                              ((0, pad_height), (0, pad_width), (0, 0)), 
                              mode='constant', constant_values=0)

        crops = []
        for y in range(0, padded_image.shape[0] - overlap_height, step_height):
            for x in range(0, padded_image.shape[1] - overlap_width, step_width):
                crop = padded_image[y:y + crop_height, x:x + crop_width]
                crops.append(crop)

        return crops
    
    def merge_crops(self, crops, original_size, crop_size=(512, 512), overlap=0):
        """
        Merges cropped masks back into a single large mask.
        """
        crop_height, crop_width = crop_size
        overlap_height, overlap_width = int(crop_height * overlap), int(crop_width * overlap)
        step_height, step_width = crop_height - overlap_height, crop_width - overlap_width

        height, width, _ = original_size
        merged_mask = np.zeros((height, width), dtype=crops[0].dtype)

        crop_idx = 0
        for y in range(0, height, step_height):
            for x in range(0, width, step_width):
                crop = crops[crop_idx]
                crop_idx += 1

                # Determine the region in the merged mask
                y1, x1 = max(y, 0), max(x, 0)
                y2, x2 = min(y + crop_height, height), min(x + crop_width, width)

                # Handle overlap by averaging
                merged_crop = merged_mask[y1:y2, x1:x2]
                overlap_crop = crop[:y2 - y1, :x2 - x1]

                # Merging with averaging in overlapping areas
                merged_mask[y1:y2, x1:x2] = np.where(merged_crop == 0, overlap_crop, (merged_crop + overlap_crop) / 2)

        return merged_mask


    def __getitem__(self, index):
        image = self.images[index]

        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image
    
    def visualize_crop(self, index):
        """
        Visualize the cropped image at the specified index.
        """
        if index >= len(self.images):
            raise ValueError(f"Index out of bounds. Dataset contains {len(self.images)} images.")

        crop = self.images[index]

        # If the image is a tensor, convert it to a NumPy array
        if isinstance(crop, torch.Tensor):
            crop = crop.permute(1, 2, 0).numpy()

        plt.imshow(crop)
        plt.title(f"Cropped Image at Index {index}")
        plt.axis('off')
        plt.savefig(f'scratch/kidney_dataset/figure_test/crop_{index}.png')

    def visualize_image_mask_pair(self, index):
        """
        Visualize the image and mask pair at the specified index side by side.
        """
        if index >= len(self.images):
            raise ValueError(f"Index out of bounds. Dataset contains {len(self.images)} images.")

        if index >= len(self.masks):
            raise ValueError(f"Index out of bounds. Dataset contains {len(self.masks)} masks.")

        image = self.images[index]
        mask = self.masks[index]

        # Convert tensors to NumPy arrays if necessary
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        if mask.shape[0] == 1:
            mask = np.squeeze(mask, 0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title(f"Cropped Image at Index {index}")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"Mask at Index {index}")
        ax[1].axis('off')

        output_dir = 'scratch/kidney_dataset/figure_test'
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(os.path.join(output_dir, f'crop_{index}.png'))

    def denoise_mask(self, mask, noise_size=3):
        """
        Denoises the mask by removing small objects.
        :param mask: The noisy mask that needs to be denoised.
        :param noise_size: The size of the structural element used for the morphological operation.
        :return: The denoised mask.
        """
        # Create the structural element for the morphological operation
        kernel = np.ones((noise_size, noise_size), np.uint8)

        # Perform opening to remove small objects
        denoised_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return denoised_mask

    def visualize_mask(self, scale_factor=0.5, denoise=-1):
        """
        Visualizes the mask by scaling it down.
        :param scale_factor: Factor to scale down the mask for visualization.
        """
        # Ensure scale factor is greater than 0 and less than or equal to 1
        scale_factor = max(0.01, min(scale_factor, 1.0))
        
        tag = ''
        if denoise > 0:
            self.masks = self.denoise_mask(self.masks, denoise)
            tag = f'_denoised_{denoise}'

        # Resize the mask
        small_height = int(self.masks.shape[0] * scale_factor)
        small_width = int(self.masks.shape[1] * scale_factor)
        small_mask = cv2.resize(self.masks, (small_width, small_height), interpolation=cv2.INTER_AREA)

        # Display the resized mask
        plt.imshow(small_mask, cmap='gray')
        plt.title(f'Mask Visualized at {scale_factor*100}% Scale')
        plt.axis('off')
        os.makedirs(f'scratch/kidney_dataset/test_masks/{self.image_id}', exist_ok=True)
        plt.savefig(f'scratch/kidney_dataset/test_masks/{self.image_id}/mask{tag}.png')

    def rle_encoding(self, mask):
        """
        Convert a mask into run-length encoding.
        mask: numpy array, 1 - mask, 0 - background
        Returns run length as string formated.
        """
        dots = np.where(mask.flatten() == 1)[0]  # Get indices of pixels with value 1
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return ' '.join([str(i) for i in run_lengths])

    def mask2rle(self, img, shape, small_mask_threshold):
        '''
        From: https://www.kaggle.com/code/kristianzeng/kidney-final-stage/notebook#Submission
        Convert mask to rle.
        img: numpy array <- 1(mask), 0(background)
        Returns run length as string formated
        
        pixels = np.array([1,1,1,0,0,1,0,1,1]) #-> rle = '1 3 6 1 8 2'
        pixels = np.concatenate([[0], pixels, [0]]) #[0,1,1,1,0,0,1,0,1,1,0]
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 #[ 1  4  6  7  8 10] bit change points
        print(runs[1::2]) #[4 7 10]
        print(runs[::2]) #[1 6 8]
        runs[1::2] -= runs[::2]
        print(runs) #[1 3 6 1 8 2]
        '''
        if img.shape != shape:
            h,w = shape
            img = cv2.resize(img, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.int8) 
        pixels = img.T.flatten()
        #pixels = np.concatenate([[0], pixels, [0]])
        pixels = np.pad(pixels, ((1, 1), ))
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        if runs[1::2].sum() <= small_mask_threshold:
            return ''
        else:
            return ' '.join(str(x) for x in runs)

    def export_rle_masks(self, denoise=100):
        """
        Convert all masks to RLE and prepare for submission.
        Returns a DataFrame with image IDs and corresponding RLE.
        """
        merged_mask = self.masks    # already merged at this point
        # rle = self.rle_encoding(self.denoise_mask(merged_mask, noise_size=denoise))
        denoised_mask = self.denoise_mask(merged_mask, noise_size=denoise)
        rle = self.mask2rle(denoised_mask, denoised_mask.shape, small_mask_threshold=0)
        return rle

def calculate_mean_std(dataset):
    # Convert images to tensor
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in tqdm(loader, desc="loader"):
        # Flatten the batch dimension, since we're considering all images in the dataset
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)
    
    # Final mean and std
    mean /= total_images
    std /= total_images
    
    return mean, std

# Assuming you have a dataset instance
dataset = KidneyDataset(path='path_to_your_data', transform=transforms.ToTensor())

if __name__ == '__main__':
    from glob import glob
    image_ids = glob("scratch/hubmap/test/*.tiff")
    rle_list = []
    for image_id in tqdm(image_ids, desc="image_ids"):
        id = os.path.basename(image_id).split('.')[0]
        dataset = KidneyDatasetTest(f"scratch/hubmap/test/{id}.tiff", f"scratch/kidney_dataset/test_masks/{id}.tiff.pkl")
        # for noise in range(40, 100):
        #     dataset.visualize_mask(denoise=noise)
        rle = dataset.export_rle_masks(denoise=100)
        rle_list.append({'img': id, 'predicted': rle})
    
    submission_df = pd.DataFrame(rle_list)
    submission_df.to_csv("scratch/kidney_dataset/submission.csv", index=False)

    # convert to submission format

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])

    # # entire training set
    # dataset = KidneyDataset('scratch/kidney_dataset/train', transform=transform)
    
    # # Calculate mean and std
    # mean, std = calculate_mean_std(dataset)

    # print(f"Mean: {mean}")
    # print(f"Std: {std}")

    # # validation split
    # total_len = len(dataset)
    # val_len = int(total_len * VALIDATION_SPLIT)
    # train_len = total_len - val_len  # ensures that train_len + val_len = total_len
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    # # create dataloaders
    # train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    # print('train set', len(train_set), 'val set', len(val_set))

    # for idx, (image_batch, mask_batch) in enumerate(val_loader):
    #     image_np = image_batch[0].numpy()
    #     mask_np = mask_batch[0].numpy()

    #     # if the image and mask have a batch dimension, remove it
    #     if image_np.shape[0] == 1:
    #         image_np = image_np[0]
    #     if mask_np.shape[0] == 1:
    #         mask_np = mask_np[0]

    #     # reshape
    #     if image_np.shape[0] == 3 or image_np.shape[0] == 1:
    #         image_np = image_np.transpose(1, 2, 0)
    #     if mask_np.shape[0] == 1:
    #         mask_np = mask_np[0]

    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    #     axes[0].imshow(image_np)
    #     axes[0].set_title('Image')
    #     axes[0].axis('off')

    #     axes[1].imshow(mask_np)
    #     axes[1].set_title('Mask')
    #     axes[1].axis('off')

    #     plt.savefig(f'scratch/kidney_dataset/figure/input_{idx}.png')

    #     plt.close(fig)