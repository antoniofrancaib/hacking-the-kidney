import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import KidneyDatasetTest
from models import UNet
import argparse
import os
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pickle as pkl
from glob import glob
from tqdm import tqdm, trange

def load_model(model_path, device, model_type="unet", num_classes=1):
    if model_type == "unet":
        model = UNet(num_classes).to(device)
    elif model_type == "unet_fourier":
        model = UNet(num_classes, fourier_up=True).to(device)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def inference(model, dataloader, device):
    all_masks = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="inference"):
            images = images.to(device)
            outputs = model(images)
            # outputs can be a dictionary or tensor based on the model
            if type(outputs) == dict:
                outputs = outputs['out']

            outputs = outputs.cpu().numpy()  # Shape: (batch_size, channels, height, width)

            for predicted_mask in outputs:
                # normalize
                min_value = predicted_mask.min()
                data_range = predicted_mask.max() - min_value
                normalized_mask = (predicted_mask - min_value) / data_range
                normalized_mask = (normalized_mask * 255).astype('uint8')

                # apply threshold
                threshold = model.threshold.detach().cpu().numpy() * 255
                binary_mask = (normalized_mask > threshold).astype('uint8')

                # Squeeze the channel dimension if it's 1
                if binary_mask.shape[0] == 1:
                    binary_mask = np.squeeze(binary_mask, 0)


                all_masks.append(binary_mask)

    return np.array(all_masks)



def main(args):
    # prepare the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device for inference.' % device)
    model = load_model(args.model_path, device, args.model, args.num_classes)

    test_files = glob(os.path.join("scratch/hubmap/test", "*.tiff"))

    for test_file in tqdm(test_files, desc="test_files"):
        args.test_data_path = test_file # set the test data path

        dataset = KidneyDatasetTest(args.test_data_path)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

        masks = inference(model, dataloader, device)
        
        # save the masks
        masks = np.array(masks)
        os.makedirs("scratch/kidney_dataset/test_masks", exist_ok=True)
        pkl.dump(masks, open(f"scratch/kidney_dataset/test_masks/{os.path.basename(test_file)}.pkl", "wb"))
        
        
    # # Assuming you want to visualize some predictions
    # dataset.masks = masks

    # for i in trange(1000, 1050, desc="visualize"):
    #     dataset.visualize_image_mask_pair(i)

    breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="unet", type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--test_data_path', default="", type=str)
    args = parser.parse_args()
    print(vars(args))


    main(args)
    """
    python inference.py \
        --model unet_fourier \
        --model_path scratch/kidney_dataset/model_unet_fourier_mixup_20231114-130856.pth \
        --test_data_path scratch/hubmap/test/2ec3f1bb9.tiff
    """
