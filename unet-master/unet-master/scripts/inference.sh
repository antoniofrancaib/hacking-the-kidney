#!/bin/bash

# python inference.py \
#     --model unet \
#     --model_path scratch/kidney_dataset/model_unet_mixup_20231114-130856.pth \
#     --test_data_path scratch/hubmap/test/2ec3f1bb9.tiff

python inference.py \
    --model unet_fourier \
    --model_path scratch/kidney_dataset/model_unet_fourier_mixup_20231114-130856.pth 