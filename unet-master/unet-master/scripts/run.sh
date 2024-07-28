#!/bin/bash

PARAM="--run_exp unet --img_interval 5"
MODEL=unet
# CUDA_VISIBLE_DEVICES=0 python main.py --model $MODEL --run_name $MODEL $PARAM &
CUDA_VISIBLE_DEVICES=0 python main.py --model $MODEL --run_name ${MODEL}_mixup $PARAM --batch_size 16 &
MODEL=unet_fourier
# CUDA_VISIBLE_DEVICES=1 python main.py --model $MODEL --run_name $MODEL $PARAM &
CUDA_VISIBLE_DEVICES=1 python main.py --model $MODEL --run_name ${MODEL}_mixup $PARAM --batch_size 8 &

# ===================== SEE THE EFFECT OF BCE LOSS =======================
MODEL=unet
# CUDA_VISIBLE_DEVICES=0 python main.py --model $MODEL --run_name unet_bce_0 --bce_weight_ratio 0 $PARAM &
# CUDA_VISIBLE_DEVICES=1 python main.py --model $MODEL --run_name unet_bce_1 --bce_weight_ratio 1 $PARAM &

# CUDA_VISIBLE_DEVICES=0 python main.py --model $MODEL --run_name thres_lr_01 --thres_lr 0.01 $PARAM 
# CUDA_VISIBLE_DEVICES=1 python main.py --model $MODEL --run_name thres_lr_001 --thres_lr 0.001 $PARAM &

# CUDA_VISIBLE_DEVICES=0 python main.py --model unet --run_name unet --run_exp eval \
# --model_path scratch/kidney_dataset/model_unet_20231021-231446.pth --eval

wait