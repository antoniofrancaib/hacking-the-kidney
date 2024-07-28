import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import timeit
import os

from datasets import *
from models import UNet
from utils import train, test
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101,  deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet101_Weights
from torch import nn

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import time

from aim import Run

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    seed_everything(42)
    # get current time
    args.cur_time = time.strftime("%Y%m%d-%H%M%S")
    if args.test:
        run = None
    else:
        run = Run(experiment=args.run_exp, log_system_params=False)
        run.name = args.run_name if args.run_name != '' else args.model

        # track hyperparameters
        hyperparams = dict()
        for key, value in vars(args).items():
            hyperparams.update({key: value})
        run["hparams"] = hyperparams

        args.run = run
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),  # This is the correct usage
        A.VerticalFlip(p=0.5),
        # A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.2), rotate=(-45, 45), shear=(-15, 15), p=0.5),
        # A.Perspective(scale=(0.2, 0.4), fit_output=True, p=0.5),
        A.Normalize(mean=[0.6090, 0.4145, 0.6645], std=[0.1277, 0.1694, 0.1009]),
        ToTensorV2(),
    ])
    
    # entire training set
    dataset = KidneyDataset('scratch/kidney_dataset/train', transform=transform)

    # validation split
    total_len = len(dataset)
    val_len = int(total_len * VALIDATION_SPLIT)
    train_len = total_len - val_len  # ensures that train_len + val_len = total_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

    if args.model == "unet":
        net = UNet(args.num_classes, use_threshold=args.use_threshold).to(device)
    elif args.model == "unet_fourier":
        net = UNet(args.num_classes, fourier_up=True, use_threshold=args.use_threshold).to(device)
    else:
        net = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # new_layer = nn.Sequential(
        #     nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.Sigmoid()
        # )
        # net.classifier[4] = new_layer   # replace the last layer with the new layer
    net.to(device)

    if args.model_path and os.path.exists(args.model_path):
        # load model weights
        state_dict = torch.load(args.model_path, map_location=device)
        if state_dict.get('threshold') is None:
            state_dict['threshold'] = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        net.load_state_dict(state_dict)
    
    if args.eval:
        test(val_loader, device, net, args, num_images=10)
        return

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    thres_optimizer = optim.Adam([net.threshold], lr=args.thres_lr)
    thres_scheduler = lr_scheduler.StepLR(thres_optimizer, step_size=args.step_size, gamma=0.1)

    test(val_loader, device, net, args, epoch=-1)
    net = train(train_loader, val_loader, device, net, optimizer, scheduler, args, thres_optimizer, thres_scheduler)
    test(val_loader, device, net, args, epoch=args.epochs-1)
    torch.save(net.state_dict(), f'scratch/kidney_dataset/model_{args.run_name}_{args.cur_time}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="unet", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=25, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--thres_lr', default=1e-2, type=float)
    parser.add_argument('--img_interval', default=10, type=int)

    parser.add_argument('--bce_weight_ratio', default=0.5, type=float)
    parser.add_argument('--use_threshold', action='store_true')
    parser.add_argument('--eval', action='store_true')
    
    parser.add_argument('--run_name', default="", type=str)
    parser.add_argument('--run_exp', default='unet', type=str)
    
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    print(vars(args))

    main(args)
