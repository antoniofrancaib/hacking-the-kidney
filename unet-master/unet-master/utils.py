import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from functools import reduce
import timeit

import os
from tqdm import tqdm, trange
from aim import Run, Image

def dice_loss(y_pred, y_true, smooth=1e-5):
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()

    intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
    union = y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (union + smooth)))

    return loss.mean()

def criterion(y_pred, y_true, metrics, bce_weight=0.5):
    y_true = y_true.permute(0, 3, 1, 2).float()
    y_pred = y_pred.float()
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)

    y_pred = torch.sigmoid(y_pred)
    dice = dice_loss(y_pred, y_true)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] = metrics.get('bce', 0) + bce.data.cpu().numpy() * y_true.size(0)
    metrics['dice'] = metrics.get('dice', 0) + dice.data.cpu().numpy() * y_true.size(0)
    metrics['loss'] = metrics.get('loss', 0) + loss.data.cpu().numpy() * y_true.size(0)

    return loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Apply mixup to a batch of images and masks using PyTorch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y, lam

def train(train_loader, val_loader, device, net, optimizer, scheduler, args, thres_optimizer=None, thres_scheduler=None):
    for epoch in trange(args.epochs, desc="epoch"):
        start_time = timeit.default_timer()

        # Each epoch has a training and validation
        metrics = dict()
        net.train() # Set model to training mode
        for index, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)

            # Apply mixup
            mixed_images, mixed_masks, _ = mixup_data(images, masks, 0.2)

            # forward
            outputs = net(mixed_images)
            
            # if type is ordered dict, use the first output
            if type(outputs) == dict:
                # primary loss from the last layer, use the first output
                primary_loss = criterion(outputs['out'], mixed_masks, metrics, args.bce_weight_ratio)
                # auxiliary loss from the intermediate layer
                aux_loss = criterion(outputs['aux'], mixed_masks, metrics, args.bce_weight_ratio)

                # obtain total loss
                loss = primary_loss + 0.4 * aux_loss  # 0.4 is a commonly used weight
            else:
                loss = criterion(outputs, mixed_masks, metrics, args.bce_weight_ratio)
            
            optimizer.zero_grad() # zero the parameter gradients
            thres_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thres_optimizer.step()
            

        print('\repoch %3d/%3d batch %3d/%3d train ' % (epoch+1, args.epochs, index, len(train_loader)), end='')
        print(' '.join(['%s %5.3f' % (k, metrics[k] / (index * args.batch_size)) for k in metrics.keys()]), end='')

        if args.run:
            for k in metrics.keys():
                args.run.track(value=metrics[k] / index * args.batch_size, name=k, epoch=epoch, context={"subset": "train"})
            args.run.track(value=net.threshold, name="threshold", epoch=epoch, context={"subset": "train"})
            print(' threshold => %5.3f' % (net.threshold.detach().cpu().numpy()))
            
        scheduler.step()
        for param_group in optimizer.param_groups:
            print(' lr %1.0e' % (param_group['lr']), end='')

        if val_loader:
            metrics = dict()
            net.eval() # Set model to evaluate mode
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                with torch.no_grad():
                    outputs = net(images)

            if type(outputs) == dict:
                # only primary loss from the last layer for validation
                loss = criterion(outputs['out'], masks, metrics, args.bce_weight_ratio)
            else:
                loss = criterion(outputs, masks, metrics)

            print(' val ', end='')
            print(' '.join(['%s %5.3f' % (k, metrics[k] / len(val_loader)) for k in metrics.keys()]), end='')
            for k in metrics.keys():
                if args.run:
                    args.run.track(value=metrics[k] / len(val_loader), name=k, epoch=epoch, step=index, context={"subset": "val"})
            
            if epoch % args.img_interval == 0:
                test(val_loader, device, net, args, epoch)

        print(' %4.1fsec' % (timeit.default_timer() - start_time))

        if args.model_path:
            torch.save(net.state_dict(), args.model_path)

    return net

def test(test_loader, device, net, args, epoch=0, num_images=3):
    net.eval()
    for index, (images, masks) in enumerate(tqdm(test_loader, desc="validating"), 1):
        images = images.to(device)
        with torch.no_grad():
            outputs = net(images)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Plot the image on the first subplot
        axes[0].imshow(images[0].cpu().numpy().transpose(1, 2, 0))

        # Plot the mask on the second subplot
        axes[1].imshow(masks[0].cpu().numpy().transpose(1, 2, 0).squeeze())

        # The third subplot is empty for now
        # conver to binary mask
        if type(outputs) == dict:
            predicted_mask = outputs['out'][0].cpu().numpy().transpose(1, 2, 0)
        else:
            predicted_mask = outputs[0].cpu().numpy().transpose(1, 2, 0)

        # normalize
        min_value = predicted_mask.min()
        data_range = predicted_mask.max() - min_value
        normalized_mask = (predicted_mask - min_value) / data_range
        normalized_mask = (normalized_mask * 255).astype('uint8')
        # apply threshold
        normalized_mask = (normalized_mask > (net.threshold.detach().cpu().numpy()*255)).astype('uint8')
        im = axes[2].imshow(normalized_mask)
        # plt.colorbar(im, ax=axes[2])

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        path = f'scratch/kidney_dataset/figure/{args.run_name}_{args.cur_time}'
        os.makedirs(path, exist_ok=True)
        fig.savefig(f'{path}/val_{epoch+1}_{index}.png')

        # Track the image if args.run is set
        if args.run:
            args.run.track(Image(fig), name="val_image", epoch=epoch, step=index)

        if index == num_images:
            break

    return net
