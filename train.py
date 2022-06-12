import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import albumentations as A

from model import build_model
from dataset import *
from metric import *



def get_args_parser():
    parser = argparse.ArgumentParser('DLP_Final_Project_Script', add_help=False)
    # basic setting
    parser.add_argument('--data-path', default='../lgg-mri-segmentation/kaggle_3m/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    # training setting
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch') # currently is not used
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Todo add learning rate scheduler
    

    # model choice
    parser.add_argument('--seg_struct', type=str, default='Unet', help="The type of segmentation model (Unet/Unet++)")
    parser.add_argument('--encoder', type=str, default='resnet50', 
                        help="The encoder to be use in the segmentation model check model.py for detailed")
    parser.add_argument('--is-swin', action='store_true')
    parser.set_defaults(is_swin=False)

    return parser


def train_model(args, model, train_loader, val_loader, loss_func, optimizer, scheduler):
    print(f"Starting Training for {args.epochs} epochs !")
    output_dir = Path(args.output_dir)
    device = args.device

    loss_history = []
    train_history = []
    val_history = []
    best_dice_coef = 0

    for epoch in range(args.start_epoch, args.epochs):
        model.train()

        losses = []
        train_iou = []

        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0            

            train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
            loss = loss_func(outputs, mask)
            losses.append(loss.item())
            train_iou.append(train_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mean_dice = compute_dice(model, val_loader, device=device)
        scheduler.step(val_mean_dice)
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_dice)

        


        if args.output_dir:
            checkpoint_paths = output_dir / 'checkpoint.pth'  
            torch.save({
                    'model': model.state_dict(),
                    'args' : args,
                    'epoch' : epoch
                }, checkpoint_paths
            )

        if best_dice_coef < val_mean_dice :
            if args.output_dir:
                checkpoint_paths = output_dir / 'best.pth'  
                torch.save({
                        'model': model.state_dict(),
                        'args' : args,
                        'epoch' : epoch
                    }, checkpoint_paths
                )
            
            best_dice_coef = val_mean_dice

        print('Epoch : {}/{} loss: {:.3f} - dice_coef: {:.3f} - val_dice_coef: {:.3f} - best_dice_coef {:.3f}'.format(
                                                                                epoch+1, args.epochs,np.array(losses).mean(),
                                                                               np.array(train_iou).mean(),
                                                                               val_mean_dice, best_dice_coef))
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(f"Epoch:{epoch} | Train_DICE:{train_history[-1]:.3f} | Val_DICE{val_history[-1]:.3f} | Best_DICE:{best_dice_coef:.3f}\n")
    return loss_history, train_history, val_history


def main(args):

    seed = args.seed
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Reading
    train_df, val_df, test_df = read_data_to_DataFrame(args.data_path) # read file into pd.Dataframe
    train_dataset, val_dataset, test_dataset = build_dataset(train_df, val_df, test_df) # convert it the dataset

    # build Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=26, shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=26, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=26, shuffle=False, num_workers=2)

    # build model
    model = build_model(seg_struct = args.seg_struct, 
                           encoder = args.encoder, 
                  decoder_channels = [256, 128, 64, 32, 16])

    model = model.to(device)

    if args.resume:
        print(f"load from checkpoint:{args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
         # testing
        test_iou = compute_dice(model, test_dataloader)
        print("Mean IoU: {:.3f}%".format(100*test_iou))
        return

    # optimizer and scheduler 
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    # training
    num_epochs = args.epochs
    loss_history, train_history, val_history = train_model(args = args, model = model, train_loader = train_dataloader, 
                                                            val_loader = val_dataloader, loss_func = bce_dice_loss, 
                                                            optimizer = optimizer, scheduler = scheduler, 
                                                        )
    
    # testing
    test_iou = compute_dice(model, test_dataloader)
    print("Mean dice coef: {:.3f}%".format(100*test_iou))




if __name__ == '__main__':
    config=[
        '--data-path', '../lgg-mri-segmentation/kaggle_3m/',
        '--epochs' , '100',
        '--output_dir', 'Unet++',
        '--lr', '1e-4',
        '--weight-decay','0.01',
        '--seg_struct', 'Unet++',
        '--encoder', 'timm-efficientnet-b5'
        #'--resume', 'Unet/checkpoint.pth',
    ]
    parser = argparse.ArgumentParser('DLP_Final', parents=[get_args_parser()])
    args = parser.parse_args(args=config)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
