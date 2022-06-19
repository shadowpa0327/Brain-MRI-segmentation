import numpy as np
import torch
import torch.nn as nn
from engine import mIoU, multiclass_dice
import segmentation_models_pytorch as smp


def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union
def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss

def ce_dice_loss(pred, label):
    dice_loss = smp.losses.DiceLoss(mode='multiclass')(pred, label)
    ce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.0)(pred, label)
    return dice_loss + ce_loss

def iou_metric(y_pred, y_true):
    intersec = (y_true * y_pred).sum()
    union = (y_true + y_pred).sum()
    iou = (intersec + 0.1) / (union- intersec + 0.1)
    return iou

def compute_dice(model, loader, threshold=0.3, device = 'cuda'):
    model.eval()
    valdice = 0
    with torch.no_grad():
        for step, (data, target) in enumerate(loader, 1):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valdice += dice

    return valdice / step

"""
Returns a tuple:
    (validation_loss, validation_mIoU)
"""
def validation_covid(model, loader, loss_func, device = 'cuda'):
    running_mIoU = 0
    running_loss = 0
    running_dice = 0
    with torch.no_grad():
        for step, (data, target) in enumerate(loader,1):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            mIoU_val = mIoU(outputs, target)
            dice_val = multiclass_dice(outputs, target)
            loss = loss_func(outputs, target).item()
            
            running_loss += loss
            running_mIoU += mIoU_val
            running_dice += dice_val

    return running_loss / step, running_mIoU / step, running_dice / step

