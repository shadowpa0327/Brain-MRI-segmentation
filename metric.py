import numpy as np
import torch
import torch.nn as nn
from engine import mIoU
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


def iou_metric(y_pred, y_true):
    intersec = (y_true * y_pred).sum()
    union = (y_true + y_pred).sum()
    iou = (intersec + 1e-10) / (union- intersec + 1e-10)
    return iou

def compute_dice(model, loader, threshold=0.3, device = 'cuda'):
    model.eval()
    valdice = 0
    valIoU = 0
    with torch.no_grad():
        for step, (data, target) in enumerate(loader, 1):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            IoU = iou_metric(out_cut, target.data.cpu().numpy())
            #tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, mode='binary', threshold=threshold)

            valdice += dice
            valIoU += IoU

    return valdice / step, valIoU / step

"""
Returns a tuple:
    (validation_loss, validation_mIoU)
"""
def validation_covid(model, loader, loss_func, device = 'cuda'):
    running_mIoU = 0
    running_loss = 0
    print(loader)
    with torch.no_grad():
        for step, (data, target) in enumerate(loader,1):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            mIoU_val = mIoU(outputs, target)
            loss = loss_func(outputs, target).item()
            
            running_loss += loss
            running_mIoU += mIoU_val

    return running_loss / step, running_mIoU / step